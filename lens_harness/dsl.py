"""Declarative graph DSL — let users *author* an agent without writing Python.

A graph spec is a JSON file. This module is the only place that knows the
format; it compiles a spec into the ordinary :class:`StateGraph`, so everything
downstream (runtime, trace, lens, diff) works unchanged. Authoring is a new
*entry point*, not a new kernel.

Zero dependencies: plain ``json``. A spec that fails to compile raises
:class:`SpecError` with a message naming the offending path in the file.

Spec format
-----------
.. code-block:: json

    {
      "name": "my-agent",
      "state": {
        "question": "str",
        "messages": ["list[dict]", "add"],
        "answer": "str",
        "steps": ["int", "overwrite"]
      },
      "llm": {"spec": "mock"},                        // optional, default mock
      "tools": {"calculator": "builtins:calculator"}, // optional registry
      "nodes": {
        "agent": {
          "type": "llm",
          "system": "You are a helpful calculator agent.",
          "max_steps": 3
        },
        "tools": {
          "type": "tool_loop",
          "pattern": "TOOL: <name> | <arg>",          // fixed wire format for now
          "final_prefix": "FINAL:"
        },
        "gate": {"type": "interrupt",
                 "question": "Approve?",
                 "resume_into": "apply"},
        "apply": {"type": "python", "fn": "my_module:do_apply"}
      },
      "entry": "agent",
      "edges": {
        "agent": {"condition": "tool_call",
                  "then": {"tools": "tools", "end": "__end__"}},
        "tools": "agent",
        "gate": "apply",
        "apply": "__end__"
      }
    }

Node types:

* ``llm`` — one OpenAI-compatible/mock call: renders ``system`` + the
  conversation from the ``messages`` channel, appends the reply. Tracks a
  ``steps`` counter and forces a final answer after ``max_steps``.
* ``tool_loop`` — executes the last assistant message's ``TOOL: name | arg``
  call against the spec's ``tools`` registry; appends a ``tool`` message.
* ``interrupt`` — writes the ``__interrupt__`` channel (human-in-the-loop);
  ``resume_into`` names the node to re-enter on resume.
* ``python`` — calls ``module:func``; the function must have the ordinary
  node signature ``(state, ctx) -> patch``.

Edge forms:

* ``"node": "target"`` — static edge (target may be ``"__end__"``).
* ``"node": {"condition": "...", "then": {key: target}}`` — conditional edge.
  Conditions are the built-in set (``tool_call``, ``always``) or
  ``python:module.func`` returning a key from ``then``.

State schema values are ``"type"`` or ``["type", "reducer"]`` where reducer is
one of ``add`` / ``add_unique`` / ``merge_dict`` / ``overwrite``.
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Callable

from .graph import END, StateGraph
from .llm import Message, load_llm
from .state import add, add_unique, merge_dict, overwrite

#: Reducers a spec may name. Nothing else is exposed — a typo must fail here,
#: not silently degrade to overwrite (see resolve_annotations' docstring).
REDUCERS = {"add": add, "add_unique": add_unique, "merge_dict": merge_dict,
            "overwrite": overwrite}

#: Channel type annotations the DSL accepts. All are plain typing forms the
#: state module can read; nothing beyond these is needed to author an agent.
_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "list[str]": list,
    "list[dict]": list,
    "list[dict[str, str]]": list,
    "dict[str, str]": dict,
    "dict[str, Any]": dict,
}

MAX_STEPS_DEFAULT = 3


class SpecError(ValueError):
    """A graph spec is malformed. The message names the offending key path."""


def _err(path: str, message: str) -> SpecError:
    return SpecError(f"{path}: {message}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_spec(path: str) -> dict[str, Any]:
    """Read a graph spec file. ``ValueError`` carries a JSON pointer hint."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(spec, dict):
        raise SpecError(f"{path}: spec must be a JSON object")
    return spec


def _resolve_ref(ref: str, kind: str) -> Callable:
    """``module:func`` → the callable. Import errors are spec errors."""
    if ":" not in ref:
        raise SpecError(f"{kind} {ref!r} must look like 'module:func'")
    module_name, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise SpecError(f"{kind} {ref!r}: {exc}") from exc


# --------------------------------------------------------------------------
# Built-in tools
# --------------------------------------------------------------------------
def _calculator(expr: str) -> str:
    import re as _re

    if not _re.fullmatch(r"[-+*/().\d\s]+", expr or ""):
        return "error: only arithmetic is supported"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - regex-guarded
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _word_count(text: str) -> str:
    return str(len((text or "").split()))


#: Tools a spec can name without importing anything.
BUILTIN_TOOLS = {
    "calculator": _calculator,
    "word_count": _word_count,
}


def _tool_table(spec: dict[str, Any]) -> dict[str, Callable[[str], str]]:
    """Resolve the spec's ``tools`` section into a name → callable table."""
    table: dict[str, Callable[[str], str]] = {}
    for name, ref in (spec.get("tools") or {}).items():
        if ref.startswith("builtin:"):
            fn = BUILTIN_TOOLS.get(ref[len("builtin:"):])
            if fn is None:
                raise _err(f"tools.{name}", f"unknown builtin tool {ref!r}; "
                           f"have {sorted(BUILTIN_TOOLS)}")
        elif ref.startswith("python:"):
            fn = _resolve_ref(ref[len("python:"):], f"tools.{name}")
        else:
            raise _err(f"tools.{name}",
                       "tool ref must start with 'builtin:' or 'python:'")
        table[name] = fn
    return table


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
def _build_schema(spec: dict[str, Any]) -> type:
    """A TypedDict built at runtime from the spec's ``state`` section.

    ``channels_from_annotations`` reads ``__annotations__``, so a dynamically
    created TypedDict behaves exactly like a hand-written one — reducer
    resolution, reserved channels, all of it is the ordinary code path.
    """
    from typing import Annotated, TypedDict

    state = spec.get("state")
    if not isinstance(state, dict) or not state:
        raise _err("state", "declare at least one channel")

    hints: dict[str, Any] = {}
    for name, decl in state.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise _err(f"state.{name}", "channel name must be a Python identifier")
        if isinstance(decl, list):
            if len(decl) != 2 or decl[1] not in REDUCERS:
                raise _err(f"state.{name}",
                           f'format is ["<type>", "<reducer>"]; reducer is one of '
                           f"{sorted(REDUCERS)}")
            base, reducer_name = decl
        else:
            base, reducer_name = decl, "overwrite"
        if base not in _TYPES:
            raise _err(f"state.{name}", f"unknown type {base!r}; have {sorted(_TYPES)}")
        hints[name] = (_TYPES[base] if reducer_name == "overwrite"
                       else Annotated[_TYPES[base], REDUCERS[reducer_name]])

    return TypedDict("SpecState", hints)  # type: ignore[misc]


# --------------------------------------------------------------------------
# Node builders
# --------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"TOOL:\s*(\w+)\s*\|\s*(.+)", re.DOTALL)


def _last_assistant(messages: list[dict[str, str]]) -> dict[str, str] | None:
    return next((m for m in reversed(messages or []) if m.get("role") == "assistant"), None)


def make_llm_node(llm: Any, *, system: str, max_steps: int,
                  answer_channel: str = "answer") -> Callable:
    """One LLM turn: render system + history, complete, append the reply.

    After ``max_steps`` turns the node forces a final answer from the last
    tool result instead of calling the model again — the visible loop guard
    the hand-written example also uses.
    """

    def llm_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        steps = state.get("steps", 0)
        messages_state = state.get("messages", [])
        if steps >= max_steps:
            final = state.get(answer_channel) or "max steps reached"
            return {
                "messages": [{"role": "assistant", "content": f"FINAL: {final}"}],
                "steps": steps + 1,
            }
        history = [Message(m.get("role", "user"), m.get("content", ""))
                   for m in messages_state]
        if system:
            history.insert(0, Message("system", system))
        response = llm.complete(history, on_token=ctx.emit_token)
        return {
            "messages": [{"role": "assistant", "content": response.text}],
            "steps": steps + 1,
        }

    llm_node.__name__ = "llm_node"
    return llm_node


def make_tool_node(tools: dict[str, Callable[[str], str]],
                   *, answer_channel: str = "answer") -> Callable:
    """Execute the last assistant message's ``TOOL: name | arg`` call."""

    def tool_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        last = _last_assistant(state.get("messages", []))
        match = TOOL_CALL_RE.search(last.get("content", "")) if last else None
        if match is None:
            return {}
        name, arg = match.group(1).strip(), match.group(2).strip()
        fn = tools.get(name)
        if fn is None:
            result = f"error: unknown tool {name!r}"
        else:
            result = fn(arg)
        return {
            "messages": [{"role": "tool", "content": f"{name}({arg}) = {result}"}],
            "tools_used": [name],
            answer_channel: result,
        }

    tool_node.__name__ = "tool_node"
    return tool_node


def make_interrupt_node(*, question: str, payload: dict[str, Any] | None = None,
                        resume_into: str) -> Callable:
    """Write ``__interrupt__``: the run pauses and waits for a human payload."""

    def interrupt_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        body = dict(payload or {})
        body.setdefault("question", question)
        return {
            "__interrupt__": {
                "node": ctx.node,
                "payload": body,
                "resume_schema": None,
                "resume_into": resume_into,
            }
        }

    interrupt_node.__name__ = "interrupt_node"
    return interrupt_node


def make_python_node(fn: Callable) -> Callable:
    return fn


# --------------------------------------------------------------------------
# Router conditions
# --------------------------------------------------------------------------
def cond_tool_call(state: dict[str, Any]) -> str:
    """Route key: ``tools`` when the last assistant message calls a tool."""
    last = _last_assistant(state.get("messages", []))
    if last and TOOL_CALL_RE.search(last.get("content", "")):
        return "tools"
    return "end"


def cond_always(state: dict[str, Any]) -> str:
    return "tools"


CONDITIONS: dict[str, Callable[[dict[str, Any]], str]] = {
    "tool_call": cond_tool_call,
    "always": cond_always,
}


def _build_condition(source: str, cond: str,
                     mapping: dict[str, str]) -> Callable:
    if cond.startswith("python:"):
        fn = _resolve_ref(cond[len("python:"):], f"edges.{source}.condition")
    elif cond in CONDITIONS:
        fn = CONDITIONS[cond]
    else:
        raise _err(f"edges.{source}.condition",
                   f"unknown condition {cond!r}; have {sorted(CONDITIONS)} or 'python:...'")
    def router(state: dict[str, Any]) -> str:
        key = fn(state)
        if key not in mapping:
            raise SpecError(
                f"router {cond!r} returned {key!r}, which is not a key of the "
                f"mapping {sorted(mapping)} declared for {source!r}"
            )
        return key
    return router


# --------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------
def build_graph(spec: dict[str, Any], *, llm_spec: str | None = None) -> StateGraph:
    """Compile a spec dict into a ``StateGraph`` — ordinary kernel path after."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    name = spec.get("name") or "spec-graph"

    schema = _build_schema(spec)
    graph = StateGraph(schema, name=name)

    tools = _tool_table(spec)
    llm_cfg = spec.get("llm") or {}
    if llm_spec:
        llm = load_llm(llm_spec)
    elif llm_cfg.get("replies") or llm_cfg.get("script") or "default" in llm_cfg:
        # An inline mock script: deterministic, offline, trace-reproducible —
        # the same MockLLM the hand-written examples drive their loops with.
        from .llm import MockLLM, ScriptedRule

        rules = [ScriptedRule(contains=r["contains"], reply=r["reply"])
                 for r in llm_cfg.get("script") or []]
        llm = MockLLM(
            script=rules,
            replies=list(llm_cfg.get("replies") or []),
            default=str(llm_cfg.get("default", "ok")),
        )
    else:
        llm = load_llm(str(llm_cfg.get("spec") or "mock"))

    nodes: dict[str, dict[str, Any]] = spec.get("nodes") or {}
    if not isinstance(nodes, dict) or not nodes:
        raise _err("nodes", "declare at least one node")

    for node_name, cfg in nodes.items():
        path = f"nodes.{node_name}"
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise _err(path, 'each node needs a "type"')
        ntype = cfg["type"]
        if ntype == "llm":
            system = str(cfg.get("system", ""))
            max_steps = int(cfg.get("max_steps", MAX_STEPS_DEFAULT))
            graph.add_node(node_name, make_llm_node(llm, system=system,
                                                    max_steps=max_steps))
        elif ntype == "tool_loop":
            graph.add_node(node_name, make_tool_node(tools))
        elif ntype == "interrupt":
            resume_into = cfg.get("resume_into")
            if not resume_into:
                raise _err(f"{path}.resume_into", "interrupt needs a resume_into node")
            graph.add_node(node_name, make_interrupt_node(
                question=str(cfg.get("question", "Approve?")),
                payload=cfg.get("payload"),
                resume_into=str(resume_into),
            ))
        elif ntype == "python":
            ref = cfg.get("fn")
            if not ref:
                raise _err(f"{path}.fn", "python node needs fn: 'module:func'")
            graph.add_node(node_name, make_python_node(
                _resolve_ref(str(ref), f"{path}.fn")))
        else:
            raise _err(f"{path}.type",
                       f"unknown node type {ntype!r}; have llm | tool_loop | interrupt | python")

    # -- edges ---------------------------------------------------------------
    edges = spec.get("edges")
    if not isinstance(edges, dict):
        raise _err("edges", 'declare edges: {"node": "target" or {"condition": ...}}')

    for source, target_spec in edges.items():
        path = f"edges.{source}"
        if isinstance(target_spec, str):
            if source == "entry":
                continue  # handled below
            target = "__end__" if target_spec == "__end__" else target_spec
            graph.add_edge(source, END if target == "__end__" else target)
        elif isinstance(target_spec, dict):
            cond = target_spec.get("condition")
            then = target_spec.get("then")
            if not cond or not isinstance(then, dict):
                raise _err(path, 'conditional edge needs {"condition": ..., "then": {...}}')
            mapping: dict[str, str] = {}
            descriptions: dict[str, str] = {}
            for key, tgt in then.items():
                real = END if tgt == "__end__" else tgt
                if real != END:
                    if real not in nodes:
                        raise _err(f"{path}.then.{key}",
                                   f"target node {real!r} is not declared")
                mapping[key] = real
                descriptions[key] = target_spec.get("descriptions", {}).get(key, key)
            router = _build_condition(source, str(cond), mapping)
            graph.add_conditional_edges(
                source, router, mapping,
                name=f"route_after_{source}",
                descriptions=descriptions,
            )
        else:
            raise _err(path, "edge must be a target string or a conditional object")

    entry = spec.get("entry") or edges.get("entry")
    if not entry:
        raise _err("entry", 'set "entry": "<node>" (or edges.entry)')
    graph.set_entry(str(entry))

    return graph


def run_spec(path: str, inputs: dict[str, Any], *, trace_dir: str = "traces",
             llm_spec: str | None = None) -> list:
    """Load a spec, run it once, write the trace — the CLI entry point."""
    import os

    from .checkpoint import CheckpointStore
    from .events import ListSink
    from .runtime import RunConfig, Runtime

    spec = load_spec(path)
    graph = build_graph(spec, llm_spec=llm_spec)
    sink = ListSink()
    store = CheckpointStore(os.path.join(trace_dir, "checkpoints"))
    runtime = Runtime(graph.compile(), store, sink)
    thread_id = os.path.splitext(os.path.basename(path))[0]
    runtime.invoke(inputs, RunConfig(thread_id=thread_id))

    os.makedirs(trace_dir, exist_ok=True)
    out = os.path.join(trace_dir, f"{runtime.emitter.run_id}.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for event in sink.events:
            fh.write(event.to_json() + "\n")
    print(f"\n[trace written to {out}]")
    return sink.events
