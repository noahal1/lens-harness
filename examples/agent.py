"""A first graph: the classic tool-calling loop, made completely visible.

The shape every agent framework hides::

    agent ──(tool_calls?)──▶ tools ──▶ agent ──(done?)──▶ END

This example is the M1 demo. It uses a deterministic mock LLM by default so the
trace is byte-for-byte reproducible; pass ``llm_spec='auto'`` to hit a real API.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from lens_harness.checkpoint import CheckpointStore
from lens_harness.events import ListSink
from lens_harness.graph import END, START, StateGraph
from lens_harness.llm import LLM, Message, MockLLM, load_llm
from lens_harness.runtime import RunConfig, Runtime
from lens_harness.state import add, add_unique


# --------------------------------------------------------------------------
# State schema — the annotations *are* the merge semantics.
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    #: Full conversation. ``add`` means every node appends; nobody overwrites.
    messages: Annotated[list[dict[str, str]], add]
    #: Which tools have run. ``add_unique`` keeps it a set with order.
    tools_used: Annotated[list[str], add_unique]
    answer: str
    #: Step budget — makes the loop terminate visibly instead of by luck.
    steps: int


# --------------------------------------------------------------------------
# A tiny tool registry — no plugin system, just a dict you can read.
# --------------------------------------------------------------------------
def _calculator(expr: str) -> str:
    """Evaluate a trivial arithmetic expression. Deliberately not ``eval``."""
    if not re.fullmatch(r"[-+*/().\d\s]+", expr or ""):
        return "error: only arithmetic is supported"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - input is regex-guarded
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _word_count(text: str) -> str:
    return str(len((text or "").split()))


TOOLS = {
    "calculator": ("Evaluate an arithmetic expression.", _calculator),
    "word_count": ("Count words in a text.", _word_count),
}


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Recognise ``TOOL: name | arg`` — a format simple enough to eyeball."""
    match = re.search(r"TOOL:\s*(\w+)\s*\|\s*(.+)", text, re.DOTALL)
    if not match:
        return None
    name, arg = match.group(1).strip(), match.group(2).strip()
    return (name, arg) if name in TOOLS else None


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def make_agent_node(llm: LLM):
    def agent_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        steps = state.get("steps", 0)
        if steps >= 3:
            return {
                "messages": [{"role": "assistant", "content": f"FINAL: {state.get('answer') or 'done'}"}],
                "answer": state.get("answer") or "done",
                "steps": steps + 1,
            }
        history = [Message(m["role"], m["content"]) for m in state.get("messages", [])]
        response = llm.complete(history, on_token=ctx.emit_token)
        return {
            "messages": [{"role": "assistant", "content": response.text}],
            "steps": steps + 1,
        }

    return agent_node


def tools_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Execute the last assistant message's tool call, if any."""
    last = next(
        (m for m in reversed(state.get("messages", [])) if m["role"] == "assistant"),
        None,
    )
    call = parse_tool_call(last["content"]) if last else None
    if call is None:
        return {}
    name, arg = call
    _desc, fn = TOOLS[name]
    result = fn(arg)
    return {
        "messages": [{"role": "tool", "content": f"{name}({arg}) = {result}"}],
        "tools_used": [name],
        "answer": result,
    }


def route_after_agent(state: dict[str, Any]) -> str:
    """The conditional edge — the function whose decision we now always show."""
    last = next(
        (m for m in reversed(state.get("messages", [])) if m["role"] == "assistant"),
        None,
    )
    if last and parse_tool_call(last["content"]):
        return "tools"
    return END


def build_graph(llm: LLM) -> Any:
    graph = StateGraph(AgentState, name="tool-loop-agent")
    graph.add_node(
        "agent",
        make_agent_node(llm),
        description="Ask the LLM what to do next (may request a tool).",
        tags=("llm",),
    )
    graph.add_node(
        "tools",
        tools_node,
        description="Run the tool the agent asked for and append the result.",
        tags=("side-effect",),
    )
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            END: END,
        },
        name="route_after_agent",
        descriptions={
            "tools": "the agent requested a tool call",
            END: "the agent produced a final answer",
        },
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# --------------------------------------------------------------------------
# Entry point used by the CLI: ``lens run examples.agent:build``
# --------------------------------------------------------------------------
DEMO_QUESTION = "What is 17*23? Then tell me how many words are in this sentence."


def build(
    *,
    inputs: dict[str, Any] | None = None,
    trace_dir: str = "traces",
    llm_spec: str | None = None,
) -> list:
    """Run the example and return the emitted events (also written to disk)."""
    import os

    question = (inputs or {}).get("question", DEMO_QUESTION)
    llm = (
        load_llm(llm_spec)
        if llm_spec
        else MockLLM(
            replies=[
                "Let me compute that. TOOL: calculator | 17*23",
                "Now the word count. TOOL: word_count | this sentence has nine words in it",
                "FINAL: 17*23 is 391, and the sentence has 9 words.",
            ],
        )
    )
    compiled = build_graph(llm)
    sink = ListSink()
    store = CheckpointStore(os.path.join(trace_dir, "checkpoints"))
    runtime = Runtime(compiled, store, sink)
    runtime.invoke(
        {
            "question": question,
            "messages": [{"role": "user", "content": question}],
            "steps": 0,
            "answer": "",
            "tools_used": [],
        },
        RunConfig(thread_id="demo"),
    )

    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{runtime.emitter.run_id}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for event in sink.events:
            fh.write(event.to_json() + "\n")
    print(f"\n[trace written to {path}]")
    return sink.events
