"""Studio: a conversational orchestration workbench on top of the ordinary kernel.

One session lives in process memory (single user, by design — see DESIGN §10,
"Studio 的写路径边界"). The chat stream carries *structured commands*, not free
text: every command mutates the JSON spec, the spec compiles through the same
:dfunc:`lens_harness.dsl.build_graph` the CLI uses, and every run writes the
same trace/checkpoint artifacts as `lens run --graph`. Studio has no executor
of its own — that is the whole point: the panel and the CLI cannot diverge.

Command envelope (POST /api/studio)::

    {"cmd": "init" | "add_node" | "set_node" | "remove_node" | "set_entry"
           | "set_edge" | "remove_edge" | "run" | "get_spec" | "reset",
     ...}

Every response is appended to the chat as an assistant message; failures carry
the real error text (SpecError, GraphError) verbatim — a mistake must be
visible where the user is typing, not swallowed.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

from . import dsl

#: Where studio runs write their traces. One dir per session, cleaned up with
#: the process (it is memory state anyway).
TRACE_DIR = "traces/studio"

#: Commands that may reference builtin node templates.
NODE_TYPES = ("llm", "tool_loop", "interrupt", "python",
              "transform", "map", "subgraph")

_TEMPLATES: dict[str, dict[str, Any]] = {
    "sequential": {
        "desc": "流水线：数据从入口节点一路改写到最后，最简单的入门模板",
        "desc_en": "Pipeline: data flows from the entry node through template rewrites to the end — the simplest starter.",
        "input": {"x": "hello"},
        "name": "studio-flow",
        "state": {"x": "str", "out": "str"},
        "nodes": {"a": {"type": "transform", "set": {"out": "got: {x}"}}},
        "entry": "a",
        "edges": {"a": "__end__"},
    },
    "llm_tool_loop": {
        "desc": "会算数的 agent：模型想用计算器就调工具，拿到结果继续想，直到给出最终答案",
        "desc_en": "A computing agent: the model calls the calculator tool when it wants, keeps reasoning, then gives a final answer.",
        "input": {"question": "17*23"},
        "llm": {"replies": ["Let me compute that. TOOL: calculator | 17*23",
                            "FINAL: computed."]},
        "tools": {"calculator": "builtin:calculator"},
        "name": "studio-agent",
        "state": {"question": "str",
                  "messages": ["list[dict[str, str]]", "add"],
                  "answer": "str", "steps": "int",
                  "tools_used": ["list[str]", "add_unique"]},
        "nodes": {"agent": {"type": "llm", "system": "You compute things."},
                  "tools": {"type": "tool_loop"}},
        "entry": "agent",
        "edges": {"agent": {"condition": "tool_call",
                            "then": {"tools": "tools", "end": "__end__"}},
                  "tools": "agent"},
    },
    "approval": {
        "desc": "人工审批：跑到关键步骤会暂停等你在界面上确认，批准后才继续执行",
        "desc_en": "Human approval: the run pauses at the gate until you approve in the UI, then continues.",
        "input": {"action": "delete prod db"},
        "name": "studio-approval",
        "state": {"action": "str", "decision": "str", "log": ["list[str]", "add"]},
        "nodes": {"gate": {"type": "interrupt", "question": "Approve?",
                           "resume_into": "apply"},
                  "apply": {"type": "transform",
                            "set": {"log": "did {action} ({decision})"}}},
        "entry": "gate",
        "edges": {"gate": "apply", "apply": "__end__"},
    },
}


class StudioSession:
    """One user's orchestration state. Not thread-safe beyond a coarse lock."""

    def __init__(self, trace_dir: str = TRACE_DIR):
        self.lock = threading.Lock()
        self.trace_dir = trace_dir
        self.spec: dict[str, Any] | None = None
        self.messages: list[dict[str, str]] = []
        self.runs: list[dict[str, Any]] = []
        self.default_input: dict[str, Any] = {}

    # -- chat ------------------------------------------------------------
    def say(self, role: str, text: str, extra: dict | None = None) -> dict:
        msg = {"role": role, "text": text, "ts": time.time()}
        if extra:
            msg.update(extra)
        self.messages.append(msg)
        return msg

    def _ok(self, text: str, **extra) -> dict:
        self.say("assistant", text, extra)
        return {"ok": True, "messages": self.messages[-1:], "spec": self.spec,
                "runs": self.runs}

    def _fail(self, err: Exception, context: str) -> dict:
        self.say("assistant", f"✗ {context}失败：{err}",
                 {"error": True, "context": context})
        return {"ok": False, "messages": self.messages[-1:], "spec": self.spec}

    # -- commands ----------------------------------------------------------
    def handle(self, payload: dict[str, Any]) -> dict:
        cmd = payload.get("cmd")
        with self.lock:
            try:
                handler = getattr(self, "_cmd_" + str(cmd), None)
                if handler is None:
                    return self._fail(ValueError(f"未知命令 {cmd!r}；可用："
                        "init/add_node/set_node/remove_node/set_entry/"
                        "set_edge/remove_edge/run/get_spec/reset"), cmd or "?")
                return handler(payload)
            except Exception as exc:  # noqa: BLE001 - errors go to the chat
                return self._fail(exc, cmd or "?")

    def _require_spec(self) -> dict[str, Any]:
        if self.spec is None:
            raise ValueError("还没有编排 spec——先发 {\"cmd\":\"init\",\"template\":\"...\"}")
        return self.spec

    def _cmd_init(self, payload: dict) -> dict:
        name = payload.get("template", "sequential")
        if name not in _TEMPLATES:
            raise ValueError(f"未知模板 {name!r}；可用：{sorted(_TEMPLATES)}")
        import copy
        spec = copy.deepcopy(_TEMPLATES[name])
        # desc / input are studio metadata, not spec fields — keep them out of
        # the JSON the user exports, or build_graph would reject the unknown keys.
        desc, default_input = spec.pop("desc", ""), spec.pop("input", {})
        self.default_input = default_input
        self.spec = spec
        self.runs = []
        self.say("user", f"init template={name}")
        note = f"✓ 已创建编排：**{name}**\n{desc}\n（{len(spec['nodes'])} 个节点，入口 {spec['entry']}）"
        return self._ok(note)

    def _cmd_templates(self, payload: dict) -> dict:
        """List the built-in templates with descriptions — for the UI buttons."""
        listing = {name: {"desc": t.get("desc", ""), "desc_en": t.get("desc_en", ""),
                          "input": t.get("input", {})}
                   for name, t in _TEMPLATES.items()}
        return {"ok": True, "templates": listing, "spec": self.spec,
                "runs": self.runs}

    def _cmd_reset(self, payload: dict) -> dict:
        self.spec = None
        self.runs = []
        self.say("user", "reset")
        return self._ok("✓ 会话已清空。")

    def _cmd_get_spec(self, payload: dict) -> dict:
        spec = self._require_spec()
        self.say("user", "get_spec")
        return self._ok("当前 spec：\n```json\n"
                        + json.dumps(spec, ensure_ascii=False, indent=2)
                        + "\n```")

    def _cmd_add_node(self, payload: dict) -> dict:
        spec = self._require_spec()
        name, cfg = payload.get("name"), payload.get("node")
        if not name or not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError('需要 {"name": "...", "node": {"type": "llm"|...}}；'
                             f"节点类型：{NODE_TYPES}")
        if str(cfg["type"]) not in NODE_TYPES:
            raise ValueError(f"未知节点类型 {cfg['type']!r}；可用：{NODE_TYPES}")
        if name in spec["nodes"]:
            raise ValueError(f"节点 {name!r} 已存在——用 set_node 覆盖")
        self.say("user", f"add_node {name} type={cfg['type']}")
        spec["nodes"][str(name)] = cfg
        return self._compile_ok(f"✓ 已添加节点 {name!r}（{cfg['type']}）")

    def _cmd_set_node(self, payload: dict) -> dict:
        spec = self._require_spec()
        name, cfg = payload.get("name"), payload.get("node")
        if not name or not isinstance(cfg, dict):
            raise ValueError('需要 {"name": "...", "node": {...完整节点配置...}}')
        if name not in spec["nodes"]:
            raise ValueError(f"节点 {name!r} 不存在——用 add_node 添加")
        self.say("user", f"set_node {name}")
        spec["nodes"][str(name)] = cfg
        return self._compile_ok(f"✓ 已更新节点 {name!r}")

    def _cmd_remove_node(self, payload: dict) -> dict:
        spec = self._require_spec()
        name = payload.get("name")
        if name not in spec.get("nodes", {}):
            raise ValueError(f"节点 {name!r} 不存在")
        self.say("user", f"remove_node {name}")
        del spec["nodes"][name]
        spec["edges"].pop(name, None)
        for src, tgt in list(spec["edges"].items()):
            if tgt == name:
                del spec["edges"][src]
            elif isinstance(tgt, dict):
                then = tgt.get("then", {})
                for k, v in list(then.items()):
                    if v == name:
                        del then[k]
        return self._compile_ok(f"✓ 已删除节点 {name!r} 及其关联边")

    def _cmd_set_entry(self, payload: dict) -> dict:
        spec = self._require_spec()
        entry = payload.get("entry")
        if entry not in spec.get("nodes", {}):
            raise ValueError(f"{entry!r} 不是已声明的节点")
        self.say("user", f"set_entry {entry}")
        spec["entry"] = entry
        return self._compile_ok(f"✓ 入口改为 {entry!r}")

    def _cmd_set_edge(self, payload: dict) -> dict:
        spec = self._require_spec()
        src, tgt = payload.get("from"), payload.get("to")
        if not src:
            raise ValueError('需要 {"from": "节点", "to": "节点" | "__end__"'
                             ' | {"condition": ..., "then": {...}}}')
        self.say("user", f"set_edge {src} → {json.dumps(tgt, ensure_ascii=False)}")
        if tgt is None:
            spec["edges"].pop(src, None)
        else:
            spec["edges"][str(src)] = tgt
        return self._compile_ok(f"✓ 已设置边 {src!r}")

    def _cmd_remove_edge(self, payload: dict) -> dict:
        spec = self._require_spec()
        src = payload.get("from")
        if src not in spec.get("edges", {}):
            raise ValueError(f"{src!r} 没有出边")
        self.say("user", f"remove_edge {src}")
        del spec["edges"][src]
        return self._compile_ok(f"✓ 已删除 {src!r} 的出边")

    def _compile_ok(self, note: str) -> dict:
        """Compile to fail loudly on a broken graph, then confirm."""
        dsl.build_graph(self.spec or {})  # raises SpecError → _fail shows it
        return self._ok(note)

    def _cmd_run(self, payload: dict) -> dict:
        spec = self._require_spec()
        inputs = payload.get("input") or self.default_input or {}
        self.say("user", f"run input={json.dumps(inputs, ensure_ascii=False)}")
        os.makedirs(self.trace_dir, exist_ok=True)
        from .events import ListSink
        from .checkpoint import CheckpointStore
        from .runtime import RunConfig, Runtime

        graph = dsl.build_graph(spec)
        dsl._SUB_TRACE_DIR = self.trace_dir
        sink = ListSink()
        store = CheckpointStore(os.path.join(self.trace_dir, "checkpoints"))
        runtime = Runtime(graph.compile(), store, sink)
        result = runtime.invoke(inputs, RunConfig(thread_id="studio"))
        trace_path = os.path.join(
            self.trace_dir, f"{result.run_id}.jsonl")
        with open(trace_path, "w", encoding="utf-8") as fh:
            for event in sink.events:
                fh.write(event.to_json() + "\n")
        record = {
            "run_id": result.run_id, "trace": trace_path, "status": result.status,
            "supersteps": result.supersteps, "error": result.error,
            "state": result.state, "ts": time.time(),
        }
        self.runs.append(record)
        line = (f"▶ 运行完成：status={result.status} supersteps={result.supersteps}"
                f"\ntrace: {trace_path}")
        if result.error:
            line += f"\nerror: {result.error}"
        final = {k: v for k, v in (result.state or {}).items()
                 if not k.startswith("__")}
        line += "\nfinal state: " + json.dumps(final, ensure_ascii=False)[:600]
        self.say("assistant", line, {"run": record})
        return {"ok": result.status == "complete", "messages": self.messages[-1:],
                "spec": self.spec, "runs": self.runs, "trace": trace_path,
                "run_id": result.run_id}


def create_session(trace_dir: str | None = None) -> StudioSession:
    return StudioSession(trace_dir or TRACE_DIR)


#: The process-wide session (single user by design — DESIGN §10).
_SESSION: StudioSession | None = None


def get_session() -> StudioSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = create_session()
    return _SESSION
