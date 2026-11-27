"""人在环路（human-in-the-loop）：一个需要人工审批的工具调用。

这个图形是所有「带护栏的 agent」都需要的形状，我们把它完全摊开：

    START ─▶ plan ──(需要审批?)──▶ request_approval ⏸
                  └──(安全)──────────────────▶ execute ✔
    request_approval ─▶ decide ──(approve)──▶ execute ✔
                              └──(deny)─────▶ reject ✘

这个示例存在的意义有两点：

1. **运行是怎么暂停的。** 节点不抛异常、不阻塞线程、也不返回 ``None``，而是返回
   ``{"__interrupt__": {...}}`` —— 一个保留通道，内核在 merge 之后读取它，然后让
   superstep 循环停下，并把 checkpoint 标记成 ``interrupted``。中断本身是**状态**，
   所以它既留在 trace 里，也留在 checkpoint 文件里。

2. **人是怎么把它继续下去的。** ``Runtime.resume(checkpoint_id, payload={...})`` 把
   payload 当作**已声明状态通道**上的 patch（``{"decision": "approve"}``），
   而不是节点名。中断 payload 里的 ``resume_into`` 才是指「下一步进哪个节点」。
   这两者一旦混用，就会报 ``unknown channel 'apply'`` —— 经典错误，也是本示例
   特意写出来的对照。

运行方式::

    py -3.13 -m lens_harness run examples.approval:build
    py -3.13 -m lens_harness show traces/<run_id>.jsonl --no-color
    py -3.13 -m lens_harness checkpoints traces/<run_id>.jsonl --no-color

``run`` 会按 ``build()`` 的真实签名传参（只传它声明过的关键字），所以这里不需要
``llm_spec`` 之类的占位参数。
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from lens_harness.checkpoint import STATUS_INTERRUPTED, CheckpointStore
from lens_harness.events import ListSink
from lens_harness.graph import END, START, StateGraph
from lens_harness.runtime import RunConfig, Runtime
from lens_harness.state import add, add_unique, merge_dict


# --------------------------------------------------------------------------
# 状态 schema —— 注解本身就是合并语义
# --------------------------------------------------------------------------
class ApprovalState(TypedDict):
    #: 人类要求 agent 做什么（自然语言）。
    task: str
    #: agent 想调用的具体工具名。
    action: str
    #: 待审批的调用，dict 形状。用 ``merge_dict``，这样某个节点只补一个字段
    #: （比如 ``reason``）时，不必把整个 call 重写一遍。
    pending: Annotated[dict[str, Any], merge_dict]
    #: 由 ``plan`` 写入、由路由器读取。把策略放在状态里而不是闭包里，trace
    #: 才能显示「这次为什么停下来」。
    requires_approval: bool
    #: 由人类的 resume payload 填入 —— 这是图在暂停之前唯一不会写的通道。
    decision: str
    result: str
    #: ``add``：每个节点只追加自己的那一行，谁都覆盖不了别人的。
    log: Annotated[list[str], add]
    #: ``add_unique``：有序集合 —— 用来证明每个节点恰好跑了一次。
    visited: Annotated[list[str], add_unique]


# --------------------------------------------------------------------------
# 一张极小的策略表 —— 没有插件系统，就是一个你能直接读完的 dict
# --------------------------------------------------------------------------
#: ``action -> (给人看的说明, 默认参数)``。
ACTIONS: dict[str, tuple[str, str]] = {
    "delete_file": ("Delete a file on disk.", "reports/2025-q4.csv"),
    "read_file": ("Read a file (no side effect).", "reports/2025-q4.csv"),
}

#: 不可通过「重跑一次」来撤销的动作，必须人类点头。
#: 为什么用集合而不是在每个节点里 if：策略集中在一处，读代码的人一眼能看全。
DESTRUCTIVE_ACTIONS = frozenset({"delete_file", "send_email"})


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------
def plan_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """决定要尝试哪个动作，以及它能不能无人值守地跑。

    返回一个**patch**而不是改 state —— 这是整个内核能并行、能时间旅行的前提。
    """
    action = state.get("action") or "delete_file"
    _desc, default_arg = ACTIONS.get(action, ("unknown action", ""))
    needs_human = action in DESTRUCTIVE_ACTIONS
    return {
        "pending": {
            "action": action,
            "args": {"path": default_arg},
            "reason": "stale export blocks the nightly rebuild",
        },
        "requires_approval": needs_human,
        "log": [
            f"plan: want to run {action} -> "
            f"{'needs a human' if needs_human else 'safe, no approval needed'}"
        ],
        "visited": ["plan"],
    }


def request_approval_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """暂停整个运行，把待审批的调用交给人类。

    返回的 dict 是一个普通 patch，只是恰好写了保留通道 ``__interrupt__``。内核在
    合并之后读到它，把它记到 checkpoint 上（``status='interrupted'``）然后停止循环 ——
    没有异常、没有阻塞的线程，也就没有任何「进程一重启就丢」的东西。
    """
    pending = state.get("pending") or {}
    return {
        "__interrupt__": {
            # 谁在提问 —— 渲染器会打印成 `⏸ interrupt at <node>`。
            "node": "request_approval",
            # 人类能看到的内容。保持小而自洽：它会被原样复制进 trace。
            "payload": {
                "question": "Approve this action?",
                "action": pending.get("action"),
                "args": pending.get("args"),
                "reason": pending.get("reason"),
            },
            # 注意：这是「恢复后进哪个节点」，不是通道名。通道由 resume 的
            # payload 携带，这里只负责目的地。
            "resume_into": "decide",
            # 给恢复者看的文档；内核不做校验。
            "resume_schema": {"decision": "str: 'approve' | 'deny'"},
        },
        "log": ["request_approval: paused, waiting for a human"],
        "visited": ["request_approval"],
    }


def decide_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """把人类的 payload 归一成一个明确的决定。

    ``state['decision']`` 在这里之所以存在，只因为 ``Runtime.resume`` 在本节点运行
    之前就通过通道 reducer 把 payload 应用了进去 —— 人类写入图状态的全部机制，
    就是这一条。
    """
    raw = str(state.get("decision") or "").strip().lower()
    if raw not in ("approve", "deny"):
        # 无法识别的值一律当拒绝：对于一个能删东西的闸门，fail closed 是唯一
        # 安全的默认值。宁可拒绝，也不要猜。
        return {
            "decision": "deny",
            "log": [f"decide: unrecognised decision {raw!r} -> deny (fail closed)"],
            "visited": ["decide"],
        }
    return {
        "log": [f"decide: human said {raw!r}"],
        "visited": ["decide"],
    }


def execute_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """真正执行动作。在真实系统里，副作用就发生在这一步。"""
    pending = state.get("pending") or {}
    action = pending.get("action", "?")
    args = pending.get("args") or {}
    return {
        "result": f"executed {action} with {args}",
        "log": [f"execute: ran {action} (approved by {state.get('decision') or 'policy'})"],
        "visited": ["execute"],
    }


def reject_node(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """拒绝路径 —— 特意做成一个真实节点，这样 trace 里能看见它发生过。"""
    action = (state.get("pending") or {}).get("action", "?")
    return {
        "result": f"refused {action}; nothing was executed",
        "log": [f"reject: {action} was not run"],
        "visited": ["reject"],
    }


# --------------------------------------------------------------------------
# 路由器 —— trace 永远会展示它们做出的判断
# --------------------------------------------------------------------------
def route_after_plan(state: dict[str, Any]) -> str:
    """返回的是 **mapping 的 key**，绝不是节点名：``mapping[key] -> 节点``。"""
    return "approval" if state.get("requires_approval") else "execute"


def route_after_decide(state: dict[str, Any]) -> str:
    return "approve" if state.get("decision") == "approve" else "deny"


def build_graph() -> Any:
    graph = StateGraph(ApprovalState, name="approval-gate")
    graph.add_node(
        "plan",
        plan_node,
        description="Pick the action and decide whether it needs a human.",
        tags=("policy",),
    )
    graph.add_node(
        "request_approval",
        request_approval_node,
        description="Write __interrupt__: pause the run and surface the pending call.",
        tags=("human",),
    )
    graph.add_node(
        "decide",
        decide_node,
        description="Normalise the human's resume payload into a decision.",
        tags=("human",),
    )
    graph.add_node(
        "execute",
        execute_node,
        description="Perform the approved action (the side effect).",
        tags=("side-effect",),
    )
    graph.add_node(
        "reject",
        reject_node,
        description="Record the refusal; no side effect happens.",
        tags=(),
    )

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        # mapping：路由器返回值 -> 节点名。人读说明放在 `descriptions` 里，
        # 绝不能塞进这里的 value —— 把 "needs a human" 写在这儿，就会路由到一个
        # 根本不存在的节点（本项目踩过这个坑）。
        {
            "approval": "request_approval",
            "execute": "execute",
        },
        name="route_after_plan",
        descriptions={
            "approval": "the action is destructive and needs a human",
            "execute": "the action is safe, so run it unattended",
        },
    )
    graph.add_edge("request_approval", "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "approve": "execute",
            "deny": "reject",
        },
        name="route_after_decide",
        descriptions={
            "approve": "the human approved the action",
            "deny": "the human refused, so skip the side effect",
        },
    )
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)
    return graph.compile()


# --------------------------------------------------------------------------
# 入口，风格对齐 examples/agent.py 的 build
# --------------------------------------------------------------------------
DEMO_TASK = "Remove the stale Q4 export so the nightly rebuild can start clean."


def build(
    *,
    inputs: dict[str, Any] | None = None,
    trace_dir: str = "traces",
    decision: str = "approve",
) -> list:
    """跑一次图、模拟一位人类，然后返回全部事件。

    完整事件序列（invoke + resume，同一个 run_id）写到
    ``<trace_dir>/<run_id>.jsonl``，这样 ``lens show`` 会把两段渲染成一个故事：
    暂停，以及人类的那一笔贡献。
    """
    data = inputs or {}
    task = data.get("task", DEMO_TASK)
    action = data.get("action", "delete_file")

    compiled = build_graph()
    sink = ListSink()
    # checkpoint 落盘不是可选项：resume 是**从 checkpoint 读回来的**，
    # 所以「中断 -> 恢复」这条链路必须经过真实的文件，才能证明它成立。
    store = CheckpointStore(os.path.join(trace_dir, "checkpoints"))
    runtime = Runtime(compiled, store, sink)
    config = RunConfig(thread_id="approval-demo")

    first = runtime.invoke(
        {
            "task": task,
            "action": action,
            "pending": {},
            "requires_approval": False,
            "decision": "",
            "result": "",
            "log": [],
            "visited": [],
        },
        config,
    )
    print(f"[1] first run status={first.status} supersteps={first.supersteps}")

    if first.status == STATUS_INTERRUPTED:
        checkpoint_id = first.checkpoints[-1].checkpoint_id
        interrupt = first.checkpoints[-1].interrupt or {}
        print(f"[2] paused at checkpoint {checkpoint_id}")
        print(f"    node={interrupt.get('node')} payload={interrupt.get('payload')}")

        # 真实应用里，这一行会被换成一个网页表单、一个 Slack 按钮或一个 CLI 提示；
        # 除此之外的一切都保持不变。
        print(f"[3] human replies: decision={decision!r}")
        resumed = runtime.resume(
            checkpoint_id,
            payload={"decision": decision},
            config=config,
        )
        print(f"[4] resumed status={resumed.status} result={resumed.state.get('result')!r}")
    else:
        # 非破坏性动作压根不会暂停 —— 这正是策略边存在的意义，所以如实打印，
        # 而不是假装发生过中断。
        print("[2] no interrupt: the action was considered safe")
        resumed = first

    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{runtime.emitter.run_id}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for event in sink.events:
            fh.write(event.to_json() + "\n")

    print(f"\n[trace written to {path}]")
    print(f"[final] status={resumed.status} log={resumed.state.get('log')}")
    return sink.events


if __name__ == "__main__":  # pragma: no cover - manual demo
    build()
