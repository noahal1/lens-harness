"""Interrupt / resume behaviour, plus the LLM adapter contract.

Kept in its own module because these are the two areas where the kernel talks
to something outside itself (a human, a network), so they deserve to fail
independently of the pure control-flow tests in ``test_kernel``.

Run with ``py -3.13 -m tests.test_flows``.
"""

from __future__ import annotations

import os
import traceback
import uuid
from typing import Annotated, TypedDict

from lens_harness.checkpoint import STATUS_COMPLETE, STATUS_INTERRUPTED, CheckpointStore
from lens_harness.events import (
    INTERRUPT,
    NODE_START,
    STATE_MERGE,
    ListSink,
)
from lens_harness.graph import END, StateGraph
from lens_harness.llm import Message, MockLLM, ScriptedRule, load_llm
from lens_harness.runtime import RunConfig, Runtime
from lens_harness.state import add

_TESTS: list[tuple[str, object]] = []

_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".lens-test-tmp"
)


def test(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


def assert_eq(actual, expected, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def assert_true(cond, what: str) -> None:
    if not cond:
        raise AssertionError(what)


class Fixture:
    def __init__(self) -> None:
        os.makedirs(_ROOT, exist_ok=True)
        self.dir = os.path.join(_ROOT, f"flow-{uuid.uuid4().hex[:10]}")
        os.makedirs(self.dir, exist_ok=True)
        self.sink = ListSink()
        self.store = CheckpointStore(os.path.join(self.dir, "ckpt"))

    @property
    def events(self):
        return self.sink.events

    def kinds(self, kind):
        return [e for e in self.events if e.kind == kind]


# --------------------------------------------------------------------------
# Interrupt
# --------------------------------------------------------------------------
class ApprovalState(TypedDict):
    request: str
    decision: str
    log: Annotated[list[str], add]


@test
def test_interrupt_pauses_run_and_marks_checkpoint_interrupted():
    """A node writing ``__interrupt__`` must stop the run, not merely record it."""
    fx = Fixture()
    g = StateGraph(ApprovalState, name="approval")

    def ask(state, ctx):
        return {"__interrupt__": {
            "node": "ask",
            "payload": {"request": state["request"]},
            "resume_schema": {"decision": "str"},
            "resume_into": "apply",
        }}

    def apply_decision(state, ctx):
        return {"log": [f"applied:{state.get('decision', '?')}"]}

    g.add_node("ask", ask)
    g.add_node("apply", apply_decision)
    g.set_entry("ask")
    g.add_edge("ask", "apply")
    g.add_edge("apply", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    result = runtime.invoke({"request": "ship it?", "log": []}, RunConfig(thread_id="t"))

    assert_eq(result.status, STATUS_INTERRUPTED, "run status")
    interrupts = fx.kinds(INTERRUPT)
    assert_eq(len(interrupts), 1, "interrupt event count")
    assert_eq(interrupts[0].data["node"], "ask", "interrupt node")
    assert_eq(interrupts[0].data["payload"], {"request": "ship it?"}, "interrupt payload")
    # The node after the interrupt must NOT have run yet.
    ran = [e.data["node"] for e in fx.kinds(NODE_START)]
    assert_eq(ran, ["ask"], "nodes that ran before the interrupt")
    # And "apply" must be queued as the next node.
    assert_eq(result.checkpoints[-1].next_nodes, ["apply"], "queued next node")


@test
def test_resume_with_payload_applies_it_and_continues():
    fx = Fixture()
    g = StateGraph(ApprovalState, name="approval2")

    def ask(state, ctx):
        return {"__interrupt__": {
            "node": "ask",
            "payload": {"request": state["request"]},
            "resume_into": "apply",
        }}

    def apply_decision(state, ctx):
        return {"log": [f"applied:{state.get('decision', '?')}"]}

    g.add_node("ask", ask)
    g.add_node("apply", apply_decision)
    g.set_entry("ask")
    g.add_edge("ask", "apply")
    g.add_edge("apply", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    first = runtime.invoke({"request": "ok?", "log": []}, RunConfig(thread_id="t"))
    ckpt = first.checkpoints[-1]

    fx.sink.events.clear()
    resumed = runtime.resume(
        ckpt.checkpoint_id,
        payload={"decision": "yes"},
        config=RunConfig(thread_id="t"),
    )

    assert_eq(resumed.state["decision"], "yes", "payload landed in state")
    assert_eq(resumed.state["log"], ["applied:yes"], "downstream saw the payload")
    # The human's contribution must be visible in the trace as a state_merge.
    merges = [e for e in fx.kinds(STATE_MERGE) if e.data.get("origin") == "resume_payload"]
    assert_eq(len(merges), 1, "resume payload state_merge events")
    assert_eq(merges[0].data["changed_channels"], ["decision"], "payload channels")


@test
def test_resume_clears_the_interrupt_so_downstream_nodes_still_run():
    """The interrupt is a one-shot signal.

    Regression: ``__interrupt__`` lives in the checkpointed state, so a resume
    that did not clear it re-triggered on the very next superstep. The symptom
    was nasty because it looked like success — the resumed node *did* run, then
    the run reported ``interrupted`` again and never reached the nodes after it.
    """
    fx = Fixture()
    g = StateGraph(ApprovalState, name="approval3")

    def ask(state, ctx):
        return {"__interrupt__": {
            "node": "ask",
            "payload": {"request": state["request"]},
            "resume_into": "apply",
        }}

    g.add_node("ask", ask)
    g.add_node("apply", lambda s, c: {"log": ["applied"]})
    g.add_node("after", lambda s, c: {"log": ["after"]})
    g.set_entry("ask")
    g.add_edge("ask", "apply")
    g.add_edge("apply", "after")
    g.add_edge("after", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    first = runtime.invoke({"request": "ok?", "log": []}, RunConfig(thread_id="t"))
    resumed = runtime.resume(
        first.checkpoints[-1].checkpoint_id,
        payload={"decision": "yes"},
        config=RunConfig(thread_id="t"),
    )

    assert_eq(resumed.status, STATUS_COMPLETE, "resumed run status")
    assert_eq(resumed.state["log"], ["applied", "after"], "every node after the resume ran")
    assert_true("__interrupt__" not in resumed.state,
                "the one-shot interrupt must not survive the resume")


@test
def test_resume_without_payload_is_rejected_when_no_interrupt():
    """Guessing where a payload goes would be worse than refusing."""
    fx = Fixture()
    g = StateGraph(ApprovalState, name="nointerrupt")
    g.add_node("only", lambda s, c: {"log": ["ran"]})
    g.set_entry("only")
    g.add_edge("only", END)
    runtime = Runtime(g.compile(), fx.store, fx.sink)
    result = runtime.invoke({"request": "x", "log": []}, RunConfig(thread_id="t"))

    try:
        runtime.resume(result.checkpoints[-1].checkpoint_id,
                       payload={"decision": "yes"},
                       config=RunConfig(thread_id="t"))
    except Exception as exc:  # GraphError
        assert "interrupt" in str(exc).lower(), f"unhelpful error: {exc}"
    else:
        raise AssertionError("resume with payload on a non-interrupt checkpoint must fail")


# --------------------------------------------------------------------------
# LLM adapters
# --------------------------------------------------------------------------
@test
def test_mock_llm_is_deterministic_and_streams_tokens():
    llm = MockLLM(replies=["hello world"], chunk_size=4)
    resp = llm.complete([Message("user", "hi")])
    assert_eq(resp.text, "hello world", "reply text")
    assert_true(resp.mocked, "mock flag must be set so the trace can show it")
    assert_true(len(resp.text.split()) == 2, "word count sanity")


@test
def test_mock_llm_scripted_rules_match_in_order():
    """Scripted replies are how the examples stay deterministic."""
    llm = MockLLM(script=[
        ScriptedRule(contains="17*23", reply="TOOL: calculator | 17*23"),
        ScriptedRule(contains="FINAL", reply="done"),
    ], default="fallback")

    first = llm.complete([Message("user", "What is 17*23?")])
    assert_eq(first.text, "TOOL: calculator | 17*23", "first scripted match")
    second = llm.complete([Message("user", "FINAL please")])
    assert_eq(second.text, "done", "second scripted match")
    third = llm.complete([Message("user", "unrelated")])
    assert_eq(third.text, "fallback", "default reply")


@test
def test_mock_llm_token_callback_receives_the_whole_text():
    llm = MockLLM(replies=["abcdefgh"], chunk_size=3)
    chunks: list[str] = []
    llm.complete([Message("user", "go")], on_token=chunks.append)
    assert_eq("".join(chunks), "abcdefgh", "streamed token concatenation")
    assert_true(len(chunks) >= 2, f"expected several chunks, got {len(chunks)}")


@test
def test_load_llm_defaults_to_mock_and_parses_openai_spec():
    saved = {k: os.environ.pop(k, None) for k in
             ("LENS_LLM", "LENS_LLM_MODEL", "LENS_LLM_API_KEY", "OPENAI_API_KEY")}
    try:
        default = load_llm("mock")
        assert_eq(type(default).__name__, "MockLLM", "explicit mock spec")

        auto = load_llm("auto")
        assert_eq(type(auto).__name__, "MockLLM", "auto without a key falls back to mock")

        remote = load_llm("openai:gpt-4o-mini", api_key="sk-test")
        assert_eq(type(remote).__name__, "OpenAICompatibleLLM", "openai spec type")
        assert_eq(remote.model, "gpt-4o-mini", "model parsed from spec")
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


# --------------------------------------------------------------------------
def main() -> int:
    failures: list[tuple[str, str]] = []
    for name, fn in _TESTS:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append((name, traceback.format_exc()))
            print(f"FAIL {name}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(_TESTS) - len(failures)}/{len(_TESTS)} passed")
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
