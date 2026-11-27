"""Kernel conformance tests — no pytest, no dependencies.

Run with ``py -3.13 -m tests.test_kernel`` (or via the project Makefile-ish
command in README). Each test names the *behaviour that must stay true*, since
this project exists to make the kernel's semantics visible, and a semantics bug
is a teaching bug.
"""

from __future__ import annotations

import os
import shutil
import traceback
import uuid
from typing import Annotated, TypedDict

from lens_harness.checkpoint import STATUS_COMPLETE, STATUS_INTERRUPTED, CheckpointStore
from lens_harness.events import (
    CHECKPOINT_CREATED,
    INTERRUPT,
    NODE_END,
    NODE_ERROR,
    ROUTE_DECISION,
    RUN_END,
    STATE_MERGE,
    ListSink,
    summarize_state,
)
from lens_harness.graph import END, GraphError, StateGraph
from lens_harness.runtime import GraphRecursionError, RunConfig, Runtime
from lens_harness.state import add, add_unique

_TESTS: list[tuple[str, object]] = []


def assert_eq(actual, expected, what: str) -> None:
    """Tiny local assertion so the suite stays dependency-free."""
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def assert_true(cond, what: str) -> None:
    if not cond:
        raise AssertionError(what)


def test(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


class Fixture:
    """Each test gets a private checkpoint dir and a fresh event sink.

    Two Windows-sandbox gotchas, both learned the hard way:

    * the OS temp dir is outside the workspace and denied, so cases live in
      ``.lens-test-tmp/`` next to the project;
    * ``tempfile.mkdtemp`` creates mode-0700 directories that a workspace ACL
      then refuses to descend into, so names are built by hand with
      ``os.makedirs`` instead.

    A test suite that only passes unsandboxed is worse than no suite.
    """

    _ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".lens-test-tmp"
    )

    def __init__(self) -> None:
        os.makedirs(self._ROOT, exist_ok=True)
        self.dir = os.path.join(self._ROOT, f"case-{uuid.uuid4().hex[:10]}")
        os.makedirs(self.dir, exist_ok=True)
        self.sink = ListSink()
        self.store = CheckpointStore(os.path.join(self.dir, "ckpt"))

    def run(self, graph, inputs, config=None):
        runtime = Runtime(graph.compile(), self.store, self.sink)
        return runtime.invoke(inputs, config or RunConfig(thread_id="t"))

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- assertions --------------------------------------------------------
    def kinds(self, kind: str) -> list:
        return [e for e in self.sink.events if e.kind == kind]



# --------------------------------------------------------------------------
# Channel / reducer semantics
# --------------------------------------------------------------------------
class AccumState(TypedDict):
    items: Annotated[list[str], add]
    tags: Annotated[list[str], add_unique]
    note: str


class LatchState(TypedDict):
    n: int


class FanState(TypedDict):
    items: Annotated[list[str], add]


@test
def test_annotated_reducers_are_detected():
    """``Annotated[..., add]`` must survive ``from __future__ import annotations``."""
    from lens_harness.state import channels_from_annotations, resolve_annotations

    channels = channels_from_annotations(resolve_annotations(AccumState))
    assert channels["items"].reducer_name == "add", channels["items"].reducer_name
    assert channels["tags"].reducer_name == "add_unique", channels["tags"].reducer_name
    assert channels["note"].reducer_name == "overwrite", channels["note"].reducer_name


@test
def test_parallel_writes_accumulate_not_overwrite():
    """Three parallel nodes appending to one channel must keep all three writes.

    This is the canonical LangGraph reduce-vs-overwrite distinction, so it is
    the one behaviour that most deserves a test.
    """
    fixture = Fixture()
    try:
        g = StateGraph(AccumState, name="fanout")
        for name in ("a", "b", "c"):
            g.add_node(name, lambda state, ctx, n=name: {"items": [n]})
        g.set_entry("a")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", END)
        g.add_edge("c", END)

        result = fixture.run(g, {"items": [], "tags": [], "note": ""},
                             RunConfig(thread_id="t", max_concurrency=4))
        # b and c run in the same superstep, then merge.
        assert sorted(result.state["items"]) == ["a", "b", "c"], result.state["items"]
    finally:
        fixture.cleanup()


@test
def test_add_unique_dedupes():
    fixture = Fixture()
    try:
        g = StateGraph(AccumState, name="dedupe")
        g.add_node("a", lambda s, c: {"tags": ["x", "y"]})
        g.add_node("b", lambda s, c: {"tags": ["y", "z"]})
        g.set_entry("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        result = fixture.run(g, {"items": [], "tags": [], "note": ""})
        assert result.state["tags"] == ["x", "y", "z"], result.state["tags"]
    finally:
        fixture.cleanup()


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------
class LoopState(TypedDict):
    n: int
    path: Annotated[list[str], add]


@test
def test_conditional_routing_records_candidates_and_choice():
    """The route event must carry *all* candidates plus what was returned."""
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="router")

        def bump(state, ctx):
            return {"n": state["n"] + 1, "path": ["bump"]}

        g.add_node("bump", bump)
        g.add_node("done", lambda s, c: {"path": ["done"]})
        g.set_entry("bump")
        g.add_conditional_edges(
            "bump",
            lambda state: "again" if state["n"] < 2 else "stop",
            {"again": "bump", "stop": "done"},
            name="loop_or_stop",
        )
        g.add_edge("done", END)

        fixture.run(g, {"n": 0, "path": []})
        # Every hop emits a route_decision — static edges included, with
        # condition=None — so filter to the conditional ones for this assertion.
        routes = [e for e in fixture.kinds(ROUTE_DECISION) if e.data["condition"] is not None]
        assert routes, "no conditional route_decision events were emitted"
        first = routes[0].data
        assert_eq(sorted(first["candidates"]), ["again", "stop"], "candidates")
        assert_eq(first["returned"], "again", "returned")
        # ``selected`` is a list: a router is allowed to fan out to several nodes.
        assert_eq(first["selected"], ["bump"], "selected")
        assert_eq(routes[-1].data["selected"], ["done"], "final selected")
    finally:
        fixture.cleanup()


@test
def test_recursion_limit_raises_instead_of_hanging():
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="infinite")
        g.add_node("spin", lambda s, c: {"n": s["n"] + 1, "path": ["spin"]})
        g.set_entry("spin")
        g.add_edge("spin", "spin")
        result = fixture.run(g, {"n": 0, "path": []},
                             RunConfig(thread_id="t", max_supersteps=5))
        assert result.status == "failed", result.status
        assert "max_supersteps" in (result.error or ""), result.error
        assert result.supersteps <= 6, result.supersteps
    finally:
        fixture.cleanup()


@test
def test_node_failure_is_captured_as_event_and_failed_status():
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="boom")

        def explode(state, ctx):
            raise ValueError("kaboom")

        g.add_node("explode", explode)
        g.set_entry("explode")
        g.add_edge("explode", END)
        result = fixture.run(g, {"n": 0, "path": []})
        errors = fixture.kinds(NODE_ERROR)
        assert errors, "no node_error event"
        assert_eq(errors[0].data["error_type"], "ValueError", "error_type")
        assert "kaboom" in errors[0].data["error"], errors[0].data["error"]
        assert result.status == "failed", result.status
    finally:
        fixture.cleanup()


# --------------------------------------------------------------------------
# Checkpoints / time travel
# --------------------------------------------------------------------------
@test
def test_checkpoints_form_a_linked_dag_with_terminal_status():
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="chain")
        g.add_node("one", lambda s, c: {"n": s["n"] + 1, "path": ["one"]})
        g.add_node("two", lambda s, c: {"n": s["n"] + 1, "path": ["two"]})
        g.set_entry("one")
        g.add_edge("one", "two")
        g.add_edge("two", END)

        result = fixture.run(g, {"n": 0, "path": []})
        assert_eq(len(result.checkpoints), 2, "checkpoint count")
        first, last = result.checkpoints
        assert_eq(first.parent_id, None, "root parent")
        assert_eq(last.parent_id, first.checkpoint_id, "child parent")
        assert_eq(last.status, STATUS_COMPLETE, "terminal status")
        assert_eq(fixture.kinds(CHECKPOINT_CREATED)[-1].data["status"],
                  STATUS_COMPLETE, "traced terminal status")
    finally:
        fixture.cleanup()


@test
def test_resume_replays_from_old_checkpoint_without_rerunning_past_nodes():
    """Time travel: resuming checkpoint 0 must not re-execute node ``one``."""
    fixture = Fixture()
    try:
        calls: list[str] = []

        class S(TypedDict):
            n: int
            path: Annotated[list[str], add]

        g = StateGraph(S, name="resume")
        g.add_node("one", lambda s, c: (calls.append("one"), {"n": s["n"] + 1, "path": ["one"]})[1])
        g.add_node("two", lambda s, c: (calls.append("two"), {"n": s["n"] + 1, "path": ["two"]})[1])
        g.set_entry("one")
        g.add_edge("one", "two")
        g.add_edge("two", END)

        compiled = g.compile()
        runtime = Runtime(compiled, fixture.store, fixture.sink)
        result = runtime.invoke({"n": 0, "path": []}, RunConfig(thread_id="t"))
        assert_eq(calls, ["one", "two"], "first run calls")

        # Resume from the *first* checkpoint: node "one" already ran, so only
        # "two" may execute again.
        calls.clear()
        resumed = runtime.resume(
            result.checkpoints[0].checkpoint_id,
            config=RunConfig(thread_id="t"),
        )
        assert_eq(calls, ["two"], "resume calls")
        assert_eq(resumed.checkpoints[0].parent_id, result.checkpoints[0].checkpoint_id,
                  "forked child")
    finally:
        fixture.cleanup()


@test
def test_store_roundtrip_survives_a_fresh_reader():
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="persist")
        g.add_node("only", lambda s, c: {"n": 7, "path": ["only"]})
        g.set_entry("only")
        g.add_edge("only", END)
        result = fixture.run(g, {"n": 0, "path": []})

        # A brand-new store instance must see what a different Runtime wrote.
        fresh = CheckpointStore(os.path.join(fixture.dir, "ckpt"))
        loaded = fresh.load(result.checkpoints[-1].checkpoint_id)
        assert loaded is not None, "checkpoint did not persist"
        assert_eq(loaded.state["n"], 7, "persisted state")
    finally:
        fixture.cleanup()


# --------------------------------------------------------------------------
# Trace integrity
# --------------------------------------------------------------------------
@test
def test_every_event_validates_and_seq_is_dense():
    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="trace")
        g.add_node("a", lambda s, c: {"n": s["n"] + 1, "path": ["a"]})
        g.set_entry("a")
        g.add_edge("a", END)
        fixture.run(g, {"n": 0, "path": []})

        from lens_harness.events import validate_event

        for event in fixture.sink.events:
            validate_event(event.kind, event.data)
        seqs = [e.seq for e in fixture.sink.events]
        assert_eq(seqs, list(range(len(seqs))), "seq must be 0..n-1 with no gaps")
        assert_eq(fixture.kinds(RUN_END)[-1].data["status"], "complete", "run_end status")
    finally:
        fixture.cleanup()


@test
def test_replay_state_summary_at_rebuilds_state_from_trace_alone():
    """The lens must be able to answer "state at checkpoint" from events only.

    The function is the *summary* rebuild (renamed from ``replay_state_at``,
    DESIGN.md §11.4): values longer than summarize_state's limits arrive
    truncated, and the full values live only in the checkpoint file.
    """
    from lens_harness.render import replay_state_summary_at

    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="replay")
        g.add_node("a", lambda s, c: {"n": 5, "path": ["a"]})
        g.set_entry("a")
        g.add_edge("a", END)
        result = fixture.run(g, {"n": 0, "path": []})

        rebuilt = replay_state_summary_at(fixture.sink.events, result.checkpoints[-1].checkpoint_id)
        assert rebuilt is not None, "replay returned None"
        assert_eq(rebuilt["n"], 5, "replayed n")
        assert_eq(rebuilt["path"], ["a"], "replayed path")
    finally:
        fixture.cleanup()


@test
def test_replay_state_summary_is_truncated_but_checkpoint_is_full():
    """The trace alone can only give a summary; the checkpoint file has it all."""
    from lens_harness.render import replay_state_summary_at

    class BigState(TypedDict):
        big: str

    fixture = Fixture()
    try:
        g = StateGraph(BigState, name="big")
        g.add_node("a", lambda s, c: {"big": "x" * 500})
        g.set_entry("a")
        g.add_edge("a", END)
        result = fixture.run(g, {"big": ""})

        rebuilt = replay_state_summary_at(fixture.sink.events, result.checkpoints[-1].checkpoint_id)
        assert rebuilt is not None
        assert_true(len(rebuilt["big"]) < 200, "trace summary is truncated")
        loaded = fixture.store.load(result.checkpoints[-1].checkpoint_id)
        assert loaded is not None
        assert_eq(len(loaded.state["big"]), 500, "checkpoint file holds the full value")
    finally:
        fixture.cleanup()


@test
def test_graph_definition_errors_are_loud():
    for label, build in [
        ("no nodes", lambda: StateGraph(LoopState, name="x").compile()),
    ]:
        try:
            build()
        except GraphError:
            pass
        else:
            raise AssertionError(f"expected GraphError for {label}")


@test
def test_diff_key_order_is_schema_order_so_traces_are_reproducible():
    """Diff keys must follow the declared schema, not set iteration order.

    Regression: ``State.diff`` iterated ``set(self.values) | set(other.values)``.
    Set order is stable within one process but varies with ``PYTHONHASHSEED``
    across runs — and the diff is serialized into the trace, so six runs of one
    example produced six different key orders and two identical-in-content
    traces diffed as different files.

    Checking "exactly schema order, deliberately not alphabetical" is what
    catches it. Note the honest limit: the old code's set order *coincided* with
    schema order for some hash seeds, so this test only fails on the seeds where
    they differ. A seed-dependent bug cannot be caught more sharply than that
    from inside one process.
    """
    from lens_harness.state import Channel, State, overwrite

    # Schema order zeta -> alpha -> mid; alphabetical would be alpha, mid, zeta.
    names = ("zeta", "alpha", "mid")
    channels = {name: Channel(name, None, overwrite, "overwrite") for name in names}
    before = State(channels, {"zeta": 1, "alpha": 2, "mid": 3})
    after = State(channels, {"zeta": 9, "alpha": 8, "mid": 7})

    assert_eq(list(before.diff(after)), list(names), "diff follows schema order")


@test
def test_resume_keeps_one_thread_identity_across_checkpoint_and_events():
    """A resumed run must not report two different thread ids.

    Regression: the new checkpoint's id was prefixed with ``cfg.thread_id``
    while the events used the parent checkpoint's thread, so resuming with a
    different config produced a trace whose ids and events disagreed.
    """
    fx = Fixture()
    g = StateGraph(LatchState, name="threadidentity")
    g.add_node("first", lambda s, c: {"n": 1})
    g.add_node("second", lambda s, c: {"n": 2})
    g.set_entry("first")
    g.add_edge("first", "second")
    g.add_edge("second", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    first = runtime.invoke({"n": 0}, RunConfig(thread_id="alpha"))
    fx.sink.events.clear()
    resumed = runtime.resume(
        first.checkpoints[-1].checkpoint_id,
        config=RunConfig(thread_id="TOTALLY-DIFFERENT"),
    )

    assert_eq(resumed.thread_id, "alpha", "result thread id")
    for ckpt in resumed.checkpoints:
        assert_eq(ckpt.thread_id, resumed.thread_id, "checkpoint thread id")
        assert_true(ckpt.checkpoint_id.startswith("alpha_"),
                    f"checkpoint id {ckpt.checkpoint_id} must carry the run's thread")
    assert_eq({e.thread_id for e in fx.sink.events}, {"alpha"}, "event thread ids")


@test
def test_parallel_write_reports_observed_concurrency_not_the_limit():
    """``max_concurrency`` must mean "how many actually ran at once"."""
    fx = Fixture()
    g = StateGraph(FanState, name="conc")
    g.add_node("seed", lambda s, c: {"items": ["seed"]})
    g.add_node("a", lambda s, c: {"items": ["a"]})
    g.add_node("b", lambda s, c: {"items": ["b"]})
    g.set_entry("seed")
    g.add_edge("seed", "a")
    g.add_edge("seed", "b")
    g.add_edge("a", END)
    g.add_edge("b", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    result = runtime.invoke({"items": []}, RunConfig(thread_id="t", max_concurrency=8))

    assert_eq(result.max_concurrency, 2, "observed parallel nodes")
    assert_eq(result.allowed_concurrency, 8, "configured ceiling")


@test
def test_conditional_edge_snapshot_carries_descriptions_not_the_mapping():
    """The static graph snapshot must expose prose, not node names.

    Regression: ``snapshot()`` put ``ce.mapping`` under the ``descriptions``
    key, so the lens saw ``tools -> "tools"`` instead of the actual sentence.
    """
    g = StateGraph(LatchState, name="snap")
    g.add_node("a", lambda s, c: {})
    g.add_node("b", lambda s, c: {})
    g.set_entry("a")
    g.add_conditional_edges(
        "a",
        lambda s: "go",
        {"go": "b"},
        name="pick",
        descriptions={"go": "the human said go"},
    )
    g.add_edge("b", END)
    snap = g.compile().graph.snapshot()
    edge = snap["conditional_edges"][0]

    assert_eq(edge["descriptions"], {"go": "the human said go"}, "descriptions")
    assert_eq(edge["mapping"], {"go": "b"}, "mapping stays separate")


@test
def test_list_checkpoints_shows_one_row_per_checkpoint_with_final_status():
    """A checkpoint is emitted twice; the listing must not show it twice."""
    from lens_harness.events import CHECKPOINT_CREATED, Event
    from lens_harness.render import list_checkpoints

    def ev(seq, cid, status):
        return Event(seq=seq, ts=0.0, run_id="r", thread_id="t", superstep=0,
                     kind=CHECKPOINT_CREATED,
                     data={"checkpoint_id": cid, "status": status, "next_nodes": []})

    events = [ev(0, "c1", "running"), ev(1, "c1", "complete"), ev(2, "c2", "running")]
    listed = list_checkpoints(events)

    assert_eq([c["checkpoint_id"] for c in listed], ["c1", "c2"], "one row per id")
    assert_eq(listed[0]["status"], "complete", "the final status wins")


@test
def test_resume_state_merge_diff_hides_reserved_channels():
    """Internal signals must not show up as meaningless diff lines."""
    fx = Fixture()
    g = StateGraph(LatchState, name="reserveddiff")
    g.add_node("ask", lambda s, c: {
        "__interrupt__": {"node": "ask", "payload": {"q": 1}, "resume_into": "done"},
    })
    g.add_node("done", lambda s, c: {"n": 5})
    g.set_entry("ask")
    g.add_edge("ask", "done")
    g.add_edge("done", END)

    runtime = Runtime(g.compile(), fx.store, fx.sink)
    result = runtime.invoke({"n": 0}, RunConfig(thread_id="t"))
    runtime.resume(result.checkpoints[-1].checkpoint_id, config=RunConfig(thread_id="t"))

    for event in fx.kinds(STATE_MERGE):
        assert_true(
            not any(name.startswith("__") for name in (event.data.get("diff") or {})),
            f"reserved channel leaked into the diff: {event.data.get('diff')}",
        )


@test
def test_summarize_state_truncates_and_hides_internals():
    summary = summarize_state({
        "big": "x" * 500,
        "lst": list(range(100)),
        "__interrupt__": {"secret": True},
        "ok": 1,
    })
    assert "__interrupt__" not in summary, "reserved channels must be hidden"
    assert summary["ok"] == 1
    assert len(summary["big"]) < 500, "long string must be truncated"


# --------------------------------------------------------------------------
# Live tailing — the read path M2's SSE lens depends on
# --------------------------------------------------------------------------
@test
def test_follow_trace_skips_a_half_written_line_then_reads_it_whole():
    """A reader must never consume a partial line.

    The writer appends ``<json>\\n`` and flushes, but those are two facts, and a
    reader can land between them. Two distinct failures hide here:

    * ``Event.from_json`` on the fragment raises ``JSONDecodeError``;
    * worse, advancing the read offset past the fragment loses the rest of the
      line forever once the writer completes it.

    This was a real bug: ``for line in fh`` handed the half-line straight to the
    parser. Terminal ``watch`` merely dropped a frame; the SSE lens would have
    dropped the connection.
    """
    import json
    import threading
    import time as _time

    from lens_harness.events import follow_trace

    fixture = Fixture()
    path = os.path.join(fixture.dir, "tail.jsonl")
    try:
        good = {
            "seq": 0, "ts": 1.0, "run_id": "r_tail", "thread_id": "t",
            "superstep": 0, "kind": "node_start",
            "data": {"node": "a", "input": {}},
        }
        line = json.dumps(good) + "\n"
        half = len(line) // 2

        def writer() -> None:
            _time.sleep(0.10)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(line[:half])
                fh.flush()
                _time.sleep(0.20)  # reader will poll inside this gap
                fh.write(line[half:])
                fh.flush()

        threading.Thread(target=writer, daemon=True).start()

        events = []
        started = _time.time()
        for ev in follow_trace(path, poll_s=0.02):
            events.append(ev)
            if events or _time.time() - started > 3.0:
                break

        assert_eq(len(events), 1, "exactly one complete event must be yielded")
        assert_eq(events[0].kind, "node_start", "parsed the right event")
        assert_eq(events[0].data["node"], "a", "payload survived intact")
    finally:
        fixture.cleanup()


@test
def test_follow_trace_ignores_blank_lines_and_keeps_reading():
    """Blank lines are not events; a later event must still arrive."""
    import json
    import threading
    import time as _time

    from lens_harness.events import follow_trace

    fixture = Fixture()
    path = os.path.join(fixture.dir, "blank.jsonl")
    try:
        def mk(seq: int, node: str) -> str:
            return json.dumps({
                "seq": seq, "ts": 1.0, "run_id": "r_b", "thread_id": "t",
                "superstep": 0, "kind": "node_start",
                "data": {"node": node, "input": {}},
            })

        def writer() -> None:
            _time.sleep(0.10)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(mk(0, "first") + "\n")
                fh.write("\n\n")  # stray blank lines
                fh.flush()
                _time.sleep(0.15)
                fh.write(mk(1, "second") + "\n")
                fh.flush()

        threading.Thread(target=writer, daemon=True).start()

        nodes = []
        started = _time.time()
        for ev in follow_trace(path, poll_s=0.02):
            nodes.append(ev.data["node"])
            if len(nodes) >= 2 or _time.time() - started > 3.0:
                break

        assert_eq(nodes, ["first", "second"], "blank lines skipped, both events seen")
    finally:
        fixture.cleanup()


@test
def test_follow_trace_recovers_when_the_file_shrinks_under_it():
    """A truncated/rotated trace must not freeze the follower forever.

    ``follow_trace`` remembers a byte offset. If the file is replaced by a
    shorter one, ``seek(pos)`` lands past EOF and the reader would see nothing
    new again — for the rest of the process — with no error to explain it.
    That is the failure mode ``serve.py`` had (DESIGN 12.9, item 4); this pins
    the same rule for the trace-level helper the CLI uses.

    The follower is driven on a worker thread on purpose. Without the fix it
    does not raise: it just never yields again, so an inline loop would hang
    the whole suite instead of failing it. A hang is not a red test.
    """
    import json
    import threading
    import time as _time

    from lens_harness.events import follow_trace

    fixture = Fixture()
    path = os.path.join(fixture.dir, "rotate.jsonl")

    def mk(seq: int, node: str) -> str:
        return json.dumps({
            "seq": seq, "ts": 1.0, "run_id": "r_rot", "thread_id": "t",
            "superstep": 0, "kind": "node_start",
            "data": {"node": node, "input": {}},
        })

    try:
        # A long first file so that `pos` ends up well past the length of the
        # replacement; otherwise the seek would land inside the new file by
        # accident and the test would pass without the fix.
        with open(path, "w", encoding="utf-8", newline="") as fh:
            for i in range(6):
                fh.write(mk(i, f"old{i}") + "\n")
            fh.flush()

        nodes: list[str] = []

        def reader() -> None:
            for ev in follow_trace(path, poll_s=0.02):
                nodes.append(ev.data["node"])
                if len(nodes) >= 7:
                    return

        def writer() -> None:
            _time.sleep(0.30)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(mk(0, "new") + "\n")
                fh.flush()

        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        threading.Thread(target=writer, daemon=True).start()
        worker.join(3.0)

        assert_true(not worker.is_alive(), "the follower must not freeze after a shrink")
        assert_eq(nodes[:6], [f"old{i}" for i in range(6)], "read the first file")
        assert_eq(
            nodes[6:], ["new"],
            "after the file shrank the follower re-read from the top",
        )
    finally:
        fixture.cleanup()


@test
def test_follow_trace_restarts_when_a_grown_rewrite_lands_mid_line():
    """A rerun that *grows* the file must not decode from the middle of a line.

    The size rule catches a shorter file and the mtime rule catches a
    same-length rewrite. Neither sees a rewrite that happened to grow: the
    remembered offset is still inside the file, so ``seek(pos)`` succeeds and
    the next ``readline`` returns the tail of some other record. That raised
    ``JSONDecodeError`` out of a live stream. The offset is only reusable if
    a newline precedes it, which is what ``offset_is_line_boundary`` checks.
    """
    import json
    import threading
    import time as _time

    from lens_harness.events import follow_trace

    fixture = Fixture()
    path = os.path.join(fixture.dir, "grown.jsonl")

    def mk(seq: int, node: str, pad: int) -> str:
        return json.dumps({
            "seq": seq, "ts": 1.0, "run_id": "r_grown", "thread_id": "t",
            "superstep": 0, "kind": "node_start",
            "data": {"node": node, "input": {"pad": "x" * pad}},
        })

    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            for i in range(3):
                fh.write(mk(i, f"old{i}", 0) + "\n")
            fh.flush()

        seqs: list[int] = []
        errors: list[str] = []

        def reader() -> None:
            try:
                for ev in follow_trace(path, poll_s=0.02):
                    seqs.append(ev.seq)
                    if len(seqs) >= 9:
                        return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        def writer() -> None:
            _time.sleep(0.30)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                for i in range(6):
                    fh.write(mk(i, f"new{i}", 40) + "\n")
                fh.flush()

        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        threading.Thread(target=writer, daemon=True).start()
        worker.join(3.0)

        assert_eq(errors, [], "a rerun must not raise out of the follower")
        assert_eq(seqs[:3], [0, 1, 2], "the first file was read first")
        assert_true(
            0 in seqs[3:],
            f"the grown replacement was re-read from the top, got {seqs}",
        )
    finally:
        fixture.cleanup()


@test
def test_follow_trace_does_not_duplicate_after_one_failed_stat():
    """One unreadable ``stat`` must not be mistaken for "the file shrank".

    Reading a failed stat as size 0 makes ``size < pos`` true for any reader
    that has already consumed a byte, so the whole trace replays as
    duplicates. A stat that raises says nothing about the file, and the poll
    is simply retried.
    """
    import json
    import threading
    import time as _time

    from lens_harness.events import follow_trace

    fixture = Fixture()
    path = os.path.join(fixture.dir, "flaky.jsonl")

    def mk(seq: int) -> str:
        return json.dumps({
            "seq": seq, "ts": 1.0, "run_id": "r_flaky", "thread_id": "t",
            "superstep": 0, "kind": "node_start",
            "data": {"node": f"n{seq}", "input": {}},
        })

    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            for i in range(3):
                fh.write(mk(i) + "\n")
            fh.flush()

        real_stat = os.stat
        calls = {"n": 0}

        def flaky_stat(target, *a, **k):
            if os.fspath(target) == os.fspath(path):
                calls["n"] += 1
                if calls["n"] in (2, 4):
                    raise OSError(13, "transient failure")
            return real_stat(target, *a, **k)

        seqs: list[int] = []

        def reader() -> None:
            for ev in follow_trace(path, poll_s=0.02):
                seqs.append(ev.seq)
                if len(seqs) >= 3:
                    return

        os.stat = flaky_stat
        try:
            worker = threading.Thread(target=reader, daemon=True)
            worker.start()
            worker.join(3.0)
        finally:
            os.stat = real_stat

        assert_eq(seqs, [0, 1, 2], "each event appears exactly once")
    finally:
        fixture.cleanup()


@test
def test_last_seq_in_file_finds_the_tail_record():
    """The resume cursor is reconciled against the file's real last seq.

    ``last_seq_in_file`` reads backwards from the end instead of scanning, so
    it must cope with a trailing newline, a half-written tail, and a file with
    no records at all.
    """
    import json

    from lens_harness.events import last_seq_in_file

    fixture = Fixture()
    path = os.path.join(fixture.dir, "tail.jsonl")

    def mk(seq: int) -> str:
        return json.dumps({
            "seq": seq, "ts": 1.0, "run_id": "r_tail", "thread_id": "t",
            "superstep": 0, "kind": "node_start",
            "data": {"node": f"n{seq}", "input": {}},
        })

    try:
        assert_eq(last_seq_in_file(path), None, "no file at all")

        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        assert_eq(last_seq_in_file(path), None, "an empty file")

        with open(path, "w", encoding="utf-8", newline="") as fh:
            for i in range(5):
                fh.write(mk(i) + "\n")
        assert_eq(last_seq_in_file(path), 4, "last complete record")

        # A half-written tail is not a record yet.
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(mk(5)[:12])
        assert_eq(last_seq_in_file(path), 4, "ignore the partial tail")

        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(mk(5)[12:] + "\n")
        assert_eq(last_seq_in_file(path), 5, "the completed record counts")
    finally:
        fixture.cleanup()


@test
def test_resume_emits_run_start_so_a_resumed_trace_is_self_describing():
    """A trace that starts at a resume must still carry the graph snapshot.

    Regression (DESIGN.md §11.3): ``resume()`` used to begin at
    ``superstep_start``, so a standalone resumed trace had no ``run_start`` —
    no graph snapshot, no entry node — and the lens could not draw it.
    """
    from lens_harness.events import RUN_START

    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="resume-meta")
        g.add_node("a", lambda s, c: {"n": s["n"] + 1, "path": ["a"]})
        g.set_entry("a")
        g.add_edge("a", END)
        runtime = Runtime(g.compile(), fixture.store, fixture.sink)
        result = runtime.invoke({"n": 0, "path": []}, RunConfig(thread_id="t"))

        before = len(fixture.kinds(RUN_START))
        runtime.resume(result.checkpoints[-1].checkpoint_id)
        starts = fixture.kinds(RUN_START)
        assert_eq(len(starts), before + 1, "resume emits exactly one run_start")
        latest = starts[-1]
        assert_true("graph" in latest.data and "entry" in latest.data.get("graph", {}),
                    "run_start carries the graph snapshot")
        assert_eq(latest.data.get("resumed_from"), result.checkpoints[-1].checkpoint_id,
                  "run_start marks what it resumed from")
    finally:
        fixture.cleanup()


@test
def test_one_runtime_can_invoke_twice_with_independent_runs():
    """A second invoke must not continue the first run's seq or run_id.

    Regression (DESIGN.md §11.11): the emitter was created once and never
    reset, so two invokes on one Runtime shared a run_id and interleaved
    their seqs in one trace.
    """
    from lens_harness.events import RUN_START

    fixture = Fixture()
    try:
        g = StateGraph(LoopState, name="twice")
        g.add_node("a", lambda s, c: {"n": s["n"] + 1, "path": ["a"]})
        g.set_entry("a")
        g.add_edge("a", END)
        runtime = Runtime(g.compile(), fixture.store, fixture.sink)
        r1 = runtime.invoke({"n": 0, "path": []}, RunConfig(thread_id="t"))
        r2 = runtime.invoke({"n": 10, "path": []}, RunConfig(thread_id="t"))

        assert_true(r1.run_id != r2.run_id, "each invoke gets its own run_id")
        starts = [e for e in fixture.sink.events if e.kind == RUN_START]
        ids = [e.run_id for e in starts]
        assert_eq(ids, [r1.run_id, r2.run_id], "two run_starts, each under its run")
        seqs = sorted(e.seq for e in fixture.sink.events if e.run_id == r2.run_id)
        assert_eq(seqs, list(range(len(seqs))), "second run's seq restarts at 0, dense")
    finally:
        fixture.cleanup()


@test
def test_error_reported_event_gives_the_reserved_channel_a_consumer():
    """Writing ``__error__`` must reach the trace, not just the checkpoint.

    Regression (DESIGN.md §11.9): the channel was accepted on write but read
    by nothing — no event, and every renderer hides ``__`` channels.
    """
    from lens_harness.events import ERROR_REPORTED

    class ErrState(TypedDict):
        n: int

    fixture = Fixture()
    try:
        g = StateGraph(ErrState, name="reported")
        g.add_node("warn", lambda s, c: {"n": 1, "__error__": {"code": "stale"}})
        g.set_entry("warn")
        g.add_edge("warn", END)
        result = fixture.run(g, {"n": 0})

        reports = fixture.kinds(ERROR_REPORTED)
        assert_eq(len(reports), 1, "one event per reporting node")
        assert_eq(reports[0].data["node"], "warn", "reporting node")
        assert_eq(reports[0].data["problem"], {"code": "stale"}, "the payload")
        # The run itself still completed: reported != fatal.
        assert_eq(result.status, "complete", "a reported problem is non-fatal")

        # A resume from the finished checkpoint must NOT re-report the old
        # problem: only a *fresh* write emits an event.
        runtime = Runtime(g.compile(), fixture.store, fixture.sink)
        runtime.resume(result.checkpoints[-1].checkpoint_id)
        assert_eq(len(fixture.kinds(ERROR_REPORTED)), 1, "no re-report on resume")
    finally:
        fixture.cleanup()


@test
def test_status_resumed_is_gone():
    """The never-written constant was removed, not wired in (DESIGN.md §11.7).

    Marking the resume anchor would clobber ``interrupted``, which the
    human-in-the-loop contract treats as a first-class state.
    """
    import lens_harness

    assert_true(not hasattr(lens_harness, "STATUS_RESUMED"),
                "STATUS_RESUMED no longer on the public surface")
    from lens_harness import checkpoint as ckpt_mod

    assert_true(not hasattr(ckpt_mod, "STATUS_RESUMED"),
                "STATUS_RESUMED gone from checkpoint.py")


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
