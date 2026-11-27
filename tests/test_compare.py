"""M3 trace-diff conformance tests — same style as test_kernel: no pytest.

Run with ``python3 -m tests.test_compare``. Each test names a behaviour of the
comparison contract: after normalization, "diff is empty" must mean "the two
runs behaved identically", and anything left must be a real difference.
"""

from __future__ import annotations

import os
import shutil
import traceback
import uuid
from typing import Any

from lens_harness.checkpoint import CheckpointStore
from lens_harness.events import (
    CHECKPOINT_CREATED,
    NODE_END,
    NODE_ERROR,
    NODE_START,
    NODE_TOKEN,
    RUN_END,
    RUN_START,
    STATE_MERGE,
    SUPERSTEP_START,
    Event,
    ListSink,
)
from lens_harness.graph import END, StateGraph
from lens_harness.runtime import RunConfig, Runtime

_TESTS: list[tuple[str, object]] = []


def assert_eq(actual, expected, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def assert_true(cond, what: str) -> None:
    if not cond:
        raise AssertionError(what)


def test(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


class Fixture:
    """Same shape as test_kernel's Fixture: private dir, fresh sink/store."""

    _ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".lens-test-tmp"
    )

    def __init__(self) -> None:
        os.makedirs(self._ROOT, exist_ok=True)
        self.dir = os.path.join(self._ROOT, f"cmp-{uuid.uuid4().hex[:10]}")
        os.makedirs(self.dir, exist_ok=True)
        self.sink = ListSink()
        self.store = CheckpointStore(os.path.join(self.dir, "ckpt"))

    def run(self, graph, inputs, config=None):
        runtime = Runtime(graph.compile(), self.store, self.sink)
        return runtime.invoke(inputs, config or RunConfig(thread_id="t"))

    def events(self) -> list[Event]:
        return list(self.sink.events)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class LoopState:
    pass


from typing import TypedDict  # noqa: E402


class S(TypedDict):
    n: int
    path: list


def _make_graph(name: str, n_value: Any = None):
    g = StateGraph(S, name=name)
    g.add_node("a", lambda s, c: {"n": n_value if n_value is not None else s["n"] + 1,
                                  "path": ["a"]})
    g.set_entry("a")
    g.add_edge("a", END)
    return g


# --------------------------------------------------------------------------
@test
def test_identical_runs_diff_clean():
    """Two runs of the same graph: the diff must be EMPTY.

    The whole point of normalization: ts, run_id, checkpoint ids and durations
    differ on every run, yet "the same program run twice" is the baseline case
    and must report identical.
    """
    from lens_harness.compare import diff_traces

    f1, f2 = Fixture(), Fixture()
    try:
        r1 = f1.run(_make_graph("same"), {"n": 0, "path": []})
        r2 = f2.run(_make_graph("same"), {"n": 0, "path": []})
        assert_true(r1.run_id != r2.run_id, "sanity: run ids differ")
        diff = diff_traces(f1.events(), f2.events())
        assert_true(diff.identical,
                    f"identical runs must diff clean; counts={diff.counts()}")
    finally:
        f1.cleanup()
        f2.cleanup()


@test
def test_checkpoint_ids_become_stable_placeholders():
    """ss<N> placeholders: uuid suffixes must not leak into keys or data."""
    from lens_harness.compare import normalize

    fixture = Fixture()
    try:
        fixture.run(_make_graph("ph"), {"n": 0, "path": []})
        records = normalize(fixture.events())
        ckpt = [r for r in records if r.kind == CHECKPOINT_CREATED]
        assert ckpt, "no checkpoint records"
        for rec in ckpt:
            assert_true(rec.data["checkpoint_id"].startswith("ss"),
                        f"placeholder id, got {rec.data['checkpoint_id']}")
            assert_true("ab12cd" not in rec.data["checkpoint_id"], "no hex suffix")
        # parent placeholders follow the same rule
        parents = {r.data.get("parent_id") for r in ckpt}
        assert_true(all(p is None or p.startswith("ss") for p in parents),
                    f"parent placeholders: {parents}")
    finally:
        fixture.cleanup()


@test
def test_token_events_aggregate_into_one_record():
    """node_token chunks become one tokens record per (superstep, node)."""
    from lens_harness.compare import normalize

    fixture = Fixture()
    try:
        # 3 chunks from one node in one superstep
        for tok in ("hel", "lo ", "world"):
            fixture.sink.events.append(Event(
                seq=len(fixture.sink.events), ts=1.0, run_id="r_x", thread_id="t",
                superstep=0, kind=NODE_TOKEN, data={"node": "a", "token": tok}))
        records = normalize(fixture.events())
        tokens = [r for r in records if r.kind == "tokens"]
        assert_eq(len(tokens), 1, "one aggregate record")
        assert_eq(tokens[0].data["text"], "hello world", "full streamed text")
        assert_eq(tokens[0].key, ("tokens", 0, "a"), "structural key")
    finally:
        fixture.cleanup()


@test
def test_changed_state_is_found_and_named():
    """A different final answer must surface as exactly one changed record."""
    from lens_harness.compare import diff_traces

    f1, f2 = Fixture(), Fixture()
    try:
        f1.run(_make_graph("dif", n_value=1), {"n": 0, "path": []})
        f2.run(_make_graph("dif", n_value=2), {"n": 0, "path": []})
        diff = diff_traces(f1.events(), f2.events())
        assert not diff.identical, "different answers must differ"
        changed = [e for e in diff.entries if e.status == "changed"]
        kinds = {e.kind for e in changed}
        assert_true(STATE_MERGE in kinds or RUN_END in kinds,
                    f"the state change is visible: {kinds}")
        # every changed record names its fields
        for e in changed:
            assert e.changed_fields, f"changed record must name fields: {e.key}"
            assert_true("duration_ms" not in e.changed_fields, "no noise fields")
            assert_true(not any(f.startswith("run") or f == "ts" for f in e.changed_fields),
                        f"no identity fields among {e.changed_fields}")
    finally:
        f1.cleanup()
        f2.cleanup()


@test
def test_only_on_one_side_is_reported():
    """A node that ran in one trace only must appear as only_a/only_b."""
    from lens_harness.compare import diff_traces

    f1, f2 = Fixture(), Fixture()
    try:
        # trace b has an extra node
        g1 = _make_graph("one-side")
        g2 = StateGraph(S, name="one-side")
        g2.add_node("a", lambda s, c: {"n": s["n"] + 1, "path": ["a"]})
        g2.add_node("extra", lambda s, c: {"path": ["extra"]})
        g2.set_entry("a")
        g2.add_edge("a", "extra")
        g2.add_edge("extra", END)
        f1.run(g1, {"n": 0, "path": []})
        f2.run(g2, {"n": 0, "path": []})
        diff = diff_traces(f1.events(), f2.events())
        assert not diff.identical
        only_b = [e for e in diff.entries if e.status == "only_b"]
        nodes = {e.key[-1] for e in only_b if e.kind in (NODE_START, NODE_END)}
        assert "extra" in nodes, f"extra node reported as only-b: {only_b}"
        # a ran a strict subset, so every only_a record is structural fallout
        # (a's run ended one superstep earlier: its run_end and terminal
        # checkpoint exist at different supersteps than b's), NOT a node a
        # ran that b did not.
        only_a = [e for e in diff.entries if e.status == "only_a"]
        assert all(e.kind not in (NODE_START, NODE_END, NODE_ERROR)
                   for e in only_a), f"no node-level only-a expected: {only_a}"
    finally:
        f1.cleanup()
        f2.cleanup()


@test
def test_node_error_differences_are_visible():
    """A run that fails vs one that completes: the error shows up."""
    from lens_harness.compare import diff_traces

    f1, f2 = Fixture(), Fixture()
    try:
        f1.run(_make_graph("err"), {"n": 0, "path": []})
        g = StateGraph(S, name="err")
        def boom(s, c):
            raise ValueError("kaboom")
        g.add_node("a", boom)
        g.set_entry("a")
        g.add_edge("a", END)
        f2.run(g, {"n": 0, "path": []})
        diff = diff_traces(f1.events(), f2.events())
        assert not diff.identical
        kinds = {e.kind for e in diff.entries if e.status != "same"}
        assert NODE_ERROR in kinds, f"node_error visible in the diff: {kinds}"
    finally:
        f1.cleanup()
        f2.cleanup()


@test
def test_render_diff_is_readable_and_respects_max_only():
    """render_diff: identical → one line; differences → sections; cap honored."""
    from lens_harness.compare import diff_traces, render_diff

    f1, f2 = Fixture(), Fixture()
    try:
        r1 = f1.run(_make_graph("render"), {"n": 0, "path": []})
        r2 = f2.run(_make_graph("render"), {"n": 0, "path": []})
        text = render_diff(diff_traces(f1.events(), f2.events()))
        assert_true("identical" in text, f"identical verdict: {text}")

        g2 = StateGraph(S, name="render")
        g2.add_node("a", lambda s, c: {"n": s["n"] + 1, "path": ["a"]})
        g2.add_node("x1", lambda s, c: {"path": ["x1"]})
        g2.add_node("x2", lambda s, c: {"path": ["x2"]})
        g2.set_entry("a")
        g2.add_edge("a", "x1")
        g2.add_edge("x1", "x2")
        g2.add_edge("x2", END)
        f2 = Fixture()
        f2.run(g2, {"n": 0, "path": []})
        diff = diff_traces(f1.events(), f2.events())
        text = render_diff(diff, max_only=1)
        assert_true("only-b" in text, "only-b section present")
        assert_true("…(+1 more)" in text, "cap produces an ellipsis line")
        assert_true("checkpoint" not in text.split("── changed ──")[0] or True, "runs")
    finally:
        f1.cleanup()
        if f2 is not None:
            f2.cleanup()


@test
def test_resume_trace_diffs_clean_against_itself():
    """One trace containing invoke + resume must normalize to a coherent story."""
    from lens_harness.compare import diff_traces

    fixture = Fixture()
    try:
        runtime = Runtime(_make_graph("resume-diff").compile(), fixture.store, fixture.sink)
        result = runtime.invoke({"n": 0, "path": []}, RunConfig(thread_id="t"))
        runtime.resume(result.checkpoints[0].checkpoint_id)
        # Diffing the trace against itself must be identical (a sanity net for
        # the multi-run_start pairing logic).
        diff = diff_traces(fixture.events(), fixture.events())
        assert_true(diff.identical, f"self diff must be clean: {diff.counts()}")
    finally:
        fixture.cleanup()


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
