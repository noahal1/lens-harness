"""DSL contract tests: a JSON spec compiles to the ordinary kernel path.

The rule under test (README "trace 对比" applies here too): a spec-built graph
and a hand-built graph are indistinguishable downstream — same events, same
trace, same checkpoints. And every authoring mistake must be a SpecError
naming the offending key path, never a silent degradation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_harness.dsl import (  # noqa: E402
    BUILTIN_TOOLS,
    SpecError,
    build_graph,
    load_spec,
    run_spec,
)
from lens_harness.graph import END  # noqa: E402
from lens_harness.llm import MockLLM  # noqa: E402


def base_spec(**overrides):
    spec = {
        "name": "t",
        "state": {"x": "str"},
        "nodes": {"a": {"type": "python", "fn": "examples.approval_actions:echo_x"}},
        "entry": "a",
        "edges": {"a": "__end__"},
    }
    spec.update(overrides)
    return spec


class TestCompile(unittest.TestCase):
    def test_compiles_and_runs_a_python_node(self):
        graph = build_graph(base_spec()).compile()
        state = graph.new_state({"x": ""})
        # Run through the real kernel, not the compiled graph directly.
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        rt = Runtime(graph, CheckpointStore(tempfile.mkdtemp()), ListSink())
        res = rt.invoke({"x": "hi"}, RunConfig(thread_id="t"))
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.state["x"], "echo:hi")

    def test_schema_reducers_are_resolved_from_the_spec(self):
        spec = base_spec(state={"items": ["list[str]", "add"],
                                "n": "int"})
        compiled = build_graph(spec).compile()
        self.assertEqual(compiled.channels["items"].reducer_name, "add")
        self.assertEqual(compiled.channels["n"].reducer_name, "overwrite")

    def test_llm_tool_loop_end_to_end(self):
        spec = {
            "name": "loop",
            "state": {"messages": ["list[dict[str, str]]", "add"],
                      "answer": "str", "steps": "int",
                      "tools_used": ["list[str]", "add_unique"]},
            "llm": {"replies": ["TOOL: calculator | 2+3",
                                "FINAL: five"]},
            "tools": {"calculator": "builtin:calculator"},
            "nodes": {"agent": {"type": "llm"},
                      "tools": {"type": "tool_loop"}},
            "entry": "agent",
            "edges": {"agent": {"condition": "tool_call",
                                "then": {"tools": "tools", "end": "__end__"}},
                      "tools": "agent"},
        }
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        rt = Runtime(build_graph(spec).compile(),
                     CheckpointStore(tempfile.mkdtemp()), ListSink())
        res = rt.invoke({"messages": [], "answer": "", "steps": 0,
                         "tools_used": []}, RunConfig(thread_id="t"))
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.state["answer"], "5")
        self.assertEqual(res.state["tools_used"], ["calculator"])
        # A FINAL: turn must not be mistaken for a tool call.
        self.assertEqual(res.supersteps, 3)  # agent, tools, agent(final)

    def test_interrupt_node_pauses_and_resume_completes(self):
        spec = {
            "name": "gate",
            "state": {"decision": "str", "log": ["list[str]", "add"]},
            "nodes": {
                "gate": {"type": "interrupt", "question": "ok?",
                         "resume_into": "apply"},
                "apply": {"type": "python",
                          "fn": "examples.approval_actions:do_apply"},
            },
            "entry": "gate",
            "edges": {"gate": "apply", "apply": "__end__"},
        }
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        store = CheckpointStore(tempfile.mkdtemp())
        rt = Runtime(build_graph(spec).compile(), store, ListSink())
        res = rt.invoke({"decision": "", "log": []}, RunConfig(thread_id="t"))
        self.assertEqual(res.status, "interrupted")
        res2 = rt.resume(res.checkpoints[-1].checkpoint_id,
                         payload={"decision": "yes"})
        self.assertEqual(res2.status, "complete")
        self.assertEqual(res2.state["log"], ["executed ? (decision=yes)"])


class TestNewTemplates(unittest.TestCase):
    """transform / map / subgraph — the declarative building blocks."""

    def _run(self, spec, inputs):
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        rt = Runtime(build_graph(spec).compile(),
                     CheckpointStore(tempfile.mkdtemp()), ListSink())
        return rt.invoke(inputs, RunConfig(thread_id="t"))

    def _run_expect_failure(self, spec, inputs):
        """Run and yield the error text; node-level SpecErrors surface as
        failed runs (kernel records node_error) — that IS the loud failure."""
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        class _Expect:
            def __enter__(self):
                res = Runtime(build_graph(spec).compile(),
                              CheckpointStore(tempfile.mkdtemp()),
                              ListSink()).invoke(inputs, RunConfig(thread_id="t"))
                assert res.status == "failed", f"expected failed, got {res.status}"
                return str(res.error)

            def __exit__(self, *a):
                return False

        return _Expect()

    def test_transform_substitutes_channels(self):
        spec = base_spec(
            state={"x": "str", "report": "str"},
            nodes={"a": {"type": "python", "fn": "examples.approval_actions:echo_x"},
                   "rep": {"type": "transform", "set": {"report": "got: {x}!"}}},
            edges={"a": "rep", "rep": "__end__"},
        )
        res = self._run(spec, {"x": "hi", "report": ""})
        self.assertEqual(res.state["report"], "got: echo:hi!")

    def test_transform_unknown_channel_ref_fails_the_node(self):
        # A bad {ref} inside a node cannot be a compile-time error (it depends
        # on runtime state), so the kernel records it as a node_error — the
        # message must still name the offending placeholder.
        spec = base_spec(
            state={"x": "str"},
            nodes={"a": {"type": "transform", "set": {"x": "{ghost}"}}},
        )
        rt = build_graph(spec).compile()
        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink
        res = Runtime(rt, CheckpointStore(tempfile.mkdtemp()), ListSink()).invoke(
            {"x": ""}, RunConfig(thread_id="t"))
        self.assertEqual(res.status, "failed")
        self.assertIn("ghost", str(res.error))

    def test_map_applies_fn_to_every_item(self):
        spec = base_spec(
            state={"x": "str", "items": ["list[str]", "overwrite"],
                   "results": ["list[str]", "overwrite"]},
            nodes={"a": {"type": "map", "fn": "examples.approval_actions:upper_item",
                         "over": "items", "into": "results"}},
        )
        res = self._run(spec, {"x": "", "items": ["a", "b"],
                               "results": []})
        self.assertEqual(res.state["results"], ["A", "B"])

    def test_map_rejects_non_list_channel(self):
        spec = base_spec(
            state={"x": "str", "items": "str", "results": "list[str]"},
            nodes={"a": {"type": "map", "fn": "examples.approval_actions:upper_item",
                         "over": "items", "into": "results"}},
        )
        with self._run_expect_failure(spec, {"x": "", "items": 42, "results": []}) as err:
            self.assertIn("must be a list", err)

    def test_subgraph_runs_child_and_maps_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = os.path.join(tmp, "child.json")
            with open(child, "w") as fh:
                json.dump({
                    "name": "child",
                    "state": {"q": "str", "marker": "str"},
                    "nodes": {"c": {"type": "python",
                                    "fn": "examples.approval_actions:child_state"}},
                    "entry": "c", "edges": {"c": "__end__"},
                }, fh)
            parent = {
                "name": "parent",
                "state": {"x": "str", "marker": "str", "sub_run": "str"},
                "nodes": {"sub": {"type": "subgraph", "spec": "child.json",
                                  "inputs": {"x": "q"},
                                  "outputs": {"marker": "marker"},
                                  "run_into": "sub_run"}},
                "entry": "sub",
                "edges": {"sub": "__end__"},
            }
            graph = build_graph(parent, spec_dir=tmp).compile()
            from lens_harness.runtime import RunConfig, Runtime
            from lens_harness.checkpoint import CheckpointStore
            from lens_harness.events import ListSink

            rt = Runtime(graph, CheckpointStore(tempfile.mkdtemp()), ListSink())
            res = rt.invoke({"x": "hello", "marker": "", "sub_run": ""},
                            RunConfig(thread_id="t"))
            self.assertEqual(res.status, "complete")
            self.assertEqual(res.state["marker"], "child saw hello")
            self.assertTrue(res.state["sub_run"].startswith("r_"))

    def test_subgraph_mapping_channels_must_be_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = os.path.join(tmp, "child.json")
            with open(child, "w") as fh:
                json.dump(base_spec(), fh)
            spec = base_spec(
                nodes={"a": {"type": "subgraph", "spec": "child.json",
                             "inputs": {"nope": "x"}}},
            )
            with self.assertRaises(SpecError) as cm:
                build_graph(spec, spec_dir=tmp)
            self.assertIn("inputs.nope", str(cm.exception))


class TestSpecErrors(unittest.TestCase):
    def assertSpecError(self, spec, fragment):
        with self.assertRaises(SpecError) as cm:
            build_graph(spec)
        self.assertIn(fragment, str(cm.exception))

    def test_bad_json_path_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            with self.assertRaises(SpecError) as cm:
                load_spec(path)
            self.assertIn("line 1", str(cm.exception))

    def test_unknown_node_type(self):
        self.assertSpecError(base_spec(nodes={"a": {"type": "magic"}}),
                             "nodes.a.type")

    def test_unknown_reducer(self):
        self.assertSpecError(
            base_spec(state={"x": ["str", "shout"]}), "reducer")

    def test_unknown_type(self):
        self.assertSpecError(base_spec(state={"x": "tensor"}), "state.x")

    def test_conditional_edge_target_must_be_declared(self):
        self.assertSpecError(
            base_spec(edges={"a": {"condition": "always",
                                   "then": {"tools": "ghost"}}}),
            "edges.a.then.tools")

    def test_router_returning_undeclared_key_fails_loudly(self):
        # A condition (python) returning a key outside the declared mapping
        # must fail at routing time with a message naming both sides.
        import tempfile as _tf

        from lens_harness.runtime import RunConfig, Runtime
        from lens_harness.checkpoint import CheckpointStore
        from lens_harness.events import ListSink

        mod = os.path.join(_tf.mkdtemp(), "badcond.py")
        with open(mod, "w") as fh:
            fh.write("def route(state):\n    return 'nowhere'\n")
        sys.path.insert(0, os.path.dirname(mod))
        try:
            spec = base_spec(
                edges={"a": {"condition": "python:badcond:route",
                             "then": {"end": "__end__"}}},
            )
            rt = Runtime(build_graph(spec).compile(),
                         CheckpointStore(_tf.mkdtemp()), ListSink())
            with self.assertRaises(SpecError) as cm:
                rt.invoke({"x": ""}, RunConfig(thread_id="t"))
            self.assertIn("'nowhere'", str(cm.exception))
        finally:
            sys.path.pop(0)
            sys.modules.pop("badcond", None)

    def test_unknown_builtin_tool(self):
        self.assertSpecError(
            base_spec(tools={"c": "builtin:nope"}), "tools.c")

    def test_python_ref_must_resolve(self):
        self.assertSpecError(
            base_spec(nodes={"a": {"type": "python", "fn": "nope:missing"}}),
            "nodes.a.fn")

    def test_interrupt_requires_resume_into(self):
        self.assertSpecError(
            base_spec(nodes={"a": {"type": "interrupt"}}), "resume_into")


class TestCliEntry(unittest.TestCase):
    def test_run_spec_writes_a_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = os.path.join(tmp, "spec.json")
            with open(spec_path, "w") as fh:
                json.dump(base_spec(), fh)
            trace_dir = os.path.join(tmp, "traces")
            events = run_spec(spec_path, {"x": "cli"}, trace_dir=trace_dir)
            kinds = [e.kind for e in events]
            self.assertIn("run_start", kinds)
            self.assertIn("run_end", kinds)
            traces = os.listdir(trace_dir)
            self.assertTrue(any(t.endswith(".jsonl") for t in traces))


if __name__ == "__main__":
    unittest.main(verbosity=2)
