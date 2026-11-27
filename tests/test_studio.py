"""Studio session tests: commands mutate the spec and always run the real kernel.

Style note: the rest of the suite is unittest, so this file is too — the first
draft used pytest-style bare functions (tmp_path fixtures) and collected zero
tests under `python3 -m unittest`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lens_harness.studio import StudioSession  # noqa: E402


class StudioTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="studio-test-")
        self.s = StudioSession(os.path.join(self.tmp, "studio_traces"))

    def run_cmd(self, **payload):
        return self.s.handle(payload)

    def test_init_then_run_sequential(self):
        self.assertTrue(self.run_cmd(cmd="init", template="sequential")["ok"])
        out = self.run_cmd(cmd="run", input={"x": "hi"})
        self.assertTrue(out["ok"])
        self.assertTrue(os.path.isfile(out["trace"]))
        events = [json.loads(l) for l in open(out["trace"], encoding="utf-8")]
        kinds = {e["kind"] for e in events}
        self.assertIn("run_start", kinds)
        self.assertIn("run_end", kinds)

    def test_llm_tool_loop_completes(self):
        self.run_cmd(cmd="init", template="llm_tool_loop")
        out = self.run_cmd(cmd="run", input={"question": "what is 2+3"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["runs"][-1]["status"], "complete")
        self.assertTrue(out["runs"][-1]["state"]["answer"])

    def test_unknown_template_fails_into_chat(self):
        out = self.run_cmd(cmd="init", template="nope")
        self.assertFalse(out["ok"])
        self.assertIn("nope", out["messages"][0]["text"])

    def test_run_without_spec_is_reported_not_raised(self):
        out = self.run_cmd(cmd="run")
        self.assertFalse(out["ok"])
        self.assertIn("init", out["messages"][0]["text"])

    def test_add_node_and_remove_keeps_spec_compiling(self):
        self.run_cmd(cmd="init", template="sequential")
        self.assertTrue(self.run_cmd(
            cmd="add_node", name="n1",
            node={"type": "transform", "set": {"x": "v: {x}"}})["ok"])
        self.assertTrue(self.run_cmd(cmd="set_edge", **{"from": "n1", "to": "__end__"})["ok"])
        self.assertTrue(self.run_cmd(cmd="remove_node", name="n1")["ok"])
        out = self.run_cmd(cmd="run", input={"x": "bye"})
        self.assertTrue(out["ok"])

    def test_broken_edge_fails_loudly(self):
        self.run_cmd(cmd="init", template="sequential")
        out = self.run_cmd(cmd="set_edge", **{"from": "a", "to": "ghost"})
        self.assertFalse(out["ok"])
        self.assertIn("ghost", out["messages"][0]["text"])

    def test_reset_clears_session(self):
        self.run_cmd(cmd="init", template="sequential")
        self.assertTrue(self.run_cmd(cmd="reset")["ok"])
        self.assertFalse(self.run_cmd(cmd="run")["ok"])

    def test_get_spec_roundtrips(self):
        self.run_cmd(cmd="init", template="sequential")
        out = self.run_cmd(cmd="get_spec")
        self.assertIn("```json", out["messages"][0]["text"])

    def test_langgraph_style_templates_complete(self):
        """supervisor / reflection / plan_execute / debate run to completion."""
        for name, probe in (("supervisor", "result"),
                            ("reflection", "verdict"),
                            ("plan_execute", "results"),
                            ("debate", "log")):
            out = self.run_cmd(cmd="init", template=name)
            self.assertTrue(out["ok"], name)
            out = self.run_cmd(cmd="run", input=self.s.default_input)
            self.assertTrue(out["ok"], name)
            record = out["runs"][-1]
            self.assertEqual(record["status"], "complete", name)
            self.assertIn(probe, record["state"], name)

    def test_init_strips_studio_metadata(self):
        self.run_cmd(cmd="init", template="supervisor")
        self.assertNotIn("desc", self.s.spec)
        self.assertNotIn("desc_en", self.s.spec)
        self.assertNotIn("input", self.s.spec)


if __name__ == "__main__":
    import os as _os
    _os.chdir(_os.path.join(_os.path.dirname(__file__), ".."))
    unittest.main()
