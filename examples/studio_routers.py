"""Router functions for the studio's LangGraph-style templates.

Studio specs reference these as ``python:examples.studio_routers:func`` in a
conditional edge's ``condition``. Each receives the run state and returns a
key that must appear in that edge's ``then`` mapping — the same contract the
DSL pins for any python router (unknown keys fail loudly at runtime).
"""


def supervisor_route(state: dict) -> str:
    """Supervisor pattern: does the worker still have tasks, or are we done?

    Reads ``tasks`` (list remaining) and ``rounds`` (safety counter): a run
    that never converges ends after 6 supervisor rounds instead of looping
    forever.
    """
    if not state.get("tasks") or state.get("rounds", 0) >= 6:
        return "end"
    return "work"


def reflection_route(state: dict) -> str:
    """Reflection pattern: retry while the critic rejects, up to 3 rounds."""
    if state.get("verdict") == "reject" and state.get("rounds", 0) < 3:
        return "retry"
    return "done"


def plan_route(state: dict) -> str:
    """Plan-and-execute: dispatch the next step or finish the plan."""
    if not state.get("plan") or state.get("rounds", 0) >= 8:
        return "finish"
    return "execute"


def handback_route(state: dict) -> str:
    """Debate pattern: alternate between the two agents for N rounds."""
    if state.get("rounds", 0) >= 4:
        return "end"
    return "reply"

def pop_task(state, ctx):
    """supervisor: pop the first task into the current slot, bump rounds."""
    tasks = list(state.get("tasks") or [])
    cur = tasks.pop(0) if tasks else ""
    return {"tasks": tasks, "current": cur,
            "rounds": state.get("rounds", 0) + 1}

def do_work(state, ctx):
    """worker: pretend to execute the current task (deterministic mock)."""
    return {"result": f"done: {state.get('current', '')}",
            "log": [f"worked on {state.get('current', '')}"]}

def critique(state, ctx):
    """reflection critic: pass on round>=2, reject earlier (demo script)."""
    r = state.get("rounds", 0)
    return {"verdict": "pass" if r >= 2 else "reject",
            "notes": f"review round {r}: " + ("ok" if r >= 2 else "needs another pass")}

def revise(state, ctx):
    return {"draft": f"v{state.get('rounds', 0) + 1}: improved draft",
            "rounds": state.get("rounds", 0) + 1}

def draft_v1(state, ctx):
    return {"draft": "v0: first draft", "rounds": 0, "verdict": "reject"}

def next_step(state, ctx):
    """plan-and-execute: shift one step off the plan into results."""
    plan = list(state.get("plan") or [])
    step = plan.pop(0) if plan else ""
    return {"plan": plan, "results": [f"ran {step}"],
            "rounds": state.get("rounds", 0) + 1}

def make_plan(state, ctx):
    plan = [s.strip() for s in str(state.get("goal", "")).split("然后") if s.strip()]
    return {"plan": plan, "rounds": 0}

def debate_a(state, ctx):
    return {"log": ["A argues"], "rounds": state.get("rounds", 0) + 1}

def debate_b(state, ctx):
    return {"log": ["B rebuts"]}
