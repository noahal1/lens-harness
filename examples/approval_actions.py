"""Python nodes for DSL examples and tests."""

from __future__ import annotations


def do_apply(state, ctx) -> dict:
    """Approval example: execute the approved action."""
    action = state.get("action", "?")
    decision = state.get("decision", "?")
    return {"log": [f"executed {action} (decision={decision})"]}


def echo_x(state, ctx) -> dict:
    """Minimal test node: writes only the declared channel ``x``."""
    return {"x": f"echo:{state.get('x', '')}"}


def upper_item(item) -> object:
    """map-node helper: uppercase a string item."""
    return str(item).upper()


def child_state(state, ctx) -> dict:
    """For the child spec: pass-through marker that the child really ran."""
    return {"marker": f"child saw {state.get('q', '')}"}
