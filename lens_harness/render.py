"""Terminal renderer — turn a trace file into something a human can read.

M1's whole point: if the trace is not readable in a terminal, no amount of UI
will save it. This renderer is the first consumer of the event contract and
therefore the first check that the contract is right.

Input is only a list of :class:`~lens_harness.events.Event`. It never imports
graph or runtime code — same constraint the Web lens will obey.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .events import (
    CHECKPOINT_CREATED,
    EVENT_SCHEMA,
    INTERRUPT,
    NODE_END,
    NODE_ERROR,
    NODE_START,
    NODE_TOKEN,
    ROUTE_DECISION,
    RUN_END,
    RUN_START,
    STATE_MERGE,
    SUPERSTEP_START,
    Event,
)

# --------------------------------------------------------------------------
# Color: one place, easy to turn off.
# --------------------------------------------------------------------------
class Palette:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, t: str) -> str: return self._wrap("2", t)
    def bold(self, t: str) -> str: return self._wrap("1", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def magenta(self, t: str) -> str: return self._wrap("35", t)
    def blue(self, t: str) -> str: return self._wrap("34", t)


def make_palette(enabled: bool | None = None) -> Palette:
    if enabled is None:
        enabled = os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"
    return Palette(enabled)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
@dataclass(slots=True)
class RenderOptions:
    show_tokens: bool = True
    show_state: bool = True
    show_diff: bool = True
    token_preview: int = 80
    indent: str = "  "


def _short(value: Any, limit: int = 100) -> str:
    """One-line rendering of an arbitrary state value."""
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
        return text if len(text) <= limit else text[:limit] + "…"
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _value_style(value: Any, pal: Palette) -> str:
    if isinstance(value, str):
        return pal.green(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pal.cyan(value)
    if isinstance(value, bool):
        return pal.magenta(value)
    return value


def render_events(
    events: Sequence[Event],
    *,
    options: RenderOptions | None = None,
    palette: Palette | None = None,
) -> str:
    opts = options or RenderOptions()
    pal = palette or make_palette()
    lines: list[str] = []
    add = lines.append

    # Group by superstep so the output reads as "what happened in this step"
    # rather than as an undifferentiated log.
    token_buffer: dict[str, list[str]] = {}
    # NODE_START records where each node's headline landed so NODE_END can
    # append its duration to that same line.
    node_line: dict[str, int] = {}

    def flush_tokens(node: str | None = None, depth: int = 1) -> None:
        """Emit buffered tokens for ``node`` (or all nodes when None).

        Tokens stream *while* the node runs, but they read best *after* the
        node's writes — so the buffer is flushed on NODE_END rather than at the
        superstep boundary (which would land them past the checkpoint line).
        """
        targets = [node] if node is not None else list(token_buffer)
        for name in targets:
            chunks = token_buffer.pop(name, [])
            if chunks:
                text = "".join(chunks)
                preview = text if len(text) <= opts.token_preview else text[: opts.token_preview] + "…"
                pad = opts.indent * depth
                add(f"{pad}{pal.dim('↳ tokens')} {pal.dim(name)}: {preview!r}")

    last_superstep: int | None = None
    for event in events:
        kind, data = event.kind, event.data

        # SUPERSTEP_START renders its own header; letting this branch fire too
        # printed every "── superstep N ──" banner twice.
        if (event.superstep != last_superstep and event.superstep >= 0
                and kind != SUPERSTEP_START):
            flush_tokens()
            add("")
            add(pal.bold(pal.blue(f"── superstep {event.superstep} " + "─" * 46)))
            last_superstep = event.superstep

        if kind == RUN_START:
            graph = data.get("graph", {})
            add(pal.bold(f"▶ run {event.run_id}") + pal.dim(f"  thread={event.thread_id}"))
            add(f"{opts.indent}graph: {pal.cyan(graph.get('name', '?'))}  "
                f"entry: {pal.cyan(graph.get('entry', '?'))}")
            node_names = [n["name"] for n in graph.get("nodes", [])]
            add(f"{opts.indent}nodes: {', '.join(node_names)}")
            channels = graph.get("channels", [])
            if channels:
                add(f"{opts.indent}channels: " + ", ".join(
                    f"{c['name']}[{c['reducer']}]" for c in channels
                ))
            if data.get("inputs"):
                add(f"{opts.indent}inputs: " + ", ".join(
                    f"{k}={_short(v, 40)}" for k, v in sorted(data["inputs"].items())
                ))

        elif kind == SUPERSTEP_START:
            nodes = data.get("nodes", [])
            add("")
            add(pal.bold(pal.blue(f"── superstep {data.get('index')} " + "─" * 46)))
            add(f"{opts.indent}plan: " + ", ".join(pal.cyan(n) for n in nodes))
            last_superstep = event.superstep

        elif kind == NODE_START:
            node_line[data["node"]] = len(lines)
            add(f"{opts.indent}{pal.bold('▸ ' + data['node'])}")

        elif kind == NODE_TOKEN:
            if opts.show_tokens:
                token_buffer.setdefault(data["node"], []).append(data["token"])

        elif kind == NODE_END:
            writes = data.get("writes") or {}
            duration = data.get("duration_ms", 0)
            # The duration belongs to the node's headline, not to a line of its
            # own: a bare "0.0 ms" floating under the writes reads like a
            # separate event. Parallel nodes finish in a different order than
            # they start, so the line to patch is looked up by name.
            idx = node_line.get(data["node"])
            if idx is not None and idx < len(lines):
                lines[idx] += "  " + pal.dim(f"{duration:.1f} ms")
            if writes:
                pairs = ", ".join(f"{k}={_short(v, 60)}" for k, v in sorted(writes.items()))
                add(f"{opts.indent}{opts.indent}{pal.dim('writes')}: {pairs}")
            flush_tokens(data["node"], depth=2)

        elif kind == NODE_ERROR:
            add(f"{opts.indent}{opts.indent}{pal.red('✖ ' + data['error_type'])}: {data['error']}")

        elif kind == STATE_MERGE and opts.show_state:
            changed = data.get("changed_channels") or []
            origin = data.get("origin")
            label = "state · resume payload" if origin == "resume_payload" else "state"
            if changed:
                add(f"{opts.indent}{pal.magenta('⟳ ' + label)} changed: "
                    + ", ".join(pal.magenta(c) for c in changed))
            else:
                add(f"{opts.indent}{pal.dim('⟳ ' + label + ' unchanged')}")
            if opts.show_diff:
                for name, delta in (data.get("diff") or {}).items():
                    before = _short(delta.get("before", "<empty>"), 60)
                    after = _short(delta.get("after", "<empty>"), 60)
                    add(f"{opts.indent}{opts.indent}{pal.dim(name)}: "
                        f"{pal.red(before)} {pal.dim('→')} {pal.green(after)}")

        elif kind == ROUTE_DECISION:
            source = data.get("source")
            candidates = data.get("candidates") or []
            selected = data.get("selected") or []
            condition = data.get("condition")
            # The star of the show: show ALL candidates, then what was chosen.
            cand_text = " | ".join(
                (pal.bold(pal.green(c)) if c in selected else pal.dim(c)) for c in candidates
            )
            if condition:
                add(f"{opts.indent}{pal.yellow('⑂ route')} {pal.dim('after')} {pal.cyan(source)} "
                    f"{pal.dim('via')} {pal.yellow(condition)}()")
                add(f"{opts.indent}{opts.indent}candidates: {cand_text}")
                add(f"{opts.indent}{opts.indent}returned:   {pal.yellow(repr(data.get('returned')))}")
                add(f"{opts.indent}{opts.indent}selected:   {pal.green(', '.join(selected) or '<none>')}")
            else:
                add(f"{opts.indent}{pal.dim('→ edge')} {pal.cyan(source)} → "
                    f"{pal.green(', '.join(selected) or '<end>')}")

        elif kind == CHECKPOINT_CREATED:
            status = data.get("status", "running")
            color = {"interrupted": pal.yellow, "failed": pal.red, "complete": pal.green}.get(
                status, pal.dim
            )
            nxt = ", ".join(data.get("next_nodes") or []) or "<end>"
            add(f"{opts.indent}{pal.dim('◆ checkpoint')} {pal.dim(data['checkpoint_id'])} "
                f"{color(status)} {pal.dim('next=' + nxt)}")

        elif kind == INTERRUPT:
            add(f"{opts.indent}{pal.yellow('⏸ interrupt')} at {pal.cyan(data.get('node'))} "
                f"{pal.dim('payload=' + _short(data.get('payload'), 80))}")

        elif kind == RUN_END:
            flush_tokens()
            status = data.get("status")
            color = {"complete": pal.green, "failed": pal.red, "interrupted": pal.yellow}.get(
                status, pal.dim
            )
            add("")
            add(pal.bold(color(f"■ run end: {status}")) +
                pal.dim(f"  supersteps={data.get('supersteps')} "
                        f"ckpt={data.get('checkpoint_id')}"))
            if data.get("error"):
                add(f"{opts.indent}{pal.red(str(data['error']))}")
            if opts.show_state and data.get("final_state"):
                add(f"{opts.indent}final state:")
                for k, v in sorted(data["final_state"].items()):
                    if k.startswith("__"):
                        continue
                    add(f"{opts.indent}{opts.indent}{k} = {_value_style(_short(v, 80), pal)}")

    flush_tokens()
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Secondary views — cheap, high value for learning
# --------------------------------------------------------------------------
def render_checkpoint_dag(events: Sequence[Event], palette: Palette | None = None) -> str:
    """ASCII tree of the checkpoint DAG: the run's *structure* at a glance."""
    pal = palette or make_palette()
    nodes = {e.data["checkpoint_id"]: e.data for e in events if e.kind == CHECKPOINT_CREATED}
    children: dict[str | None, list[str]] = {}
    for cid, data in nodes.items():
        children.setdefault(data.get("parent_id"), []).append(cid)

    lines = [pal.bold("checkpoint DAG")]
    def walk(cid: str, depth: int, last: bool) -> None:
        data = nodes[cid]
        status = data.get("status", "running")
        marker = {"interrupted": "⏸", "failed": "✖", "complete": "●"}.get(status, "○")
        color = {"interrupted": pal.yellow, "failed": pal.red, "complete": pal.green}.get(
            status, pal.dim
        )
        prefix = "" if depth == 0 else ("  " * (depth - 1) + ("└─ " if last else "├─ "))
        nxt = ",".join(data.get("next_nodes") or []) or "end"
        ss = data.get("superstep")
        ss_label = "-" if ss is None or ss < 0 else str(ss)
        lines.append(f"{prefix}{color(marker)} ss{ss_label} "
                     f"{pal.dim(cid)} {pal.dim('→ ' + nxt)}")
        for i, child in enumerate(children.get(cid, [])):
            walk(child, depth + 1, i == len(children.get(cid, [])) - 1)

    for root in children.get(None, []):
        walk(root, 0, True)
    return "\n".join(lines)


def render_summary(events: Sequence[Event], palette: Palette | None = None) -> str:
    """Counts per event kind — a quick sanity check on a trace."""
    pal = palette or make_palette()
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    lines = [pal.bold("event summary")]
    width = max(len(k) for k in counts) if counts else 0
    for kind in sorted(counts, key=lambda k: (-counts[k], k)):
        lines.append(f"  {kind:<{width}}  {counts[kind]}")
    return "\n".join(lines)


def replay_state_at(events: Sequence[Event], checkpoint_id: str) -> dict[str, Any] | None:
    """State as of a checkpoint, reconstructed from the trace alone.

    Note what this *cannot* need: no graph, no runtime, no re-execution. That
    is the payoff of "trace is the source of truth".

    A checkpoint is committed *after* the superstep's ``state_merge``, so the
    last merge at or before the checkpoint's ``seq`` is exactly the state it
    captured.
    """
    target_seq: int | None = None
    for event in events:
        if event.kind == CHECKPOINT_CREATED and event.data["checkpoint_id"] == checkpoint_id:
            target_seq = event.seq
            break
    if target_seq is None:
        return None

    best: dict[str, Any] | None = None
    for event in events:
        if event.seq > target_seq:
            break
        if event.kind == STATE_MERGE:
            best = event.data.get("state")
    return best


def list_checkpoints(events: Sequence[Event]) -> list[dict[str, Any]]:
    """Newest record per checkpoint id, in first-seen order.

    A checkpoint is emitted twice: once when it is created (status ``running``)
    and once when it reaches a terminal status. Listing the raw events therefore
    printed the same id twice, which reads like two checkpoints. The last record
    for an id carries its final status, so it is the one worth showing.
    """
    order: list[str] = []
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.kind != CHECKPOINT_CREATED:
            continue
        cid = event.data.get("checkpoint_id")
        if cid not in latest:
            order.append(cid)
        latest[cid] = event.data
    return [latest[cid] for cid in order]
