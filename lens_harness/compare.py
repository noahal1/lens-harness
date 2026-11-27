"""Trace comparison — the M3 lens: what *really* changed between two runs?

A diff of two trace files is useless as-is: five fields differ on every run
by design (``ts``, ``run_id``, ``checkpoint_id``, ``parent_id``,
``duration_ms`` — measured on real reruns, see README), so a raw textual diff
is never empty and tells you nothing. This module answers the question the
README could only describe by hand: after normalizing the noise away, which
events actually differ?

Same contract as ``render.py`` (principle two): the input is lists of
:class:`~lens_harness.events.Event` and this file imports nothing from the
kernel but ``events``. No graph, no runtime, no checkpoint files — two traces
are compared against each other alone.

Normalization
-------------
Per event, the noise fields are dropped or replaced:

* ``ts`` / ``run_id`` — dropped; identity is not behavior.
* ``checkpoint_id`` / ``parent_id`` — replaced by ``ss<N>`` placeholders
  (N = the checkpoint's superstep). Real ids contain a random hex suffix, but
  "the checkpoint produced by superstep 2" is a stable *position* in a story
  that ran the same way twice.
* ``duration_ms`` — dropped; node timing is machine noise, not behavior.
* ``node_token`` — the per-chunk events are aggregated into one record per
  ``(superstep, node)`` holding the full streamed text. Comparing chunks
  pairwise would diff the chunking, not the content.

Alignment
---------
Each record gets a structural key — ``(kind, superstep, node)`` for node
events, ``(kind, index)`` for supersteps, and so on — and records are paired
by key in first-seen order. A key present on one side only is a real
behavioral difference (a route that was never taken, a step that never ran).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .events import (
    CHECKPOINT_CREATED,
    EVENT_SCHEMA,
    NODE_END,
    NODE_ERROR,
    NODE_START,
    NODE_TOKEN,
    ROUTE_DECISION,
    RUN_START,
    STATE_MERGE,
    SUPERSTEP_START,
    Event,
)

#: Fields that differ on every run by construction. Dropping them is what
#: makes "the diff is empty" mean "the runs behaved identically".
NOISE_FIELDS = frozenset({"ts", "run_id", "duration_ms"})

#: Event-level fields replaced by stable placeholders instead of dropped.
#: A checkpoint's *identity within the story* is its superstep, not its uuid.
PLACEHOLDER_FIELDS = ("checkpoint_id", "parent_id")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Record:
    """One comparable unit: a normalized event (or an aggregated token run)."""

    key: tuple
    kind: str
    data: dict[str, Any]
    #: Order within its trace, for stable output.
    ordinal: int


def _canon(value: Any) -> Any:
    """JSON-safe canonical form so two dicts with equal content compare equal."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr))


def _normalize_data(data: dict[str, Any], id_map: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in NOISE_FIELDS:
            continue
        if k in PLACEHOLDER_FIELDS and isinstance(v, str):
            out[k] = id_map.get(v, v if v is None else _placeholder(v, id_map))
        else:
            out[k] = _canon(v)
    return out


def _placeholder(checkpoint_id: str, id_map: dict[str, str]) -> str:
    """``thread_0003_ab12cd`` → ``ss3`` — stable, readable, uuid-free."""
    parts = checkpoint_id.rsplit("_", 2)
    ss = parts[-2] if len(parts) == 3 else checkpoint_id
    try:
        return f"ss{int(ss)}"
    except ValueError:
        return checkpoint_id


def normalize(events: Sequence[Event]) -> list[Record]:
    """Turn a trace into comparable records: noise gone, tokens aggregated."""
    records: list[Record] = []
    id_map: dict[str, str] = {}
    run_ordinals: dict[str, int] = {}
    token_runs: dict[tuple, Record] = {}
    token_order: list[tuple] = []

    for ordinal, event in enumerate(events):
        kind, data = event.kind, event.data
        ss = event.superstep

        if kind == NODE_TOKEN:
            # Aggregate: one record per (superstep, node) with the full text.
            # The event's own data carries nothing else worth keeping.
            key = ("tokens", ss, data["node"])
            rec = token_runs.get(key)
            if rec is None:
                rec = Record(key=key, kind="tokens", data={"text": ""}, ordinal=ordinal)
                token_runs[key] = rec
                token_order.append(key)
            rec.data["text"] += data["token"]
            continue

        if kind == RUN_START:
            # A trace may hold more than one run_start (a resume emits one).
            # Pair them positionally: first with first, resume with resume.
            n = run_ordinals.get("run_start", 0)
            run_ordinals["run_start"] = n + 1
            key = (kind, n)
        elif kind == SUPERSTEP_START:
            key = (kind, data.get("index"))
        elif kind in (NODE_START, NODE_END, NODE_ERROR):
            key = (kind, ss, data.get("node"))
        elif kind == ROUTE_DECISION:
            key = (kind, ss, data.get("source"))
        elif kind == STATE_MERGE:
            key = (kind, ss)
        elif kind == CHECKPOINT_CREATED:
            # Both lifecycle events of one checkpoint share a placeholder id,
            # so pair on (placeholder, status): running↔running, final↔final.
            cid = data.get("checkpoint_id") or ""
            ph = _placeholder(cid, id_map)
            id_map[cid] = ph
            key = (kind, ph, data.get("status"))
        else:
            # interrupt / error_reported and any future kind: structural key.
            key = (kind, ss, data.get("node"))

        records.append(Record(key=key, kind=kind, data=_normalize_data(data, id_map),
                              ordinal=ordinal))

    # Token runs keep their first-appearance order relative to other records.
    for key in token_order:
        records.append(token_runs[key])
    return records


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------
@dataclass(slots=True)
class DiffEntry:
    """One aligned pair (or one-sided record) and how it compares."""

    key: tuple
    kind: str
    status: str  # same | changed | only_a | only_b
    a: dict[str, Any] | None = None
    b: dict[str, Any] | None = None
    #: Channels/fields that differ, for a quick headline.
    changed_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TraceDiff:
    entries: list[DiffEntry]

    def counts(self) -> dict[str, int]:
        out = {"same": 0, "changed": 0, "only_a": 0, "only_b": 0}
        for e in self.entries:
            out[e.status] += 1
        return out

    @property
    def identical(self) -> bool:
        c = self.counts()
        return c["changed"] == 0 and c["only_a"] == 0 and c["only_b"] == 0


def diff_traces(events_a: Sequence[Event], events_b: Sequence[Event]) -> TraceDiff:
    """Align two normalized traces by structural key, first-seen order."""
    ra = normalize(events_a)
    rb = normalize(events_b)

    # Group by key, preserving each side's order for positional pairing when
    # a key legitimately repeats (e.g. two runs in one trace).
    by_key_a: dict[tuple, list[Record]] = {}
    for rec in ra:
        by_key_a.setdefault(rec.key, []).append(rec)
    by_key_b: dict[tuple, list[Record]] = {}
    for rec in rb:
        by_key_b.setdefault(rec.key, []).append(rec)

    entries: list[DiffEntry] = []
    seen: set[tuple] = set()
    for rec in ra:
        if rec.key in seen:
            continue
        seen.add(rec.key)
        group_a = by_key_a[rec.key]
        group_b = by_key_b.pop(rec.key, [])
        for i, rec_a in enumerate(group_a):
            rec_b = group_b[i] if i < len(group_b) else None
            entries.append(_pair(rec_a, rec_b))
    # Whatever b still holds exists only on that side.
    for key, group in by_key_b.items():
        for rec in group:
            entries.append(_pair(None, rec))

    return TraceDiff(entries=entries)


def _pair(rec_a: Record | None, rec_b: Record | None) -> DiffEntry:
    if rec_a is not None and rec_b is None:
        return DiffEntry(key=rec_a.key, kind=rec_a.kind, status="only_a", a=rec_a.data)
    if rec_a is None and rec_b is not None:
        return DiffEntry(key=rec_b.key, kind=rec_b.kind, status="only_b", b=rec_b.data)
    assert rec_a is not None and rec_b is not None
    changed = sorted(
        k for k in set(rec_a.data) | set(rec_b.data) if rec_a.data.get(k) != rec_b.data.get(k)
    )
    return DiffEntry(
        key=rec_a.key,
        kind=rec_a.kind,
        status="changed" if changed else "same",
        a=rec_a.data,
        b=rec_b.data,
        changed_fields=changed,
    )


# --------------------------------------------------------------------------
# Terminal rendering
# --------------------------------------------------------------------------
def _short(value: Any, limit: int = 72) -> str:
    text = json.dumps(value, ensure_ascii=False, default=repr) if not isinstance(value, str) else value
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "…"


def render_diff(diff: TraceDiff, *, palette: Any = None, max_only: int = 20) -> str:
    """Human-readable diff report. ``palette`` is a render.Palette or None."""
    dim = bold = red = green = yellow = (lambda t: t) if palette is None else (
        lambda t: t)
    if palette is not None:
        dim, bold, red, green, yellow = palette.dim, palette.bold, palette.red, palette.green, palette.yellow

    counts = diff.counts()
    total = len(diff.entries)
    lines: list[str] = []
    if diff.identical:
        lines.append(bold(green(f"■ identical: {total} aligned records, "
                                "every behavioral field matches")))
        return "\n".join(lines)

    lines.append(bold("trace diff") + dim(f"  {total} aligned records"))
    lines.append(f"  same: {counts['same']}   "
                 f"{red('changed')}: {counts['changed']}   "
                 f"{yellow('only-a')}: {counts['only_a']}   "
                 f"{yellow('only-b')}: {counts['only_b']}")
    lines.append("")

    if counts["changed"]:
        lines.append(bold("── changed ──"))
        for e in diff.entries:
            if e.status != "changed":
                continue
            lines.append(f"  {yellow(e.kind)} {_key_label(e.key)}"
                         f"  {dim(','.join(e.changed_fields))}")
            for f_ in e.changed_fields:
                lines.append(f"    {dim('a')}: {_short((e.a or {}).get(f_))}")
                lines.append(f"    {dim('b')}: {_short((e.b or {}).get(f_))}")
        lines.append("")

    for side, tag, other in (("only_a", "only-a", "b"), ("only_b", "only-b", "a")):
        items = [e for e in diff.entries if e.status == side]
        if not items:
            continue
        lines.append(bold(f"── {tag} (only in trace {'a' if side == 'only_a' else 'b'}) ──"))
        for e in items[:max_only]:
            data = e.a if side == "only_a" else e.b
            lines.append(f"  {yellow(e.kind)} {_key_label(e.key)}  {dim(_short(data))}")
        if len(items) > max_only:
            lines.append(dim(f"  …(+{len(items) - max_only} more)"))
        lines.append("")

    return "\n".join(lines).rstrip()


def _key_label(key: tuple) -> str:
    parts = [str(p) for p in key[1:]]
    return " ".join(parts) if parts else "-"
