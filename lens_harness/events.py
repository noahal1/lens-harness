"""Event model — the single source of truth for lens-harness observability.

Every observable thing the kernel does becomes exactly one ``Event``, and a
run becomes one JSONL file. The lens UI is a pure function of that file
(see DESIGN.md §2), which is why this module is the most important contract
in the project: change an event's schema and every consumer changes with it.

Wire format — one JSON object per line::

    {"seq":0,"ts":1732600000.123,"run_id":"r_ab12","thread_id":"t1",
     "superstep":-1,"kind":"run_start","data":{...}}
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol, TextIO

# --------------------------------------------------------------------------
# Event kinds. One constant per row of the DESIGN.md §5 table.
# --------------------------------------------------------------------------
RUN_START = "run_start"
RUN_END = "run_end"
SUPERSTEP_START = "superstep_start"
NODE_START = "node_start"
NODE_TOKEN = "node_token"
NODE_END = "node_end"
STATE_MERGE = "state_merge"
ROUTE_DECISION = "route_decision"
CHECKPOINT_CREATED = "checkpoint_created"
INTERRUPT = "interrupt"
NODE_ERROR = "node_error"

#: ``kind -> required data keys``. Used by :func:`validate_event` so a typo in
#: the kernel fails loudly instead of producing a trace the lens cannot read.
EVENT_SCHEMA: dict[str, frozenset[str]] = {
    RUN_START: frozenset({"graph", "inputs"}),
    RUN_END: frozenset({"status", "supersteps", "final_state"}),
    SUPERSTEP_START: frozenset({"index", "nodes"}),
    NODE_START: frozenset({"node", "input"}),
    NODE_TOKEN: frozenset({"node", "token"}),
    NODE_END: frozenset({"node", "writes", "duration_ms"}),
    STATE_MERGE: frozenset({"patches", "changed_channels", "state"}),
    ROUTE_DECISION: frozenset(
        {"source", "condition", "candidates", "returned", "selected"}
    ),
    CHECKPOINT_CREATED: frozenset({"checkpoint_id", "parent_id", "next_nodes", "status"}),
    INTERRUPT: frozenset({"checkpoint_id", "node", "payload", "resume_schema"}),
    NODE_ERROR: frozenset({"node", "error_type", "error", "traceback"}),
}


class EventSchemaError(ValueError):
    """Raised when an event is missing keys its kind requires."""


def validate_event(kind: str, data: dict[str, Any]) -> None:
    required = EVENT_SCHEMA.get(kind)
    if required is None:
        raise EventSchemaError(f"unknown event kind {kind!r}")
    missing = required - data.keys()
    if missing:
        raise EventSchemaError(
            f"event {kind!r} missing required keys: {sorted(missing)}"
        )


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable line of a trace."""

    seq: int
    ts: float
    run_id: str
    thread_id: str
    superstep: int
    kind: str
    data: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "run_id": self.run_id,
                "thread_id": self.thread_id,
                "superstep": self.superstep,
                "kind": self.kind,
                "data": self.data,
            },
            ensure_ascii=False,
            default=_json_default,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "Event":
        raw = json.loads(line)
        return cls(
            seq=raw["seq"],
            ts=raw["ts"],
            run_id=raw["run_id"],
            thread_id=raw["thread_id"],
            superstep=raw["superstep"],
            kind=raw["kind"],
            data=raw["data"],
        )


def _json_default(obj: Any) -> Any:
    """Best-effort JSON coercion for values living inside a state snapshot."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


# --------------------------------------------------------------------------
# Sinks — where events go. The kernel only knows this protocol.
# --------------------------------------------------------------------------
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullSink:
    """Drops everything. Keeps kernels cheap when nobody is watching."""

    def emit(self, event: Event) -> None:  # noqa: ARG002
        return


class ListSink:
    """Keeps events in memory — the natural choice in tests."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]


class JsonlSink:
    """Appends events to ``<dir>/<run_id>.jsonl``; flushes every line.

    Flushing matters: the CLI/lens tail this file *while* the run is still
    going, which is the whole point of a live trace.
    """

    def __init__(self, directory: str, run_id: str) -> None:
        import os

        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{run_id}.jsonl")
        self._fh: TextIO = open(self.path, "w", encoding="utf-8")

    def emit(self, event: Event) -> None:
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TeeSink:
    """Fan one stream out to several sinks (e.g. file + in-memory for tests)."""

    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            sink.emit(event)


# --------------------------------------------------------------------------
# Emitter — assigns seq/ts/run_id/thread_id and validates before sending.
# --------------------------------------------------------------------------
class Emitter:
    """Kernel-side helper. One per run; ``superstep`` is advanced by the loop.

    Nodes in one superstep may run on worker threads, so ``emit`` is
    lock-guarded: without it two nodes can take the same ``seq`` and the trace
    loses its total order (``seq`` is what every consumer sorts by).
    """

    def __init__(
        self,
        sink: EventSink,
        run_id: str | None = None,
        thread_id: str = "default",
    ) -> None:
        import threading

        self.sink = sink
        self.run_id = run_id or f"r_{uuid.uuid4().hex[:8]}"
        self.thread_id = thread_id
        self.superstep = -1
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, kind: str, /, **data: Any) -> Event:
        validate_event(kind, data)
        with self._lock:
            event = Event(
                seq=self._seq,
                ts=time.time(),
                run_id=self.run_id,
                thread_id=self.thread_id,
                superstep=self.superstep,
                kind=kind,
                data=data,
            )
            self._seq += 1
            self.sink.emit(event)
        return event


# --------------------------------------------------------------------------
# Reading a trace back.
# --------------------------------------------------------------------------
def read_trace(path: str) -> list[Event]:
    """Load a whole trace file into memory (traces are small by design)."""
    events: list[Event] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(Event.from_json(line))
    return events


class TraceRotation:
    """Decide whether the file a reader is positioned in is still that file.

    Both readers of a live trace -- :func:`follow_trace` for the terminal and
    ``serve.TraceFeed`` for the browser -- need this answer, and they must
    reach the *same* answer: if one restarts a rotated file and the other does
    not, the two views of one trace disagree. So the rule lives here once.

    Three things mean "a different file", and all three are needed:

    * the inode changed -- the writer replaced the file (temp file + rename);
    * the file is shorter than the read offset -- truncated in place;
    * the length is unchanged but ``mtime`` moved -- rewritten in place at
      exactly the same size, which a rerun of a deterministic example really
      does produce. A normal append grows the file, so it cannot trigger this.

    A ``stat`` that fails (the path vanished, or is not readable yet) says
    nothing about the file and must *not* be read as "shrank to zero": that
    would replay the whole trace as duplicates on one transient error.
    """

    __slots__ = ("_ino", "_size", "_mtime")

    def __init__(self) -> None:
        self._ino: int | None = None
        self._size = -1
        self._mtime: int | None = None

    def observe(self, st: Any, pos: int) -> bool:
        """Record ``st``; return True when the reader must restart at 0."""
        rotated = False
        if self._ino is None:
            self._ino = st.st_ino
        elif st.st_ino != self._ino:
            rotated = True
        elif st.st_size < pos:
            rotated = True
        elif st.st_size == self._size and st.st_mtime_ns != self._mtime:
            rotated = True
        if rotated:
            self._ino = st.st_ino
        self._size = st.st_size
        self._mtime = st.st_mtime_ns
        return rotated

    def forget(self) -> None:
        """Drop the remembered identity after the path stopped being stat-able."""
        self._size = -1

def offset_is_line_boundary(path: str, pos: int) -> bool:
    """True when ``pos`` sits just after a newline, so a reader may resume there.

    Size and mtime catch a file that was replaced or truncated, but not one
    that was *rewritten in place and grew*: ``st_size`` is larger than the old
    offset, so "shrank" never fires, and the byte at ``pos`` is now in the
    middle of some other line. Reading from there yields a JSONDecodeError out
    of a live stream. The offset is only reusable if the byte before it is the
    newline that ended a record -- true for every append, and false for a
    rewrite that shifted the text underneath the reader.
    """
    if pos <= 0:
        return True
    try:
        with open(path, "rb") as fh:
            fh.seek(pos - 1)
            return fh.read(1) == b"\n"
    except OSError:
        return False

def last_seq_in_file(path: str) -> int | None:
    """Seq of the last complete record in ``path``, or None if there is none.

    A resume cursor (``Last-Event-ID``) is a claim about *a particular file*:
    "I have seen everything up to seq N". If the trace was rerun while the
    browser was away, the new file's seqs start at 0 again and never reach
    N, so honoring the cursor would hide the entire new run from a client
    that is perfectly healthy. The last record's seq is the cheapest check
    that the claim can still be true: within one run seq is dense and
    increasing, so the final record carries the largest seq in the file.

    Reads backwards from the end rather than scanning the file, because this
    runs once per connection and a trace can be large.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return None
            chunk = 65536
            while True:
                start = max(0, size - chunk)
                fh.seek(start)
                blob = fh.read(size - start)
                lines = blob.split(b"\n")
                # split() leaves one trailing element either way: an empty one
                # when the blob ends at a newline, or the half-written tail when
                # it does not. Dropping it handles both. The first element is
                # itself a fragment unless the scan started at byte 0.
                lines = lines[:-1]
                if start > 0 and lines:
                    lines = lines[1:]
                for raw in reversed(lines):
                    if not raw.strip():
                        continue
                    try:
                        return int(json.loads(raw.decode("utf-8"))["seq"])
                    except (ValueError, KeyError, TypeError,
                            UnicodeDecodeError, json.JSONDecodeError):
                        return None
                if start == 0:
                    return None
                chunk *= 2
    except OSError:
        return None

def follow_trace(path: str, poll_s: float = 0.1) -> Iterator[Event]:
    """Tail a trace file, yielding events as they are appended (live mode).

    Only *complete* lines are consumed. The writer appends ``<json>\\n`` per
    event and flushes, but a reader can still open the file in the window
    between those two facts and see a line with no trailing newline yet:

    * ``Event.from_json`` on that fragment raises ``JSONDecodeError``, and
    * advancing the read position past it would silently lose the rest of the
      line once the writer finished it.

    So a trailing fragment is left unconsumed and re-read on the next poll.
    This matters most for the SSE lens, where one raise kills the live stream.

    The remembered ``pos`` is a byte offset, so it is only meaningful for the
    file it came from. A rerun truncates or replaces the trace under the
    reader, and then ``seek(pos)`` can land past the new EOF (nothing new ever
    again, silently) or in the middle of a line (``JSONDecodeError`` out of a
    live stream). So the file *identity* is tracked, not just the offset:

    * a failed ``stat`` is treated as "not measurable right now" and the poll
      is retried -- it must never be read as "the file shrank to zero", or one
      transient error would replay the whole trace as duplicates;
    * a change in ``st_ino`` or ``st_mtime_ns`` means a *different* file, so
      both the offset and the ``after`` cursor are reset and reading restarts
      from the top. Size alone is not enough: a rerun of a deterministic
      example produces a replacement of *exactly the same length*.
    """
    import os
    import time as _time

    pos = 0
    rotation = TraceRotation()
    while True:
        try:
            st = os.stat(path)
        except OSError:
            rotation.forget()
            _time.sleep(poll_s)
            continue
        if rotation.observe(st, pos) or not offset_is_line_boundary(path, pos):
            pos = 0
        with open(path, "r", encoding="utf-8") as fh:
            fh.seek(pos)
            while True:
                line = fh.readline()
                if not line:
                    break  # EOF
                if not line.endswith("\n"):
                    # Half-written tail: stop here and retry from `pos` later.
                    break
                pos = fh.tell()
                line = line.strip()
                if line:
                    yield Event.from_json(line)
        _time.sleep(poll_s)


def summarize_state(state: dict[str, Any], *, max_str: int = 120) -> dict[str, Any]:
    """Compact, JSON-safe view of a state for event payloads.

    Full state lives in the checkpoint; events carry a *summary* so traces stay
    readable and small. The lens joins the two by checkpoint id.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("__"):
            continue
        if isinstance(value, str) and len(value) > max_str:
            out[key] = value[:max_str] + f"…({len(value)} chars)"
        elif isinstance(value, (list, tuple)) and len(value) > 8:
            out[key] = list(value[:8]) + [f"…(+{len(value) - 8} more)"]
        else:
            out[key] = value
    return out
