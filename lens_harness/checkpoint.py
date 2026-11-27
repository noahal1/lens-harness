"""Checkpoints — a DAG of snapshots, not a line of logs.

The one idea that makes time travel possible: every checkpoint keeps
``parent_id``. A linear run looks like a chain, but the moment you resume from
an old checkpoint (fork experiment, see DESIGN.md §6) the structure becomes a
real DAG — two children of one parent, two different futures.

Storage is deliberately dumb: one flat JSON file per checkpoint,
``<dir>/<checkpoint_id>.json``, with no index — ``all()`` just scans the
directory. A learning tool should let you ``cat`` a checkpoint and read it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from .events import _json_default

#: Checkpoint lifecycle. ``interrupted`` is a first-class state, not an error —
#: it is the anchor a human-in-the-loop resume starts from.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_RESUMED = "resumed"


@dataclass(slots=True)
class Checkpoint:
    """Immutable snapshot of a run at a superstep boundary."""

    checkpoint_id: str
    parent_id: str | None
    run_id: str
    thread_id: str
    superstep: int
    state: dict[str, Any]
    next_nodes: list[str]
    status: str = STATUS_RUNNING
    ts: float = field(default_factory=time.time)
    #: Set when status == interrupted: what the graph is waiting for.
    interrupt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "superstep": self.superstep,
            "state": self.state,
            "next_nodes": self.next_nodes,
            "status": self.status,
            "ts": self.ts,
            "interrupt": self.interrupt,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Checkpoint":
        return cls(**raw)

    def lineage(self, store: "CheckpointStore") -> list["Checkpoint"]:
        """Walk ``parent_id`` up to the root — the run's history, oldest first."""
        chain: list[Checkpoint] = [self]
        seen = {self.checkpoint_id}
        current = self
        while current.parent_id:
            parent = store.load(current.parent_id)
            if parent is None or parent.checkpoint_id in seen:
                break
            chain.append(parent)
            seen.add(parent.checkpoint_id)
            current = parent
        chain.reverse()
        return chain

    def children(self, store: "CheckpointStore") -> list["Checkpoint"]:
        """Siblings sharing this parent — i.e. forked experiments from here."""
        return [c for c in store.all() if c.parent_id == self.checkpoint_id]


class CheckpointStore:
    """File-backed checkpoint store: flat ``<dir>/<checkpoint_id>.json`` files."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._cache: dict[str, Checkpoint] = {}

    def _path(self, checkpoint_id: str) -> str:
        return os.path.join(self.directory, f"{checkpoint_id}.json")

    # -- write -------------------------------------------------------------
    def put(self, checkpoint: Checkpoint) -> Checkpoint:
        path = self._path(checkpoint.checkpoint_id)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(checkpoint.to_dict(), fh, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)  # atomic: a truncated checkpoint can never be read
        self._cache[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def create(
        self,
        *,
        run_id: str,
        thread_id: str,
        superstep: int,
        state: dict[str, Any],
        next_nodes: list[str],
        parent_id: str | None,
        status: str = STATUS_RUNNING,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=f"{thread_id}_{superstep:04d}_{uuid.uuid4().hex[:6]}",
            parent_id=parent_id,
            run_id=run_id,
            thread_id=thread_id,
            superstep=superstep,
            state=state,
            next_nodes=list(next_nodes),
            status=status,
        )
        return self.put(checkpoint)

    # -- read --------------------------------------------------------------
    def load(self, checkpoint_id: str) -> Checkpoint | None:
        if checkpoint_id in self._cache:
            return self._cache[checkpoint_id]
        path = self._path(checkpoint_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            checkpoint = Checkpoint.from_dict(json.load(fh))
        self._cache[checkpoint_id] = checkpoint
        return checkpoint

    def all(self, thread_id: str | None = None) -> list[Checkpoint]:
        out: list[Checkpoint] = []
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.directory, name), "r", encoding="utf-8") as fh:
                try:
                    checkpoint = Checkpoint.from_dict(json.load(fh))
                except (json.JSONDecodeError, TypeError):
                    continue
            if thread_id is None or checkpoint.thread_id == thread_id:
                out.append(checkpoint)
        out.sort(key=lambda c: (c.ts, c.superstep))
        return out

    def latest(self, thread_id: str) -> Checkpoint | None:
        candidates = self.all(thread_id)
        return candidates[-1] if candidates else None

    # -- DAG queries used by the lens --------------------------------------
    def dag(self, thread_id: str | None = None) -> dict[str, Any]:
        """Nodes + parent links, ready to render as a DAG."""
        items = self.all(thread_id)
        nodes = [
            {
                "id": c.checkpoint_id,
                "superstep": c.superstep,
                "status": c.status,
                "next_nodes": c.next_nodes,
                "ts": c.ts,
            }
            for c in items
        ]
        edges = [
            {"source": c.parent_id, "target": c.checkpoint_id}
            for c in items
            if c.parent_id
        ]
        return {"nodes": nodes, "edges": edges}
