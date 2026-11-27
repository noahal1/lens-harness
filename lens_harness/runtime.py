"""The runtime — the superstep loop. This file is the heart of the project.

Read this top to bottom and you have read lens-harness. Everything else is
either a data structure it uses or a renderer of what it emits.

The BSP loop::

    while active:
        plan    = dedupe(active)                # nodes scheduled this step
        writes  = run all plan nodes            # in parallel (threads)
        merge   writes into state via reducers  # -> state_merge event
        ckpt    = checkpoint(state, next)       # -> DAG grows by one node
        active  = next                          # -> route_decision events

A node never mutates state; it returns a patch. That single rule is what makes
parallel fan-out safe and time travel possible.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .checkpoint import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    Checkpoint,
    CheckpointStore,
)
from .events import (
    CHECKPOINT_CREATED,
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
    Emitter,
    EventSink,
    NullSink,
    summarize_state,
)
from .graph import END, CompiledGraph, GraphError, NodeContext
from .state import MISSING, State

#: Guards against an infinite loop in a hand-written graph — a learning tool
#: should fail with a clear message, not hang.
DEFAULT_MAX_SUPERSTEPS = 100


class GraphRecursionError(RuntimeError):
    """Raised when a run exceeds ``max_supersteps`` (almost always a cycle bug)."""


@dataclass(slots=True)
class RunResult:
    run_id: str
    thread_id: str
    status: str
    state: dict[str, Any]
    checkpoints: list[Checkpoint]
    supersteps: int
    #: The most nodes that ran at once in any single superstep.
    max_concurrency: int = 1
    #: The ceiling that was configured, for comparison against the above.
    allowed_concurrency: int = 1
    error: str | None = None


@dataclass(slots=True)
class RunConfig:
    thread_id: str = "default"
    max_supersteps: int = DEFAULT_MAX_SUPERSTEPS
    #: >1 enables real parallel fan-out inside a superstep.
    max_concurrency: int = 4
    #: Artificial per-node latency is NOT added here; nodes own their own time.
    configurable: dict[str, Any] = field(default_factory=dict)


class Runtime:
    """Executes a :class:`CompiledGraph` and emits one event per observable act."""

    def __init__(
        self,
        compiled: CompiledGraph,
        store: CheckpointStore,
        sink: EventSink | None = None,
    ) -> None:
        self.compiled = compiled
        self.store = store
        self.sink = sink or NullSink()
        self.emitter = Emitter(self.sink)

    # -- public API --------------------------------------------------------
    def invoke(
        self,
        inputs: dict[str, Any],
        config: RunConfig | None = None,
    ) -> RunResult:
        cfg = config or RunConfig()
        state = self.compiled.new_state(inputs)
        self.emitter.thread_id = cfg.thread_id
        run_id = self.emitter.run_id

        self.emitter.emit(
            RUN_START,
            graph=self.compiled.graph.snapshot(),
            inputs=summarize_state(inputs),
            config={"thread_id": cfg.thread_id, "max_supersteps": cfg.max_supersteps},
        )

        active = [self.compiled.graph.entry]
        return self._drive(
            state=state,
            active=active,
            parent=None,
            cfg=cfg,
            run_id=run_id,
            superstep=0,
        )

    def resume(
        self,
        checkpoint_id: str,
        *,
        payload: dict[str, Any] | None = None,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Continue a run from an ``interrupted`` (or any) checkpoint.

        This is also the fork-experiment primitive: resume an *old* checkpoint
        with a patched payload and the DAG grows a second branch, because the
        new checkpoint's ``parent_id`` points at the old one.
        """
        cfg = config or RunConfig()
        checkpoint = self.store.load(checkpoint_id)
        if checkpoint is None:
            raise FileNotFoundError(f"no such checkpoint: {checkpoint_id}")

        state = self.compiled.new_state(checkpoint.state)
        # An interrupt is a one-shot signal, but it lives in the state that was
        # checkpointed — so without clearing it the resumed run re-triggers on
        # the very next superstep and stops forever right after the node it
        # resumed into. (Symptom: resume "works", runs one node, then reports
        # interrupted again.) The payload itself is already preserved on
        # ``checkpoint.interrupt``, so nothing is lost by clearing the channel.
        state.values.pop("__interrupt__", None)
        resume_node = checkpoint.interrupt.get("resume_into") if checkpoint.interrupt else None
        if payload is not None:
            if resume_node is None:
                raise GraphError(
                    "resume() with a payload needs an interrupt to resume into; "
                    "this checkpoint has none"
                )
            # ``resume_into`` names the node to enter, not a channel: the payload
            # is a patch over *declared state channels*. Writing it under the
            # node name fails with "unknown channel 'apply'".
            if not isinstance(payload, dict):
                raise GraphError(
                    f"resume payload must be a dict of channel -> value, got {type(payload).__name__}"
                )
            changed = state.apply_patch(payload)
            # A resume payload is itself a state write; surface it as an event so
            # the trace shows the human's contribution, not just the graph's.
            self.emitter.thread_id = checkpoint.thread_id
            self.emitter.emit(
                STATE_MERGE,
                patches=summarize_state(payload),
                changed_channels=changed,
                state=summarize_state(state.values),
                origin="resume_payload",
            )

        active = list(checkpoint.next_nodes) or [resume_node] if resume_node else list(
            checkpoint.next_nodes
        )
        result = self._drive(
            state=state,
            active=active,
            parent=checkpoint,
            cfg=cfg,
            run_id=checkpoint.run_id,
            superstep=checkpoint.superstep + 1,
        )
        return result

    # -- internals ---------------------------------------------------------
    def _drive(
        self,
        *,
        state: State,
        active: list[str],
        parent: Checkpoint | None,
        cfg: RunConfig,
        run_id: str,
        superstep: int,
    ) -> RunResult:
        """The superstep loop — the whole kernel, in one place.

        ``invoke`` and ``resume`` both come here. Keeping a single copy is
        deliberate: a second, near-identical loop is exactly how a debugger
        ends up lying about one code path. The only difference between a fresh
        run and a resume is the starting ``state``/``active``/``parent``.
        """
        self.emitter.run_id = run_id
        self.emitter.thread_id = parent.thread_id if parent is not None else cfg.thread_id
        thread_id = self.emitter.thread_id
        checkpoints: list[Checkpoint] = []
        status = STATUS_COMPLETE
        error: str | None = None
        max_conc = max(1, cfg.max_concurrency)
        # What the run *actually* used, as opposed to what it was allowed to use.
        # The field is reported as ``max_concurrency`` and a 2-node run with a
        # limit of 4 never ran 4 anything.
        observed_conc = 0

        try:
            while active and superstep < cfg.max_supersteps:
                self.emitter.superstep = superstep
                plan = list(dict.fromkeys(active))  # dedupe, keep order
                self.emitter.emit(SUPERSTEP_START, index=superstep, nodes=plan)

                before = state.clone()
                observed_conc = max(observed_conc, min(max_conc, len(plan)))
                writes, failed = self._run_superstep(plan, cfg, state)

                if failed is not None:
                    status = STATUS_FAILED
                    error = failed
                    parent = self._commit(
                        state=state, before=before, writes=writes,
                        next_nodes=[], parent=parent, status=STATUS_FAILED,
                    )
                    checkpoints.append(parent)
                    break

                next_active = self._merge_and_route(
                    state=state, before=before, writes=writes, plan=plan
                )

                interrupted = self._interrupt_payload(state)
                parent = self._commit(
                    state=state, before=before, writes=writes,
                    next_nodes=next_active, parent=parent,
                    status=STATUS_INTERRUPTED if interrupted else None,
                    interrupt=interrupted,
                )
                checkpoints.append(parent)

                if interrupted is not None:
                    status = STATUS_INTERRUPTED
                    break

                active = next_active
                superstep += 1
            else:
                if active:
                    raise GraphRecursionError(
                        f"run exceeded max_supersteps={cfg.max_supersteps}; "
                        f"still active: {active}"
                    )
        except GraphRecursionError as exc:
            status = STATUS_FAILED
            error = str(exc)
        except GraphError as exc:
            status = STATUS_FAILED
            error = f"{type(exc).__name__}: {exc}"

        # The terminal checkpoint is the one the run *ended* on, so its status
        # must say how the run ended — not "running".
        if parent is not None and parent.status != status:
            parent.status = status
            self.store.put(parent)
            self.emitter.emit(
                CHECKPOINT_CREATED,
                checkpoint_id=parent.checkpoint_id,
                parent_id=parent.parent_id,
                next_nodes=parent.next_nodes,
                status=parent.status,
                superstep=parent.superstep,
            )

        self.emitter.superstep = superstep
        self.emitter.emit(
            RUN_END,
            status=status,
            supersteps=superstep,
            final_state=summarize_state(state.values),
            checkpoint_id=parent.checkpoint_id if parent else None,
            error=error,
        )
        return RunResult(
            run_id=run_id,
            thread_id=thread_id,
            status=status,
            state=state.to_dict(),
            checkpoints=checkpoints,
            supersteps=superstep,
            max_concurrency=observed_conc,
            allowed_concurrency=max_conc,
            error=error,
        )

    def _run_superstep(
        self,
        plan: list[str],
        cfg: RunConfig,
        state: State,
    ) -> tuple[list[tuple[str, dict[str, Any] | None]], str | None]:
        """Run every node in ``plan``; each sees the *same* pre-merge snapshot.

        Failures are returned, never stored on ``self``: two nodes in one
        superstep can fail on different threads and a shared attribute would
        lose one of them.
        """
        snapshot = state.to_dict()  # one frozen view for all parallel nodes

        def run_one(name: str) -> tuple[str, dict[str, Any] | None, str | None]:
            return self._run_node(name, snapshot, cfg)

        if len(plan) == 1:
            results = [run_one(plan[0])]
        else:
            max_workers = min(max(1, cfg.max_concurrency), len(plan))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lens-node") as pool:
                results = list(pool.map(run_one, plan))

        writes = [(name, patch) for name, patch, _err in results]
        failures = [err for _name, _patch, err in results if err is not None]
        failure = "; ".join(failures) if failures else None
        return writes, failure

    def _run_node(
        self,
        name: str,
        snapshot: dict[str, Any],
        cfg: RunConfig,
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        """Run one node. Returns ``(name, patch, failure_message_or_None)``."""
        node = self.compiled.graph.nodes[name]
        emitter = self.emitter
        emitter.emit(NODE_START, node=name, input=summarize_state(snapshot))

        ctx = NodeContext(
            run_id=emitter.run_id,
            thread_id=emitter.thread_id,
            superstep=emitter.superstep,
            node=name,
            emit_token=lambda tok: emitter.emit(NODE_TOKEN, node=name, token=tok),
            config=dict(cfg.configurable),
        )

        import time as _time

        started = _time.perf_counter()
        try:
            patch = node.fn(snapshot, ctx)
        except Exception as exc:  # noqa: BLE001 - node errors are data, not crashes
            emitter.emit(
                NODE_ERROR,
                node=name,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            return name, None, f"{name}: {type(exc).__name__}: {exc}"

        duration_ms = (_time.perf_counter() - started) * 1000
        emitter.emit(
            NODE_END,
            node=name,
            writes=summarize_state(patch or {}),
            duration_ms=round(duration_ms, 2),
        )
        return name, patch, None

    def _merge_and_route(
        self,
        *,
        state: State,
        before: State,
        writes: list[tuple[str, dict[str, Any] | None]],
        plan: list[str],
    ) -> list[str]:
        """Apply all patches, then resolve the outgoing edges of every node.

        Each write is applied **individually**. Collapsing them into one dict
        first (``merged.update(patch)``) looks equivalent and is not: with two
        parallel nodes both writing ``messages``, the second *replaces* the
        first before the reducer ever runs, so one node's output silently
        vanishes. The reducer is the merge, so it must see every write.
        """
        flattened: dict[str, Any] = {}
        for _node, patch in writes:
            if patch:
                flattened.update(patch)  # for the human-facing summary only

        changed: list[str] = []
        for _node, patch in writes:
            if patch:
                for name in state.apply_patch(patch):
                    if name not in changed:
                        changed.append(name)

        self.emitter.emit(
            STATE_MERGE,
            patches=summarize_state(flattened),
            changed_channels=changed,
            state=summarize_state(state.values),
            # Reserved channels are internal signals, not state you read back.
            # summarize_state already hides them, which made them render as
            # "__interrupt__: null → null" — a diff line that says nothing and
            # invites the reader to wonder what went wrong. Leave them out.
            diff={
                name: {
                    "before": summarize_state({name: b}).get(name),
                    "after": summarize_state({name: a}).get(name),
                }
                for name, (b, a) in before.diff(state).items()
                if not name.startswith("__")
            },
        )

        nxt: list[str] = []
        for node in plan:
            targets, route_info = self.compiled.next_targets(node, state)
            if route_info is not None:
                self.emitter.emit(ROUTE_DECISION, **route_info)
            elif targets:
                self.emitter.emit(
                    ROUTE_DECISION,
                    source=node,
                    condition=None,
                    candidates=targets,
                    returned=targets,
                    selected=targets,
                )
            nxt.extend(t for t in targets if t != END)
        return list(dict.fromkeys(nxt))

    def _interrupt_payload(self, state: State) -> dict[str, Any] | None:
        """A node signals an interrupt by writing the ``__interrupt__`` channel."""
        payload = state.values.get("__interrupt__")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return {"payload": payload}
        return payload

    def _commit(
        self,
        *,
        state: State,
        before: State,
        writes: list[tuple[str, dict[str, Any] | None]],
        next_nodes: list[str],
        parent: Checkpoint | None,
        status: str | None = None,
        interrupt: dict[str, Any] | None = None,
    ) -> Checkpoint:
        checkpoint = self.store.create(
            run_id=self.emitter.run_id,
            # Must match the ``thread_id`` stamped on the events, which on a
            # resume comes from the parent checkpoint, not from today's config.
            # Using ``cfg.thread_id`` here made a resumed run's checkpoint id
            # say one thread while every event said another.
            thread_id=self.emitter.thread_id,
            superstep=self.emitter.superstep,
            state=state.to_dict(),
            next_nodes=next_nodes,
            parent_id=parent.checkpoint_id if parent else None,
            status=status or "running",
        )
        checkpoint.interrupt = interrupt
        if interrupt is not None:
            self.store.put(checkpoint)
            self.emitter.emit(
                INTERRUPT,
                checkpoint_id=checkpoint.checkpoint_id,
                node=interrupt.get("node", "?"),
                payload=interrupt.get("payload"),
                resume_schema=interrupt.get("resume_schema"),
            )
        self.emitter.emit(
            CHECKPOINT_CREATED,
            checkpoint_id=checkpoint.checkpoint_id,
            parent_id=checkpoint.parent_id,
            next_nodes=checkpoint.next_nodes,
            status=checkpoint.status,
            superstep=checkpoint.superstep,
        )
        return checkpoint
