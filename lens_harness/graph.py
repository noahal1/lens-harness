"""Graph definition — nodes, edges, conditional routing, and compilation.

Deliberately thin. A graph here is a *description*; it holds no execution
state. Execution lives entirely in ``runtime.py``, which is what makes the
"trace is the only source of truth" split clean.

Two edge kinds, matching the concept we want to teach:

* ``add_edge(a, b)``            — b always runs after a
* ``add_conditional_edges(a, f, mapping)`` — f(state) picks one of the
  declared targets. The **full candidate set is recorded on the graph**, so
  the lens can show "3 possible routes, this run took route 2" instead of
  just highlighting a line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .state import RESERVED_CHANNELS, Channel, State

END = "__end__"
START = "__start__"

#: A node receives a read-only state dict and returns a patch (or None).
NodeFn = Callable[[dict[str, Any], "NodeContext"], dict[str, Any] | None]
#: A router receives a read-only state dict and returns one target or many.
RouterFn = Callable[[dict[str, Any]], str | Iterable[str]]


class GraphError(ValueError):
    """Malformed graph: unknown node, unreachable route, duplicate name…"""


@dataclass(slots=True)
class NodeContext:
    """Per-node scratch space handed to a node function.

    Intentionally tiny: what a node is *allowed* to see about the run. The
    emit helpers exist so an LLM node can stream tokens without importing the
    event module.
    """

    run_id: str
    thread_id: str
    superstep: int
    node: str
    emit_token: Callable[[str], None]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Node:
    name: str
    fn: NodeFn
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class ConditionalEdge:
    source: str
    router: RouterFn
    #: ``router return value -> node to run``. Keys are the complete candidate set.
    mapping: dict[str, str]
    router_name: str
    #: ``router return value -> human description``, for the trace only.
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def candidates(self) -> list[str]:
        return list(self.mapping)


class StateGraph:
    """Builder for a Pregel-style graph over one state schema."""

    def __init__(self, schema: type, *, name: str = "graph") -> None:
        self.schema = schema
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional: dict[str, ConditionalEdge] = {}
        self.entry: str | None = None
        self._channels: dict[str, Channel] | None = None

    # -- declaration -------------------------------------------------------
    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        description: str = "",
        tags: Sequence[str] = (),
    ) -> "StateGraph":
        if name in (START, END):
            raise GraphError(f"node name {name!r} is reserved")
        if name in self.nodes:
            raise GraphError(f"duplicate node {name!r}")
        self.nodes[name] = Node(name, fn, description, tuple(tags))
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        if source == START:
            self.entry = target
            return self
        self._require_node(source, role="edge source")
        if target != END:
            self._require_node(target, role="edge target")
        self.edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        mapping: Mapping[str, str],
        name: str | None = None,
        descriptions: Mapping[str, str] | None = None,
    ) -> "StateGraph":
        """Route from ``source`` by ``router``'s return value.

        ``mapping`` is ``router return value -> target node``, matching
        LangGraph. Descriptions are a *separate* channel (``descriptions``)
        because overloading the mapping values with prose is how a graph ends
        up routing to a node literally named "the agent requested a tool call".
        """
        self._require_node(source, role="router source")
        for target in mapping.values():
            if target != END:
                self._require_node(target, role="route target")
        self.conditional[source] = ConditionalEdge(
            source=source,
            router=router,
            mapping=dict(mapping),
            router_name=name or getattr(router, "__name__", "router"),
            descriptions=dict(descriptions or {}),
        )
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self._require_node(name, role="entry")
        self.entry = name
        return self

    def _require_node(self, name: str, *, role: str) -> None:
        if name not in self.nodes:
            raise GraphError(f"{role} {name!r} is not a declared node")

    # -- compilation -------------------------------------------------------
    def compile(self) -> "CompiledGraph":
        if not self.nodes:
            raise GraphError("cannot compile a graph with no nodes")
        if self.entry is None:
            raise GraphError("graph has no entry; call add_edge(START, ...) or set_entry()")
        # Every node must be able to go somewhere, otherwise a run can stall
        # silently at that node — a class of bug worth failing loudly on.
        for node in self.nodes:
            if node not in self.edges and node not in self.conditional:
                raise GraphError(
                    f"node {node!r} has no outgoing edge; add add_edge({node!r}, END) "
                    "if it is terminal"
                )
        from .state import channels_from_annotations, resolve_annotations

        self._channels = channels_from_annotations(resolve_annotations(self.schema))
        return CompiledGraph(self)

    def snapshot(self) -> dict[str, Any]:
        """Static graph description embedded in ``run_start``.

        The lens draws the static graph from this alone — it never imports the
        graph-definition code.
        """
        return {
            "name": self.name,
            "entry": self.entry,
            "nodes": [
                {
                    "name": n.name,
                    "description": n.description,
                    "tags": list(n.tags),
                }
                for n in self.nodes.values()
            ],
            "edges": [{"source": s, "target": t} for s, ts in self.edges.items() for t in ts],
            "conditional_edges": [
                {
                    "source": ce.source,
                    "router": ce.router_name,
                    "candidates": ce.candidates,
                    "descriptions": dict(ce.descriptions),
                    "mapping": dict(ce.mapping),
                }
                for ce in self.conditional.values()
            ],
            "channels": [
                {"name": c.name, "reducer": c.reducer_name}
                for c in (self._channels or {}).values()
            ],
        }


@dataclass(slots=True)
class CompiledGraph:
    """A validated, executable graph. Stateless: safe to reuse across runs."""

    graph: StateGraph

    @property
    def channels(self) -> dict[str, Channel]:
        assert self.graph._channels is not None
        return self.graph._channels

    def new_state(self, initial: dict[str, Any] | None = None) -> State:
        state = State(self.channels)
        if initial:
            # Reserved channels (``__interrupt__``/``__error__``) live in a
            # checkpoint exactly like declared ones, so restoring a paused run
            # must not reject them — otherwise every resume of an interrupt
            # fails on its own saved state.
            unknown = set(initial) - set(self.channels) - set(RESERVED_CHANNELS)
            if unknown:
                raise GraphError(
                    f"input wrote unknown channels {sorted(unknown)}; "
                    f"declared: {sorted(self.channels)}"
                )
            state.values.update(initial)
        return state

    def next_targets(self, node: str, state: State) -> tuple[list[str], dict[str, Any] | None]:
        """Resolve where to go after ``node``.

        Returns ``(targets, route_info)`` where ``route_info`` is the audit
        record of a conditional decision — ``None`` for a static edge. The
        kernel turns ``route_info`` into a ``route_decision`` event, which is
        the transparency feature we care most about.
        """
        ce = self.graph.conditional.get(node)
        if ce is not None:
            read_only = state.to_dict()
            returned = ce.router(read_only)
            # What the router *returns* is a mapping key; the node to run is the
            # mapped value. Conflating the two is what makes a router look like
            # it "randomly" jumps to a node that does not exist.
            keys = [returned] if isinstance(returned, str) else list(returned)
            for key in keys:
                if key not in ce.mapping:
                    raise GraphError(
                        f"router {ce.router_name!r} returned {key!r}, which is not a "
                        f"declared candidate {sorted(ce.mapping)}"
                    )
            targets = [ce.mapping[key] for key in keys]
            return targets, {
                "source": node,
                "condition": ce.router_name,
                "candidates": ce.candidates,
                "descriptions": dict(ce.descriptions),
                "returned": returned,
                "selected": targets,
            }
        return list(self.graph.edges.get(node, [])), None
