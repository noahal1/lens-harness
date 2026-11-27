"""State channels and reducers — the merge semantics of a superstep.

The single idea worth learning here:

    a node does not *mutate* state, it returns a **patch** (a partial dict);
    the kernel merges patches into channels using each channel's reducer.

Everything else in the kernel follows from that. ``add``-style reducers make
parallel fan-out safe; a plain overwrite reducer makes the last writer win and
is therefore a *bug generator* in fan-out graphs (see ``tests/test_state.py``).
"""

from __future__ import annotations

import copy
from typing import Annotated, Any, Callable, TypeVar, get_args, get_origin

T = TypeVar("T")

#: A reducer takes (current_value, patch_value) and returns the merged value.
Reducer = Callable[[Any, Any], Any]


class _Missing:
    """Sentinel: the channel has never received a value."""

    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

#: Channels the kernel itself understands. A node may write them without
#: declaring them in the state schema.
#:
#: ``__interrupt__`` — suspend the run and hand a payload to the human.
#: ``__error__``     — record a structured, non-fatal problem for the lens.
RESERVED_CHANNELS = ("__interrupt__", "__error__")


# --------------------------------------------------------------------------
# Built-in reducers
# --------------------------------------------------------------------------
def overwrite(current: Any, patch: Any) -> Any:
    """Default reducer: last write wins. Simple, and lossy under fan-out."""
    return patch


def add(current: Any, patch: Any) -> Any:
    """List concatenation / numeric addition — the classic ``add_messages`` base."""
    if current is MISSING:
        return copy.deepcopy(patch)
    if isinstance(current, (list, tuple)) and isinstance(patch, (list, tuple)):
        return list(current) + list(patch)
    return current + patch


def add_unique(current: Any, patch: Any) -> Any:
    """Ordered set union — handy for "visited nodes" style channels."""
    merged = add(current, patch)
    if isinstance(merged, list):
        seen: set[Any] = set()
        out = []
        for item in merged:
            key = item if isinstance(item, (str, int, float, bool, type(None))) else repr(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
    return merged


def merge_dict(current: Any, patch: Any) -> Any:
    """Shallow dict merge — partial updates to a dict-shaped channel."""
    if current is MISSING:
        return dict(patch)
    out = dict(current)
    out.update(patch)
    return out


BUILTIN_REDUCERS: dict[str, Reducer] = {
    "overwrite": overwrite,
    "add": add,
    "add_unique": add_unique,
    "merge_dict": merge_dict,
}


# --------------------------------------------------------------------------
# Channel spec
# --------------------------------------------------------------------------
class Channel:
    """One named slot in the state, plus how patches to it are merged."""

    __slots__ = ("name", "annotation", "reducer", "reducer_name")

    def __init__(
        self,
        name: str,
        annotation: Any = Any,
        reducer: Reducer = overwrite,
        reducer_name: str | None = None,
    ) -> None:
        self.name = name
        self.annotation = annotation
        self.reducer = reducer
        self.reducer_name = reducer_name or _reducer_label(reducer)

    def __repr__(self) -> str:
        return f"Channel({self.name!r}, reducer={self.reducer_name})"


def _reducer_label(fn: Reducer) -> str:
    for name, candidate in BUILTIN_REDUCERS.items():
        if candidate is fn:
            return name
    return getattr(fn, "__name__", repr(fn))


def resolve_annotations(schema: type) -> dict[str, Any]:
    """Get a schema's annotations with ``Annotated[...]`` *structurally* intact.

    This is subtle and worth reading. If the defining module uses
    ``from __future__ import annotations`` (very common), every annotation is a
    **string** — so ``get_origin("Annotated[list[str], add]")`` is ``None`` and
    every channel silently falls back to ``overwrite``. That bug is invisible
    until a fan-out node starts losing writes.

    ``typing.get_type_hints(..., include_extras=True)`` evaluates those strings
    and keeps the ``Annotated`` metadata, so the reducer survives. On failure
    (unresolvable forward refs) we fall back to the raw annotations rather than
    crashing a graph definition over a type hint.
    """
    import typing

    try:
        return typing.get_type_hints(schema, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolvable forward ref / missing import
        return dict(getattr(schema, "__annotations__", {}))


def channels_from_annotations(annotations: dict[str, Any]) -> dict[str, Channel]:
    """Build channels from a TypedDict / dataclass style annotation mapping.

    ``Annotated[list[str], add]`` declares a channel with the ``add`` reducer;
    a bare annotation gets ``overwrite``. This is the exact mechanism
    LangGraph uses, reproduced in ~30 lines so it can be read and stepped
    through.
    """
    channels: dict[str, Channel] = {}
    for name, annotation in annotations.items():
        if name.startswith("_"):
            continue
        reducer: Reducer = overwrite
        reducer_name: str | None = None
        base = annotation
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            base = args[0]
            meta = args[1:]
            for item in meta:
                if callable(item) and not isinstance(item, type):
                    reducer = item
                    break
        channels[name] = Channel(name, base, reducer, _reducer_label(reducer))
    return channels


# --------------------------------------------------------------------------
# The state container used by the kernel
# --------------------------------------------------------------------------
class State:
    """Mutable holder of channel values. Cloning yields an independent copy.

    ``apply_patch`` returns the *names of channels that actually changed*, which
    is what the ``state_merge`` event reports — "what did this superstep really
    touch" is a question every debugging session asks.
    """

    __slots__ = ("channels", "values")

    def __init__(self, channels: dict[str, Channel], values: dict[str, Any] | None = None):
        self.channels = channels
        self.values: dict[str, Any] = dict(values or {})

    # -- construction ------------------------------------------------------
    @classmethod
    def from_typed_dict(cls, schema: type, initial: dict[str, Any] | None = None) -> "State":
        return cls(channels_from_annotations(resolve_annotations(schema)), initial)

    def clone(self) -> "State":
        return State(self.channels, copy.deepcopy(self.values))

    # -- access ------------------------------------------------------------
    def get(self, name: str, default: Any = None) -> Any:
        value = self.values.get(name, MISSING)
        return default if value is MISSING else value

    def __getitem__(self, name: str) -> Any:
        if name not in self.values:
            raise KeyError(f"channel {name!r} is empty")
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def to_dict(self) -> dict[str, Any]:
        """Plain read-only view for nodes: never hand a node the live State."""
        return copy.deepcopy(self.values)

    # -- merge -------------------------------------------------------------
    def apply_patch(self, patch: dict[str, Any]) -> list[str]:
        """Merge a node's patch in, returning the channels that changed.

        Reserved channels (``__interrupt__``, ``__error__``) are accepted even
        though they are not part of the declared schema: they are how a node
        talks to the kernel rather than to the next node.
        """
        changed: list[str] = []
        for name, patch_value in patch.items():
            channel = self.channels.get(name)
            if channel is None:
                if name in RESERVED_CHANNELS:
                    self.values[name] = patch_value
                    changed.append(name)
                    continue
                raise KeyError(
                    f"node wrote unknown channel {name!r}; "
                    f"declared channels: {sorted(self.channels)}"
                )
            current = self.values.get(name, MISSING)
            merged = channel.reducer(current, patch_value)
            if merged is not current or current is MISSING or merged != current:
                self.values[name] = merged
                changed.append(name)
        return changed

    def diff(self, other: "State") -> dict[str, tuple[Any, Any]]:
        """``channel -> (before, after)`` for every channel whose value differs.

        Sorted, not set-ordered: the result is serialized into the trace, so
        iterating a ``set`` baked a different key order into every run. Two runs
        of the same example then produced traces that were identical in content
        but diffed as different files. Declared channels come first in schema
        order, then anything else (reserved channels) alphabetically.
        """
        declared = [name for name in self.channels if name in self.values or name in other.values]
        extra = sorted((set(self.values) | set(other.values)) - set(declared))
        out: dict[str, tuple[Any, Any]] = {}
        for name in declared + extra:
            before = self.values.get(name, MISSING)
            after = other.values.get(name, MISSING)
            if before is not after and before != after:
                out[name] = (before, after)
        return out
