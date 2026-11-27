"""lens-harness — a readable LangGraph-style orchestration kernel with a trace
that can be replayed, diffed and time-travelled.

Public surface is intentionally small. If you are learning the internals, read
in this order:

  1. ``events.py``    — the trace contract (the source of truth)
  2. ``state.py``     — channels, reducers, patches
  3. ``graph.py``     — declaration + compiled graph
  4. ``runtime.py``   — the superstep loop (the heart)
  5. ``checkpoint.py``— the DAG that enables time travel
  6. ``render.py``    — a renderer, i.e. one *consumer* of the trace
  7. ``serve.py``     — the same trace over SSE, for ``web/index.html``

``render`` and ``serve`` are consumers: they import from ``events`` and never
the other way round. ``serve`` is deliberately *not* re-exported here — it
pulls in ``http.server`` and only ``lens serve`` needs it.
"""

from .checkpoint import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    Checkpoint,
    CheckpointStore,
)
from .events import (
    Event,
    EventSchemaError,
    EventSink,
    JsonlSink,
    ListSink,
    NullSink,
    TeeSink,
    follow_trace,
    read_trace,
    summarize_state,
)
from .graph import END, START, GraphError, NodeContext, StateGraph
from .llm import (
    LLM,
    LLMResponse,
    Message,
    MockLLM,
    OpenAICompatibleLLM,
    RecordingLLM,
    load_llm,
)
from .runtime import GraphRecursionError, RunConfig, RunResult, Runtime
from .state import MISSING, Channel, State, add, add_unique, merge_dict, overwrite

__version__ = "0.1.0"

__all__ = [
    # graph
    "StateGraph", "NodeContext", "GraphError", "START", "END",
    # state
    "State", "Channel", "MISSING", "add", "add_unique", "overwrite", "merge_dict",
    # runtime
    "Runtime", "RunConfig", "RunResult", "GraphRecursionError",
    # checkpoints
    "Checkpoint", "CheckpointStore", "STATUS_RUNNING", "STATUS_COMPLETE",
    "STATUS_INTERRUPTED", "STATUS_FAILED",
    # events
    "Event", "EventSink", "JsonlSink", "ListSink", "NullSink", "TeeSink",
    "EventSchemaError", "read_trace", "follow_trace", "summarize_state",
    # llm
    "LLM", "LLMResponse", "Message", "MockLLM", "RecordingLLM",
    "OpenAICompatibleLLM", "load_llm",
]
