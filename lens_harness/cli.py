"""``python -m lens_harness`` — inspect a trace, or run an example.

Commands::

    python -m lens_harness show traces/<file>.jsonl        # render a trace
    python -m lens_harness dag traces/<file>.jsonl         # checkpoint DAG
    python -m lens_harness summary traces/<file>.jsonl     # event counts
    python -m lens_harness checkpoints traces/<file>.jsonl # list checkpoints
    python -m lens_harness state traces/<file>.jsonl [<ckpt_id>]
    python -m lens_harness state traces/<file>.jsonl --at <superstep>
    python -m lens_harness watch traces/<file>.jsonl       # follow a live run
    python -m lens_harness diff traces/a.jsonl traces/b.jsonl  # compare two traces
    python -m lens_harness serve traces/<file>.jsonl       # the web view of a trace
    python -m lens_harness serve a.jsonl b.jsonl           # the diff in a browser
    python -m lens_harness run examples.agent:build --input '{"question":"..."}'

``show``/``dag``/``checkpoints``/``watch``/``serve`` take only a trace. ``state``
takes the trace first and then optionally a checkpoint id, or ``--at N`` for the
state that superstep produced. ``diff`` takes exactly two traces and exits 1
when they differ (GNU diff convention). ``serve`` opens a browser page that is
a pure function of the same trace the terminal renderer reads; ``--follow``
keeps reading a run in progress; a second trace argument switches it to the
comparison page (/compare + /api/diff).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from typing import Any

from .events import Event, read_trace
from .render import (
    RenderOptions,
    list_checkpoints,
    make_palette,
    render_checkpoint_dag,
    render_events,
    render_summary,
    replay_state_summary_at,
)


def _load(path: str | None) -> list[Event]:
    # ``path`` is optional on the ``state`` subcommand, so a missing argument
    # arrives here as None and must not reach ``os.path.exists`` (it raises
    # "path should be string, bytes, os.PathLike or integer, not NoneType").
    if path is None:
        raise SystemExit(
            "no trace given: pass a trace file, e.g. "
            "`state traces/<run_id>.jsonl --at 2`"
        )
    if not os.path.exists(path):
        raise SystemExit(f"no such trace: {path}")
    return read_trace(path)


def _force_utf8_stdout() -> None:
    """Windows consoles default to GBK/cp1252; our box-drawing glyphs are not.

    Reconfiguring is safer than asking the user to set PYTHONIOENCODING, and
    ``errors="replace"`` guarantees the CLI never crashes on a stray glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _palette(args: argparse.Namespace) -> Any:
    return make_palette(False if getattr(args, "no_color", False) else None)


def cmd_show(args: argparse.Namespace) -> int:
    events = _load(args.trace)
    print(render_events(
        events,
        options=RenderOptions(
            show_tokens=not args.no_tokens,
            show_state=not args.no_state,
            show_diff=not args.no_diff,
        ),
        palette=_palette(args),
    ))
    return 0


def cmd_dag(args: argparse.Namespace) -> int:
    print(render_checkpoint_dag(_load(args.trace), palette=_palette(args)))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    print(render_summary(_load(args.trace), palette=_palette(args)))
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    for ckpt in list_checkpoints(_load(args.trace)):
        print(f"{ckpt['checkpoint_id']}  ss={ckpt['superstep']:<3} "
              f"{ckpt['status']:<12} next={','.join(ckpt['next_nodes']) or 'end'}")
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    events = _load(args.trace)
    target = args.checkpoint_id
    # --at N is sugar for "the checkpoint that produced superstep N", which is
    # what you actually want when reading a trace top to bottom and you spot
    # something odd at a given step.
    if target is None and args.at is not None:
        matches = [c for c in list_checkpoints(events)
                   if c.get("superstep") == args.at]
        if not matches:
            raise SystemExit(f"no checkpoint at superstep {args.at}")
        target = matches[-1]["checkpoint_id"]
    if target is None:
        raise SystemExit("pass a checkpoint id, or --at <superstep>")
    # The checkpoint file has the FULL state; the trace only carries a summary
    # (summarize_state truncates long strings and lists). Prefer the file, and
    # say so when falling back to the summary so nobody mistakes truncation
    # for the real value (DESIGN.md §11.4).
    state = _full_state_from_checkpoint(args.trace, target)
    source = "checkpoint file"
    if state is None:
        state = replay_state_summary_at(events, target)
        source = "trace summary (long values truncated)"
    if state is None:
        raise SystemExit(f"checkpoint not found in trace: {target}")
    print(f"# full state from {source}")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def _full_state_from_checkpoint(trace_path: str, checkpoint_id: str) -> dict[str, Any] | None:
    """Read the complete state out of ``<trace_dir>/checkpoints/<id>.json``.

    The trace stores summaries; only the checkpoint file has every value. The
    path convention is the one :class:`CheckpointStore` writes — the trace's
    sibling ``checkpoints/`` directory — and a missing file is normal (old
    traces, or a checkpoint dir deleted on purpose), hence the ``None``.
    """
    trace_dir = os.path.dirname(trace_path) or "."
    path = os.path.join(trace_dir, "checkpoints", f"{checkpoint_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("state")


def cmd_watch(args: argparse.Namespace) -> int:
    """Follow a trace file while a run is still writing it."""
    from .events import follow_trace

    pal = _palette(args)
    print(pal.dim(f"following {args.trace} (Ctrl-C to stop)"))
    buffer: list[Event] = []
    try:
        for event in follow_trace(args.trace):
            buffer.append(event)
            # Re-rendering the whole buffer keeps the renderer stateless and the
            # contract simple; traces are small enough that this is fine.
            print("\033[H\033[2J", end="")
            print(render_events(buffer, palette=pal))
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two traces after normalizing the fields that always change.

    Exit code follows GNU diff: 1 means "differences found" (a signal, not a
    failure), 0 means the two runs behaved identically after normalization.
    """
    from .compare import diff_traces, render_diff

    events_a = _load(args.trace)
    events_b = _load(args.trace_b)
    diff = diff_traces(events_a, events_b)
    print(render_diff(diff, palette=_palette(args), max_only=args.max_only))
    return 1 if not diff.identical else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Import ``module:attr`` and call it to get the events of one run."""
    target = args.target
    if ":" not in target:
        raise SystemExit("target must look like 'module.submodule:attribute'")
    module_name, attr = target.split(":", 1)
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    inputs = json.loads(args.input) if args.input else {}

    # Pass only the keyword arguments the target actually declares. Every
    # example writes its own ``build()`` and they legitimately differ (the
    # approval demo has no LLM, so no ``llm_spec``); hard-coding one signature
    # made ``run`` work for exactly one example.
    wanted = {
        "inputs": inputs,
        "trace_dir": args.trace_dir,
        "llm_spec": args.llm,
    }
    try:
        accepted = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # builtins and some C callables
        accepted = wanted.keys()
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    events = factory(**kwargs)
    print(render_events(events, options=RenderOptions(show_tokens=not args.no_tokens)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve one trace over HTTP so the browser can render it."""
    from .serve import serve

    try:
        return serve(
            args.trace,
            host=args.host,
            port=args.port,
            follow=args.follow,
            compare_path=args.trace_b,
        )
    except FileNotFoundError:
        raise SystemExit(f"no such trace: {args.trace}")
    except OSError as exc:
        # Almost always "port already in use"; say what to do about it.
        raise SystemExit(f"cannot serve on {args.host}:{args.port}: {exc}")


def _add_color_flag(parser: argparse.ArgumentParser) -> None:
    """``--no-color`` must work both before and after the subcommand."""
    parser.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                        help="disable ANSI colors")


def build_parser() -> argparse.ArgumentParser:
    # RawDescriptionHelpFormatter keeps the newlines in ``__doc__``. The default
    # formatter re-wraps the description and flattens every newline, so `--help`
    # printed the whole command list as one unreadable paragraph.
    parser = argparse.ArgumentParser(
        prog="lens",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_color_flag(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="render a trace")
    _add_color_flag(p_show)
    p_show.add_argument("trace")
    p_show.add_argument("--no-tokens", action="store_true")
    p_show.add_argument("--no-state", action="store_true")
    p_show.add_argument("--no-diff", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_dag = sub.add_parser("dag", help="print the checkpoint DAG")
    _add_color_flag(p_dag)
    p_dag.add_argument("trace")
    p_dag.set_defaults(func=cmd_dag)

    p_sum = sub.add_parser("summary", help="event counts")
    _add_color_flag(p_sum)
    p_sum.add_argument("trace")
    p_sum.set_defaults(func=cmd_summary)

    p_ckpt = sub.add_parser("checkpoints", help="list checkpoints")
    _add_color_flag(p_ckpt)
    p_ckpt.add_argument("trace")
    p_ckpt.set_defaults(func=cmd_checkpoints)

    p_state = sub.add_parser("state", help="state at a checkpoint")
    _add_color_flag(p_state)
    p_state.add_argument("trace", nargs="?")
    p_state.add_argument("checkpoint_id", nargs="?")
    p_state.add_argument("--at", type=int, default=None,
                         help="superstep whose resulting state you want")
    p_state.set_defaults(func=cmd_state)

    p_watch = sub.add_parser("watch", help="follow a live trace")
    _add_color_flag(p_watch)
    p_watch.add_argument("trace")
    p_watch.set_defaults(func=cmd_watch)

    p_diff = sub.add_parser("diff", help="compare two traces")
    _add_color_flag(p_diff)
    p_diff.add_argument("trace")
    p_diff.add_argument("trace_b")
    p_diff.add_argument("--max-only", type=int, default=20,
                        help="cap on listed only-a/only-b records")
    p_diff.set_defaults(func=cmd_diff)

    p_serve = sub.add_parser("serve", help="render a trace in the browser")
    _add_color_flag(p_serve)
    p_serve.add_argument("trace")
    p_serve.add_argument("trace_b", nargs="?", default=None,
                         help="second trace: serve the comparison page")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000, help="0 picks a free port")
    p_serve.add_argument("--follow", action="store_true",
                         help="keep following a trace a run is still writing")
    p_serve.set_defaults(func=cmd_serve)

    p_run = sub.add_parser("run", help="run an example that returns events")
    _add_color_flag(p_run)
    p_run.add_argument("target")
    p_run.add_argument("--input", default="{}")
    p_run.add_argument("--trace-dir", default="traces")
    p_run.add_argument("--llm", default=None, help="mock | auto | openai:<model>")
    p_run.add_argument("--no-tokens", action="store_true")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
