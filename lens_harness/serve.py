"""The Web lens: serve a trace to a browser over Server-Sent Events.

This module is the *only* new moving part M2 adds, and it is deliberately dumb.
It does not know what a superstep is, what a reducer does, or how to draw a
graph. It knows one thing: how to hand a browser the bytes of a trace, one
:class:`~lens_harness.events.Event` at a time, in order.

Everything visual lives in ``lens_harness/web/index.html``, which is a pure
consumer of that stream — the same constraint ``render.py`` obeys (see
DESIGN.md §2, principle two). Neither file imports ``graph``, ``runtime``,
``checkpoint`` or ``llm``: a lens that could *ask the kernel* what happened
would stop being a proof that the trace is self-sufficient.

Why SSE and not WebSocket
-------------------------
SSE is one-way (server → browser), which is exactly the shape of the problem:
the trace already happened; the browser only reads. It is also a plain HTTP
response with a text body, so it needs no handshake, no framing library and no
new dependency — ``EventSource`` is a baseline browser API. A WebSocket would
add a protocol and a second code path for zero extra capability.

Wire format
-----------
One SSE frame per event, ``id`` = the trace's own ``seq``::

    id: 7
    event: lens
    data: {"seq":7,"kind":"node_end",...}

Because ``seq`` is dense and monotonic (``Emitter`` assigns it under a lock),
the browser's automatic reconnect carries ``Last-Event-ID: 7`` and the server
resumes at 8. Live-view reconnects therefore never duplicate or skip events.
Nothing is *persisted* between connections to make that true — the cursor is the
client's, echoed back on each reconnect — but a live stream is not stateless
either: each open connection owns a bounded queue, a stop :class:`threading.Event`
and one producer thread, all created per request and torn down with it.

Two modes, one code path
------------------------
* **replay** (default) — read the file to EOF, send ``done``, close the socket.
  A finished trace becomes a scrubber.
* **follow** — keep polling; new lines are streamed as they are flushed. The
  run is live and the connection stays open until the client leaves.

Both are the same generator; ``follow`` only decides whether to stop at EOF.

A stream that stops early is not a stream that finished. If the trace turns out
to be malformed the server sends ``event: error`` and **no** ``done`` frame, so a
client can tell truncation from completion.
"""

from __future__ import annotations

import json
import os
import queue
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

from .events import (Event, TraceRotation, last_seq_in_file,
                    offset_is_line_boundary)

#: Loopback by default. Binding this server to a public interface would publish
#: the user's traces to the network, so it is a deliberate opt-in.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: How often the follower re-checks the trace file. 100 ms is below the
#: threshold where a live run looks choppy and far above a wasted spin.
DEFAULT_POLL_S = 0.1

#: Idle seconds before a ``: ping`` comment is written. Keeps the connection
#: provably alive and lets the page distinguish "quiet" from "dead".
PING_AFTER_S = 15.0

#: While waiting for the next event the handler also watches the socket, so a
#: tab that closed on a *quiet* file is noticed in about this long instead of
#: only at the next write (which may be a ping a whole PING_AFTER_S away).
DISCONNECT_CHECK_S = 0.25

_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
_COMPARE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "compare.html")


# --------------------------------------------------------------------------
# The feed: a trace file, turned into an ordered stream of events.
# --------------------------------------------------------------------------
class TraceFeed:
    """Iterate a trace file as events, optionally following it as it grows.

    The one subtle rule (learned from a real bug in ``follow_trace``, see
    DESIGN.md §11.16) is that **only complete lines are consumed**. A writer
    flushes ``<json>\\n`` per event, but a reader can still open the file in the
    window between the JSON and the newline. Parsing that fragment raises, and
    advancing past it would lose the rest of the line forever. So a trailing
    fragment stays unconsumed and is re-read on the next pass. This rule is
    deliberately shared with :func:`lens_harness.events.follow_trace`: the two
    readers must agree, because one feeds the terminal and one feeds the browser.

    Two more cases matter once the file is *live*:

    * the file can be **truncated or replaced** (a rerun writes the same path),
      leaving the remembered byte offset past the new EOF — a ``seek`` there
      yields nothing forever, so a shrunken file restarts from the top;
    * a follower must be **stoppable** while it is idle. ``stop``, when given,
      is waited on instead of a bare ``sleep`` so a closed browser tab does not
      leave a thread polling a file for nobody.
    """

    def __init__(
        self,
        path: str,
        *,
        after: int = -1,
        follow: bool = False,
        poll_s: float = DEFAULT_POLL_S,
        stop: threading.Event | None = None,
    ) -> None:
        self.path = path
        self.after = after
        self.follow = follow
        self.poll_s = poll_s
        self.stop = stop

    def __iter__(self) -> Iterator[Event]:
        pos = 0
        after = self.after
        rotation = TraceRotation()
        cursor_checked = False
        while True:
            if self.stop is not None and self.stop.is_set():
                return
            # The rotation rule is shared with follow_trace (see TraceRotation):
            # a rerun may truncate or replace the trace under us, and then
            # `seek(pos)` lands either past the new EOF (nothing new, forever,
            # silently) or in the middle of a line (JSONDecodeError out of a
            # live stream). Restarting must also clear the `after` cursor: the
            # new run starts at seq 0, so keeping the old cursor would hide it.
            try:
                st = os.stat(self.path)
            except OSError:
                rotation.forget()
                if not os.path.exists(self.path):
                    if not self.follow:
                        raise FileNotFoundError(f"no such trace: {self.path}")
                if _sleep_or_stop(self.stop, self.poll_s):
                    return
                continue
            if not cursor_checked:
                # A resume cursor names a seq in a *particular* trace. If the
                # trace was rerun while the client was away, the new run's
                # seqs start over and never reach that number, so honoring the
                # cursor would hide the whole new run from a healthy client.
                cursor_checked = True
                if after >= 0:
                    tail = last_seq_in_file(self.path)
                    if tail is None or tail < after:
                        after = -1
            if rotation.observe(st, pos) or not offset_is_line_boundary(
                    self.path, pos):
                pos = 0
                after = -1
            with open(self.path, "r", encoding="utf-8") as fh:
                fh.seek(pos)
                while True:
                    line = fh.readline()
                    if not line:
                        break  # EOF
                    if not line.endswith("\n"):
                        break  # half-written tail: retry from `pos` next poll
                    pos = fh.tell()
                    if not line.strip():
                        continue
                    event = Event.from_json(line)
                    if event.seq <= after:
                        continue
                    yield event
            if not self.follow:
                return
            if _sleep_or_stop(self.stop, self.poll_s):
                return


def _sleep_or_stop(stop: threading.Event | None, seconds: float) -> bool:
    """Wait ``seconds``, waking early if ``stop`` fires. True means stop.

    ``Event.wait`` returns as soon as the event is set, which is what lets a
    producer thread die promptly instead of finishing a full poll interval
    after its connection closed. With no event the call is a plain ``sleep``.
    """
    if stop is None:
        time.sleep(seconds)
        return False
    return stop.wait(seconds)


# --------------------------------------------------------------------------
# SSE framing
# --------------------------------------------------------------------------
def sse_frame(event: Event) -> bytes:
    """Encode one event as an SSE frame.

    The payload is the trace line itself — ``Event.to_json()`` — so the browser
    parses exactly the bytes that would have been written to the JSONL file.
    There is no second serialisation to drift out of sync with the first.

    Two framing hazards, both handled by construction: JSON escapes ``\\n``
    inside strings, so a ``data:`` value can never contain a raw newline; and
    the encoder is UTF-8, which is the SSE default charset.
    """
    payload = event.to_json()
    return f"id: {event.seq}\nevent: lens\ndata: {payload}\n\n".encode("utf-8")


def sse_event(name: str, data: dict) -> bytes:
    """Encode a non-trace frame (``meta``/``done``/``error``)."""
    payload = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {name}\ndata: {payload}\n\n".encode("utf-8")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _parse_cursor(source: str, value: str, *, minimum: int) -> int:
    """Parse one resume cursor, or raise ``ValueError`` with a usable message.

    ``int()`` alone is too forgiving: it accepts ``" 12"``, ``"+3"`` and
    ``"1_000"``. The cursor is a protocol value the browser echoes back, so
    require plain ASCII digits and mean it. ``minimum`` is the smallest legal
    value: 0 for ``Last-Event-ID`` (ids we send are always ``seq >= 0``) and -1
    for ``?after=``, whose ``-1`` is the documented "from the beginning".
    """
    body = value[1:] if value.startswith("-") else value
    if not (body.isascii() and body.isdigit()):
        raise ValueError(
            f"{source}{value!r} is not a resume cursor; expected an integer "
            f">= {minimum}"
        )
    number = int(value)
    if number < minimum:
        raise ValueError(
            f"{source}{value!r} is not a resume cursor; expected an integer "
            f">= {minimum}"
        )
    return number


class LensHandler(BaseHTTPRequestHandler):
    """Serves the page and the stream. One thread per connection."""

    #: Shared, read-only after construction — set by :func:`make_server`.
    feed_path: str = ""
    feed_follow: bool = False
    #: Second trace for the M3 comparison page; None means single-trace mode.
    compare_path: str | None = None
    server_version = "lens-harness"
    protocol_version = "HTTP/1.1"  # the probe measured real streaming under this

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # BaseHTTPRequestHandler writes a line per request to stderr, which
        # buries the "lens is live at ..." message a user actually needs.
        if os.environ.get("LENS_SERVE_LOG"):
            super().log_message(fmt, *args)

    def _start_response(
        self,
        status: int,
        content_type: str | None,
        content_length: int | None,
    ) -> None:
        """Write the status line and headers.

        ``content_type=None`` / ``content_length=None`` omit those headers
        entirely, which is what a 204 requires: a 204 has no body, so it must
        not define one.
        """
        self.send_response(status)
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        # The page changes while you work on it; never serve a stale lens.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self._start_response(status, content_type, len(body))
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        """A bodiless response (204): no Content-Type, no Content-Length."""
        self._start_response(status, None, None)

    def _peer_gone(self) -> bool:
        """True when the client has closed its end of a streaming response.

        A write is the usual way to discover a dead peer, but on a quiet trace
        there is nothing to write for up to :data:`PING_AFTER_S` seconds, so a
        closed tab would keep a producer polling a file for nobody. Polling the
        socket catches that in about :data:`DISCONNECT_CHECK_S`.

        Readable data is *not* treated as a disconnect: an HTTP/1.1 client is
        not expected to send anything on a one-way stream, so whatever is there
        is ignored — it is only used as a signal that the socket is still alive.

        An empty read means the peer shut its sending direction. On this stream
        that is taken as "gone", which is the safe reading: nothing the server
        does is driven by client input, and an ``EventSource`` never half-closes,
        so the only realistic sender of ``b""`` is a departing tab.
        """
        conn = getattr(self, "connection", None)
        if conn is None:
            return False
        try:
            readable, _, _ = select.select([conn], [], [], 0)
        except (OSError, ValueError):
            return True  # the socket is already unusable
        if not readable:
            return False
        try:
            data = conn.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True
        return not data  # b"" is an orderly shutdown: the client left

    # -- routes ------------------------------------------------------------
    def _route_path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self._route_path()
        try:
            if path == "/":
                self._serve_index()
            elif path == "/compare":
                self._serve_compare()
            elif path == "/api/diff":
                self._serve_api_diff()
            elif path == "/events":
                self._serve_events()
            elif path == "/api/health":
                self._serve_health()
            elif path == "/favicon.ico":
                self._send_empty(204)
            else:
                self._send_bytes(
                    404,
                    b"not found: try / or /events",
                    "text/plain; charset=utf-8",
                )
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The tab closed mid-write. Normal, and never worth a traceback.
            return

    def do_HEAD(self) -> None:  # noqa: N802 (stdlib naming)
        """Answer ``HEAD`` with the same headers as ``GET`` and no body.

        ``BaseHTTPRequestHandler`` has no default ``do_HEAD``, so without this
        every ``HEAD`` was a 501. ``HEAD /events`` is answered from the headers
        alone: a HEAD must never block on a live stream.
        """
        path = self._route_path()
        try:
            if path == "/":
                try:
                    length = os.path.getsize(_INDEX_PATH)
                except OSError:
                    self._start_response(500, "text/plain; charset=utf-8", None)
                    return
                self._start_response(200, "text/html; charset=utf-8", length)
            elif path == "/events":
                try:
                    self._client_after()
                except ValueError as exc:
                    body = f"bad resume cursor: {exc}\n".encode("utf-8")
                    self._start_response(400, "text/plain; charset=utf-8", len(body))
                    return
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
            elif path == "/api/health":
                payload = self._health_payload()
                self._start_response(200, "application/json", len(payload))
            elif path == "/favicon.ico":
                self._send_empty(204)
            else:
                body = b"not found: try / or /events"
                self._start_response(404, "text/plain; charset=utf-8", len(body))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _health_payload(self) -> bytes:
        # Health is the one endpoint a monitor polls repeatedly, so it answers
        # with the name, not the location: a basename is all the page needs, and
        # an absolute path would leak the user's directory layout to anything
        # that can reach the port. (The ``meta`` frame still carries ``path``,
        # but only a client that has already opened the stream can see it.)
        return json.dumps(
            {
                "ok": True,
                "trace": os.path.basename(self.feed_path),
                "follow": self.feed_follow,
            }
        ).encode("utf-8")

    def _serve_health(self) -> None:
        self._send_bytes(200, self._health_payload(), "application/json")

    def _serve_index(self) -> None:
        try:
            with open(_INDEX_PATH, "rb") as fh:
                body = fh.read()
        except FileNotFoundError:
            self._send_bytes(
                500,
                f"lens page missing: {_INDEX_PATH}\n"
                "the package is incomplete; reinstall or restore lens_harness/web/"
                .encode("utf-8"),
                "text/plain; charset=utf-8",
            )
            return
        self._send_bytes(200, body, "text/html; charset=utf-8")

    def _serve_compare(self) -> None:
        if self.compare_path is None:
            self._send_bytes(
                404,
                b"no comparison configured: start the server with two traces, "
                b"e.g. `lens serve a.jsonl b.jsonl`",
                "text/plain; charset=utf-8",
            )
            return
        try:
            with open(_COMPARE_PATH, "rb") as fh:
                body = fh.read()
        except FileNotFoundError:
            self._send_bytes(
                500,
                f"compare page missing: {_COMPARE_PATH}\n".encode("utf-8"),
                "text/plain; charset=utf-8",
            )
            return
        self._send_bytes(200, body, "text/html; charset=utf-8")

    def _serve_api_diff(self) -> None:
        """The precomputed M3 comparison as one JSON document.

        Computed per request from the two trace files: both are small by
        design and a running `watch`/rerun may have rewritten them, so a
        stale cached diff would lie about what the files hold *now*. 504
        when no second trace was configured.
        """
        from .compare import diff_traces, normalize

        if self.compare_path is None:
            self._send_bytes(504, b"no second trace configured", "text/plain; charset=utf-8")
            return
        try:
            from .events import read_trace

            events_a = read_trace(self.feed_path)
            events_b = read_trace(self.compare_path)
        except (OSError, ValueError) as exc:
            self._send_bytes(
                500,
                f"cannot read traces: {type(exc).__name__}: {exc}\n".encode("utf-8"),
                "text/plain; charset=utf-8",
            )
            return

        diff = diff_traces(events_a, events_b)
        payload = {
            "a": {"trace": os.path.basename(self.feed_path), "events": len(events_a),
                  "records": len(normalize(events_a))},
            "b": {"trace": os.path.basename(self.compare_path), "events": len(events_b),
                  "records": len(normalize(events_b))},
            "identical": diff.identical,
            "entries": [
                {
                    "kind": e.kind,
                    "key": [str(p) for p in e.key],
                    "status": e.status,
                    "changed_fields": e.changed_fields,
                    "a": e.a,
                    "b": e.b,
                }
                for e in diff.entries
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(200, body, "application/json")

    # -- the stream --------------------------------------------------------
    def _client_after(self) -> int:
        """Resume point: ``Last-Event-ID`` (browser reconnect) beats ``?after=``.

        Parsed once per request and validated: a cursor the browser could not
        have produced (``abc``, ``1e9``, ``-5``, ``+3``) is a client bug, not a
        reason to silently replay the whole trace. ``ValueError`` carries the
        message the caller turns into a 400.

        A negative ``Last-Event-ID`` cannot come from ``EventSource`` (the ids we
        send are always ``seq >= 0``), but ``?after=-1`` is the documented
        "from the beginning" request, so it stays legal for the query parameter
        only.
        """
        last = self.headers.get("Last-Event-ID")
        if last is not None:
            return _parse_cursor("Last-Event-ID: ", last, minimum=0)
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        for pair in query.split("&"):
            key, _, value = pair.partition("=")
            if key == "after":
                return _parse_cursor("?after=", value, minimum=-1)
        return -1

    def _serve_events(self) -> None:
        # One parse for both the meta frame and the producer.
        try:
            after = self._client_after()
        except ValueError as exc:
            body = f"bad resume cursor: {exc}\n".encode("utf-8")
            self._send_bytes(400, body, "text/plain; charset=utf-8")
            return

        # A stream has no Content-Length, so it can only end by closing the
        # socket. HTTP/1.1 would otherwise leave this connection in keep-alive
        # and the client would hang after `done` waiting for more bytes.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        # Proxies that buffer would turn a live stream into one delayed blob.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        follow = self.feed_follow
        mode = "follow" if follow else "replay"
        self.wfile.write(
            sse_event(
                "meta",
                {
                    "trace": os.path.basename(self.feed_path),
                    "path": self.feed_path,
                    "mode": mode,
                    "after": after,
                },
            )
        )
        self.wfile.flush()

        # Producer thread → queue, so the request thread can time out and ping.
        # Without this the handler blocks inside the file poll and can never
        # write a keep-alive, and a closed tab is only noticed on the next event.
        out: queue.Queue[object] = queue.Queue(maxsize=2048)
        stop = threading.Event()
        done = object()

        def produce() -> None:
            try:
                for event in TraceFeed(
                    self.feed_path,
                    after=after,
                    follow=follow,
                    stop=stop,  # so a quiet poll is interrupted, not leaked
                ):
                    # The queue is bounded: an abandoned consumer fills it and
                    # a bare put() would block this daemon thread forever.
                    while True:
                        if stop.is_set():
                            return
                        try:
                            out.put(event, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:  # noqa: BLE001 — reported to the page
                while True:
                    try:
                        out.put(exc, timeout=0.5)
                        break
                    except queue.Full:
                        if stop.is_set():
                            return
            finally:
                # Never block here either: if the consumer is gone, nobody will
                # read the sentinel and a blocking put would strand the thread.
                while True:
                    try:
                        out.put(done, timeout=0.5)
                        break
                    except queue.Full:
                        if stop.is_set():
                            return

        threading.Thread(target=produce, daemon=True).start()

        truncated = False
        waited = 0.0
        try:
            while True:
                try:
                    item = out.get(timeout=DISCONNECT_CHECK_S)
                    waited = 0.0
                except queue.Empty:
                    if self._peer_gone():
                        return
                    waited += DISCONNECT_CHECK_S
                    if waited < PING_AFTER_S:
                        continue
                    waited = 0.0
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if item is done:
                    break
                if isinstance(item, BaseException):
                    # Do not follow this with `done`: an error is not an ending.
                    # The page keys "replay complete" off `done`, so sending it
                    # here would turn truncated output into a clean finish.
                    truncated = True
                    self.wfile.write(
                        sse_event(
                            "error",
                            {
                                "message": f"{type(item).__name__}: {item}",
                                "truncated": True,
                            },
                        )
                    )
                    self.wfile.flush()
                    break
                self.wfile.write(sse_frame(item))  # type: ignore[arg-type]
                self.wfile.flush()

            if not follow and not truncated:
                self.wfile.write(sse_event("done", {"events": "eof"}))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        finally:
            # Tell the producer to stop *and* unblock it if it is stuck on a full
            # queue; then close the socket. Without the close a keep-alive client
            # waits for a body that will never come.
            stop.set()
            self.close_connection = True


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def make_server(
    trace_path: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    follow: bool = False,
    compare_path: str | None = None,
) -> ThreadingHTTPServer:
    """Build (do not start) a server for one trace file.

    In replay mode the trace must already be a readable *file* — not merely a
    path that exists. A directory (or an unreadable file) would otherwise be
    accepted here and only fail once a browser is attached, as a mid-stream
    ``event: error`` on a page that had already drawn a blank graph. Following
    is different: the run may not have created the trace yet, so a missing path
    is legal and the feed waits for it.

    With ``compare_path`` the server additionally serves ``/compare`` — a
    second page that renders the M3 diff of the two traces — and ``/api/diff``,
    the precomputed comparison as JSON. Both traces must exist; a comparison of
    a live file makes no sense (it would be stale before it rendered).
    """
    if not follow:
        if not os.path.isfile(trace_path) or not os.access(trace_path, os.R_OK):
            raise FileNotFoundError(
                f"no readable trace file at {trace_path}; serve an existing "
                "trace or pass --follow to wait for one being written"
            )
    if compare_path is not None and (
        not os.path.isfile(compare_path) or not os.access(compare_path, os.R_OK)
    ):
        raise FileNotFoundError(f"no readable trace file at {compare_path} (the second trace)")
    handler = type(
        "BoundLensHandler",
        (LensHandler,),
        {
            "feed_path": trace_path,
            "feed_follow": follow,
            "compare_path": compare_path,
        },
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(
    trace_path: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    follow: bool = False,
    compare_path: str | None = None,
) -> int:
    """Serve until interrupted. Returns a process exit code."""
    httpd = make_server(trace_path, host=host, port=port, follow=follow,
                        compare_path=compare_path)
    actual_port = httpd.server_address[1]
    mode = "live (following the file)" if follow else "replay (finished trace)"
    print(f"lens serving {trace_path}")
    print(f"  mode : {mode}")
    print(f"  open : http://{host}:{actual_port}/")
    if compare_path:
        print(f"  diff : http://{host}:{actual_port}/compare  ({os.path.basename(compare_path)})")
    print("  stop : Ctrl-C")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: bound to {host}, not loopback — traces are visible on the network")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - convenience only
    import sys

    raise SystemExit(serve(sys.argv[1] if len(sys.argv) > 1 else "traces"))
