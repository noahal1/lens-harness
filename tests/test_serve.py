"""Transport tests for ``lens_harness.serve`` — no pytest, no dependencies.

Run with ``py -3.13 -m tests.test_serve``.

The web view is the part of this project that is hardest to trust by reading,
because "it looked right in the browser" is not evidence. So the suite pins the
things that would silently break it:

* the frame bytes (a wrong ``id:`` breaks resume; a wrong ``data:`` breaks JSON);
* the half-written-line rule (the same bug class ``follow_trace`` had);
* resume semantics (``Last-Event-ID`` beats ``?after=``);
* the page being genuinely self-contained (no CDN link, no build step);
* the page's *purity*: it must not be able to reach the runtime.

Everything runs against a real ``ThreadingHTTPServer`` on port 0, in-process, so
the tests exercise the same request path the browser does.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid

from lens_harness.events import Event, read_trace
from lens_harness.serve import (
    TraceFeed,
    make_server,
    sse_event,
    sse_frame,
)

_TESTS: list[tuple[str, object]] = []

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_TRACE = os.path.join(HERE, "traces", "r_70b323a6.jsonl")
PAGE = os.path.join(HERE, "lens_harness", "web", "index.html")


def assert_eq(actual, expected, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def assert_true(cond, what: str) -> None:
    if not cond:
        raise AssertionError(what)


def test(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


class Fixture:
    """A private directory plus a live server on an ephemeral port.

    Same two Windows-sandbox constraints as ``tests/test_kernel.py``: scratch
    files must live inside the workspace, and directories are created by hand
    because ``tempfile.mkdtemp``'s 0700 mode is refused by the workspace ACL.
    """

    _ROOT = os.path.join(HERE, ".lens-test-tmp")

    def __init__(self, source: str = DEMO_TRACE) -> None:
        os.makedirs(self._ROOT, exist_ok=True)
        self.dir = os.path.join(self._ROOT, f"serve-{uuid.uuid4().hex[:10]}")
        os.makedirs(self.dir, exist_ok=True)
        self.trace = os.path.join(self.dir, "t.jsonl")
        if source:
            shutil.copyfile(source, self.trace)
        self._server = None

    # -- server ------------------------------------------------------------
    def serve(self, *, follow: bool = False, path: str | None = None):
        self._server = make_server(path or self.trace, port=0, follow=follow)
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # -- requests ----------------------------------------------------------
    def get(self, path: str, headers: dict | None = None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def frames(self, path: str = "/events", headers: dict | None = None):
        status, _, body = self.get(path, headers)
        assert_eq(status, 200, f"GET {path} status")
        raw = body.decode("utf-8")
        out = []
        for chunk in raw.split("\n\n"):
            if not chunk.strip():
                continue
            name, data, ident = None, None, None
            for line in chunk.split("\n"):
                if line.startswith(":"):
                    continue
                key, _, value = line.partition(": ")
                if key == "event":
                    name = value
                elif key == "data":
                    data = value
                elif key == "id":
                    ident = value
            out.append({"event": name, "data": data, "id": ident})
        return out

    def cleanup(self) -> None:
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------
@test
def test_sse_frame_carries_the_trace_line_verbatim():
    """The data field must be the exact JSONL line, not a re-serialisation.

    This is the guarantee that lets the page and the terminal renderer read the
    same bytes. A ``json.dumps`` round-trip here would look identical until
    someone added a field with a non-default encoder.
    """
    events = read_trace(DEMO_TRACE)
    frame = sse_frame(events[0]).decode("utf-8")
    assert_true(frame.startswith(f"id: {events[0].seq}\n"), "frame starts with the seq")
    assert_true("\nevent: lens\n" in frame, "frame has the lens event name")
    assert_true(frame.endswith("\n\n"), "frame ends with a blank line")
    payload = frame.split("\ndata: ", 1)[1].rstrip("\n")
    assert_eq(payload, events[0].to_json(), "payload is Event.to_json() verbatim")
    assert_eq(Event.from_json(payload).seq, events[0].seq, "payload round-trips")


@test
def test_sse_frame_never_contains_a_raw_newline_in_data():
    """A raw newline inside ``data:`` would silently split one event in two."""
    for event in read_trace(DEMO_TRACE):
        frame = sse_frame(event).decode("utf-8").rstrip("\n")
        for line in frame.split("\n"):
            assert_true(line.startswith(("id: ", "event: ", "data: ")),
                        "every line is a known field")


@test
def test_sse_event_escapes_its_payload():
    frame = sse_event("meta", {"trace": 'a"b.jsonl', "mode": "replay"}).decode("utf-8")
    assert_true(frame.startswith("event: meta\n"), "named frame")
    assert_true(frame.endswith("\n\n"), "terminated")
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert_eq(data["trace"], 'a"b.jsonl', "quotes survive")


# --------------------------------------------------------------------------
# The feed
# --------------------------------------------------------------------------
@test
def test_trace_feed_replays_every_event_in_order():
    events = list(TraceFeed(DEMO_TRACE))
    assert_eq(len(events), 49, "demo trace has 49 events")
    assert_eq([e.seq for e in events], list(range(49)), "seq is dense and ordered")


@test
def test_trace_feed_after_filters_inclusive_of_the_cursor():
    """``after=N`` means "I have seen N" — so it must resume at N+1."""
    events = list(TraceFeed(DEMO_TRACE, after=40))
    assert_eq([e.seq for e in events], list(range(41, 49)), "resumes after the cursor")


@test
def test_trace_feed_waits_for_a_half_written_line():
    """A reader can open the file between the JSON and its newline.

    The bug this pins: if the fragment is consumed, the rest of that line is
    lost forever. The feed must leave ``pos`` *before* the fragment.
    """
    fx = Fixture(source=None)
    try:
        lines = [json.loads(l) for l in open(DEMO_TRACE, encoding="utf-8") if l.strip()]
        first = json.dumps(lines[0], ensure_ascii=False)
        second = json.dumps(lines[1], ensure_ascii=False)
        with open(fx.trace, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(first + "\n")
            fh.write(second[: len(second) // 2])   # deliberately truncated
            fh.flush()

        got = []
        for event in TraceFeed(fx.trace):
            got.append(event.seq)
        assert_eq(got, [0], "only the complete line is yielded")

        with open(fx.trace, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(second[len(second) // 2 :] + "\n")
        got = [e.seq for e in TraceFeed(fx.trace)]
        assert_eq(got, [0, 1], "the completed line is read whole, not re-parsed as a fragment")
    finally:
        fx.cleanup()


@test
def test_trace_feed_missing_file_is_an_error_unless_following():
    fx = Fixture(source=None)
    try:
        raised = False
        try:
            list(TraceFeed(fx.trace))
        except FileNotFoundError:
            raised = True
        assert_true(raised, "a missing trace raises FileNotFoundError")

        # Following a file that does not exist yet is legal: the run may not have
        # created it. It must yield nothing and stay alive rather than raise --
        # and "stay alive" has to be OBSERVED, not assumed: pulling on the
        # iterator is what actually drives the code (asserting `iter(feed)` is
        # not None would never execute a line of it).
        feed = TraceFeed(fx.trace, follow=True, poll_s=0.01)
        it = iter(feed)
        box: list = []

        def pull():
            try:
                box.append(next(it))
            except StopIteration:
                box.append("stopped")
            except FileNotFoundError as exc:
                box.append(exc)

        worker = threading.Thread(target=pull, daemon=True)
        worker.start()
        worker.join(0.5)
        assert_true(worker.is_alive(), "follow mode waits for a trace that is not there yet")

        with open(fx.trace, "w", encoding="utf-8") as fh:
            fh.write(Event(seq=0, ts=0.0, run_id="r", thread_id="t", superstep=-1,
                           kind="run_start", data={"graph": {}, "inputs": {}}).to_json() + "\n")
        worker.join(2.0)
        assert_true(not worker.is_alive(), "the follower wakes up once the line lands")
        assert_true(isinstance(box and box[0], Event), f"follow delivered the event, got {box}")
    finally:
        fx.cleanup()


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------
@test
def test_health_reports_the_trace_and_mode():
    fx = Fixture().serve()
    try:
        status, _, body = fx.get("/api/health")
        assert_eq(status, 200, "health status")
        data = json.loads(body)
        assert_eq(data["ok"], True, "ok")
        assert_true(data["trace"].endswith("t.jsonl"), "names the trace")
        assert_eq(data["follow"], False, "replay mode")
    finally:
        fx.cleanup()


@test
def test_events_streams_every_event_then_done():
    fx = Fixture().serve()
    try:
        frames = fx.frames()
        assert_eq(frames[0]["event"], "meta", "first frame is meta")
        events = [f for f in frames if f["event"] == "lens"]
        assert_eq(len(events), 49, "all 49 events streamed")
        assert_eq([int(f["id"]) for f in events], list(range(49)), "ids are the seq values")
        assert_eq(frames[-1]["event"], "done", "a replay ends with done")
    finally:
        fx.cleanup()


@test
def test_events_resumes_from_last_event_id():
    """This is what makes EventSource's automatic reconnect cheap."""
    fx = Fixture().serve()
    try:
        frames = fx.frames(headers={"Last-Event-ID": "40"})
        events = [f for f in frames if f["event"] == "lens"]
        assert_eq([int(f["id"]) for f in events], list(range(41, 49)), "resumed at 41")
    finally:
        fx.cleanup()


@test
def test_last_event_id_beats_the_query_parameter():
    fx = Fixture().serve()
    try:
        frames = fx.frames("/events?after=10", headers={"Last-Event-ID": "45"})
        ids = [int(f["id"]) for f in frames if f["event"] == "lens"]
        assert_eq(ids, [46, 47, 48], "the header wins over ?after=")
    finally:
        fx.cleanup()


@test
def test_unknown_paths_are_404_and_say_where_to_go():
    fx = Fixture().serve()
    try:
        status, _, body = fx.get("/nope")
        assert_eq(status, 404, "404 status")
        assert_true(b"/events" in body, "the message names the real routes")
    finally:
        fx.cleanup()


@test
def test_index_is_served_and_not_cached():
    fx = Fixture().serve()
    try:
        status, headers, body = fx.get("/")
        assert_eq(status, 200, "index status")
        assert_eq(headers["Content-Type"], "text/html; charset=utf-8", "content type")
        assert_eq(headers["Cache-Control"], "no-store", "never cached")
        assert_true(b"<script>" in body, "it is the lens page")
    finally:
        fx.cleanup()


@test
def test_two_clients_each_get_an_independent_full_stream():
    """A second tab must not steal the first one's events."""
    fx = Fixture().serve()
    try:
        results: dict[str, list] = {}

        def grab(name: str) -> None:
            results[name] = [f for f in fx.frames() if f["event"] == "lens"]

        threads = [threading.Thread(target=grab, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        for name in ("a", "b"):
            assert_eq(len(results.get(name, [])), 49, f"client {name} got every event")
    finally:
        fx.cleanup()


@test
def test_follow_mode_streams_what_arrives_later():
    """``--follow`` must keep the connection open and deliver new lines."""
    fx = Fixture(source=None)
    try:
        with open(fx.trace, "w", encoding="utf-8", newline="\n") as fh:
            pass
        fx.serve(follow=True)
        got: list[int] = []

        def consume() -> None:
            req = urllib.request.Request(fx.base + "/events", headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith("id: "):
                        got.append(int(line[4:]))
                    if len(got) >= 2:
                        return

        t = threading.Thread(target=consume, daemon=True)
        t.start()

        import time

        time.sleep(0.4)  # let the request reach the handler before writing
        lines = [l for l in open(DEMO_TRACE, encoding="utf-8") if l.strip()][:2]
        with open(fx.trace, "a", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(line if line.endswith("\n") else line + "\n")
                fh.flush()
        t.join(timeout=20)
        assert_eq(got[:2], [0, 1], "both late events arrived, in order")
    finally:
        fx.cleanup()


# --------------------------------------------------------------------------
# The stream must end, and end honestly
# --------------------------------------------------------------------------
def _produce_threads() -> list:
    return [t for t in threading.enumerate() if "produce" in t.name]


@test
def test_events_socket_reaches_eof_after_done():
    """A finished stream must close the socket instead of keeping it alive.

    ``text/event-stream`` has no Content-Length, so EOF is the only way a client
    can learn the stream is over. urllib cannot see this bug: its HTTP handler
    asks for ``Connection: close`` by itself, so every test above passes either
    way. A raw socket asks for nothing and keeps the default keep-alive -- which
    is exactly what a browser does. Before the fix this recv() blocked until the
    timeout with the full 19 KB already received.
    """
    fx = Fixture().serve()
    try:
        port = fx._server.server_address[1]
        with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
            sock.sendall(
                b"GET /events HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Accept: text/event-stream\r\n"
                b"\r\n"
            )
            sock.settimeout(5.0)
            chunks: list[bytes] = []
            closed = False
            try:
                while True:
                    data = sock.recv(65536)
                    if not data:
                        closed = True
                        break
                    chunks.append(data)
            except socket.timeout:
                closed = False
        raw = b"".join(chunks)
        assert_true(closed, "server closed the socket after done (no keep-alive hang)")
        assert_true(b"event: done" in raw, "the done frame was sent")
        assert_eq(raw.count(b"event: lens"), 49, "every event still streamed")
    finally:
        fx.cleanup()


@test
def test_aborted_follow_clients_leave_no_threads_behind():
    """A closed tab on a quiet follow stream must not strand its producer.

    The producer blocks in the poll loop, and it also blocks on the bounded
    queue when nobody reads. Both waits have to notice the stop event; a bare
    ``time.sleep`` and an unbounded ``put`` each leak one thread (and one open
    file handle) per abandoned tab, for the life of the server -- invisible
    until a long session starts wedging.
    """
    baseline = len(_produce_threads())
    fx = Fixture(source=None)
    try:
        with open(fx.trace, "w", encoding="utf-8", newline="\n"):
            pass
        fx.serve(follow=True)
        port = fx._server.server_address[1]

        for _ in range(3):
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            try:
                sock.sendall(
                    b"GET /events HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Accept: text/event-stream\r\n"
                    b"\r\n"
                )
                sock.settimeout(10)
                seen = b""
                while b"event: meta" not in seen:  # the producer thread exists now
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    seen += chunk
            finally:
                sock.close()  # abandon it, the way a closed tab does

        deadline = time.time() + 10.0
        while time.time() < deadline and len(_produce_threads()) > baseline:
            time.sleep(0.1)
        assert_eq(len(_produce_threads()), baseline,
                  "no producer thread survives its client")
    finally:
        fx.cleanup()


@test
def test_truncate_then_append_still_delivers_new_events():
    """A trace replaced by a new run must be read from the start again.

    ``TraceFeed`` remembers a byte offset. When the file shrinks (a rotated or
    re-run trace), that offset points past EOF and ``seek`` silently finds
    nothing: the view freezes while the file has content, and no error is ever
    raised. The offset has to be reset when the file is shorter than it was.
    """
    fx = Fixture()
    fx.serve(follow=True)
    got: list[int] = []
    try:
        def consume() -> None:
            req = urllib.request.Request(fx.base + "/events")
            with urllib.request.urlopen(req, timeout=30) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith("id: "):
                        got.append(int(line[4:]))

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        deadline = time.time() + 10.0
        while time.time() < deadline and len(got) < 49:
            time.sleep(0.05)
        assert_eq(len(got), 49, "the original run streamed first")

        fresh = [
            Event(seq=1000 + i, ts=0.0, run_id="r2", thread_id="t",
                  superstep=0, kind="node_token",
                  data={"node": "agent", "token": f"t{i}"}).to_json()
            for i in range(3)
        ]
        with open(fx.trace, "w", encoding="utf-8", newline="\n") as fh:
            for line in fresh:
                fh.write(line + "\n")
            fh.flush()

        deadline = time.time() + 10.0
        while time.time() < deadline and 1002 not in got:
            time.sleep(0.05)
        assert_true(1002 in got,
                    f"the replaced trace was read from the start, got {got[-6:]}")
        # `1002 in got` alone is too weak: re-streaming the whole first run
        # before the new one would also satisfy it. The offset must not have
        # been reused at all, and no seq may be delivered twice.
        assert_eq(got.count(1000), 1, "the first record of the new run appeared once")
        assert_eq(got, list(dict.fromkeys(got)), "no seq was delivered twice")
    finally:
        fx.cleanup()


@test
def test_resume_cursor_does_not_hide_a_rerun_trace():
    """A stale `Last-Event-ID` must not freeze the view on a rerun trace.

    A resume cursor is a claim about one particular file ("I have seen up to
    seq N"). If the trace was rerun while the browser was away, the new run
    starts at seq 0 and never reaches N, so honouring the cursor would hide
    every event of a perfectly good run — and unlike a crash, it looks like
    an idle run: nothing arrives, nothing errors.

    ``TraceFeed`` therefore reconciles the cursor against the file's last
    seq once, and drops it when the file cannot contain that seq.
    """
    fx = Fixture()
    try:
        fresh = [
            Event(seq=i, ts=0.0, run_id="r_new", thread_id="t",
                  superstep=0, kind="node_token",
                  data={"node": "agent", "token": f"t{i}"}).to_json()
            for i in range(2)
        ]
        with open(fx.trace, "w", encoding="utf-8", newline="\n") as fh:
            for line in fresh:
                fh.write(line + "\n")
            fh.flush()

        # The browser reconnects claiming seq 7 of a run that no longer exists.
        feed = TraceFeed(fx.trace, follow=True, poll_s=0.02, after=7)
        got: list[int] = []

        def reader() -> None:
            for ev in feed:
                got.append(ev.seq)
                if len(got) >= 2:
                    return

        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        worker.join(5.0)

        assert_eq(got, [0, 1], "the rerun was delivered despite the stale cursor")
    finally:
        fx.cleanup()


@test
def test_malformed_line_ends_with_error_and_no_done():
    """A broken line is a failure to finish, not a finish.

    The page keys "replay complete" off the ``done`` frame. Sending ``done``
    after an error would make a truncated stream (say 2 of 49 events) look
    clean, which is the worst outcome for a tool whose whole job is to be
    trusted about what happened.
    """
    fx = Fixture(source=None)
    try:
        first = [l for l in open(DEMO_TRACE, encoding="utf-8") if l.strip()][0]
        with open(fx.trace, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(first if first.endswith("\n") else first + "\n")
            fh.write("this is not json\n")
        fx.serve(follow=True)
        frames = fx.frames()
        names = [f["event"] for f in frames]
        assert_true("error" in names, f"an error frame was sent, got {names}")
        assert_true("done" not in names, f"no done frame after an error, got {names}")
        error = next(f for f in frames if f["event"] == "error")
        payload = json.loads(error["data"])
        assert_eq(payload.get("truncated"), True, "the error says the stream was cut")
    finally:
        fx.cleanup()


@test
def test_a_malformed_resume_cursor_is_a_400_not_a_silent_replay():
    """A cursor the browser could not have produced is a client bug.

    Silently replaying the whole trace would make a broken client look like a
    working one; silently returning zero events is worse still.
    """
    fx = Fixture().serve()
    try:
        for bad in ("abc", "1e9", "-5", "+3"):
            status, _, body = fx.get("/events?after=" + urllib.parse.quote(bad))
            assert_eq(status, 400, f"?after={bad} is rejected")
            assert_true(b"bad resume cursor" in body, f"?after={bad} says why")
        status, _, _ = fx.get("/events", {"Last-Event-ID": "abc"})
        assert_eq(status, 400, "a bad Last-Event-ID is rejected too")
    finally:
        fx.cleanup()


@test
def test_serve_rejects_a_directory_not_just_a_missing_path():
    """``os.path.exists`` is not enough: a directory exists and is not a trace.

    Accepted here, the mistake only surfaces mid-stream as an ``event: error``,
    long after the startup banner promised it was serving something.
    """
    fx = Fixture()
    try:
        try:
            make_server(fx.dir, port=0)
        except FileNotFoundError as exc:
            assert_true("no readable trace file" in str(exc),
                        f"explains why, got {exc}")
        else:
            raise AssertionError("serving a directory should fail before any request")

        from lens_harness import cli

        try:
            cli.main(["serve", fx.dir, "--port", "0"])
        except SystemExit as exc:
            assert_true("no such trace" in str(exc), f"clear CLI message, got {exc}")
        else:
            raise AssertionError("the CLI must refuse a directory too")
    finally:
        fx.cleanup()


# --------------------------------------------------------------------------
# The page itself
# --------------------------------------------------------------------------
@test
def test_page_is_self_contained():
    """Zero dependencies is a promise; a CDN <script> would quietly break it.

    It must also work offline, so no http(s) URL may appear as a *resource*.
    Every way a browser can be told to fetch something is listed: the first
    version only checked <script src>, <link href> and @import, which would
    have let an <img>, an <iframe> or a protocol-relative //cdn... through.
    """
    html = open(PAGE, encoding="utf-8").read()

    patterns = (
        r"<script[^>]+src=",
        r"<link[^>]+href=",
        r"<img[^>]+src=",
        r"<iframe[^>]+src=",
        r"<source[^>]+src=",
        r"<video[^>]+src=",
        r"<audio[^>]+src=",
        r"@import\s+url\(",
        r"url\(\s*['\"]?(?!\#)",
    )
    for pattern in patterns:
        assert_true(re.search(pattern, html) is None,
                    f"no external resource: {pattern}")

    for m in re.finditer(r"https?://[^\s\"'<>)]+", html):
        url = m.group(0)
        assert_true(url.startswith(("http://127.0.0.1", "http://localhost")),
                    f"no off-box URL in the page: {url}")
    # protocol-relative URLs fetch off-box without ever saying http:
    assert_true(re.search(r"[\"'(]//[a-zA-Z0-9]", html) is None,
                "no protocol-relative URL in the page")


@test
def test_page_cannot_reach_the_runtime():
    """The purity constraint, enforced.

    The page talks to the trace and to nothing else. If it ever mentions the
    kernel modules, the architecture has drifted: the UI would then have a
    second source of truth.

    This is a lexical tripwire, not a proof. The real proof is structural: the
    page has no way to reach the kernel, because the only endpoint it calls is
    /events and the only bytes it gets are Event JSON. The tripwire still earns
    its keep -- the drift it catches ("just import the reducer, it is right
    there") is exactly the drift that reads as harmless in review.
    """
    html = open(PAGE, encoding="utf-8").read()
    for forbidden in ("lens_harness", "runtime.py", "graph.py", "checkpoint.py",
                      "state_merge(", "superstep_start("):
        assert_true(forbidden not in html, f"page must not reference {forbidden}")


@test
def test_page_starts_over_when_a_rerun_rewinds_seq():
    """A rerun restarts seq at 0, so the page must not append a second run.

    The event list is indexed by position but every label prints a seq, and
    the scrubber's range is the array length. Appending a rerun therefore
    shows duplicate seqs, a counter that disagrees with the newest seq, and a
    scrubber that no longer lines up with its own label. Seeing a seq that
    does not advance is the signal; the fix is to start the model over, which
    is always valid because the page is a pure function of the event prefix.
    """
    html = open(PAGE, encoding="utf-8").read()
    push = html[html.index("function push("):]
    push = push[:push.index("function setConn(")]
    assert_true(
        "ev.seq <= S.events[S.events.length - 1].seq" in push,
        "push() compares the incoming seq with the newest one",
    )
    assert_true("S.events = [];" in push, "push() can start the model over")
    assert_true(
        push.index("S.events = [];") < push.index("S.events.push(ev)"),
        "the model is reset before the new event is stored",
    )


@test
def test_cursor_parser_requires_ascii_digits():
    """`isdigit()` alone accepts digits that `int()` will happily convert.

    `"\u0664\u0660".isdigit()` is True and `int()` reads it as 40, so an
    Arabic-Indic Last-Event-ID would be accepted even though the docstring
    promises plain ASCII. The check has to be both.
    """
    from lens_harness.serve import _parse_cursor

    assert_eq(_parse_cursor("Last-Event-ID: ", "40", minimum=0), 40, "plain digits")
    assert_eq(_parse_cursor("?after=", "-1", minimum=-1), -1, "the documented -1")
    for bad in ("\u0664\u0660", "\uff11\uff12", "abc", "", "+3", "1_0", "1.5"):
        try:
            _parse_cursor("Last-Event-ID: ", bad, minimum=0)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r} as a resume cursor")


@test
def test_page_declares_its_purity_and_has_the_three_views():
    html = open(PAGE, encoding="utf-8").read()
    assert_true("pure function of the event stream" in html.lower()
                or "PURE FUNCTION OF THE EVENT STREAM" in html, "states the contract")
    for view in ('data-tab="state"', 'data-tab="diff"', 'data-tab="writes"',
                 'data-tab="events"', 'data-tab="cps"'):
        assert_true(view in html, f"inspector has {view}")
    for part in ('id="graph"', 'id="steps"', 'id="range"'):
        assert_true(part in html, f"page has {part}")


@test
def test_page_resolves_candidates_through_mapping_not_prose():
    """``candidates`` are router *return values*, not node names.

    Two generations of graph snapshots exist: newer ones carry ``mapping``
    (candidate -> node) plus prose ``descriptions``; older ones carry no
    ``mapping`` and put the node name in ``descriptions``. Both pinned traces in
    ``traces/`` are the older generation, so if the page ever resolved candidates
    from prose alone it would still look right on those and break on a fresh
    trace -- the edge would point at a node that does not exist and vanish.

    The resolver therefore has to consult ``mapping`` first. Pinned here because
    this failure is invisible on the very traces used for visual checking.
    """
    html = open(PAGE, encoding="utf-8").read()

    assert_true("ce.mapping" in html, "resolver consults the mapping field")
    assert_true(html.count("candTarget(") >= 4, "every caller goes through the resolver")
    # the old, wrong idiom: jump straight from a candidate to its description
    assert_true("ce.descriptions[key]) || key" not in html,
                "candidates must not be resolved from prose alone")


@test
def test_page_handles_every_event_kind_the_kernel_can_emit():
    """A new event kind must not render as a blank row."""
    from lens_harness.events import EVENT_SCHEMA

    html = open(PAGE, encoding="utf-8").read()
    for kind in EVENT_SCHEMA:
        assert_true(f'"{kind}"' in html, f"page handles {kind}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@test
def test_cli_serve_subcommand_is_wired():
    from lens_harness.cli import build_parser

    args = build_parser().parse_args(["serve", "traces/x.jsonl"])
    assert_eq(args.command, "serve", "command name")
    assert_eq(args.host, "127.0.0.1", "default host is loopback")
    assert_eq(args.port, 8000, "default port")
    assert_eq(args.follow, False, "replay by default")
    assert_true(callable(args.func), "has a func")


@test
def test_cli_serve_rejects_a_missing_trace_with_a_readable_message():
    from lens_harness import cli

    try:
        cli.main(["serve", os.path.join(HERE, "traces", "definitely-not-here.jsonl"),
                  "--port", "0"])
    except SystemExit as exc:
        assert_true("no such trace" in str(exc), f"clear message, got {exc}")
        return
    raise AssertionError("serving a missing trace should exit, not start a server")


def main() -> int:
    failures = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"ok  {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    total = len(_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
