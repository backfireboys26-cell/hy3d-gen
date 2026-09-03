"""auth_gate.py - the container's front door: bearer auth + health, stdlib only.

The patched Hunyuan3D api_server binds 127.0.0.1:${UPSTREAM_PORT} INSIDE the container and is
never exposed directly. This gate listens on ${PORT} (RunPod LOAD_BALANCER convention) and:

  (a) forwards every request verbatim to the api_server (same method, path, body, so
      generate3d.py's POST /send + GET /status/{uid} contract passes through unchanged);
  (b) enforces the app token ONLY when HY3D_TOKEN is set in the environment - unset means
      OPEN, which is the local-container test mode. The token is runtime env, never baked into
      the image (the vault never holds secret values). The token is read from
      `X-HY3D-Token: <token>` FIRST and from `Authorization: Bearer <token>` as a fallback:
      a RunPod load-balancer endpoint already consumes `Authorization: Bearer <RUNPOD_API_KEY>`
      as the platform layer, so an app layer on the SAME header could only work if the two
      secrets were the same value (Phase C finding, 2026-09-02). Two headers = two independent
      walls, and a leaked app token alone cannot reach the endpoint;
  (c) serves ${HEALTH_CHECK_PATH} (default /ping) WITHOUT auth, on ${PORT_HEALTH}
      (default = PORT; a distinct PORT_HEALTH gets its own health-only listener).

Health is honest, not decorative: 200 only when the api_server is actually accepting
connections. The upstream loads multi-GB weights BEFORE it starts listening, so a probe of
the socket is exactly "model loaded and serving" - returning a blanket 200 would route load-
balancer traffic into a still-loading worker.

  (d) keeps the LOAD BALANCER's 30 MB payload cap out of the way: the upstream /status/{uid}
      returns the whole GLB as base64 (an octree-384 mesh is 25-31 MB raw, 33-42 MB encoded -
      over the cap, so the client could never even see "completed"). When the encoded body
      exceeds STATUS_INLINE_MAX the gate answers {"status":"completed","download":"/result/<uid>",
      "size":N,"sha256":...} instead and serves the raw bytes in ranges from GET
      /result/{uid}?offset=&length= (chunks capped at RESULT_CHUNK_MAX). Small meshes pass
      through verbatim, so a local container keeps the byte-identical contract.
  (e) health follows RunPod's contract: 204 (empty) while the api_server is still loading =
      "initializing", 200 = healthy; anything else is read as UNHEALTHY and a worker that
      is unhealthy for its whole first weight download gets terminated and relaunched (billed).
      The 200 body names the served model so a client can pick its step ladder honestly.
      Health is NOT blind to the generation loop (2026-09-02 fix round): once the socket
      accepts, the gate reads the api_server's GET /queue and answers 503 {"status":
      "unhealthy","reason":...} when it reports healthy:false - a job past HY3D_JOB_MAX_S or a
      dead loop thread - so the load balancer recycles a wedged worker instead of routing to
      a 'healthy' one that never finishes anything. A /queue that cannot be read at all
      (timeout, non-JSON) is 200 with "queue":"unreadable" - a GIL stall during marching
      cubes is not a wedge - unless it has stayed unreadable for HEALTH_UNREADABLE_MAX_S
      (default 120) straight, which is.
  (f) upstream status codes and the headers the contract depends on pass through VERBATIM and
      are never cached: the api_server's 429 (queue full) keeps its Retry-After, its 400
      (bad parameter) and 404 (unknown uid -> {"status":"not_found"}) reach the client as-is.
      The only thing the gate remembers is a completed result it has already decoded for
      ranged download - a 404 today is a 404 on the next poll too, never a stale answer.
  (g) 2026-09-03 round 3: an early answer (401, health, a bad Content-Length) first DRAINS the
      request body - on a keep-alive connection an unread body was parsed as the next
      request's start line and the client's next call got an HTML 400; a body over
      GATE_DRAIN_MAX (64 MiB) or unparseable is answered with Connection: close instead. A
      result is latched for ranged download only when the api_server's own declared size +
      sha256 match the bytes it served (a mismatch is a 502, never a wrong 'completed'), and the
      api_server only says 'completed' once its job record does (its export is atomic). Health
      also reads the api_server's watchdog liveness: watchdog_alive:false, or a last tick older
      than 3 ticks, is a 503 - a worker whose watchdog is dead could wedge unnoticed.

Dependency-light on purpose: stdlib only (http.server + http.client), so the gate can never
be the thing that breaks when the ML stack's pins move.
"""
import base64
import hashlib
import hmac
import http.client
import json
import os
import socket
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PORT = int(os.environ.get("PORT", "8080"))
PORT_HEALTH = int(os.environ.get("PORT_HEALTH", str(PORT)))
HEALTH_PATH = os.environ.get("HEALTH_CHECK_PATH", "/ping")
UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "8081"))
TOKEN = os.environ.get("HY3D_TOKEN") or None

# /status responses carry the whole GLB as base64 (tens of MB at high octree); read fully in
# memory - fine for a worker container, and it keeps the proxy loop trivial. Generation itself
# is async upstream ( /send returns a uid instantly ), so per-request time here is I/O only.
UPSTREAM_TIMEOUT_S = 600
MODEL_PATH = os.environ.get("MODEL_PATH", "")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "")
# a /status body above this many bytes is rewritten into a download handle (RunPod LB cap: 30 MB)
STATUS_INLINE_MAX = int(os.environ.get("STATUS_INLINE_MAX", str(16 * 1024 * 1024)))
RESULT_CHUNK_MAX = int(os.environ.get("RESULT_CHUNK_MAX", str(16 * 1024 * 1024)))
# a request body up to this size is read and discarded before an early answer (401 etc.) so the
# keep-alive connection stays in sync; anything larger is answered with Connection: close
GATE_DRAIN_MAX = int(os.environ.get("GATE_DRAIN_MAX", str(64 * 1024 * 1024)))
# decoded results kept for ranged download, newest last; a worker serves one client at a time
_RESULTS = OrderedDict()
_RESULTS_LOCK = threading.Lock()
_RESULTS_KEEP = 4


def upstream_listening() -> bool:
    try:
        with socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), timeout=2):
            return True
    except OSError:
        return False


# the api_server's /queue is read on every health probe; a slow answer is tolerated for this
# long before "unreadable" itself counts as unhealthy (a wedged event loop, not a GIL stall)
HEALTH_QUEUE_TIMEOUT_S = float(os.environ.get("HEALTH_QUEUE_TIMEOUT_S", "5"))
HEALTH_UNREADABLE_MAX_S = float(os.environ.get("HEALTH_UNREADABLE_MAX_S", "120"))
_unreadable_since = [None]   # monotonic time the queue first became unreadable, or None


def upstream_queue_state():
    """The api_server's /queue as a dict, or None when it cannot be read right now."""
    try:
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=HEALTH_QUEUE_TIMEOUT_S)
        conn.request("GET", "/queue")
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        if resp.status != 200:
            return None
        obj = json.loads(data)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def health_verdict():
    """(http_code, body_or_None) for the health path - see docstring (e)."""
    if not upstream_listening():
        return 204, None
    state = upstream_queue_state()
    now = time.monotonic()
    if state is None:
        since = _unreadable_since[0]
        if since is None:
            _unreadable_since[0] = since = now
        if now - since > HEALTH_UNREADABLE_MAX_S:
            return 503, {"status": "unhealthy", "model": MODEL_PATH, "subfolder": SUBFOLDER,
                         "reason": f"api_server /queue unreadable for {now - since:.0f}s "
                                   f"(> HEALTH_UNREADABLE_MAX_S={HEALTH_UNREADABLE_MAX_S:.0f})"}
        return 200, {"status": "ok", "model": MODEL_PATH, "subfolder": SUBFOLDER, "queue": "unreadable"}
    _unreadable_since[0] = None
    if state.get("healthy") is False:
        return 503, {"status": "unhealthy", "model": MODEL_PATH, "subfolder": SUBFOLDER,
                     "reason": state.get("reason") or "api_server reports healthy:false",
                     "queue": state}
    stalled = watchdog_stalled(state)
    if stalled:
        return 503, {"status": "unhealthy", "model": MODEL_PATH, "subfolder": SUBFOLDER,
                     "reason": stalled, "queue": state}
    return 200, {"status": "ok", "model": MODEL_PATH, "subfolder": SUBFOLDER, "queue": state}


def watchdog_stalled(state: dict):
    """A reason string when the api_server's watchdog is dead or has not ticked for more than
    3 ticks (its own arithmetic, so a stale age is caught even if watchdog_alive lied), else
    None. An api_server that does not report the fields at all (older build) is not judged."""
    age, tick = state.get("watchdog_age_s"), state.get("watchdog_tick_s")
    if state.get("watchdog_alive") is False:
        return (f"api_server watchdog is not alive (last tick {age}s ago, tick {tick}s) - a wedged "
                "generation would go undetected, recycling")
    if isinstance(age, (int, float)) and isinstance(tick, (int, float)) and tick > 0 and age > 3 * tick:
        return (f"api_server watchdog stalled: last tick {age:.0f}s ago (> 3 x {tick:.0f}s tick) - a "
                "wedged generation would go undetected, recycling")
    return None


APP_TOKEN_HEADER = "X-HY3D-Token"


def _same(a: str, b: str) -> bool:
    # bytes, not str: a non-ASCII header made compare_digest raise (TypeError) and drop the
    # connection with a traceback instead of a clean 401 (audit 2026-09-02 #17)
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


def _authorized(app_header: "str | None", auth_header: "str | None") -> bool:
    if TOKEN is None:
        return True
    if app_header:
        return _same(app_header.strip(), TOKEN)
    if auth_header:
        return _same(auth_header, f"Bearer {TOKEN}")
    return False


def _remember(uid: str, data: bytes, sha: str) -> None:
    with _RESULTS_LOCK:
        _RESULTS[uid] = (data, sha)
        while len(_RESULTS) > _RESULTS_KEEP:
            _RESULTS.popitem(last=False)


def _recall(uid: str):
    with _RESULTS_LOCK:
        return _RESULTS.get(uid)


def _integrity_mismatch(obj: dict, raw: bytes, sha: str):
    """A reason when the api_server's declared size/sha256 (2026-09-03 /status fields) do not
    match the bytes it served - such a result is never latched; None when they match or the
    api_server (an older build) declared nothing."""
    want_size, want_sha = obj.get("size"), (obj.get("sha256") or "")
    if isinstance(want_size, int) and not isinstance(want_size, bool) and want_size != len(raw):
        return (f"result integrity mismatch: api_server declared size {want_size} but served "
                f"{len(raw)} bytes - not latched, poll /status again")
    if isinstance(want_sha, str) and want_sha and want_sha.lower() != sha:
        return ("result integrity mismatch: api_server declared sha256 "
                f"{want_sha[:12]}... but the served bytes hash to {sha[:12]}... - not latched, poll /status again")
    return None


class _Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # concise worker-log lines, no reverse DNS stalls
        print(f"[auth_gate] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            # http.server closes after this response; tell the client so it does not reuse
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _discard_body(self) -> None:
        """Consume the request body before an early (non-proxied) answer - docstring (g).
        HTTP/1.1 keep-alive: an unread body stays in the socket and becomes the NEXT request's
        start line. A body over GATE_DRAIN_MAX, a chunked body, or an unparseable
        Content-Length is not read; the connection is closed after the answer instead."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        chunked = "chunked" in (self.headers.get("Transfer-Encoding") or "").lower()
        if length < 0 or length > GATE_DRAIN_MAX or chunked:
            self.close_connection = True
            return
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)

    def _health(self) -> None:
        code, body = health_verdict()
        if body is None:
            # 204 = "initializing" in RunPod's health contract (200 healthy, other = unhealthy)
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if code != 200:
            print(f"[auth_gate] health -> {code}: {body.get('reason')}", flush=True)
        self._json(code, body)


class GateHandler(_Base):
    """Main listener: health (no auth) + authenticated reverse proxy."""

    def _handle(self) -> None:
        if self.path.split("?", 1)[0] == HEALTH_PATH:
            self._discard_body()
            self._health()
            return
        if not _authorized(self.headers.get(APP_TOKEN_HEADER), self.headers.get("Authorization")):
            self._discard_body()
            self._json(401, {"error": f"unauthorized: this endpoint requires {APP_TOKEN_HEADER}: <token> "
                                      "(or Authorization: Bearer <token>)"})
            return
        self._proxy()

    def _upstream(self, method: str, path: str, body, headers: dict):
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=UPSTREAM_TIMEOUT_S)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, data

    def _path_uid(self, head: str):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == head and self.command == "GET":
            return parts[1]
        return None

    def _proxy(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True       # an unsized body cannot be drained: close
            self._json(400, {"error": "bad Content-Length"})
            return
        body = self.rfile.read(length) if length else None
        result_uid = self._path_uid("result")
        if result_uid:
            self._serve_result(result_uid)
            return
        try:
            headers = {}
            if self.headers.get("Content-Type"):
                headers["Content-Type"] = self.headers["Content-Type"]
            resp, data = self._upstream(self.command, self.path, body, headers)
        except OSError as e:
            self._json(502, {"error": f"upstream api_server unreachable: {e}"})
            return
        status_uid = self._path_uid("status")
        if status_uid and resp.status == 200 and len(data) > STATUS_INLINE_MAX:
            try:
                obj = json.loads(data)
            except ValueError:
                obj = None
            if isinstance(obj, dict) and obj.get("status") == "completed" and obj.get("model_base64"):
                raw = base64.b64decode(obj["model_base64"])
                sha = hashlib.sha256(raw).hexdigest()
                why = _integrity_mismatch(obj, raw, sha)
                if why:
                    self._json(502, {"error": why, "uid": status_uid})
                    return
                _remember(status_uid, raw, sha)
                self._json(200, {"status": "completed", "download": f"/result/{status_uid}",
                                 "size": len(raw), "sha256": sha})
                return
        self.send_response(resp.status)
        self.send_header("Content-Type",
                         resp.getheader("Content-Type") or "application/json")
        self.send_header("Content-Length", str(len(data)))
        # the queue's back-off advice (429) must survive the hop - a client that only sees the
        # status code would guess; a client that sees Retry-After waits exactly as long as asked
        retry_after = resp.getheader("Retry-After")
        if retry_after:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(data)

    def _serve_result(self, uid: str) -> None:
        """Raw GLB bytes in ranges: ?offset=&length= (length capped at RESULT_CHUNK_MAX). A
        uid the gate has not decoded yet (worker restarted, memory evicted) is re-fetched from
        the upstream /status once; a uid that is not completed there is a 404, never a guess."""
        hit = _recall(uid)
        if hit is None:
            try:
                resp, data = self._upstream("GET", f"/status/{uid}", None, {})
                obj = json.loads(data) if resp.status == 200 else {}
            except (OSError, ValueError) as e:
                self._json(502, {"error": f"upstream api_server unreachable: {e}"})
                return
            if not (isinstance(obj, dict) and obj.get("status") == "completed" and obj.get("model_base64")):
                self._json(404, {"error": f"no completed result for uid {uid}",
                                 "upstream": obj.get("status") if isinstance(obj, dict) else None})
                return
            raw = base64.b64decode(obj["model_base64"])
            hit = (raw, hashlib.sha256(raw).hexdigest())
            why = _integrity_mismatch(obj, raw, hit[1])
            if why:
                self._json(502, {"error": why, "uid": uid})
                return
            _remember(uid, *hit)
        raw, sha = hit
        q = parse_qs(urlsplit(self.path).query)
        try:
            offset = int(q.get("offset", ["0"])[0])
            length = int(q.get("length", [str(len(raw))])[0])
        except ValueError:
            self._json(400, {"error": "offset/length must be integers"})
            return
        if offset < 0 or offset > len(raw) or length < 0:
            self._json(416, {"error": "range out of bounds", "size": len(raw)})
            return
        length = min(length, RESULT_CHUNK_MAX, len(raw) - offset)
        chunk = raw[offset:offset + length]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("X-Result-Size", str(len(raw)))
        self.send_header("X-Result-Sha256", sha)
        self.send_header("X-Result-Offset", str(offset))
        self.end_headers()
        self.wfile.write(chunk)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _handle


class HealthOnlyHandler(_Base):
    """Extra listener when PORT_HEALTH != PORT: health path only, never proxies, no auth."""

    def _handle(self) -> None:
        self._discard_body()
        if self.path.split("?", 1)[0] == HEALTH_PATH:
            self._health()
        else:
            self._json(404, {"error": f"health listener serves only {HEALTH_PATH}"})

    do_GET = do_POST = _handle


def main() -> None:
    mode = "OPEN (no HY3D_TOKEN set - local test mode)" if TOKEN is None else "bearer-auth"
    print(f"[auth_gate] :{PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT} · {mode} · "
          f"health {HEALTH_PATH} on :{PORT_HEALTH}", flush=True)
    servers = [ThreadingHTTPServer(("0.0.0.0", PORT), GateHandler)]
    if PORT_HEALTH != PORT:
        servers.append(ThreadingHTTPServer(("0.0.0.0", PORT_HEALTH), HealthOnlyHandler))
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
