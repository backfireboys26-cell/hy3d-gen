"""auth_gate.py - the container's front door: bearer auth + health, stdlib only.

The patched Hunyuan3D api_server binds 127.0.0.1:${UPSTREAM_PORT} INSIDE the container and is
never exposed directly. This gate listens on ${PORT} (RunPod LOAD_BALANCER convention) and:

  (a) forwards every request verbatim to the api_server (same method, path, body, so
      generate3d.py's POST /send + GET /status/{uid} contract passes through unchanged);
  (b) enforces `Authorization: Bearer ${HY3D_TOKEN}` ONLY when HY3D_TOKEN is set in the
      environment - unset means OPEN, which is the local-container test mode. The token is
      runtime env, never baked into the image (the vault never holds secret values);
  (c) serves ${HEALTH_CHECK_PATH} (default /ping) WITHOUT auth, on ${PORT_HEALTH}
      (default = PORT; a distinct PORT_HEALTH gets its own health-only listener).

Health is honest, not decorative: 200 only when the api_server is actually accepting
connections. The upstream loads multi-GB weights BEFORE it starts listening, so a probe of
the socket is exactly "model loaded and serving" - returning a blanket 200 would route load-
balancer traffic into a still-loading worker.

Dependency-light on purpose: stdlib only (http.server + http.client), so the gate can never
be the thing that breaks when the ML stack's pins move.
"""
import hmac
import http.client
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


def upstream_listening() -> bool:
    try:
        with socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), timeout=2):
            return True
    except OSError:
        return False


def _authorized(header_value: "str | None") -> bool:
    if TOKEN is None:
        return True
    if not header_value:
        return False
    return hmac.compare_digest(header_value, f"Bearer {TOKEN}")


class _Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # concise worker-log lines, no reverse DNS stalls
        print(f"[auth_gate] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _health(self) -> None:
        if upstream_listening():
            self._json(200, {"status": "ok"})
        else:
            self._json(503, {"status": "loading", "detail":
                             "api_server not accepting connections yet (weights loading?)"})


class GateHandler(_Base):
    """Main listener: health (no auth) + authenticated reverse proxy."""

    def _handle(self) -> None:
        if self.path.split("?", 1)[0] == HEALTH_PATH:
            self._health()
            return
        if not _authorized(self.headers.get("Authorization")):
            self._json(401, {"error": "unauthorized: this endpoint requires "
                                      "Authorization: Bearer <token>"})
            return
        self._proxy()

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT,
                                              timeout=UPSTREAM_TIMEOUT_S)
            headers = {}
            if self.headers.get("Content-Type"):
                headers["Content-Type"] = self.headers["Content-Type"]
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
        except OSError as e:
            self._json(502, {"error": f"upstream api_server unreachable: {e}"})
            return
        self.send_response(resp.status)
        self.send_header("Content-Type",
                         resp.getheader("Content-Type") or "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _handle


class HealthOnlyHandler(_Base):
    """Extra listener when PORT_HEALTH != PORT: health path only, never proxies, no auth."""

    def _handle(self) -> None:
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
