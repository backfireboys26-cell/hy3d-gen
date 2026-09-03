"""stub_api_server.py - stands in for /app/api_server.py in test-start-sh.sh (no ML stack).

Accepts the exact argv start.sh passes, prints the env start.sh is supposed to have decided
(HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / U2NET_HOME), then serves the two routes auth_gate's
health path needs: GET /queue (healthy unless STUB_UNHEALTHY=1) and a 404 for anything else.
Knobs: STUB_LOAD_S=<s> delays listening (health must read 204 meanwhile); STUB_DIE=<code>
exits with that code instead of serving (start.sh must take the container down with it).
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", default="8081")
ap.add_argument("--model_path", default="")
ap.add_argument("--subfolder", default="")
ap.add_argument("--device", default="")
a = ap.parse_args()
print(f"[stub-api] args={sys.argv[1:]} HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
      f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')} U2NET_HOME={os.environ.get('U2NET_HOME')}",
      flush=True)
time.sleep(float(os.environ.get("STUB_LOAD_S", "0")))
if os.environ.get("STUB_DIE"):
    print(f"[stub-api] dying on purpose with {os.environ['STUB_DIE']}", flush=True)
    sys.exit(int(os.environ["STUB_DIE"]))
HEALTHY = os.environ.get("STUB_UNHEALTHY") != "1"


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/queue":
            code, obj = 200, {"in_flight": 0, "queued": 0, "queue_max": 4, "loop_alive": HEALTHY,
                              "in_flight_age_s": None, "job_max_s": 900, "stuck": not HEALTHY,
                              "healthy": HEALTHY, "reason": None if HEALTHY else "stub says wedged"}
        else:
            code, obj = 404, {"status": "not_found"}
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


print(f"[stub-api] listening on {a.host}:{a.port} healthy={HEALTHY}", flush=True)
HTTPServer((a.host, int(a.port)), H).serve_forever()
