"""harness.py - contract test of the hardened api_server + auth_gate WITHOUT a model.

Imports the patched api_server.py (upstream f2456e0 + patches/rsv4-stack.patch) with hy3dgen on
the path, swaps the ModelWorker for a stub that sleeps GEN_S seconds and writes a tiny .glb,
serves it with uvicorn on UP_PORT, runs auth_gate.py in front (one OPEN gate, one token gate),
and probes through the gates. Every claim the docker/README.md contract table makes is a case
here, plus the failure shapes a 3 am page is made of: a burst, a deleted result, the prune
bound, ranged results, the token wall, a SystemExit inside generate(), a generation that hangs
(watchdog -> honest error + 503 health + exit 3), and - round 3 (2026-09-03) - a result export
that takes seconds (verifier P2: the result must be published ATOMICALLY, /status must say
'processing' until the rename even when a half-written file sits at the final path, and the
gate must never latch a partial), a 401 that leaves the keep-alive connection usable (the body
is drained), and the watchdog's own liveness (a dead/stalled watchdog -> 503 health).

Where things are (all overridable, so the same file runs on rsv4, omen, or INSIDE the image):
  --api-dir / HY3D_API_DIR   directory holding api_server.py          (default /app)
  --repo    / HY3D_REPO      Hunyuan3D-2GP checkout providing hy3dgen (default = api-dir)
  --gate    / HY3D_GATE      auth_gate.py to put in front             (default <api-dir>/auth_gate.py)
  --log     / HY3D_HARNESS_LOG  log path; the cwd becomes its directory (api_server writes
                                gradio_cache/ there)                   (default ./harness.log)
Run inside the published image (no GPU needed): tests/run-harness-in-image.sh <image>
On rsv4/omen against the vault tree (fresh f2456e0 checkout + the vault patch, C:\\ai3d\\venv):
  powershell -NoProfile -File tests/run-harness-local.ps1 [-OutDir <dir>]
or by hand:
  C:\\ai3d\\venv\\Scripts\\python.exe harness.py --api-dir <patched upstream> --repo <patched upstream> ^
      --gate <vault>\\engine\\generation\\docker\\auth_gate.py --log <dir>\\harness.log
Exit code 1 on any failure. `--hang-child` is the helper mode the exit-3 case spawns.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--api-dir", default=os.environ.get("HY3D_API_DIR", "/app"))
ap.add_argument("--repo", default=os.environ.get("HY3D_REPO"))
ap.add_argument("--gate", default=os.environ.get("HY3D_GATE"))
ap.add_argument("--log", default=os.environ.get("HY3D_HARNESS_LOG", "harness.log"))
ap.add_argument("--hang-child", action="store_true", help=argparse.SUPPRESS)
ap.add_argument("--child-port", type=int, default=18299, help=argparse.SUPPRESS)
args = ap.parse_args()
API_DIR = os.path.abspath(args.api_dir)
REPO = os.path.abspath(args.repo or API_DIR)
GATE_PY = os.path.abspath(args.gate or os.path.join(API_DIR, "auth_gate.py"))
LOG = os.path.abspath(args.log)
UP_PORT, GATE_PORT, TOK_PORT, SGATE_PORT = 18099, 18098, 18097, 18096
GEN_S = 1.5
CRASH_SEED, HANG_SEED, SYSEXIT_SEED = 666, 555, 777
SLOW_SEED, SLOW_DIRECT_SEED = 444, 445      # round 3: a slow export (atomic) / a half-written final file
SLOW_PAUSE_S = 2.5
SLOW_RESULT = b"glTF-slow-" + bytes(range(256)) * 24   # 6154 B: over the small gate's inline cap
SLOW_SHA = hashlib.sha256(SLOW_RESULT).hexdigest()

os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.chdir(os.path.dirname(LOG))
sys.path.insert(0, REPO)
sys.path.insert(0, API_DIR)
# hy3d_models.py (the served-model catalog api_server imports) sits beside auth_gate.py: in
# /app inside the image, in <vault>/engine/generation/docker in this tree
sys.path.insert(0, os.path.dirname(GATE_PY))
# the harness fixes the knobs it depends on; the hang case lowers JOB_MAX_S at runtime
os.environ["HY3D_QUEUE_MAX"] = "4"
os.environ["HY3D_JOB_ETA_S"] = "7"
os.environ["HY3D_WATCHDOG_S"] = "1"
if not args.hang_child:
    os.environ["HY3D_JOB_MAX_S"] = "900"
    os.environ["HY3D_STUCK_EXIT"] = "0"     # in-process: prove the state, do not die
else:
    os.environ["HY3D_JOB_MAX_S"] = "2"
    os.environ["HY3D_STUCK_EXIT"] = "1"     # the child MUST die with exit 3 ...
    os.environ["HY3D_STUCK_EXIT_GRACE_S"] = "3"  # ... after a short honest-state window

import api_server as s  # noqa: E402

HAS_R3 = hasattr(s, "publish_result") and hasattr(s, "result_path")
if not HAS_R3:
    # CONTROL MODE: a pre-round-3 api_server (e.g. the cu124-20260903-336a1a3 image) has no atomic
    # publish helper. Mimic its inline, non-atomic export so cases O/O2 measure the OLD behaviour
    # honestly (FAIL rows) instead of dying on a missing attribute; N and O3 guard themselves.
    s.result_path = lambda uid, type="glb": os.path.join(s.SAVE_DIR, f"{uid}.{type}")

    def _old_publish(mesh, uid, type="glb"):
        mesh.export(s.result_path(uid, type), file_type=type)
        return s.result_path(uid, type)
    s.publish_result = _old_publish
    print("[harness] CONTROL MODE: api_server has no publish_result/result_path (pre-round-3) - the round-3 cases are expected to FAIL",
          flush=True)


class SlowMesh:
    """Stands in for the trimesh object the pipeline returns: export() writes the result in two
    halves with a pause between them - what a 30 MB octree-512 export looks like in time."""

    def __init__(self, data, pause_s):
        self.data, self.pause_s = data, pause_s

    def export(self, path, file_type=None):
        assert file_type == "glb", f"publish_result must pass file_type (got {file_type!r})"
        with open(path, "wb") as f:
            f.write(self.data[: len(self.data) // 2])
            f.flush()
            time.sleep(self.pause_s)
            f.write(self.data[len(self.data) // 2:])


class StubWorker:
    def __init__(self):
        self.seen = []                      # (kind, model, view slots) per job, for case Q/R
        self.resident = ["stub-resident"]   # what /queue + /ping report as 'loaded'

    def generate(self, uid, params):
        seed = params.get("seed")
        if seed == CRASH_SEED:
            raise RuntimeError("stub crash on purpose")
        if seed == SYSEXIT_SEED:
            raise SystemExit("stub SystemExit on purpose")
        if seed == HANG_SEED:
            time.sleep(10 ** 6)
        if seed == SLOW_SEED:
            # the REAL publish path (api_server.publish_result) with a slow exporter
            s.publish_result(SlowMesh(SLOW_RESULT, SLOW_PAUSE_S), uid, "glb")
            return
        if seed == SLOW_DIRECT_SEED:
            # the WORST writer: half the bytes straight to the final path, a pause, the rest -
            # the job record must still say 'processing' and the gate must not latch the half
            with open(s.result_path(uid), "wb") as f:
                f.write(SLOW_RESULT[: len(SLOW_RESULT) // 2])
                f.flush()
                time.sleep(SLOW_PAUSE_S)
                f.write(SLOW_RESULT[len(SLOW_RESULT) // 2:])
            return
        time.sleep(GEN_S)
        self.seen.append(("mesh", params.get("model"), sorted(params.get("views") or {})))
        with open(os.path.join(s.SAVE_DIR, f"{uid}.glb"), "wb") as f:
            f.write(b"glTF-stub-" + str(uid).encode())

    def imagine(self, uid, params):
        """The /imagine half of the worker: the SAME atomic publish, a tiny PNG per view."""
        time.sleep(GEN_S / 3)
        self.seen.append(("views", params.get("model"), sorted(params["views"])))
        views = {v: base64.b64encode(STUB_PNG + v.encode()).decode() for v in params["views"]}
        result = {"kind": "views", "model": params["model"], "views": views, "size": params["size"],
                  "steps": params["steps"], "guidance_scale": params["guidance_scale"],
                  "seed": params["seed"], "elevation": params["elevation"], "seconds": 0.1}
        s.publish_bytes(json.dumps(result).encode(), uid, "json")

    def loaded(self):
        return list(self.resident)


def serve(port):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(s.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


def req(method, path, body=None, port=GATE_PORT, headers=None, timeout=30):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    h = dict(headers or {})
    if data:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            hdr = {k.lower(): v for k, v in resp.headers.items()}
            try:
                return resp.status, hdr, (json.loads(raw) if raw else {})
            except ValueError:
                return resp.status, hdr, {"_raw": raw[:80]}
    except urllib.error.HTTPError as e:
        raw = e.read()
        hdr = {k.lower(): v for k, v in e.headers.items()}
        try:
            return e.code, hdr, json.loads(raw)
        except ValueError:
            return e.code, hdr, {"_raw": raw[:200].decode(errors="replace")}


def raw_get(path, port=GATE_PORT):
    """(status, headers, the WHOLE body bytes) - for /result chunks."""
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def raw_http(sock, request_bytes):
    """Send one raw HTTP/1.1 request on an open socket, return (head bytes, body bytes)."""
    if request_bytes:
        sock.sendall(request_bytes)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return buf, b""
        buf += chunk
    head, rest = buf.split(b"\r\n\r\n", 1)
    m = re.search(rb"content-length:\s*(\d+)", head, re.I)
    n = int(m.group(1)) if m else 0
    while len(rest) < n:
        chunk = sock.recv(65536)
        if not chunk:
            break
        rest += chunk
    return head, rest[:n]


STUB_PNG = b"\x89PNG\r\n\x1a\n"
IMG = base64.b64encode(STUB_PNG + b"\0" * 64).decode()
VIEWS = {"front": IMG, "left": IMG, "right": IMG}
good = {"image": IMG, "octree_resolution": 128, "num_inference_steps": 5, "guidance_scale": 5.0,
        "seed": 1, "unknown_field": "ignored"}

if args.hang_child:
    # helper: a stub server that hangs its first job; the watchdog must take the PROCESS down
    s.worker = StubWorker()
    s.start_generation_loop()
    serve(args.child_port)
    for _ in range(100):
        try:
            c, _, b = req("POST", "/send", {**good, "seed": HANG_SEED}, port=args.child_port)
            if c == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    print(f"[hang-child] hanging job sent -> {c} {b}; waiting for the watchdog", flush=True)
    time.sleep(60)
    print("[hang-child] STILL ALIVE after 60 s - the watchdog did not exit", flush=True)
    os._exit(99)

# PREFLIGHT (2026-09-04): a harness run that dies before its teardown leaves its gates bound to
# these ports. The next run's gates then fail to bind and DIE, the STALE ones answer instead - with
# the previous run's token - and every token case reads 401 while the open-gate cases still pass:
# a 5-case phantom failure that looks like a gate bug. A port already listening is fatal here, with
# the pids to kill, before a single case runs.
busy = None if args.hang_child else []
for name, port in [] if busy is None else (("upstream", UP_PORT), ("open gate", GATE_PORT), ("token gate", TOK_PORT),
                   ("small-inline gate", SGATE_PORT)):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        busy.append(f"{name} :{port}")
    except OSError:
        pass
if busy:
    sys.__stdout__.write(
        f"[harness] FATAL: {', '.join(busy)} already listening - a stale harness gate would answer "
        f"with the WRONG token and fail the token cases for the wrong reason. Kill it first "
        f"(Windows: Get-NetTCPConnection -State Listen -LocalPort {UP_PORT},{GATE_PORT},{TOK_PORT},"
        f"{SGATE_PORT} | Select LocalPort,OwningProcess).\n")
    sys.__stdout__.flush()
    os._exit(2)

s.worker = StubWorker()
gen_thread = s.start_generation_loop()
serve(UP_PORT)

TOKEN = "harness-token-" + uuid.uuid4().hex[:8]
base_env = dict(os.environ, UPSTREAM_PORT=str(UP_PORT), HEALTH_CHECK_PATH="/ping")
base_env.pop("HY3D_TOKEN", None)
gate_log = open(LOG + ".gate.txt", "w", encoding="utf-8")
gate = subprocess.Popen([sys.executable, GATE_PY], env=dict(base_env, PORT=str(GATE_PORT)),
                        stdout=gate_log, stderr=subprocess.STDOUT)
# the token gate also carries a 1 MiB drain cap so case P can prove the over-cap branch
tgate = subprocess.Popen([sys.executable, GATE_PY], env=dict(base_env, PORT=str(TOK_PORT), HY3D_TOKEN=TOKEN,
                                                             GATE_DRAIN_MAX=str(1 << 20)),
                         stdout=gate_log, stderr=subprocess.STDOUT)
# a gate whose inline cap is tiny, so every stub result goes through the latch-for-ranged-download path
sgate = subprocess.Popen([sys.executable, GATE_PY], env=dict(base_env, PORT=str(SGATE_PORT), STATUS_INLINE_MAX="64"),
                         stdout=gate_log, stderr=subprocess.STDOUT)

for name, proc in (("open gate", gate), ("token gate", tgate), ("small-inline gate", sgate)):
    time.sleep(0.2)
    if proc.poll() is not None:
        sys.__stdout__.write(f"[harness] FATAL: the {name} process exited with {proc.returncode} "
                             f"before any case ran - see {LOG}.gate.txt\n")
        sys.__stdout__.flush()
        os._exit(2)

out = open(LOG, "w", encoding="utf-8")
results = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    out.write(line + "\n"); out.flush()
    sys.__stdout__.write(line + "\n"); sys.__stdout__.flush()


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def wait_drain(budget=60):
    t0 = time.time()
    while time.time() - t0 < budget:
        q = req("GET", "/queue")[2]
        if q.get("in_flight") == 0 and q.get("queued") == 0:
            return True
        time.sleep(0.3)
    return False


def send_until_accepted(body, tries=400, port=GATE_PORT, headers=None):
    for _ in range(tries):
        c, h, b = req("POST", "/send", body, port=port, headers=headers)
        if c == 200:
            return b["uid"]
        time.sleep(0.05)
    raise RuntimeError("never accepted")


def wait_terminal(uid, budget=30):
    t0 = time.time()
    while time.time() - t0 < budget:
        c, h, b = req("GET", f"/status/{uid}")
        if b.get("status") in ("completed", "error") or c == 404:
            return c, b
        time.sleep(0.2)
    return c, b


# --- A. health ---
code, body = None, {}
for _ in range(150):
    try:
        code, _, body = req("GET", "/ping")
        if code == 200 and req("GET", "/ping", port=TOK_PORT)[0] == 200 and req("GET", "/ping", port=SGATE_PORT)[0] == 200:
            break
    except Exception:
        pass
    time.sleep(0.2)
check("A1 gate health 200 once upstream listens", code == 200, str(body)[:160])
check("A2 health body carries the api_server queue state (healthy, loop_alive)",
      isinstance(body.get("queue"), dict) and body["queue"].get("healthy") is True
      and body["queue"].get("loop_alive") is True, str(body.get("queue")))

# --- B. validation, before the queue is busy ---
for bad, field in [
    ({**good, "octree_resolution": 9999}, "octree_resolution"),
    ({**good, "octree_resolution": 63}, "octree_resolution"),
    ({**good, "octree_resolution": "abc"}, "octree_resolution"),
    ({**good, "octree_resolution": True}, "octree_resolution"),
    ({**good, "num_inference_steps": 0}, "num_inference_steps"),
    ({**good, "num_inference_steps": 101}, "num_inference_steps"),
    ({**good, "guidance_scale": 99}, "guidance_scale"),
    ({**good, "guidance_scale": "x"}, "guidance_scale"),
    ({**good, "seed": 1.5}, "seed"),
    ({**good, "seed": "x"}, "seed"),
    ({**good, "seed": 2 ** 64}, "seed"),
    ({**good, "mc_algo": "foo"}, "mc_algo"),
    ({**good, "type": "obj"}, "type"),
    ({k: v for k, v in good.items() if k != "image"}, "image"),
    ({**good, "image": ""}, "image"),
    ([1, 2, 3], "body"),
]:
    code, hdr, body = req("POST", "/send", bad)
    check(f"B 400 for bad {field}", code == 400 and body.get("field") == field and field in body.get("error", ""),
          f"-> {code} {body}")
code, hdr, body = req("POST", "/send", b"{not json")
check("B 400 for invalid JSON body", code == 400 and body.get("field") == "body", f"-> {code} {body}")
code, hdr, body = req("POST", "/send", {**good, "octree_resolution": "256", "seed": 7.0, "guidance_scale": "4.5"})
check("B coercions accepted (octree '256', seed 7.0, guidance '4.5')", code == 200 and body.get("uid"), f"-> {code} {body}")
first_uid = body.get("uid")
code, hdr, body = req("POST", "/generate", good)
check("B /generate -> 410", code == 410, f"-> {code} {body}")

# --- C. queue: 5 more rapid sends: with the one above in flight -> 4 queued, then 429 ---
accepted = [first_uid]
codes, last = [], None
for i in range(5):
    code, hdr, body = req("POST", "/send", {**good, "seed": 100 + i})
    codes.append(code)
    if code == 200:
        accepted.append(body["uid"])
    else:
        last = (code, hdr, body)
check("C sends 2..5 accepted (200), 6th refused", codes[:4] == [200] * 4 and codes[4] == 429, f"codes {codes}")
c, h, b = last or (None, {}, {})
ra = h.get("retry-after")
check("C 429 carries Retry-After through the gate", ra is not None and ra.isdigit() and int(ra) >= 5, f"Retry-After={ra}")
check("C 429 JSON body names queue state", b.get("status") == "busy" and b.get("queue_max") == 4 and b.get("queued") == 4
      and "retry_after_s" in b, f"{b}")
st = [req("GET", f"/status/{u}")[2] for u in accepted]
check("C first uid is 'processing'", st[0].get("status") == "processing", f"{st[0]}")
check("C queued uids say 'queued' with positions 1..4",
      [x.get("status") for x in st[1:]] == ["queued"] * 4 and [x.get("position") for x in st[1:]] == [1, 2, 3, 4],
      f"{st[1:]}")
code, hdr, q = req("GET", "/queue")
check("C /queue reports 1 in flight + 4 queued, healthy, with the in-flight age",
      q.get("in_flight") == 1 and q.get("queued") == 4 and q.get("queue_max") == 4 and q.get("healthy") is True
      and q.get("stuck") is False and isinstance(q.get("in_flight_age_s"), (int, float)) and q.get("job_max_s") == 900,
      f"{q}")

# --- D. unknown uid -> 404 through the gate, never cached ---
for path in ["/status/nonexistent", f"/status/{uuid.uuid4()}"]:
    code, hdr, body = req("GET", path)
    check(f"D 404 not_found for {path}", code == 404 and body.get("status") == "not_found", f"-> {code} {body}")
code, hdr, body = req("GET", "/status/../../etc/passwd")
check("D 404 for a traversal path (router or handler)", code == 404, f"-> {code} {body}")
code, hdr, body = req("GET", f"/result/{uuid.uuid4()}")
check("D /result/<unknown> -> 404 via gate", code == 404, f"-> {code} {body}")

# --- E. all five complete; queue drains; crash -> error; 404 stays 404 ---
t0 = time.time()
done = set()
while time.time() - t0 < 60 and len(done) < len(accepted):
    for u in accepted:
        if u in done:
            continue
        code, hdr, body = req("GET", f"/status/{u}")
        if body.get("status") == "completed":
            ok = base64.b64decode(body["model_base64"]).startswith(b"glTF-stub-")
            done.add(u)
            check(f"E uid {u[:8]} completed with bytes", ok)
        elif body.get("status") not in ("queued", "processing"):
            check(f"E uid {u[:8]} unexpected state", False, f"{body}")
            done.add(u)
    time.sleep(0.3)
check("E all 5 accepted jobs completed within 60 s", len(done) == 5 and all(r[1] for r in results[-5:]),
      f"{time.time() - t0:.1f}s")
check("E queue empty after drain", wait_drain(10), f"{req('GET', '/queue')[2]}")
code, hdr, body = req("POST", "/send", {**good, "seed": CRASH_SEED})
check("E send accepted again after drain", code == 200, f"-> {code}")
c, body = wait_terminal(body.get("uid"), 10)
check("E crashed job reports status error with the exception", body.get("status") == "error" and "stub crash" in body.get("error", ""),
      f"{body}")
code, hdr, body = req("GET", f"/status/{first_uid}")
check("E completed uid still completed (file wins)", body.get("status") == "completed")
code, hdr, body = req("GET", "/status/nonexistent")
check("E repeat /status/nonexistent still 404", code == 404 and body.get("status") == "not_found")

# --- F. burst: 40 concurrent sends -> exactly 5 accepted ---
codes, lock = [], threading.Lock()


def burst(i):
    c, h, b = req("POST", "/send", {**good, "seed": 1000 + i})
    with lock:
        codes.append(c)


ths = [threading.Thread(target=burst, args=(i,)) for i in range(40)]
[t.start() for t in ths]
[t.join() for t in ths]
check("F burst of 40 concurrent sends -> exactly 5 accepted, 35 x 429", codes.count(200) == 5 and codes.count(429) == 35,
      f"200s={codes.count(200)} 429s={codes.count(429)} other={[c for c in codes if c not in (200, 429)]}")
q = req("GET", "/queue")[2]
check("F /queue shows 1 in flight + 4 queued after the burst", q.get("in_flight") == 1 and q.get("queued") == 4, f"{q}")
check("F queue drains", wait_drain(), "")

# --- G. a completed job whose file vanished is terminal, never 'processing' ---
uid = send_until_accepted({**good, "seed": 2})
c, b = wait_terminal(uid)
check("G job completed", b.get("status") == "completed")
os.remove(os.path.join(s.SAVE_DIR, f"{uid}.glb"))
c, h, b = req("GET", f"/status/{uid}")
check("G completed uid whose file was deleted -> terminal error", c == 200 and b.get("status") == "error" and "missing" in b.get("error", ""),
      f"-> {c} {b}")

# --- H. the job table is bounded ---
for i in range(600):
    send_until_accepted({**good, "seed": CRASH_SEED})
wait_drain()
with s._JOBS_LOCK:
    n = len(s._JOBS)
check("H _JOBS bounded after 600 terminal jobs", n <= 512 + 5, f"len(_JOBS)={n}")

# --- I. ranged /result ---
uid = send_until_accepted({**good, "seed": 3})
wait_terminal(uid)
c, h, b = req("GET", f"/result/{uid}?offset=-1")
check("I /result offset=-1 -> 416", c == 416, f"-> {c} {b}")
c, h, b = req("GET", f"/result/{uid}?offset=abc")
check("I /result offset=abc -> 400", c == 400, f"-> {c} {b}")
c, h, b = req("GET", f"/result/{uid}?offset=5&length=4")
check("I /result chunk carries size/sha/offset headers and the right bytes",
      c == 200 and h.get("x-result-offset") == "5" and h.get("x-result-size") and h.get("x-result-sha256")
      and b.get("_raw") == b"stub", f"-> {c} off={h.get('x-result-offset')} {b}")

# --- Q. MODEL SELECTION (2026-09-04): one worker, three shape models, chosen per request ---
served = list(s.SERVED)
check("Q the worker serves the single-view dit, both multiview dits and zero123-xl",
      served == ["hunyuan3d-dit-v2-mini-turbo", "hunyuan3d-dit-v2-0", "hunyuan3d-dit-v2-mv",
                 "hunyuan3d-dit-v2-mv-turbo", "zero123-xl"], f"{served}")
check("Q the defaults route a body without 'model': image -> the single dit, views -> the NON-turbo mv, imagine -> zero123",
      s.DEFAULTS == {"image": "hunyuan3d-dit-v2-mini-turbo", "views": "hunyuan3d-dit-v2-mv",
                     "imagine": "zero123-xl"}, f"{s.DEFAULTS}")
c, h, b = req("POST", "/send", {"views": VIEWS, "seed": 11, "octree_resolution": 128})
check("Q a {views} body is accepted and defaults to the multiview model",
      c == 200 and b.get("model") == "hunyuan3d-dit-v2-mv" and b.get("kind") == "mesh", f"-> {c} {b}")
mv_uid = b.get("uid")
c, h, b = req("POST", "/send", {"views": VIEWS, "model": "dit-v2-mv-turbo", "seed": 12})
check("Q model 'dit-v2-mv-turbo' (a short alias) selects the turbo multiview dit",
      c == 200 and b.get("model") == "hunyuan3d-dit-v2-mv-turbo", f"-> {c} {b}")
turbo_uid = b.get("uid")
c, h, b = req("POST", "/send", {**good, "model": "dit-v2-0", "seed": 13})
check("Q an {image} body with model 'dit-v2-0' selects the full single-view dit",
      c == 200 and b.get("model") == "hunyuan3d-dit-v2-0", f"-> {c} {b}")
single_uid = b.get("uid")
c, h, b = req("POST", "/send", {**good, "model": "dit-v2-mv", "seed": 14})
check("Q an {image} body on a multiview model is accepted (the image fills the front slot)",
      c == 200 and b.get("model") == "hunyuan3d-dit-v2-mv", f"-> {c} {b}")
front_uid = b.get("uid")
for body, field, needle in [
    ({"views": VIEWS, "model": "dit-v2-0"}, "model", "no view slots"),
    ({"views": VIEWS, "model": "zero123-xl"}, "model", "POST /imagine"),
    ({**good, "model": "dit-v2-99"}, "model", "unknown"),
    ({**good, "model": 7}, "model", "must be a string"),
    ({**good, "views": VIEWS}, "views", "not both"),
    ({"views": {"top": IMG}}, "views", "unknown view"),
    ({"views": {}}, "views", "non-empty object"),
    ({"views": {"front": ""}}, "views", "non-empty base64"),
    ({"views": [IMG]}, "views", "non-empty object"),
]:
    c, h, b = req("POST", "/send", body)
    check(f"Q 400 for {field}: {needle}",
          c == 400 and b.get("field") == field and needle in b.get("error", ""), f"-> {c} {b}")
for uid_, what in [(mv_uid, "views/mv"), (turbo_uid, "views/mv-turbo"),
                   (single_uid, "image/dit-v2-0"), (front_uid, "image/mv")]:
    c, b = wait_terminal(uid_, 60)
    check(f"Q the {what} job completed", b.get("status") == "completed", f"{b.get('status')} {b.get('error')}")
seen4 = [x for x in s.worker.seen if x[0] == "mesh"][-4:]
check("Q each job reached the worker under the model it named, with the slots it sent "
      "(an {image} body on the mv model arrives as the FRONT slot alone)",
      seen4 == [("mesh", "hunyuan3d-dit-v2-mv", ["front", "left", "right"]),
                ("mesh", "hunyuan3d-dit-v2-mv-turbo", ["front", "left", "right"]),
                ("mesh", "hunyuan3d-dit-v2-0", []),
                ("mesh", "hunyuan3d-dit-v2-mv", ["front"])], f"{seen4}")
wait_drain(60)

# --- R. POST /imagine (Zero123-XL view synthesis) on the SAME queue, clamps and record ---
c, h, b = req("POST", "/imagine", {"image": IMG, "views": ["left", "right", "back"], "size": 256, "steps": 75})
check("R /imagine accepted: uid + kind 'views' + the zero123 model",
      c == 200 and b.get("uid") and b.get("kind") == "views" and b.get("model") == "zero123-xl", f"-> {c} {b}")
im_uid = b.get("uid")
c, h, b = req("GET", f"/status/{im_uid}")
check("R /status of a queued/processing /imagine job says so, tagged kind 'views'",
      c == 200 and b.get("status") in ("queued", "processing") and b.get("kind") == "views", f"-> {c} {b}")
c, b = wait_terminal(im_uid, 40)
vs = b.get("views") or {}
check("R /status -> completed with the three requested views as base64 PNGs",
      b.get("status") == "completed" and sorted(vs) == ["back", "left", "right"]
      and all(base64.b64decode(v).startswith(STUB_PNG) for v in vs.values())
      and b.get("size") == 256 and b.get("steps") == 75 and b.get("model") == "zero123-xl",
      f"views={sorted(vs)} size={b.get('size')} steps={b.get('steps')} status={b.get('status')}")
c, h, b = req("POST", "/imagine", {"image": IMG, "views": ["left"], "size": 128, "steps": 4, "seed": 5})
im2 = b.get("uid")
c, b = wait_terminal(im2, 40)
check("R a single-view /imagine at another size/steps/seed round-trips those fields",
      b.get("status") == "completed" and sorted(b.get("views") or {}) == ["left"]
      and b.get("size") == 128 and b.get("steps") == 4 and b.get("seed") == 5,
      f"status={b.get('status')} size={b.get('size')} steps={b.get('steps')} seed={b.get('seed')}")
for body, field, needle in [
    ({"views": ["left"]}, "image", "required"),
    ({"image": IMG, "views": ["top"]}, "views", "unknown view"),
    ({"image": IMG, "views": []}, "views", "non-empty list"),
    ({"image": IMG, "views": "left"}, "views", "non-empty list"),
    ({"image": IMG, "size": 100}, "size", "multiple of 8"),
    ({"image": IMG, "size": 4096}, "size", "within"),
    ({"image": IMG, "steps": 0}, "steps", "within"),
    ({"image": IMG, "steps": "x"}, "steps", "must be an integer"),
    ({"image": IMG, "elevation": 900}, "elevation", "within"),
    ({"image": IMG, "model": "dit-v2-0"}, "model", "POST /send"),
    ({"image": IMG, "model": "nope"}, "model", "unknown"),
    ([1, 2], "body", "JSON object"),
]:
    c, h, b = req("POST", "/imagine", body)
    check(f"R 400 for {field}: {needle}",
          c == 400 and b.get("field") == field and needle in b.get("error", ""), f"-> {c} {b}")
c, h, b = req("POST", "/imagine", b"{nope")
check("R 400 for invalid JSON on /imagine", c == 400 and b.get("field") == "body", f"-> {c} {b}")
# /imagine shares the ONE single-flight queue with /send: fill it, then /imagine must 429
fill = [req("POST", "/send", {**good, "seed": 200 + i})[0] for i in range(5)]
c, h, b = req("POST", "/imagine", {"image": IMG, "views": ["back"]})
check("R /imagine rides the SAME single-flight queue: 429 + Retry-After once /send filled it",
      fill == [200] * 5 and c == 429 and (h.get("retry-after") or "").isdigit() and b.get("status") == "busy",
      f"fill={fill} -> {c} Retry-After={h.get('retry-after')}")
check("R the /imagine job reached the worker as a views job with its slots",
      ("views", "zero123-xl", ["back", "left", "right"]) in s.worker.seen,
      f"{[x for x in s.worker.seen if x[0] == 'views']}")
wait_drain(60)

# --- S. /ping and /queue NAME every served model (the client picks its model and ladder from it) ---
c, h, b = req("GET", "/ping")
check("S /ping lists every served model and the ones resident right now",
      c == 200 and b.get("status") == "ok" and b.get("models") == served and b.get("loaded") == ["stub-resident"],
      f"models={b.get('models')} loaded={b.get('loaded')}")
check("S /ping keeps 'model' + 'subfolder' naming the served repos/subfolders (the 2026-09-03 client reads exactly these)",
      "tencent/Hunyuan3D-2mv" in (b.get("model") or "") and "hunyuan3d-dit-v2-mv" in (b.get("subfolder") or ""),
      f"model={b.get('model')!r} subfolder={b.get('subfolder')!r}")
_MV_ID_RE = re.compile(r"(^|[-_/])2?mv([-_/]|$)")
check("S both /ping name-fields pass generate3d.py's POSITIVE multiview guard (an 'mv' token, no contradiction)",
      bool(_MV_ID_RE.search((b.get("model") or "").lower())) and bool(_MV_ID_RE.search((b.get("subfolder") or "").lower())),
      f"model={b.get('model')!r} subfolder={b.get('subfolder')!r}")
check("S the subfolder list ends with a NON-turbo dit, so the client's step ladder defaults to quality",
      not (b.get("subfolder") or "").endswith("-turbo"), f"subfolder={b.get('subfolder')!r}")
q = req("GET", "/queue")[2]
check("S /queue carries the same model state beside the health fields",
      q.get("models") == served and q.get("loaded") == ["stub-resident"] and isinstance(q.get("defaults"), dict),
      f"models={q.get('models')} loaded={q.get('loaded')}")

# --- T. residency: pipelines stay loaded, the LRU one is evicted when a load will not fit ---
try:
    import collections
    w = object.__new__(s.ModelWorker)           # no rembg/u2net download in a contract test
    w.device, w.pipelines, w.lock = "cuda", collections.OrderedDict(), threading.RLock()
    w._load = lambda name: f"pipe:{name}"
    # free VRAM RESPONDS to residency (~5.4 GB of weights each), so an eviction really frees room
    # and the loop is measured, not asserted: a fixed number would either never evict or evict
    # everything regardless of the rule.
    card = [24576.0]
    w._free_mb = lambda: (card[0] - 5400.0 * len(w.pipelines), card[0])
    for n in ("hunyuan3d-dit-v2-0", "hunyuan3d-dit-v2-mv", "hunyuan3d-dit-v2-mv-turbo"):
        w.ensure(n)
    check("T the three shape models stay resident TOGETHER on a 24 GB card (~16 GB of weights)",
          w.loaded() == ["hunyuan3d-dit-v2-0", "hunyuan3d-dit-v2-mv", "hunyuan3d-dit-v2-mv-turbo"], f"{w.loaded()}")
    w.ensure("hunyuan3d-dit-v2-0")              # touch: it becomes the most recently used
    check("T ensure() on a resident model returns it without reloading and refreshes its LRU position",
          w.loaded() == ["hunyuan3d-dit-v2-mv", "hunyuan3d-dit-v2-mv-turbo", "hunyuan3d-dit-v2-0"], f"{w.loaded()}")
    w.ensure("zero123-xl")
    check("T zero123 joins them - a fourth model that still fits is NOT an eviction",
          w.loaded() == ["hunyuan3d-dit-v2-mv", "hunyuan3d-dit-v2-mv-turbo", "hunyuan3d-dit-v2-0", "zero123-xl"],
          f"{w.loaded()}")
    w.ensure("hunyuan3d-dit-v2-mini-turbo")     # a fifth: HY3D_MAX_LOADED is 4
    check(f"T a fifth model past HY3D_MAX_LOADED ({s.MAX_LOADED}) evicts the LEAST recently used one only",
          w.loaded() == ["hunyuan3d-dit-v2-mv-turbo", "hunyuan3d-dit-v2-0", "zero123-xl",
                         "hunyuan3d-dit-v2-mini-turbo"], f"{w.loaded()}")
    card[0] = 8192.0                             # an 8 GB Pascal: one 5.4 GB dit + 2 GB margin fills it
    w.pipelines.clear()
    w.ensure("hunyuan3d-dit-v2-0")
    w.ensure("hunyuan3d-dit-v2-mv")
    check("T on an 8 GB card the SAME rule holds one model at a time (VRAM, not the count, evicts)",
          w.loaded() == ["hunyuan3d-dit-v2-mv"], f"{w.loaded()}")
except Exception as e:
    check("T the worker keeps loaded pipelines resident and evicts LRU", False, repr(e))

# --- O. a slow result export (verifier P2): atomic publish, record-first /status, no partial latch ---


def poll_during(uid, seconds, port=GATE_PORT):
    seen, t0 = [], time.time()
    while time.time() - t0 < seconds:
        c, h, b = req("GET", f"/status/{uid}", port=port)
        seen.append(b.get("status") if c == 200 else f"http{c}")
        time.sleep(0.1)
    return seen


uid = send_until_accepted({**good, "seed": SLOW_SEED})
final, partial = s.result_path(uid), s.result_path(uid) + ".partial"
t0 = time.time()
while time.time() - t0 < 10 and not os.path.exists(partial):
    time.sleep(0.02)
mid_partial, mid_final = os.path.exists(partial), os.path.exists(final)
seen = poll_during(uid, SLOW_PAUSE_S - 0.8)
seen_small = [req("GET", f"/status/{uid}", port=SGATE_PORT)[2] for _ in range(3)]
check("O publish_result writes a .partial sibling and NOTHING at the final path while the export runs",
      mid_partial and not mid_final, f"partial={mid_partial} final={mid_final}")
check("O /status says 'processing' for the whole slow export (never 'completed' before the rename)",
      len(seen) >= 5 and set(seen) == {"processing"}, f"{len(seen)} polls: {sorted(set(seen))}")
check("O the low-inline gate hands out no download handle during the export",
      all(x.get("status") == "processing" and "download" not in x for x in seen_small), f"{seen_small}")
c, b = wait_terminal(uid, 15)
raw = base64.b64decode(b.get("model_base64") or "")
check("O completed carries the WHOLE result plus size/sha256 fields that match it",
      b.get("status") == "completed" and raw == SLOW_RESULT and b.get("size") == len(SLOW_RESULT) and b.get("sha256") == SLOW_SHA,
      f"status={b.get('status')} bytes={len(raw)} size={b.get('size')} sha_ok={b.get('sha256') == SLOW_SHA}")
check("O .partial is gone after the rename and the final file is present",
      not os.path.exists(partial) and os.path.exists(final) and os.path.getsize(final) == len(SLOW_RESULT))
c, h, b = req("GET", f"/status/{uid}", port=SGATE_PORT)
check("O the low-inline gate's download handle names the whole result's size + sha256",
      c == 200 and b.get("download") == f"/result/{uid}" and b.get("size") == len(SLOW_RESULT) and b.get("sha256") == SLOW_SHA, f"-> {c} {b}")
c, h, raw = raw_get(f"/result/{uid}?offset=0&length=100000", port=SGATE_PORT)
check("O ranged /result serves exactly the whole bytes with the matching X-Result-Sha256",
      c == 200 and raw == SLOW_RESULT and h.get("x-result-sha256") == SLOW_SHA and h.get("x-result-size") == str(len(SLOW_RESULT)),
      f"-> {c} {len(raw)} B sha_hdr_ok={h.get('x-result-sha256') == SLOW_SHA}")

# O2: the worst writer - a half-written file sitting at the FINAL path - must still read 'processing'
uid2 = send_until_accepted({**good, "seed": SLOW_DIRECT_SEED})
final2 = s.result_path(uid2)
t0 = time.time()
while time.time() - t0 < 10 and not os.path.exists(final2):
    time.sleep(0.02)
time.sleep(0.1)
half = os.path.getsize(final2) if os.path.exists(final2) else -1
seen = poll_during(uid2, SLOW_PAUSE_S - 0.8)
seen_small = [req("GET", f"/status/{uid2}", port=SGATE_PORT)[2] for _ in range(3)]
check("O2 a half-written file at the FINAL path still reads 'processing' - the job record decides, not the file",
      0 < half < len(SLOW_RESULT) and len(seen) >= 5 and set(seen) == {"processing"}, f"file={half} B of {len(SLOW_RESULT)}; {len(seen)} polls: {sorted(set(seen))}")
check("O2 the low-inline gate never latched the half-written bytes",
      all(x.get("status") == "processing" and "download" not in x for x in seen_small), f"{seen_small}")
c, b = wait_terminal(uid2, 15)
raw = base64.b64decode(b.get("model_base64") or "")
check("O2 'completed' arrives only after the writer finished, with the whole bytes",
      b.get("status") == "completed" and raw == SLOW_RESULT and b.get("size") == len(SLOW_RESULT), f"status={b.get('status')} bytes={len(raw)}")
c, h, b = req("GET", f"/status/{uid2}", port=SGATE_PORT)
c2, h2, raw2 = raw_get(f"/result/{uid2}?offset=0&length=100000", port=SGATE_PORT)
check("O2 the gate's handle + ranged bytes after completion are the whole result",
      c == 200 and b.get("sha256") == SLOW_SHA and b.get("size") == len(SLOW_RESULT) and raw2 == SLOW_RESULT, f"-> {c} {b}")

# O3: the gate's own integrity guard, unit-tested on the module (a lying upstream is a 502, never a latch)
try:
    spec = importlib.util.spec_from_file_location("auth_gate_mod", GATE_PY)
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    mm = gm._integrity_mismatch
    sha_abc = hashlib.sha256(b"abc").hexdigest()
    check("O3 _integrity_mismatch: matching size+sha -> None; declared size 5 for 3 bytes -> reason; wrong sha -> reason; nothing declared -> None",
          mm({"size": 3, "sha256": sha_abc}, b"abc", sha_abc) is None
          and "declared size 5" in (mm({"size": 5}, b"abc", sha_abc) or "")
          and "sha256" in (mm({"sha256": "00" * 32}, b"abc", sha_abc) or "")
          and mm({}, b"abc", sha_abc) is None)
except Exception as e:
    check("O3 the gate has an _integrity_mismatch guard", False, repr(e))
wait_drain(30)

# --- P. a 401 drains the request body: the keep-alive connection stays usable ---
body = json.dumps(good).encode()
try:
    sock = socket.create_connection(("127.0.0.1", TOK_PORT), timeout=10)
    head1, b1 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                               b"Content-Length: %d\r\n\r\n" % len(body) + body)
    check("P 401 (no token) on a keep-alive connection, without Connection: close",
          head1.startswith(b"HTTP/1.1 401") and b"connection: close" not in head1.lower(), head1.splitlines()[0] if head1 else b"EMPTY")
    head2, b2 = raw_http(sock, b"GET /ping HTTP/1.1\r\nHost: x\r\n\r\n")
    check("P the NEXT request on the SAME connection is parsed cleanly: GET /ping -> 200 (not an HTML 400 from leftover JSON)",
          head2.startswith(b"HTTP/1.1 200"), (head2.splitlines()[0] if head2 else b"EMPTY") + b" " + b2[:60])
    head3, b3 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: x\r\nX-HY3D-Token: %s\r\nContent-Type: application/json\r\n"
                               b"Content-Length: %d\r\n\r\n" % (TOKEN.encode(), len(body)) + body)
    check("P then a valid tokened POST /send on the same connection is accepted (200 + uid)",
          head3.startswith(b"HTTP/1.1 200") and b'"uid"' in b3, (head3.splitlines()[0] if head3 else b"EMPTY") + b" " + b3[:80])
    sock.close()
except Exception as e:
    check("P keep-alive 401 sequence", False, repr(e))
try:
    sock = socket.create_connection(("127.0.0.1", TOK_PORT), timeout=10)
    head4, b4 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                               b"Content-Length: 2000000\r\n\r\n{")
    tail = sock.recv(16)
    sock.close()
    check("P a body over GATE_DRAIN_MAX (1 MiB here) is answered 401 + Connection: close and the server closes the socket",
          head4.startswith(b"HTTP/1.1 401") and b"connection: close" in head4.lower() and tail == b"", f"{head4.splitlines()[0] if head4 else b'EMPTY'} tail={tail!r}")
except Exception as e:
    check("P over-cap body -> 401 + close", False, repr(e))
try:
    sock = socket.create_connection(("127.0.0.1", TOK_PORT), timeout=10)
    head5, b5 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: x\r\nX-HY3D-Token: %s\r\nContent-Length: abc\r\n\r\n" % TOKEN.encode())
    tail = sock.recv(16)
    sock.close()
    check("P an unparseable Content-Length is a 400 + Connection: close (cannot be drained)",
          head5.startswith(b"HTTP/1.1 400") and b"connection: close" in head5.lower() and tail == b"", f"{head5.splitlines()[0] if head5 else b'EMPTY'} tail={tail!r}")
except Exception as e:
    check("P bad Content-Length -> 400 + close", False, repr(e))
wait_drain(30)

# --- J. the token wall ---
c, h, b = req("GET", "/ping", port=TOK_PORT)
check("J token gate: /ping open without token", c == 200, f"-> {c}")
c, h, b = req("POST", "/send", good, port=TOK_PORT)
check("J token gate: /send without token -> 401", c == 401, f"-> {c} {b}")
c, h, b = req("POST", "/send", good, port=TOK_PORT, headers={"X-HY3D-Token": "wrong"})
check("J token gate: wrong token -> 401", c == 401, f"-> {c}")
try:
    sock = socket.create_connection(("127.0.0.1", TOK_PORT), timeout=5)
    sock.sendall("POST /send HTTP/1.1\r\nHost: x\r\nX-HY3D-Token: t\u00f6k\u00e9n\r\nContent-Length: 2\r\n"
                 "Content-Type: application/json\r\n\r\n{}".encode("latin-1"))
    raw = sock.recv(4096).decode(errors="replace")
    sock.close()
    check("J token gate: non-ASCII token -> clean 401", raw.startswith("HTTP/1.1 401"), raw.splitlines()[0] if raw else "EMPTY")
except Exception as e:
    check("J token gate: non-ASCII token -> clean 401", False, repr(e))
c, h, b = req("POST", "/send", {**good, "octree_resolution": 1}, port=TOK_PORT, headers={"X-HY3D-Token": TOKEN})
check("J token gate: correct token passes upstream 400 verbatim", c == 400 and b.get("field") == "octree_resolution", f"-> {c} {b}")
c, h, b = req("POST", "/send", good, port=TOK_PORT, headers={"Authorization": f"Bearer {TOKEN}"})
check("J token gate: Bearer fallback accepted", c == 200, f"-> {c} {b}")
ra = None
for i in range(6):
    c, h, b = req("POST", "/send", {**good, "seed": 50 + i}, port=TOK_PORT, headers={"X-HY3D-Token": TOKEN})
    if c == 429:
        ra = h.get("retry-after")
        break
check("J token gate: 429 + Retry-After through the token gate", c == 429 and ra and ra.isdigit(), f"-> {c} Retry-After={ra}")
wait_drain(120)

# --- K. a SystemExit inside generate() must not kill the loop ---
uid = send_until_accepted({**good, "seed": SYSEXIT_SEED})
c, b = wait_terminal(uid, 10)
check("K SystemExit inside generate -> status error naming it", b.get("status") == "error" and "SystemExit" in b.get("error", ""), f"{b}")
check("K generation loop thread survived", gen_thread.is_alive(), "")
uid = send_until_accepted({**good, "seed": 8})
c, b = wait_terminal(uid, 15)
check("K the next job still completes", b.get("status") == "completed", f"{b.get('status')}")
q = req("GET", "/queue")[2]
check("K /queue still healthy", q.get("healthy") is True and q.get("loop_alive") is True, f"{q}")

# --- N. the watchdog's own liveness: visible in /queue, gated by the health path ---
q = req("GET", "/queue")[2]
check("N /queue reports watchdog_alive:true, a fresh watchdog_age_s (<= 3 ticks of 1 s) and watchdog_tick_s",
      q.get("watchdog_alive") is True and isinstance(q.get("watchdog_age_s"), (int, float)) and q["watchdog_age_s"] <= 3
      and q.get("watchdog_tick_s") == 1, f"{q}")
try:
    s._WATCHDOG_STOP.set()
    s._WATCHDOG_THREAD.join(5)
    time.sleep(3.5)
    q = req("GET", "/queue")[2]
    check("N a stopped watchdog reads watchdog_alive:false with an age past 3 ticks",
          q.get("watchdog_alive") is False and (q.get("watchdog_age_s") or 0) > 3, f"{q}")
    c, h, b = req("GET", "/ping")
    check("N gate health -> 503 naming the watchdog while it is dead", c == 503 and "watchdog" in (b.get("reason") or ""), f"-> {c} {b}")
    s.start_watchdog()
    time.sleep(0.5)
    c, h, b = req("GET", "/ping")
    q = b.get("queue") or {}
    check("N restarting the watchdog restores 200 health with watchdog_alive:true", c == 200 and q.get("watchdog_alive") is True, f"-> {c} {q}")
except Exception as e:
    check("N the api_server exposes a stoppable/restartable watchdog (_WATCHDOG_STOP, start_watchdog)", False, repr(e))

# --- L. a hung generation: watchdog -> honest errors, unhealthy /queue, 503 health, no new work ---
s.JOB_MAX_S = 3
hang_uid = send_until_accepted({**good, "seed": HANG_SEED})
time.sleep(0.5)
behind_uid = send_until_accepted({**good, "seed": 4})
t0 = time.time()
while time.time() - t0 < 15:
    if req("GET", "/queue")[2].get("stuck"):
        break
    time.sleep(0.3)
q = req("GET", "/queue")[2]
log(f"L {time.time() - t0:.1f}s after the budget: /queue={q}")
check("L watchdog declares the worker stuck within the budget + tick",
      q.get("stuck") is True and q.get("healthy") is False and "HY3D_JOB_MAX_S" in (q.get("reason") or ""), f"{q}")
c, h, b = req("GET", f"/status/{hang_uid}")
check("L the hung job reads status error naming the budget", b.get("status") == "error" and "HY3D_JOB_MAX_S" in b.get("error", ""), f"{b}")
c, h, b = req("GET", f"/status/{behind_uid}")
check("L the job queued behind it is failed too, not 'queued' forever", b.get("status") == "error", f"{b}")
c, h, b = req("GET", "/ping")
check("L gate health -> 503 unhealthy with the reason", c == 503 and b.get("status") == "unhealthy" and "HY3D_JOB_MAX_S" in b.get("reason", ""),
      f"-> {c} {b}")
c, h, b = req("POST", "/send", {**good, "seed": 9})
check("L a wedged worker refuses new work (503), never 'queued'", c == 503 and b.get("status") == "unhealthy", f"-> {c} {b}")

# --- M. the same hang in a fresh process with HY3D_STUCK_EXIT=1 -> the PROCESS exits 3 ---
child_env = dict(os.environ, HY3D_API_DIR=API_DIR, HY3D_REPO=REPO, HY3D_GATE=GATE_PY)
child_log = os.path.join(os.path.dirname(LOG), "hang-child.log")
t0 = time.time()
with open(child_log, "w", encoding="utf-8") as cl:
    try:
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--hang-child", "--api-dir", API_DIR,
                             "--repo", REPO, "--gate", GATE_PY, "--log", os.path.join(os.path.dirname(LOG), "hang-child-run.log")],
                            env=child_env, stdout=cl, stderr=subprocess.STDOUT, timeout=90).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
tail = open(child_log, encoding="utf-8", errors="replace").read()
check("M watchdog exits the process with 3 once a job outlives HY3D_JOB_MAX_S (HY3D_STUCK_EXIT=1)",
      rc == 3 and "WATCHDOG" in tail, f"rc={rc} in {time.time() - t0:.1f}s; log tail: {tail[-300:]!r}")

gate.terminate()
tgate.terminate()
sgate.terminate()
gate_log.close()
fails = [n for n, ok in results if not ok]
log(f"SUMMARY: {len(results) - len(fails)}/{len(results)} passed; failures: {fails}")
out.close()
os._exit(1 if fails else 0)
