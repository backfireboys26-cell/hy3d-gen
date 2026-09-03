"""harness.py - contract test of the hardened api_server + auth_gate WITHOUT a model.

Imports the patched api_server.py (upstream f2456e0 + patches/rsv4-stack.patch) with hy3dgen on
the path, swaps the ModelWorker for a stub that sleeps GEN_S seconds and writes a tiny .glb,
serves it with uvicorn on UP_PORT, runs auth_gate.py in front (one OPEN gate, one token gate),
and probes through the gates. Every claim the docker/README.md contract table makes is a case
here, plus the failure shapes a 3 am page is made of: a burst, a deleted result, the prune
bound, ranged results, the token wall, a SystemExit inside generate(), a generation that hangs
(watchdog -> honest error + 503 health + exit 3).

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
import json
import os
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
UP_PORT, GATE_PORT, TOK_PORT = 18099, 18098, 18097
GEN_S = 1.5
CRASH_SEED, HANG_SEED, SYSEXIT_SEED = 666, 555, 777

os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.chdir(os.path.dirname(LOG))
sys.path.insert(0, REPO)
sys.path.insert(0, API_DIR)
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


class StubWorker:
    def generate(self, uid, params):
        seed = params.get("seed")
        if seed == CRASH_SEED:
            raise RuntimeError("stub crash on purpose")
        if seed == SYSEXIT_SEED:
            raise SystemExit("stub SystemExit on purpose")
        if seed == HANG_SEED:
            time.sleep(10 ** 6)
        time.sleep(GEN_S)
        with open(os.path.join(s.SAVE_DIR, f"{uid}.glb"), "wb") as f:
            f.write(b"glTF-stub-" + str(uid).encode())


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


IMG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 64).decode()
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

s.worker = StubWorker()
gen_thread = s.start_generation_loop()
serve(UP_PORT)

TOKEN = "harness-token-" + uuid.uuid4().hex[:8]
base_env = dict(os.environ, UPSTREAM_PORT=str(UP_PORT), HEALTH_CHECK_PATH="/ping")
base_env.pop("HY3D_TOKEN", None)
gate_log = open(LOG + ".gate.txt", "w", encoding="utf-8")
gate = subprocess.Popen([sys.executable, GATE_PY], env=dict(base_env, PORT=str(GATE_PORT)),
                        stdout=gate_log, stderr=subprocess.STDOUT)
tgate = subprocess.Popen([sys.executable, GATE_PY], env=dict(base_env, PORT=str(TOK_PORT), HY3D_TOKEN=TOKEN),
                         stdout=gate_log, stderr=subprocess.STDOUT)

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
        if code == 200 and req("GET", "/ping", port=TOK_PORT)[0] == 200:
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
gate_log.close()
fails = [n for n, ok in results if not ok]
log(f"SUMMARY: {len(results) - len(fails)}/{len(results)} passed; failures: {fails}")
out.close()
os._exit(1 if fails else 0)
