"""probe_real.py - prove the contract against a RUNNING container with the REAL model (GPU).

Usage: python probe_real.py <base_url> <image.png> <log_path> [--mode full|wedge] [--ready-s N]
                            [--q-steps N] [--wedge-budget-s N]

--mode full (default), in order; every request/response is logged, exit 1 on any FAIL:
  1. waits for /ping 200 (204 while loading is recorded); /queue must say healthy, loop alive
  2. octree 9999 / steps 0 / guidance 99 / seed "x" / no image -> 400 naming the field
  3. /status/nonexistent and /status/<random uuid> -> 404 {"status":"not_found"} through the gate
  4. six overlapping /send: #1 at 5 steps (octree 64), #2-#5 at --q-steps (default 2) ->
     [200 x5, 429] + Retry-After + busy JSON; #1 'processing', #2-#5 'queued' positions 1..4;
     404 while busy; /queue in_flight 1 queued 4 with an in-flight age
  5. drain: #1 completes with a GLB; #2-#5 end in the HONEST error naming steps/octree
     ("no surface extracted at 2 steps / octree 64 ... raise num_inference_steps") - never
     'processing' forever, never the bare AttributeError; /queue empty + healthy after
  6. a normal small generation (octree 128, 5 steps) completes; GLB magic; saved next to the log
--mode wedge: the container was started with a tiny HY3D_JOB_MAX_S (e.g. 20) and a long
  HY3D_STUCK_EXIT_GRACE_S (e.g. 90): one 5-step send -> within budget + 2 ticks /queue says
  stuck:true healthy:false naming HY3D_JOB_MAX_S; the uid's /status -> error naming it; /ping ->
  503 unhealthy; /send -> 503; then the gate disappears (exit 3 -> start.sh took the container
  down) within the grace window + 60 s. The container's exit code is read by the caller
  (docker wait / inspect), not here.
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("image")
ap.add_argument("log")
ap.add_argument("--mode", choices=["full", "wedge"], default="full")
ap.add_argument("--ready-s", type=int, default=1500, help="cold start budget (Pascal-in-WSL: up to ~5 min)")
ap.add_argument("--q-steps", type=int, default=2, help="steps for queue jobs #2-#5 (2 = the honest-error path)")
ap.add_argument("--wedge-budget-s", type=int, default=20, help="the HY3D_JOB_MAX_S the container was started with")
ap.add_argument("--wedge-grace-s", type=int, default=90, help="the HY3D_STUCK_EXIT_GRACE_S the container was started with")
a = ap.parse_args()
BASE, LOG = a.base.rstrip("/"), a.log
out = open(LOG, "w", encoding="utf-8")
results = []


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    out.write(line + "\n"); out.flush(); print(line, flush=True)


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def req(method, path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            code, hdr = resp.status, {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        raw = e.read(); code, hdr = e.code, {k.lower(): v for k, v in e.headers.items()}
    try:
        js = json.loads(raw) if raw else {}
    except ValueError:
        js = {"_raw": raw[:200].decode(errors="replace")}
    shown = {k: (v if k != "model_base64" else f"<{len(v)} b64 chars>") for k, v in js.items()} if isinstance(js, dict) else js
    sb = "" if body is None else json.dumps({k: (v if k != "image" else "<b64>") for k, v in body.items()})
    log(f"{method} {path} {sb} -> {code} {json.dumps(shown)[:400]} ({time.time() - t0:.2f}s)")
    return code, hdr, js


def wait_terminal(uid, budget_s, every=10):
    t0 = time.time()
    body = {}
    while time.time() - t0 < budget_s:
        code, hdr, body = req("GET", f"/status/{uid}")
        if code != 200 or body.get("status") in ("completed", "error"):
            return code, body, time.time() - t0
        time.sleep(every)
    return 200, body, time.time() - t0


def summary():
    fails = [n for n, ok in results if not ok]
    log(f"SUMMARY: {len(results) - len(fails)}/{len(results)} passed; failures: {fails}")
    out.close()
    sys.exit(1 if fails else 0)


IMG = base64.b64encode(open(a.image, "rb").read()).decode()
small = {"image": IMG, "octree_resolution": 64, "num_inference_steps": 5, "guidance_scale": 5.0}

# 1. readiness
t0 = time.time()
code, body, saw204 = None, {}, False
while time.time() - t0 < a.ready_s:
    try:
        code, hdr, body = req("GET", "/ping", timeout=10)
        saw204 = saw204 or code == 204
        if code == 200:
            break
    except Exception as e:
        log(f"GET /ping -> {type(e).__name__}: {e}")
    time.sleep(15)
check("gate /ping 200 (model loaded)", code == 200, f"after {time.time() - t0:.0f}s; 204-while-loading seen: {saw204}; {body}")
if code != 200:
    summary()
q = body.get("queue") if isinstance(body, dict) else None
check("/ping body carries /queue: healthy, loop_alive, job_max_s", isinstance(q, dict) and q.get("healthy") is True
      and q.get("loop_alive") is True and isinstance(q.get("job_max_s"), int), f"{q}")

if a.mode == "wedge":
    code, hdr, body = req("POST", "/send", {**small, "seed": 11})
    check("wedge: send accepted", code == 200 and body.get("uid"), f"{code} {body}")
    uid = body.get("uid")
    t0 = time.time()
    q = {}
    while time.time() - t0 < a.wedge_budget_s + 60:
        code, hdr, q = req("GET", "/queue", timeout=10)
        if q.get("stuck"):
            break
        time.sleep(3)
    check("wedge: /queue stuck:true healthy:false within budget + 60 s, reason names HY3D_JOB_MAX_S",
          q.get("stuck") is True and q.get("healthy") is False and "HY3D_JOB_MAX_S" in (q.get("reason") or ""),
          f"after {time.time() - t0:.0f}s: {q}")
    code, hdr, body = req("GET", f"/status/{uid}")
    check("wedge: the in-flight uid reads status error naming the budget",
          code == 200 and body.get("status") == "error" and "HY3D_JOB_MAX_S" in body.get("error", ""), f"{code} {body}")
    code, hdr, body = req("GET", "/ping")
    check("wedge: /ping -> 503 unhealthy with the reason (load balancer recycles)",
          code == 503 and body.get("status") == "unhealthy" and "HY3D_JOB_MAX_S" in body.get("reason", ""), f"{code} {body}")
    code, hdr, body = req("POST", "/send", {**small, "seed": 12})
    check("wedge: /send while wedged -> 503, never 'queued'", code == 503 and body.get("status") == "unhealthy", f"{code} {body}")
    t0 = time.time()
    gone = False
    while time.time() - t0 < a.wedge_grace_s + 60:
        try:
            req("GET", "/ping", timeout=5)
        except Exception as e:
            gone = True
            log(f"gate unreachable after {time.time() - t0:.0f}s: {type(e).__name__}: {e}")
            break
        time.sleep(5)
    check("wedge: the process exited (gate gone) within the grace window + 60 s", gone, f"{time.time() - t0:.0f}s")
    summary()

# 2. validation (queue idle)
for bad, field in [({**small, "octree_resolution": 9999}, "octree_resolution"),
                   ({**small, "num_inference_steps": 0}, "num_inference_steps"),
                   ({**small, "guidance_scale": 99}, "guidance_scale"),
                   ({**small, "seed": "x"}, "seed"),
                   ({k: v for k, v in small.items() if k != "image"}, "image")]:
    code, hdr, body = req("POST", "/send", bad)
    check(f"400 for bad {field}", code == 400 and body.get("field") == field and field in body.get("error", ""), f"{code} {body}")

# 3. unknown uid -> 404
for path in ["/status/nonexistent", f"/status/{uuid.uuid4()}"]:
    code, hdr, body = req("GET", path)
    check(f"404 not_found for {path}", code == 404 and body.get("status") == "not_found", f"{code} {body}")

# 4. six overlapping sends
accepted, codes, last = [], [], None
t_send = time.time()
for i in range(6):
    steps = 5 if i == 0 else a.q_steps
    code, hdr, body = req("POST", "/send", {**small, "seed": 10 + i, "num_inference_steps": steps})
    codes.append(code)
    if code == 200:
        accepted.append(body["uid"])
    else:
        last = (code, hdr, body)
log(f"6 sends issued in {time.time() - t_send:.2f}s -> {codes}")
check("sends 1-5 accepted, 6th refused with 429", codes == [200] * 5 + [429], f"{codes}")
if last:
    c, h, b = last
    ra = h.get("retry-after")
    check("429 carries Retry-After through the gate", ra is not None and ra.isdigit(), f"Retry-After={ra}")
    check("429 JSON body names the queue state", b.get("status") == "busy" and b.get("queue_max") == 4 and b.get("queued") == 4, f"{b}")
sts = [req("GET", f"/status/{u}")[2] for u in accepted]
check("first send is 'processing'", bool(sts) and sts[0].get("status") == "processing", f"{sts[0] if sts else ''}")
check("sends 2-5 are 'queued' with positions 1..4",
      [x.get("status") for x in sts[1:]] == ["queued"] * 4 and [x.get("position") for x in sts[1:]] == [1, 2, 3, 4], f"{sts[1:]}")
code, hdr, q = req("GET", "/queue")
check("/queue: 1 in flight + 4 queued, healthy, with an in-flight age",
      q.get("in_flight") == 1 and q.get("queued") == 4 and q.get("healthy") is True and q.get("stuck") is False
      and isinstance(q.get("in_flight_age_s"), (int, float)), f"{q}")
code, hdr, body = req("GET", "/status/nonexistent")
check("404 while busy, too", code == 404 and body.get("status") == "not_found")

# 5. drain
t0 = time.time()
done = {}
while time.time() - t0 < 3600 and len(done) < len(accepted):
    for u in accepted:
        if u in done:
            continue
        code, hdr, body = req("GET", f"/status/{u}")
        st = body.get("status")
        if st == "completed":
            n = len(base64.b64decode(body["model_base64"])) if body.get("model_base64") else body.get("size")
            done[u] = ("completed", n)
            log(f"  uid {u[:8]} completed: {n} bytes at +{time.time() - t0:.0f}s")
        elif st == "error" or code != 200:
            done[u] = ("error", body.get("error"))
            log(f"  uid {u[:8]} ERROR at +{time.time() - t0:.0f}s: {body.get('error')}")
    time.sleep(10)
check("all 5 accepted jobs reached a terminal state", len(done) == len(accepted), f"{len(done)}/{len(accepted)} in {time.time() - t0:.0f}s")
first = done.get(accepted[0]) if accepted else None
check("job #1 (5 steps) completed with a GLB", first is not None and first[0] == "completed" and first[1] and first[1] > 1000, f"{first}")
errs = [done.get(u) for u in accepted[1:]]
want = f"no surface extracted at {a.q_steps} steps / octree 64"
check(f"jobs #2-#5 ({a.q_steps} steps) ended in the honest RuntimeError naming steps/octree + the advice",
      all(e and e[0] == "error" and want in (e[1] or "") and "raise num_inference_steps (>= 5 on the non-turbo dit)" in (e[1] or "")
          for e in errs), f"{errs}")
check("no bare AttributeError leaked", all(not (e and "NoneType" in (e[1] or "")) for e in errs))
code, hdr, q = req("GET", "/queue")
check("queue empty + healthy after the drain", q.get("in_flight") == 0 and q.get("queued") == 0 and q.get("healthy") is True, f"{q}")

# 6. a normal small generation
code, hdr, body = req("POST", "/send", {"image": IMG, "octree_resolution": 128, "num_inference_steps": 5, "guidance_scale": 5.0, "seed": 101})
check("normal send accepted", code == 200, f"{code} {body}")
uid = body.get("uid")
code, body, took = wait_terminal(uid, 1800)
glb = base64.b64decode(body["model_base64"]) if body.get("status") == "completed" and body.get("model_base64") else None
check("normal generation (octree 128, 5 steps) completed", glb is not None, f"{len(glb) if glb else body} in {took:.0f}s")
if glb:
    p = LOG.rsplit(".", 1)[0] + "-normal.glb"
    open(p, "wb").write(glb)
    check("GLB magic", glb[:4] == b"glTF", f"{glb[:4]!r} -> {p}")
summary()
