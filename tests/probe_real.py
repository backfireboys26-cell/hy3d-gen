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
  7. (round 3, 2026-09-03) with --token: every request carries X-HY3D-Token; a request without
     it is a 401; then over ONE raw keep-alive socket: 401 (no token, JSON body) followed by
     GET /ping -> 200 and a tokened POST /send -> 200 (the gate drained the 401's body).
     /queue carries watchdog_alive:true + watchdog_age_s (<= 3 ticks) + watchdog_tick_s.
  8. (round 3) --big-octree N (default 0 = skip): one job at octree N / 5 steps, polled every
     --big-poll-s seconds (default 0.25; the 2026-09-04 GPU runs used 0.05) from 'processing'
     to the end; the FIRST 'completed' must carry a GLB whose header
     length (bytes 8..12) equals the byte count, whose size/sha256 fields match, and whose
     sha is stable on the next polls - /status never said 'completed' on a partial file.
--mode wedge: the container was started with a tiny HY3D_JOB_MAX_S (e.g. 20) and a long
  HY3D_STUCK_EXIT_GRACE_S (e.g. 90): one 5-step send -> within budget + 2 ticks /queue says
  stuck:true healthy:false naming HY3D_JOB_MAX_S; the uid's /status -> error naming it; /ping ->
  503 unhealthy; /send -> 503; then the gate disappears (exit 3 -> start.sh took the container
  down) within the grace window + 60 s. The container's exit code is read by the caller
  (docker wait / inspect), not here.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import socket
import struct
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
ap.add_argument("--token", default=os.environ.get("HY3D_PROBE_TOKEN") or None,
                help="X-HY3D-Token for a gate started with HY3D_TOKEN (env HY3D_PROBE_TOKEN); never logged")
ap.add_argument("--big-octree", type=int, default=0, help="round-3 slow-export case: octree for one 5-step job (0 = skip)")
ap.add_argument("--big-budget-s", type=int, default=3600, help="wall budget for the --big-octree job")
ap.add_argument("--big-poll-s", type=float, default=0.25, help="/status poll interval for the --big-octree job (s)")
a = ap.parse_args()
BASE, LOG = a.base.rstrip("/"), a.log
TOKEN = a.token
out = open(LOG, "w", encoding="utf-8")
results = []


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    out.write(line + "\n"); out.flush(); print(line, flush=True)


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def req(method, path, body=None, timeout=120, token="default"):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    tok = TOKEN if token == "default" else token
    if tok:
        headers["X-HY3D-Token"] = tok
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
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
    if token != "default":
        path = path + "  [no token]" if not tok else path + "  [token]"
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
check("/queue carries the watchdog's liveness: watchdog_alive:true, watchdog_age_s <= 3 ticks, watchdog_tick_s",
      isinstance(q, dict) and q.get("watchdog_alive") is True and isinstance(q.get("watchdog_tick_s"), (int, float))
      and isinstance(q.get("watchdog_age_s"), (int, float)) and q["watchdog_age_s"] <= 3 * q["watchdog_tick_s"],
      f"alive={q.get('watchdog_alive') if isinstance(q, dict) else None} age={q.get('watchdog_age_s') if isinstance(q, dict) else None} tick={q.get('watchdog_tick_s') if isinstance(q, dict) else None}")

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

# 1b. the token wall on the real gate + the keep-alive 401 sequence (round 3)
if TOKEN:
    code, hdr, body = req("POST", "/send", {**small, "seed": 1}, token=None)
    check("token gate: POST /send without X-HY3D-Token -> 401", code == 401, f"{code} {body}")
    code, hdr, body = req("GET", "/ping", token=None)
    check("token gate: /ping stays open without a token", code == 200, f"{code}")
    m = re.match(r"https?://([^/:]+)(?::(\d+))?", BASE)
    host, port = m.group(1), int(m.group(2) or (443 if BASE.startswith("https") else 80))
    body_b = json.dumps({**small, "seed": 2}).encode()

    def raw_http(sock, request_bytes):
        sock.sendall(request_bytes)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return buf, b""
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        mm = re.search(rb"content-length:\s*(\d+)", head, re.I)
        n = int(mm.group(1)) if mm else 0
        while len(rest) < n:
            chunk = sock.recv(65536)
            if not chunk:
                break
            rest += chunk
        return head, rest[:n]

    try:
        sock = socket.create_connection((host, port), timeout=30)
        h1, b1 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                                % (host.encode(), len(body_b)) + body_b)
        log(f"keep-alive #1 POST /send (no token, {len(body_b)} B body) -> {h1.splitlines()[0] if h1 else b'EMPTY'!r}")
        h2, b2 = raw_http(sock, b"GET /ping HTTP/1.1\r\nHost: %s\r\n\r\n" % host.encode())
        log(f"keep-alive #2 GET /ping on the SAME socket -> {h2.splitlines()[0] if h2 else b'EMPTY'!r} {b2[:80]!r}")
        h3, b3 = raw_http(sock, b"POST /send HTTP/1.1\r\nHost: %s\r\nX-HY3D-Token: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                                % (host.encode(), TOKEN.encode(), len(body_b)) + body_b)
        log(f"keep-alive #3 POST /send (token) on the SAME socket -> {h3.splitlines()[0] if h3 else b'EMPTY'!r} {b3[:120]!r}")
        sock.close()
        check("keep-alive: 401 without a token, then GET /ping on the same socket -> 200 (the 401 drained its body)",
              h1.startswith(b"HTTP/1.1 401") and h2.startswith(b"HTTP/1.1 200"))
        check("keep-alive: then a tokened POST /send on the same socket -> 200 + uid",
              h3.startswith(b"HTTP/1.1 200") and b'"uid"' in b3)
        try:
            ka_uid = json.loads(b3).get("uid")
        except ValueError:
            ka_uid = None
        if ka_uid:
            code, body, took = wait_terminal(ka_uid, 3600)
            check("keep-alive: that job reached a terminal state", body.get("status") in ("completed", "error"), f"{body.get('status')} in {took:.0f}s")
    except Exception as e:
        check("keep-alive 401 sequence", False, repr(e))

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


def glb_whole(data):
    """(ok, detail): a GLB's header carries its total length at bytes 8..12 - a truncated
    file's header claims more bytes than it has."""
    if len(data) < 12 or data[:4] != b"glTF":
        return False, f"not a GLB ({data[:4]!r}, {len(data)} B)"
    total = struct.unpack("<I", data[8:12])[0]
    return total == len(data), f"header length {total} vs {len(data)} bytes"


# 7. the slow-export case (round 3): a big octree so the export takes seconds; /status polled fast
if a.big_octree:
    code, hdr, body = req("POST", "/send", {"image": IMG, "octree_resolution": a.big_octree, "num_inference_steps": 5,
                                           "guidance_scale": 5.0, "seed": 101})
    check(f"big-octree {a.big_octree} send accepted", code == 200, f"{code} {body}")
    big_uid = body.get("uid")
    t0 = time.time()
    states, first_done, polls = [], None, 0
    while time.time() - t0 < a.big_budget_s:
        try:
            r = urllib.request.Request(BASE + f"/status/{big_uid}", headers={"X-HY3D-Token": TOKEN} if TOKEN else {})
            with urllib.request.urlopen(r, timeout=600) as resp:
                raw = resp.read(); code = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read(); code = e.code
        polls += 1
        try:
            js = json.loads(raw)
        except ValueError:
            js = {"_raw": raw[:100]}
        st = js.get("status") if code == 200 else f"http{code}"
        if not states or states[-1][1] != st:
            states.append((round(time.time() - t0, 2), st))
            log(f"  big-octree {big_uid[:8]} state -> {st} at +{time.time() - t0:.2f}s (poll {polls})")
        if st in ("completed", "error") or code != 200:
            first_done = (time.time() - t0, code, js)
            break
        time.sleep(a.big_poll_s)
    check("big-octree job reached a terminal state within the budget", first_done is not None,
          f"{polls} polls at {a.big_poll_s}s in {time.time() - t0:.0f}s; transitions {states}")
    if first_done:
        took, code, js = first_done
        if js.get("status") == "completed":
            if js.get("model_base64"):
                data = base64.b64decode(js["model_base64"])
            else:
                # the gate's download handle (result over its inline cap): fetch in ranges
                size, want = int(js.get("size") or 0), (js.get("sha256") or "").lower()
                buf = bytearray()
                while len(buf) < size:
                    r = urllib.request.Request(BASE + f"{js['download']}?offset={len(buf)}&length={16 * 1024 * 1024}",
                                               headers={"X-HY3D-Token": TOKEN} if TOKEN else {})
                    with urllib.request.urlopen(r, timeout=600) as resp:
                        part = resp.read()
                    if not part:
                        break
                    buf += part
                data = bytes(buf)
                log(f"  big-octree result via ranged /result: {len(data)} B, handle size {size}, sha_ok={hashlib.sha256(data).hexdigest() == want}")
            ok, detail = glb_whole(data)
            sha = hashlib.sha256(data).hexdigest()
            check(f"big-octree FIRST 'completed' (poll {polls}, +{took:.1f}s) is a WHOLE GLB: header length == byte count",
                  ok, f"{detail}; {len(data)} B")
            check("big-octree size/sha256 fields match the served bytes",
                  js.get("size") == len(data) and (js.get("sha256") or "").lower() == sha, f"size={js.get('size')} vs {len(data)}")
            shas = []
            for _ in range(3):
                c2, h2, js2 = req("GET", f"/status/{big_uid}")
                if js2.get("model_base64"):
                    shas.append(hashlib.sha256(base64.b64decode(js2["model_base64"])).hexdigest())
                else:
                    shas.append((js2.get("sha256") or "").lower())
                time.sleep(0.5)
            check("big-octree the sha256 is stable on the next 3 polls (nothing was still being written)", all(x == sha for x in shas), f"{[x[:12] for x in shas]}")
            check("big-octree every poll before the first 'completed' read 'processing' (or 'queued')",
                  all(st in ("queued", "processing") for _, st in states[:-1]) and states[-1][1] == "completed", f"{states}")
            pb = LOG.rsplit(".", 1)[0] + f"-big{a.big_octree}.glb"
            open(pb, "wb").write(data)
            log(f"  big-octree GLB saved -> {pb} ({len(data)} B, sha256 {sha[:16]})")
        else:
            check("big-octree job completed", False, f"{code} {js}")
summary()
