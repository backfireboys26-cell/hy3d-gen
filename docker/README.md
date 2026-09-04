---
name: forge-generation-docker
description: The deployable FORGE generation container - one worker serving the single-view dit, both multiview dits and Zero123-XL view synthesis behind one queue, plus the stdlib auth gate; build/run commands for WSL2 and the RunPod LOAD_BALANCER endpoint it targets.
type: reference
updated: 2026-09-04
---

# FORGE generation container (pillar-3 Phase B)

The same patched stack `engine/generation/README.md` describes, packaged for any linux/amd64
GPU host. One image serves the whole `POST /send` / `GET /status/{uid}` contract that
`generate3d.py` speaks - point `GEN3D_ENDPOINT` at wherever it runs (a local container, a
RunPod endpoint) and the client is identical.

**One worker, every model (2026-09-04).** The same process serves the single-view dit named by
`MODEL_PATH`/`HY3D_SUBFOLDER`, the full multiview `tencent/Hunyuan3D-2mv` (`hunyuan3d-dit-v2-mv`),
its step-distilled twin (`hunyuan3d-dit-v2-mv-turbo`) and Zero123-XL view synthesis - selected
**per request**, loaded lazily, kept resident, evicted least-recently-used when the card cannot
hold another (a 24 GB 4090 holds all four, ~19 GB; an 8 GB Pascal holds one at a time, and every
eviction is logged with the free/needed numbers). `HY3D_MODELS` picks the set; `hy3d_models.py`
is the ONE catalog of which repo/subfolder/files each model is - `api_server.py` imports it and
`start.sh` shells out to its `check`/`prefetch` CLI, so the offline decision and the model list
can never drift apart.

| Body | Runs |
|---|---|
| `POST /send {"image": <b64>}` | the single-view dit (`HY3D_SUBFOLDER`) |
| `POST /send {"views": {"front": <b64>, "left"?, "back"?, "right"?}}` | `hunyuan3d-dit-v2-mv` |
| either, plus `"model": "dit-v2-0" \| "dit-v2-mv" \| "dit-v2-mv-turbo"` | that model (short aliases and full names both work) |
| `POST /send {"image"}` with an mv model | that model, the image filling the FRONT slot |
| `POST /imagine {"image": <b64>, "views": ["left","right","back"], "size": 256, "steps": 75}` | Zero123-XL; `/status/{uid}` -> `{"status":"completed","kind":"views","views":{"left": <b64 png>, ...}}` |

Every one of them rides the SAME single-flight queue, clamps, watchdog, atomic publish and
record-gated `/status` as before - a `/imagine` call while `/send` has filled the queue is the
same `429` + `Retry-After`.

## Contents

| File | Job |
|---|---|
| `Dockerfile` | python:3.12-slim + torch 2.5.1 cu124 wheels + Hunyuan3D-2GP @ `f2456e0` + the vaulted container requirements under TWO constraint files + `rsv4-stack.patch` (applied after pip so a patch change rebuilds from cache). Weights NOT baked (`HF_HOME`, default `/runpod-volume/hf`). |
| `constraints.txt` | every package of the published image, pinned (`pip freeze` of `cu124-20260902b`, torch flavor left to the index). Regenerate from a built image when a dependency is deliberately moved. |
| `hy3d_models.py` | the ONE model catalog: name -> repo/subfolder/kind/VRAM, the files each model opens, the served set from `MODEL_PATH` + `HY3D_MODELS`, the `/ping` label strings, and the `list` / `check` / `prefetch` CLI `start.sh` runs (stdlib-only at import, so it answers before the ML stack is touched). |
| `zero123/` | the `POST /imagine` loader: `nvs.py` (manual component assembly from `kxic/zero123-xl`, fp16, attention slicing, no xformers) plus the diffusers-0.32.2 community `pipeline_zero1to3.py` and the torch-only `kornia.py` shim it needs beside it - `kornia` is deliberately not a pin. |
| `auth_gate.py` | stdlib-only front door on `$PORT`: bearer auth (only when `HY3D_TOKEN` set), honest `$HEALTH_CHECK_PATH` (204 while the api_server socket is not yet accepting = weights loading; 200 once it serves AND its `/queue` says `healthy:true`, with `models` = every served model and `loaded` = the ones resident right now, copied from `/queue`; **503 `{"status":"unhealthy","reason"}` when `/queue` says `healthy:false`** - a job past `HY3D_JOB_MAX_S` or a dead loop - **or when the api_server's watchdog is dead / has not ticked for 3 ticks** - so the load balancer recycles a wedged worker), reverse proxy to the loopback api_server; passes 400/404/429 (+`Retry-After`) through verbatim, caches nothing but decoded results for ranged download - and latches one only when the api_server's declared `size`/`sha256` match the bytes (a mismatch is a 502). Every early answer (401, health, bad `Content-Length`) drains the request body first so a keep-alive connection stays in sync (round 3). |
| `start.sh` | runs api_server on `127.0.0.1:$UPSTREAM_PORT` + auth_gate on `$PORT`; either process dying kills the container, loudly (an ERR trap names the line; nothing exits silently). Refuses to start without `/runpod-volume`; keeps u2net on the volume; sets `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` by itself when the cache already holds the dit + turbo-VAE snapshot, and on an EMPTY or partial cache logs `HF cache incomplete, staying ONLINE; missing: <paths>` and boots online so the volume self-populates (`../tests/test-start-sh.sh` case B - the 2026-09-02 round-1 copy died here with exit 1 and no output). |
| `../tests/` | the no-GPU contract tests (`harness.py` with a stubbed pipeline, `test-start-sh.sh` against a real image); baked into the image at `/app/tests`. See `../tests/README.md`. |

## Environment

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | gate listen port (RunPod LB sets this) |
| `PORT_HEALTH` | `= PORT` | health listener port; if different, a health-only listener is added |
| `HEALTH_CHECK_PATH` | `/ping` | unauthenticated health path; 204 = still loading, 200 = serving and healthy, 503 = wedged (recycle) |
| `HY3D_TOKEN` | *(unset)* | bearer token. **Unset = OPEN** (local test mode). Set at runtime only - never baked, never in the vault (DPAPI store name on the client side: `gen3d_endpoint_token`) |
| `HF_HOME` | `/runpod-volume/hf` | weight cache - point at the mounted volume; first warm run downloads once |
| `MODEL_PATH` | `tencent/Hunyuan3D-2mini` | HF model repo |
| `HY3D_SUBFOLDER` | `hunyuan3d-dit-v2-mini-turbo` | dit subfolder of the SINGLE-VIEW default (`hunyuan3d-dit-v2-0` on `tencent/Hunyuan3D-2` for the quality model) |
| `HY3D_MODELS` | `dit-v2-0,dit-v2-mv,dit-v2-mv-turbo,zero123-xl` | the models served BESIDE that default, comma-separated (catalog names or short aliases). Empty = the single-view model alone. An unknown name is fatal at boot, naming it |
| `HY3D_PRELOAD` | *(the single-view default)* | models loaded BEFORE the server listens (health stays 204 meanwhile); `none` = everything lazy |
| `HY3D_MAX_LOADED` | `4` | pipelines resident at once before an LRU eviction |
| `HY3D_VRAM_MARGIN_MB` | `2048` | activation headroom required free on top of a model's weights before it loads; short of it, the LRU resident is evicted (logged with the numbers) |
| `HY3D_PREFETCH` | `1` | on an incomplete cache, download the missing weights BEFORE serving (so a multi-GB fetch never lands inside one job's `HY3D_JOB_MAX_S`), then re-check; `0` = the old lazy behaviour |
| `HY3D_ZERO123_DIR` | `<api_server dir>/zero123` | where the `/imagine` loader and its two support files live |
| `HY3D_DEVICE` | `cuda` | `cpu` for GPU-less contract tests (WSL) |
| `UPSTREAM_PORT` | `8081` | loopback port the api_server binds inside the container |
| `HY3D_QUEUE_MAX` | `4` | jobs the api_server queues behind the ONE in flight (matches the endpoint's REQUEST_COUNT 4); the next `/send` is a 429 |
| `HY3D_JOB_ETA_S` | `30` | seconds per outstanding job used for the 429's `Retry-After` (advice only) |
| `HY3D_JOB_MAX_S` | `900` | wall-clock budget for ONE generation (= the endpoint's request timeout). A job still `processing` past it means the worker is wedged: every outstanding job -> `error` naming the budget, `/queue` -> `stuck:true healthy:false`, `/ping` -> 503, `/send` -> 503. `0` disables (never on the endpoint). **Slow local host: raise this AND `HEALTH_UNREADABLE_MAX_S` together** (see below) - on Pascal-in-WSL a dit-v2-0 job at octree 384 takes minutes and a long marching-cubes stall can hold the GIL past the gate's 120 s unreadable window; the round-3 GPU-1 proof ran with `HY3D_JOB_MAX_S=1800 HEALTH_UNREADABLE_MAX_S=600` |
| `HY3D_WATCHDOG_S` | `5` | watchdog tick (seconds); `/queue` reports `watchdog_alive` / `watchdog_age_s` / `watchdog_tick_s` and the gate answers 503 when the age passes 3 ticks |
| `HY3D_STUCK_EXIT` | `1` | once wedged, exit the api_server with 3 after the grace window so start.sh takes the container down and the platform relaunches it; `0` keeps the wedged worker up (unhealthy, refusing work) for inspection |
| `HY3D_STUCK_EXIT_GRACE_S` | `15` | seconds the wedged worker holds the honest state (503s, `error` statuses) before exit 3, so a client mid-poll reads the terminal answer |
| `HEALTH_QUEUE_TIMEOUT_S` / `HEALTH_UNREADABLE_MAX_S` | `5` / `120` | gate: how long one `/queue` read may take, and how long `/queue` may stay unreadable (a GIL stall during marching cubes is not a wedge) before that itself is a 503. **`HEALTH_UNREADABLE_MAX_S` is the pair of `HY3D_JOB_MAX_S`: raise both on a slow local host** (the api_server's job budget says when a job is a hang; the gate's unreadable window says how long the api_server may go quiet inside one) |
| `GATE_DRAIN_MAX` | `64 MiB` | gate: a request body up to this size is read and discarded before an early answer (401, health, bad `Content-Length`) so the keep-alive connection stays usable; a larger or chunked body gets `Connection: close` instead |
| `HF_HUB_OFFLINE` | *(auto)* | unset = start.sh decides from the cache (offline when complete); `0`/`1` set by the operator always wins |
| `HY3D_ALLOW_EPHEMERAL_CACHE` | `0` | `1` lets a local run start without `/runpod-volume` mounted |

## The `/send` - `/status` contract (as hardened 2026-09-02)

| Call | Answer |
|---|---|
| `POST /send` valid body | `200 {"uid", "status":"queued", "position", "kind":"mesh", "model"}` - ONE generation runs at a time; up to `HY3D_QUEUE_MAX` wait FIFO; `model` echoes which model the body selected |
| `POST /imagine` valid body | `200 {"uid", "status":"queued", "position", "kind":"views", "model":"zero123-xl"}` - the SAME queue as `/send` (a `/imagine` behind a full queue is the same 429) |
| `GET /status/{uid}` of an `/imagine` job | `200 {"status":"completed","kind":"views","views":{"left": <b64 png>, ...},"size","steps","guidance_scale","seed","elevation","seconds"}` - published atomically as `<uid>.json`, record-gated exactly like a mesh |
| `POST /send {"views"}` with a single-image `model`, an unknown `model`, a shape model on `/imagine` (or the reverse), both `image` and `views`, an unknown view slot, an empty `views` object | `400 {"error":"<field>: <why>", "field":"model"\|"views"\|...}` naming the field and the route that WOULD serve it |
| `POST /imagine` with `views` outside `left/right/back`, `size` not a multiple of 8 or outside 64..512, `steps` outside 1..200, `elevation` outside -90..90 | `400` naming the field |
| `POST /send` when 1 in flight + `HY3D_QUEUE_MAX` queued | `429` + `Retry-After: <s>` + `{"status":"busy","in_flight","queued","queue_max","retry_after_s"}` |
| `POST /send` with `octree_resolution` outside 64..512, `num_inference_steps` outside 1..100, `guidance_scale` outside 0..30, a non-integer `seed`, no `image`, `mc_algo` not mc/dmc, `type` not glb, or a non-object body | `400 {"error":"<field>: <why>", "field":"<field>"}` - unknown fields are ignored, never fatal |
| `GET /status/{uid}` | `200 {"status":"queued","position"}` / `{"status":"processing"}` / `{"status":"error","error"}` / `{"status":"completed","model_base64","size","sha256"}` (or the gate's `download` handle over 16 MiB). **The job record decides while a job is queued/processing** - the result file is published atomically (`<uid>.glb.partial` -> `os.replace` -> `<uid>.glb`, same filesystem) and `completed` is answered only once the loop recorded it (or the record was pruned and the whole file is the witness); a file at the final path never turns a running job into `completed` (round 3, verifier P2) |
| `GET /status/{uid}` for a uid this worker never accepted | `404 {"status":"not_found"}` - a replaced worker fails the client in one poll instead of burning its timeout |
| `GET /status/{uid}` for a job that extracted no surface (1-2 steps of the non-turbo dit) | `200 {"status":"error","error":"RuntimeError: no surface extracted at N steps / octree M (the pipeline returned no mesh) - raise num_inference_steps (>= 5 on the non-turbo dit) or try another seed"}` |
| `GET /queue` | `{"in_flight","queued","queue_max","loop_alive","in_flight_age_s","job_max_s","stuck","watchdog_alive","watchdog_age_s","watchdog_tick_s","healthy","reason","models","loaded","defaults","model","subfolder"}` - `healthy:false` (+`reason`) once a job outlives `HY3D_JOB_MAX_S` or the loop thread died; the gate turns that into a 503 health, and also 503s when `watchdog_alive` is false or `watchdog_age_s` > 3 x `watchdog_tick_s` |
| any request without a valid token (gate with `HY3D_TOKEN` set) | `401` - the body is drained first, so the next request on the same keep-alive connection is parsed cleanly (a body over `GATE_DRAIN_MAX` gets `Connection: close`) |
| any request while wedged | `/status/<every outstanding uid>` -> `error` naming the budget; `POST /send` -> `503 {"status":"unhealthy","error"}`; `/ping` -> 503; then (default) the process exits 3 after `HY3D_STUCK_EXIT_GRACE_S` and start.sh takes the container down |
| a `SystemExit`/`BaseException` inside `generate()` | recorded as `status:error` naming it; the loop thread survives and the next job runs |
| `POST /generate` (upstream's synchronous path) | `410` - it bypassed the queue |

## Build (WSL2 Ubuntu on rsv4)

```bash
# context = engine/generation, so patches/ and requirements/ COPY in
cd /mnt/c/Users/Nickz/Documents/Brain/engine/generation
docker build -f docker/Dockerfile -t hy3d-gen:cu124 .
# CPU-torch variant of the SAME file (plumbing tests on a GPU-less builder):
docker build -f docker/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t hy3d-gen:cpu .
```

## Run locally (WSL contract test)

```bash
mkdir -p ~/hy3d-hf
docker run --rm --network=host \
  -e HF_HOME=/hf -v ~/hy3d-hf:/hf \
  -e UPSTREAM_PORT=18081 \
  hy3d-gen:cu124            # add -e HY3D_TOKEN=... to exercise the auth path;
                            # -e HY3D_DEVICE=cpu on a GPU-less builder;
                            # --gpus all -e CUDA_VISIBLE_DEVICES=<n> with nvidia-container-toolkit
```

Two WSL gotchas, both measured on rsv4 (2026-08-31):
- **Bridge-network downloads corrupt** (sha256 mismatch on >100 MB wheels; the same URL
  hashes correctly outside the bridge). Build AND run with `--network=host`.
- **Host network shares the Windows port space** (mirrored networking): the local rsv4
  Hunyuan server already owns :8081, so the in-container api_server must move -
  `-e UPSTREAM_PORT=18081`. On RunPod neither applies (no host networking, nothing on 8081).

Then from Windows, drive it through the REAL client - the same call that will hit the rented
host:

```powershell
$env:GEN3D_ENDPOINT = "http://localhost:8080"
# single view (the model named by HY3D_SUBFOLDER)
engine\forge-scripts\parts\.venv\Scripts\python.exe engine\forge-scripts\generate3d.py `
  --image front.png --out-dir out --seeds 101 --octree 128 --no-render --no-anatomy
# multiview: --side routes to the views body, and the client's POSITIVE mv guard passes because
# /ping now NAMES the mv model in both 'model' and 'subfolder'
engine\forge-scripts\parts\.venv\Scripts\python.exe engine\forge-scripts\generate3d.py `
  --image front.png --side side.png --out-dir out-mv --seeds 101 --octree 256 --no-render --no-anatomy
```

Zero123-XL view synthesis has no client flag yet - it is one HTTP call (header names only; the
token header is only needed when `HY3D_TOKEN` is set):

```bash
curl -s -X POST "$GEN3D_ENDPOINT/imagine" -H 'Content-Type: application/json' -H 'X-HY3D-Token: <token>' \
     -d '{"image":"<base64 png>","views":["left","right","back"],"size":256,"steps":75}'   # -> {"uid": ...}
curl -s "$GEN3D_ENDPOINT/status/<uid>" -H 'X-HY3D-Token: <token>'                          # -> {"status":"completed","views":{...}}
```

## Image tags

CI (`backfireboys26-cell/hy3d-gen`, `.github/workflows/build.yml`) pushes two tags per lane on
every push to `main`: the floating lane tag (`cu124`, `hy3d21`, `trellis2`, `pixal3d`) and an
IMMUTABLE `<lane>-<utc date>-<short sha>` (e.g. `cu124-20260902-abc1234`). The endpoint is
pinned to an immutable tag only; the floating tag is for the bake-off lanes and local pulls.
The current pin is quoted in the "Round 2" section of `projects/2026-08-31-gen-hosting-proof.md`
(read it from `get-endpoint`, never from memory). Before re-pinning, run `../tests/` against the
new tag (`run-harness-in-image.sh`, `test-start-sh.sh`) and the real-model probe on GPU 1.

## RunPod deployment (post spend-gate; see `projects/2026-08-31-gen-hosting-proof.md`)

LOAD_BALANCER serverless endpoint, image run byte-for-byte unmodified. Endpoint spec (submit
via the REST v2 create-endpoint call - `docs.runpod.io/api-reference-v2` is authoritative for
exact field names at deploy time):

```jsonc
{
  "name": "hy3d-gen",
  "type": "LOAD_BALANCER",              // routes HTTPS straight to our HTTP server
  "template": {
    "imageName": "<registry>/hy3d-gen:cu124",
    "env": {
      "PORT": "8080",
      "PORT_HEALTH": "8080",
      "HEALTH_CHECK_PATH": "/ping",
      "HY3D_TOKEN": "<value of DPAPI secret gen3d_endpoint_token - runtime env only>",
      "HF_HOME": "/runpod-volume/hf",
      "MODEL_PATH": "tencent/Hunyuan3D-2",
      "HY3D_SUBFOLDER": "hunyuan3d-dit-v2-0",
      "HY3D_MODELS": "dit-v2-0,dit-v2-mv,dit-v2-mv-turbo,zero123-xl"
    }
  },
  "gpuTypeIds": ["ADA_24"],             // 4090 pool, $1.10/hr active
  "networkVolumeId": "<60 GB volume - weights land here on the first warm run>",
  "workersMin": 0,                      // scale-to-zero: $0 idle
  "workersMax": 1,
  "idleTimeout": 5
}
```

Client side: store the token as `gen3d_endpoint_token` via `brain-secrets.ps1`, set
`GEN3D_ENDPOINT=https://<ENDPOINT_ID>.api.runpod.ai`, and every `generate3d.py` call carries
`Authorization: Bearer <token>` automatically.

## Honest scope note

WSL validates the **HTTP contract** (auth gate, health, /send//status shapes, the same
`generate3d.py` client end-to-end) - it does NOT validate GPU quality or speed. The cu124
CUDA path (sm_89 kernels, VRAM behavior, octree-512 headroom) is validated on the rented card
itself; that is the $0.40 smoke test behind the spend gate, not something a GPU-less WSL run
can claim. See "Phase B results" in `projects/2026-08-31-gen-hosting-proof.md` for exactly
what the local run proved.
