---
name: forge-generation-docker
description: The deployable FORGE generation container (patched Hunyuan3D-2GP + stdlib auth gate) - build/run commands for WSL2 and the RunPod LOAD_BALANCER endpoint it targets.
type: reference
updated: 2026-09-02
---

# FORGE generation container (pillar-3 Phase B)

The same patched stack `engine/generation/README.md` describes, packaged for any linux/amd64
GPU host. One image serves the whole `POST /send` / `GET /status/{uid}` contract that
`generate3d.py` speaks - point `GEN3D_ENDPOINT` at wherever it runs (a local container, a
RunPod endpoint) and the client is identical.

## Contents

| File | Job |
|---|---|
| `Dockerfile` | python:3.12-slim + torch 2.5.1 cu124 wheels + Hunyuan3D-2GP @ `f2456e0` + the vaulted container requirements under TWO constraint files + `rsv4-stack.patch` (applied after pip so a patch change rebuilds from cache). Weights NOT baked (`HF_HOME`, default `/runpod-volume/hf`). |
| `constraints.txt` | every package of the published image, pinned (`pip freeze` of `cu124-20260902b`, torch flavor left to the index). Regenerate from a built image when a dependency is deliberately moved. |
| `auth_gate.py` | stdlib-only front door on `$PORT`: bearer auth (only when `HY3D_TOKEN` set), honest `$HEALTH_CHECK_PATH` (200 only when the api_server socket accepts - i.e. weights loaded), reverse proxy to the loopback api_server; passes 400/404/429 (+`Retry-After`) through verbatim, caches nothing but decoded results for ranged download. |
| `start.sh` | runs api_server on `127.0.0.1:$UPSTREAM_PORT` + auth_gate on `$PORT`; either process dying kills the container. Refuses to start without `/runpod-volume`; keeps u2net on the volume; sets `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` by itself when the cache already holds the dit + turbo-VAE snapshot (logs which path is missing otherwise). |

## Environment

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | gate listen port (RunPod LB sets this) |
| `PORT_HEALTH` | `= PORT` | health listener port; if different, a health-only listener is added |
| `HEALTH_CHECK_PATH` | `/ping` | unauthenticated health path; 200 = api_server serving, 503 = still loading |
| `HY3D_TOKEN` | *(unset)* | bearer token. **Unset = OPEN** (local test mode). Set at runtime only - never baked, never in the vault (DPAPI store name on the client side: `gen3d_endpoint_token`) |
| `HF_HOME` | `/runpod-volume/hf` | weight cache - point at the mounted volume; first warm run downloads once |
| `MODEL_PATH` | `tencent/Hunyuan3D-2mini` | HF model repo |
| `HY3D_SUBFOLDER` | `hunyuan3d-dit-v2-mini-turbo` | dit subfolder (`hunyuan3d-dit-v2-0` on `tencent/Hunyuan3D-2` for the quality model) |
| `HY3D_DEVICE` | `cuda` | `cpu` for GPU-less contract tests (WSL) |
| `UPSTREAM_PORT` | `8081` | loopback port the api_server binds inside the container |
| `HY3D_QUEUE_MAX` | `4` | jobs the api_server queues behind the ONE in flight (matches the endpoint's REQUEST_COUNT 4); the next `/send` is a 429 |
| `HY3D_JOB_ETA_S` | `30` | seconds per outstanding job used for the 429's `Retry-After` (advice only) |
| `HF_HUB_OFFLINE` | *(auto)* | unset = start.sh decides from the cache (offline when complete); `0`/`1` set by the operator always wins |
| `HY3D_ALLOW_EPHEMERAL_CACHE` | `0` | `1` lets a local run start without `/runpod-volume` mounted |

## The `/send` - `/status` contract (as hardened 2026-09-02)

| Call | Answer |
|---|---|
| `POST /send` valid body | `200 {"uid", "status":"queued", "position"}` - ONE generation runs at a time; up to `HY3D_QUEUE_MAX` wait FIFO |
| `POST /send` when 1 in flight + `HY3D_QUEUE_MAX` queued | `429` + `Retry-After: <s>` + `{"status":"busy","in_flight","queued","queue_max","retry_after_s"}` |
| `POST /send` with `octree_resolution` outside 64..512, `num_inference_steps` outside 1..100, `guidance_scale` outside 0..30, a non-integer `seed`, no `image`, `mc_algo` not mc/dmc, `type` not glb, or a non-object body | `400 {"error":"<field>: <why>", "field":"<field>"}` - unknown fields are ignored, never fatal |
| `GET /status/{uid}` | `200 {"status":"queued","position"}` / `{"status":"processing"}` / `{"status":"error","error"}` / `{"status":"completed","model_base64"}` (or the gate's `download` handle over 16 MiB) |
| `GET /status/{uid}` for a uid this worker never accepted | `404 {"status":"not_found"}` - a replaced worker fails the client in one poll instead of burning its timeout |
| `GET /queue` | `{"in_flight","queued","queue_max"}` |
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
engine\forge-scripts\parts\.venv\Scripts\python.exe engine\forge-scripts\generate3d.py `
  --image front.png --out-dir out --seeds 101 --octree 128 --no-render --no-anatomy
```

## Image tags

CI (`backfireboys26-cell/hy3d-gen`, `.github/workflows/build.yml`) pushes two tags per lane on
every push to `main`: the floating lane tag (`cu124`, `hy3d21`, `trellis2`, `pixal3d`) and an
IMMUTABLE `<lane>-<utc date>-<short sha>` (e.g. `cu124-20260902-abc1234`). The endpoint is
pinned to an immutable tag only; the floating tag is for the bake-off lanes and local pulls.

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
      "HF_HOME": "/runpod-volume/hf"
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
