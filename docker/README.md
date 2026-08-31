---
name: forge-generation-docker
description: The deployable FORGE generation container (patched Hunyuan3D-2GP + stdlib auth gate) - build/run commands for WSL2 and the RunPod LOAD_BALANCER endpoint it targets.
type: reference
updated: 2026-08-31
---

# FORGE generation container (pillar-3 Phase B)

The same patched stack `engine/generation/README.md` describes, packaged for any linux/amd64
GPU host. One image serves the whole `POST /send` / `GET /status/{uid}` contract that
`generate3d.py` speaks - point `GEN3D_ENDPOINT` at wherever it runs (a local container, a
RunPod endpoint) and the client is identical.

## Contents

| File | Job |
|---|---|
| `Dockerfile` | python:3.12-slim + torch 2.5.1 cu124 wheels + Hunyuan3D-2GP @ `f2456e0` + `rsv4-stack.patch` + the vaulted container requirements. Weights NOT baked (`HF_HOME`, default `/runpod-volume/hf`). |
| `auth_gate.py` | stdlib-only front door on `$PORT`: bearer auth (only when `HY3D_TOKEN` set), honest `$HEALTH_CHECK_PATH` (200 only when the api_server socket accepts - i.e. weights loaded), reverse proxy to the loopback api_server. |
| `start.sh` | runs api_server on `127.0.0.1:$UPSTREAM_PORT` + auth_gate on `$PORT`; either process dying kills the container. |

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
