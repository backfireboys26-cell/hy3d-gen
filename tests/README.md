---
name: forge-generation-tests
description: Contract tests for the hy3d-gen container (patched api_server + auth_gate + start.sh) that run WITHOUT a GPU - the harness with a stubbed pipeline, and start.sh's branch matrix against a real image.
type: reference
updated: 2026-09-04
---

# hy3d-gen contract tests (no GPU)

What the `docker/README.md` contract table promises, made executable. Mirrored to
`backfireboys26-cell/hy3d-gen:/tests` and baked into every image at `/app/tests` (Dockerfile), so
anyone holding the image can prove it without this machine.

| File | Proves | Run |
|---|---|---|
| `harness.py` | the api_server + gate contract with the ModelWorker stubbed: 400 per field, 5 accepted + 6th 429 (+`Retry-After`), positions, 404 `not_found`, drain, crash -> `error`, `/generate` 410, 40-way burst, deleted result -> terminal, `_JOBS` bound, ranged `/result`, **O: a slow export is published atomically (`.partial` -> rename), `/status` says `processing` until the rename even with a half-written file at the final path, a low-inline gate never latches a partial and its handle/ranged bytes are the whole result, the gate's integrity guard**, **P: a 401 drains the body so the next request on the same keep-alive connection is parsed cleanly; over-cap / unparseable bodies get `Connection: close`**, the token wall, **a SystemExit inside generate() keeps the loop alive**, **N: `/queue` carries `watchdog_alive`/`watchdog_age_s`/`watchdog_tick_s`; a stopped watchdog -> 503 health, a restarted one -> 200**, and - 2026-09-04 - **Q: model selection ({views} defaults to the mv dit, `model` aliases select, a views body on a single-image model / an unknown model / both `image` and `views` / a bad view slot are 400s naming the field, and each job reaches the worker under the model and slots it named), R: `POST /imagine` (uid + `kind:"views"`, three base64 PNGs back, 12 clamp 400s, and a 429 proving it rides the SAME single-flight queue), S: `/ping` + `/queue` list every served model and the `model`/`subfolder` strings still pass generate3d.py's POSITIVE mv guard and default its ladder to quality, T: residency - four models resident on a 24 GB card, LRU order refreshed by use, a fifth evicting only the least recently used, and one-at-a-time on an 8 GB card**, **a hung generation -> watchdog: every outstanding job `error` naming `HY3D_JOB_MAX_S`, `/queue` `stuck:true healthy:false`, `/ping` 503, `/send` 503**, and **`HY3D_STUCK_EXIT=1` exits the process with 3** (spawned child) | `run-harness-local.ps1` (Windows, C:\ai3d\venv) or `run-harness-in-image.sh <image>` (WSL/Linux docker) |
| `test-start-sh.sh <image> [overlay]` | start.sh's 18 branches against a real image with `stub_api_server.py` in place of the ML server: unmounted volume -> exit 3; **EMPTY cache -> 'incomplete' line AND the server boots** (the 2026-09-02 blocker); partial cache; complete -> offline env reaches the server; operator override; unknown model; server death -> same exit code; gate death; `healthy:false` -> 503; 204-while-loading; dangling refs/main; **an unknown `HY3D_MODELS` name -> exit 3 naming it; the default served set on a single-model cache -> the missing list names the mv weights AND the zero123 components; `HY3D_PREFETCH=1` -> the prefetch is attempted and named per file, then the worker boots ONLINE when it could not fetch them** | WSL: `bash tests/test-start-sh.sh ghcr.io/backfireboys26-cell/hy3d-gen:<tag>` (`1` as 2nd arg overlays this tree's start.sh/auth_gate.py) |
| `stub_api_server.py` | stands in for `/app/api_server.py` in `test-start-sh.sh`: same argv, prints the env start.sh decided, serves `/queue` (`STUB_UNHEALTHY=1` -> healthy:false, `STUB_LOAD_S` delays listening, `STUB_DIE=<code>` exits) | used by the script |
| `ping.py <code> [url]` | one GET of the health path from inside the image (no curl there); exit 0 iff the code matches | used by the script |

Every case prints `[PASS]`/`[FAIL]` and the scripts exit 1 on any failure. Expected: harness
SUMMARY `N/N passed` (138/138 on 2026-09-04), start.sh script `0 failure(s) across 18 cases`.

Not covered here (needs the real model on a GPU; see `projects/2026-08-31-gen-hosting-proof.md`):
the 2-step "no surface extracted" RuntimeError text, real generation timings, VRAM placement.
`probe_real.py --mode full` covers those against a running container; its round-3 additions
(`--token`, a keep-alive 401 -> valid request sequence, watchdog fields, and `--big-octree N`:
a rapid `/status` poll through a multi-second export whose first `completed` must carry a
GLB whose header length equals the byte count) are the real-model witnesses of cases O/P/N.
