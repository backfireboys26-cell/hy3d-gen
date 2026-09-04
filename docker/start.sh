#!/bin/bash
# start.sh - container entry: the patched api_server on loopback + auth_gate on $PORT.
# Either process dying kills the container LOUDLY: a gate without a server (or a server
# without its front door) must never keep looking alive from outside.
set -Eeuo pipefail
# A silent death IS the bug class here (2026-09-02 fix round: an `[[ ... ]] &&` that returned 1
# inside a function killed the script under set -e with ZERO output on every empty volume).
# Whatever dies under errexit from now on says where.
trap 'echo "[start] FATAL: start.sh aborted at line ${LINENO} (exit $?) - see the command above" >&2' ERR

UPSTREAM_PORT="${UPSTREAM_PORT:-8081}"
MODEL_PATH="${MODEL_PATH:-tencent/Hunyuan3D-2mini}"
SUBFOLDER="${HY3D_SUBFOLDER:-hunyuan3d-dit-v2-mini-turbo}"
DEVICE="${HY3D_DEVICE:-cuda}"
export MODEL_PATH HY3D_SUBFOLDER="${SUBFOLDER}"
# the ONE catalog of served models (repo/subfolder/kind/VRAM + the cache check and prefetch);
# api_server.py imports it, this script shells out to its CLI - never a second mapping here
MODELS_PY="${HY3D_MODELS_PY:-/app/hy3d_models.py}"
[[ -f "${MODELS_PY}" ]] || MODELS_PY="$(dirname "$0")/hy3d_models.py"

# The weight cache MUST land on the persistent volume. A missing mount used to be silently
# papered over by mkdir on the ephemeral disk: every cold start re-downloaded, the paid volume
# stayed empty, and nothing said so (audit 2026-09-02 #11). Refuse loudly instead - unless the
# operator opts out for a local run (HY3D_ALLOW_EPHEMERAL_CACHE=1).
HF_ROOT="${HF_HOME:-/runpod-volume/hf}"
if [[ "${HF_ROOT}" == /runpod-volume/* ]] && ! mountpoint -q /runpod-volume 2>/dev/null \
   && [[ "${HY3D_ALLOW_EPHEMERAL_CACHE:-0}" != "1" ]]; then
    echo "FATAL: /runpod-volume is not mounted - weights would land on ephemeral disk (set HY3D_ALLOW_EPHEMERAL_CACHE=1 to override)" >&2
    exit 3
fi
export HF_HOME="${HF_ROOT}"
mkdir -p "${HF_ROOT}"
# rembg's u2net.onnx (~176 MB) is fetched from GitHub on first use; keep it on the volume too,
# so a cold start neither re-downloads it nor dies on a GitHub hiccup (audit #10).
export U2NET_HOME="${U2NET_HOME:-${HF_ROOT}/u2net}"
mkdir -p "${U2NET_HOME}"

# Which models this worker serves (the single-image default + HY3D_MODELS). An unknown name is
# fatal HERE, before any weight is touched, naming it - never a worker that boots serving fewer
# models than the operator asked for.
if ! SERVED_JSON="$(python "${MODELS_PY}" list 2>&1)"; then
    echo "FATAL: ${SERVED_JSON}" >&2
    exit 3
fi
echo "[start] serving: ${SERVED_JSON}"

# HF offline mode, self-proving (audit 2026-09-02, endpoint hygiene "HF_HUB_OFFLINE=1 only if
# the volume is proven complete"): go offline ONLY when every hub file the loaders will open is
# already in the cache - for EVERY served model (each dit subfolder's config.yaml +
# model.fp16.safetensors, the turbo VAE enable_flashvdm swaps in, and zero123-xl's components).
# hy3d_models.py owns that file list; this script only reads its verdict. Then huggingface_hub /
# transformers never touch huggingface.co at boot. Anything missing -> PREFETCH it (so a fresh
# volume self-populates BEFORE the first job, instead of a multi-GB download landing inside one
# job's HY3D_JOB_MAX_S budget), then re-check; whatever is still missing is named and the worker
# boots ONLINE so the lazy loaders can still fetch it. HY3D_PREFETCH=0 skips the prefetch (the
# old lazy behaviour); an operator-set HF_HUB_OFFLINE (0 or 1) always wins over all of it.
cache_check() { python "${MODELS_PY}" check 2>&1 || true; }
if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
    echo "[start] HF_HUB_OFFLINE=${HF_HUB_OFFLINE} set by operator - honoring it"
else
    STATUS="$(cache_check)"
    if [[ "${STATUS}" != COMPLETE* && "${HY3D_PREFETCH:-1}" == "1" ]]; then
        echo "[start] HF cache incomplete, prefetching before serving; missing: ${STATUS#MISSING }"
        python "${MODELS_PY}" prefetch || echo "[start] prefetch exited non-zero - continuing with what is cached" >&2
        STATUS="$(cache_check)"
    fi
    if [[ "${STATUS}" == COMPLETE* ]]; then
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        echo "[start] HF cache complete under ${HF_ROOT} (${STATUS#COMPLETE }) -> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    else
        echo "[start] HF cache incomplete, staying ONLINE; missing: ${STATUS#MISSING }"
    fi
fi
if [[ -s "${U2NET_HOME}/u2net.onnx" ]]; then
    echo "[start] u2net.onnx present in ${U2NET_HOME} (no GitHub fetch)"
else
    echo "[start] u2net.onnx absent from ${U2NET_HOME} - rembg will fetch it from GitHub on first use"
fi

python /app/api_server.py --host 127.0.0.1 --port "${UPSTREAM_PORT}" \
    --model_path "${MODEL_PATH}" --subfolder "${SUBFOLDER}" --device "${DEVICE}" &
API_PID=$!

python /app/auth_gate.py &
GATE_PID=$!
echo "[start] api_server pid ${API_PID} on 127.0.0.1:${UPSTREAM_PORT}, auth_gate pid ${GATE_PID}"

set +e
trap - ERR
wait -n "${API_PID}" "${GATE_PID}"
EXIT=$?
if kill -0 "${API_PID}" 2>/dev/null; then WHO="auth_gate"; else WHO="api_server"; fi
echo "[start] ${WHO} exited with ${EXIT} - taking the container down" >&2
kill "${API_PID}" "${GATE_PID}" 2>/dev/null
exit "${EXIT}"
