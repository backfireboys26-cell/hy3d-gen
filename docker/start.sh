#!/bin/bash
# start.sh - container entry: the patched api_server on loopback + auth_gate on $PORT.
# Either process dying kills the container LOUDLY: a gate without a server (or a server
# without its front door) must never keep looking alive from outside.
set -euo pipefail

UPSTREAM_PORT="${UPSTREAM_PORT:-8081}"
MODEL_PATH="${MODEL_PATH:-tencent/Hunyuan3D-2mini}"
SUBFOLDER="${HY3D_SUBFOLDER:-hunyuan3d-dit-v2-mini-turbo}"
DEVICE="${HY3D_DEVICE:-cuda}"

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
mkdir -p "${HF_ROOT}"
# rembg's u2net.onnx (~176 MB) is fetched from GitHub on first use; keep it on the volume too,
# so a cold start neither re-downloads it nor dies on a GitHub hiccup (audit #10).
export U2NET_HOME="${U2NET_HOME:-${HF_ROOT}/u2net}"
mkdir -p "${U2NET_HOME}"

python /app/api_server.py --host 127.0.0.1 --port "${UPSTREAM_PORT}" \
    --model_path "${MODEL_PATH}" --subfolder "${SUBFOLDER}" --device "${DEVICE}" &
API_PID=$!

python /app/auth_gate.py &
GATE_PID=$!

set +e
wait -n "${API_PID}" "${GATE_PID}"
EXIT=$?
kill "${API_PID}" "${GATE_PID}" 2>/dev/null
exit "${EXIT}"
