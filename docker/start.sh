#!/bin/bash
# start.sh - container entry: the patched api_server on loopback + auth_gate on $PORT.
# Either process dying kills the container LOUDLY: a gate without a server (or a server
# without its front door) must never keep looking alive from outside.
set -euo pipefail

UPSTREAM_PORT="${UPSTREAM_PORT:-8081}"
MODEL_PATH="${MODEL_PATH:-tencent/Hunyuan3D-2mini}"
SUBFOLDER="${HY3D_SUBFOLDER:-hunyuan3d-dit-v2-mini-turbo}"
DEVICE="${HY3D_DEVICE:-cuda}"

mkdir -p "${HF_HOME:-/runpod-volume/hf}"

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
