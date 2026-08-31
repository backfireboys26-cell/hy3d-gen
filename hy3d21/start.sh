#!/bin/bash
# hy3d21/start.sh - container entry: the shape-only 2.1 server on loopback + auth_gate
# on $PORT. Either process dying kills the container LOUDLY (same rule as the cu124 image).
set -euo pipefail

UPSTREAM_PORT="${UPSTREAM_PORT:-8081}"

mkdir -p "${HF_HOME:-/runpod-volume/hf}" \
         "${HY3DGEN_MODELS:-/runpod-volume/hy3dgen}" \
         "${U2NET_HOME:-/runpod-volume/u2net}"

python /app/hy3d21_server.py --host 127.0.0.1 --port "${UPSTREAM_PORT}" &
API_PID=$!

python /app/auth_gate.py &
GATE_PID=$!

set +e
wait -n "${API_PID}" "${GATE_PID}"
EXIT=$?
kill "${API_PID}" "${GATE_PID}" 2>/dev/null
exit "${EXIT}"
