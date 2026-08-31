#!/bin/bash
# pixal3d/start.sh - container entry: Pixal3D server on loopback + auth_gate on $PORT.
# Either process dying kills the container LOUDLY (same rule as the other lanes).
set -euo pipefail

UPSTREAM_PORT="${UPSTREAM_PORT:-8081}"

mkdir -p "${HF_HOME:-/runpod-volume/hf}" "${TORCH_HOME:-/runpod-volume/torch}"

python /app/pixal3d_server.py --host 127.0.0.1 --port "${UPSTREAM_PORT}" &
API_PID=$!

python /app/auth_gate.py &
GATE_PID=$!

set +e
wait -n "${API_PID}" "${GATE_PID}"
EXIT=$?
kill "${API_PID}" "${GATE_PID}" 2>/dev/null
exit "${EXIT}"
