#!/bin/bash
# run-harness-in-image.sh <image> [out_dir] - the contract harness INSIDE a hy3d-gen image, no GPU.
# The image's own /app/api_server.py (upstream + rsv4-stack.patch, as baked) and /app/auth_gate.py
# are what is tested; only the ModelWorker is stubbed. This tree's tests/ is bind-mounted over
# /app/tests so the harness version is the one you are reading, whatever the image baked.
# Logs land in out_dir (default ./harness-out): harness.log, harness.log.gate.txt, hang-child.log.
set -euo pipefail
IMG="${1:?usage: run-harness-in-image.sh <image> [out_dir]}"
OUT="$(mkdir -p "${2:-./harness-out}" && cd "${2:-./harness-out}" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d /tmp/hy3d-harness-XXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$HERE"/*.py "$STAGE/"
sed -i 's/\r$//' "$STAGE"/*.py
docker run --rm --network=none -v "$STAGE:/app/tests:ro" -v "$OUT:/out" --entrypoint python "$IMG" \
    /app/tests/harness.py --api-dir /app --repo /app --gate /app/auth_gate.py --log /out/harness.log
echo "harness exit $? - logs in $OUT"
