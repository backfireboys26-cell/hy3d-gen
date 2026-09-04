#!/bin/bash
# run-real-wsl.sh <image> <name> [overlay=0|1] [extra docker -e args...] - start a hy3d-gen image with
# the REAL model on ONE local GPU under WSL2 Docker, weights from a local ext4 volume that mimics the
# RunPod layout (/runpod-volume/<HF_HOME tail>). Pair with probe_real.py from the Windows side.
#
# Machine knobs (env, rsv4 defaults): HY3D_GPU=1 (CUDA_VISIBLE_DEVICES), HY3D_VOL=/srv (mounted at
# /runpod-volume), HY3D_HF=/runpod-volume/forge-hf (HF_HOME), HY3D_MODEL=tencent/Hunyuan3D-2,
# HY3D_SUB=hunyuan3d-dit-v2-0 (the endpoint's model), UPSTREAM_PORT=18081 (host networking shares
# the Windows port space, :8081 is the local generator), GATE_PORT=8080.
# overlay=1 bind-mounts THIS tree's docker/start.sh + docker/auth_gate.py + docker/hy3d_models.py +
# docker/zero123/ and a patched api_server.py
# (env HY3D_API_SERVER=<path>, default: apply patches/rsv4-stack.patch to a fresh f2456e0 clone of
# HY3D_UPSTREAM_GIT, default /mnt/c/ai3d/Hunyuan3D-2GP) over the image's copies.
# HF_ENDPOINT is black-holed: if the boot were to make any hub call it would fail loudly.
set -euo pipefail
IMG="${1:?usage: run-real-wsl.sh <image> <name> [overlay] [-e K=V ...]}"
NAME="${2:?name}"
OVERLAY="${3:-0}"
shift 3 2>/dev/null || shift $#
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$(cd "$HERE/.." && pwd)"
GPU="${HY3D_GPU:-1}"; VOL="${HY3D_VOL:-/srv}"; HF="${HY3D_HF:-/runpod-volume/forge-hf}"
MODEL="${HY3D_MODEL:-tencent/Hunyuan3D-2}"; SUB="${HY3D_SUB:-hunyuan3d-dit-v2-0}"
UP="${UPSTREAM_PORT:-18081}"; GP="${GATE_PORT:-8080}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
MOUNTS=()
if [[ "$OVERLAY" == "1" ]]; then
    STAGE="${HY3D_STAGE:-/srv/hy3d-stage}"
    mkdir -p "$STAGE"
    cp "$GEN/docker/auth_gate.py" "$GEN/docker/start.sh" "$GEN/docker/hy3d_models.py" "$STAGE/"
    rm -rf "$STAGE/zero123"; cp -r "$GEN/docker/zero123" "$STAGE/zero123"
    if [[ -n "${HY3D_API_SERVER:-}" ]]; then
        cp "$HY3D_API_SERVER" "$STAGE/api_server.py"
    else
        rm -rf "$STAGE/upstream"
        git clone -q "${HY3D_UPSTREAM_GIT:-/mnt/c/ai3d/Hunyuan3D-2GP}" "$STAGE/upstream"
        git -C "$STAGE/upstream" config core.autocrlf false
        git -C "$STAGE/upstream" checkout -q f2456e0
        git -C "$STAGE/upstream" rm -q --cached -r . && git -C "$STAGE/upstream" reset -q --hard
        git -C "$STAGE/upstream" apply "$GEN/patches/rsv4-stack.patch"
        cp "$STAGE/upstream/api_server.py" "$STAGE/api_server.py"
    fi
    sed -i 's/\r$//' "$STAGE"/*.py "$STAGE"/*.sh
    chmod +x "$STAGE/start.sh"
    sha256sum "$STAGE"/api_server.py "$STAGE"/auth_gate.py "$STAGE"/hy3d_models.py \
              "$STAGE"/start.sh "$STAGE"/zero123/nvs.py
    MOUNTS=(-v "$STAGE/api_server.py:/app/api_server.py:ro"
            -v "$STAGE/auth_gate.py:/app/auth_gate.py:ro"
            -v "$STAGE/hy3d_models.py:/app/hy3d_models.py:ro"
            -v "$STAGE/zero123:/app/zero123:ro"
            -v "$STAGE/start.sh:/app/start.sh:ro")
fi
docker run -d --name "$NAME" --network=host --runtime=nvidia --gpus all \
    -e CUDA_VISIBLE_DEVICES="$GPU" \
    -v "$VOL:/runpod-volume" -e HF_HOME="$HF" \
    -e PORT="$GP" -e UPSTREAM_PORT="$UP" -e HEALTH_CHECK_PATH=/ping \
    -e MODEL_PATH="$MODEL" -e HY3D_SUBFOLDER="$SUB" \
    -e HF_ENDPOINT=http://127.0.0.1:9 \
    -e HY3D_QUEUE_MAX=4 -e HY3D_JOB_ETA_S=90 \
    "${MOUNTS[@]}" "$@" "$IMG" >/dev/null
echo "started $NAME from $IMG overlay=$OVERLAY gpu=$GPU extra: $*"
sleep 6
docker ps -a --filter "name=$NAME" --format '{{.ID}} {{.Image}} {{.Status}}'
docker logs "$NAME" 2>&1 | grep -E "^\[start\]|^\[auth_gate\] :|FATAL" | head -8
