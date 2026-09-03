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

# HF offline mode, self-proving (audit 2026-09-02, endpoint hygiene "HF_HUB_OFFLINE=1 only if
# the volume is proven complete"): go offline ONLY when every hub file the loader will open is
# already in the cache - the dit subfolder's config.yaml + model.fp16.safetensors and the turbo
# VAE the pipeline swaps in (enable_flashvdm), both under the repo's refs/main snapshot. Then
# huggingface_hub / transformers never touch huggingface.co at boot: no per-file HEAD calls, no
# dependence on the hub being up. Anything missing -> stay online and SAY which path, so a fresh
# volume still self-populates (proven by ../tests/test-start-sh.sh case B: empty cache ->
# 'incomplete' line AND the api_server boots). An operator-set HF_HUB_OFFLINE (0 or 1) always
# wins. The image encoder is built from the config inline in config.yaml (no hub repo), rembg's
# u2net comes from GitHub via U2NET_HOME (not an HF call) - so these two paths are the whole
# hub surface.
case "${MODEL_PATH##*/}" in
    Hunyuan3D-2|Hunyuan3D-2mv) VAE_REPO="tencent/Hunyuan3D-2";     VAE_SUBFOLDER="hunyuan3d-vae-v2-0-turbo" ;;
    Hunyuan3D-2mini)           VAE_REPO="tencent/Hunyuan3D-2mini"; VAE_SUBFOLDER="hunyuan3d-vae-v2-mini-turbo" ;;
    *)                         VAE_REPO="";                         VAE_SUBFOLDER="" ;;
esac
hf_snapshot_dir() {  # <repo_id> -> the refs/main snapshot dir, or nothing; ALWAYS status 0
    # (a non-zero return propagates through snap="$(...)" and, under errexit, kills the script
    # before the 'incomplete' branch is reached - the 2026-09-02 empty-volume regression)
    local repo_dir="${HF_ROOT}/hub/models--${1//\//--}" rev
    rev="$(cat "${repo_dir}/refs/main" 2>/dev/null || true)"
    if [[ -n "${rev}" && -d "${repo_dir}/snapshots/${rev}" ]]; then
        echo "${repo_dir}/snapshots/${rev}"
    fi
    return 0
}
if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
    echo "[start] HF_HUB_OFFLINE=${HF_HUB_OFFLINE} set by operator - honoring it"
else
    missing=""
    snap="$(hf_snapshot_dir "${MODEL_PATH}")"
    for f in config.yaml model.fp16.safetensors; do
        if [[ -z "${snap}" || ! -s "${snap}/${SUBFOLDER}/${f}" ]]; then
            missing="${missing} ${MODEL_PATH}/${SUBFOLDER}/${f}"
        fi
    done
    if [[ -n "${VAE_REPO}" ]]; then
        vsnap="$(hf_snapshot_dir "${VAE_REPO}")"
        for f in config.yaml model.fp16.safetensors; do
            if [[ -z "${vsnap}" || ! -s "${vsnap}/${VAE_SUBFOLDER}/${f}" ]]; then
                missing="${missing} ${VAE_REPO}/${VAE_SUBFOLDER}/${f}"
            fi
        done
    else
        missing="${missing} (unknown VAE mapping for ${MODEL_PATH})"
    fi
    if [[ -z "${missing}" ]]; then
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        echo "[start] HF cache complete under ${HF_ROOT} (${SUBFOLDER} + ${VAE_SUBFOLDER}) -> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    else
        echo "[start] HF cache incomplete, staying ONLINE; missing:${missing}"
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
