#!/bin/bash
# test-start-sh.sh <image> [overlay] - prove start.sh's branches against a real image, NO GPU.
#
# The ML api_server is replaced by tests/stub_api_server.py (bind-mounted over /app/api_server.py):
# it takes the same argv, prints the env start.sh decided, and serves /queue so auth_gate's health
# path has something honest to read. Everything else is the real image: bash, mountpoint, python,
# auth_gate.py and - unless overlay=1 - the baked start.sh.
#   overlay=1 bind-mounts THIS tree's docker/start.sh + docker/auth_gate.py over the image's (proof
#             of the vault copies before CI has baked them; the 2026-09-02 fix round's use).
# HF_ENDPOINT is black-holed (127.0.0.1:9) so an ONLINE branch can never download anything here.
#
# Cases (each prints PASS/FAIL; exit 1 on any FAIL):
#   A  HF_HOME under /runpod-volume, volume NOT mounted        -> exit 3 + FATAL line
#   B  EMPTY cache                                             -> 'HF cache incomplete ... missing:' names 4 paths,
#                                                                 the api_server STILL boots, /ping -> 200 (the
#                                                                 2026-09-02 BLOCKER: the old start.sh died here,
#                                                                 exit 1, zero output)
#   C  partial cache (configs present, models zero bytes)      -> missing = the two model.fp16.safetensors only
#   D  fully populated cache (presence/size check, not content)-> offline chosen; the api_server sees HF_HUB_OFFLINE=1
#   E  operator HF_HUB_OFFLINE=0                               -> 'honoring it', api_server sees 0
#   F  unknown MODEL_PATH (no VAE mapping)                     -> stays ONLINE naming the unknown mapping
#   G  api_server dies (STUB_DIE=7)                            -> start.sh exits 7 saying api_server exited
#   H  auth_gate dies at import (PORT=notaport)                -> start.sh exits non-zero saying auth_gate exited
#   I  api_server reports healthy:false (STUB_UNHEALTHY=1)     -> /ping 503 {"status":"unhealthy"} (stuck-job path)
#   J  api_server slow to listen (STUB_LOAD_S=6)               -> /ping 204 while loading, 200 after
#   K  refs/main points at a MISSING snapshot                  -> incomplete (not a crash), boots
#   L  HY3D_MODELS names an unknown model                     -> exit 3 naming it (before any weight)
#   M  the DEFAULT served set on a single-model cache         -> missing names the mv weights + zero123 parts
#   N  incomplete cache with HY3D_PREFETCH=1                  -> prefetch attempted and NAMED per file, then
#                                                                boots ONLINE when it could not fetch them
set -uo pipefail
IMG="${1:?usage: test-start-sh.sh <image> [overlay]}"
OVERLAY="${2:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d /tmp/hy3d-startsh-XXXX)"
trap 'rm -rf "$STAGE"' EXIT
# staged, CR-stripped copies: a /mnt/c checkout may carry CRLF, and bash refuses a CRLF script
cp "$HERE/stub_api_server.py" "$HERE/ping.py" "$STAGE/"
MOUNTS=(-v "$STAGE/stub_api_server.py:/app/api_server.py:ro" -v "$STAGE/ping.py:/app/ping.py:ro")
if [[ "$OVERLAY" == "1" ]]; then
    cp "$HERE/../docker/start.sh" "$HERE/../docker/auth_gate.py" "$HERE/../docker/hy3d_models.py" "$STAGE/"
    MOUNTS+=(-v "$STAGE/start.sh:/app/start.sh:ro" -v "$STAGE/auth_gate.py:/app/auth_gate.py:ro"
             -v "$STAGE/hy3d_models.py:/app/hy3d_models.py:ro")
fi
sed -i 's/\r$//' "$STAGE"/*
chmod +x "$STAGE"/*.sh 2>/dev/null || true
echo "== image $IMG overlay=$OVERLAY =="
docker inspect --format 'id={{.Id}} created={{.Created}}' "$IMG" || { echo "FAIL: image not present"; exit 1; }
sha256sum "$STAGE"/* | sed 's#'"$STAGE"'/#staged #'

# HY3D_MODELS= (empty) -> only the single-image model named by MODEL_PATH/HY3D_SUBFOLDER, and
# HY3D_PREFETCH=0 -> no download attempt: cases A-K are about start.sh's BRANCHES, one model's
# cache, exactly as they were before the multi-model build (L/M/N below cover the new behaviour).
BH=(-e HF_ENDPOINT=http://127.0.0.1:9 -e HY3D_ALLOW_EPHEMERAL_CACHE=1 -e HF_HOME=/tmp/hf
    -e MODEL_PATH=tencent/Hunyuan3D-2 -e HY3D_SUBFOLDER=hunyuan3d-dit-v2-0 -e PORT=8080 -e UPSTREAM_PORT=8081
    -e HY3D_MODELS= -e HY3D_PREFETCH=0)
# populate a fake cache inside the container: mk <dit_cfg> <dit_model> <vae_cfg> <vae_model> (each 'x' = 1 byte, '' = 0 bytes, '-' = absent)
MK='mk(){ R=/tmp/hf/hub/models--tencent--Hunyuan3D-2; mkdir -p $R/refs $R/snapshots/abc/hunyuan3d-dit-v2-0 $R/snapshots/abc/hunyuan3d-vae-v2-0-turbo; echo abc > $R/refs/main;
 i=1; for p in hunyuan3d-dit-v2-0/config.yaml hunyuan3d-dit-v2-0/model.fp16.safetensors hunyuan3d-vae-v2-0-turbo/config.yaml hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors; do
   v="${!i}"; case "$v" in -) ;; "") : > $R/snapshots/abc/$p ;; *) echo x > $R/snapshots/abc/$p ;; esac; i=$((i+1)); done; }; set -- "$@"; '
# run start.sh for N seconds, then ping the gate with the expected code; prints start.sh output + PING line + exit
RUN='run_and_ping(){ want=$1; secs=$2; ( timeout "$secs" /app/start.sh; echo "exit=$?" ) > /tmp/o 2>&1 &
  for i in $(seq 1 40); do sleep 0.25; python /app/ping.py "$want" > /tmp/p 2>&1 && break; done; cat /tmp/p; wait; cat /tmp/o; }; '

DEFAULT_MODELS="dit-v2-0,dit-v2-mv,dit-v2-mv-turbo,zero123-xl"   # = hy3d_models.DEFAULT_MODELS
fails=0
check() { if [[ "$2" == "1" ]]; then echo "[PASS] $1"; else echo "[FAIL] $1"; fails=$((fails + 1)); fi; }
runc() { docker run --rm --entrypoint bash "${MOUNTS[@]}" "$@"; }

echo "== A: /runpod-volume not mounted -> exit 3 =="
out="$(runc -e HF_HOME=/runpod-volume/hf "$IMG" -c 'timeout 20 /app/start.sh; echo "exit=$?"' 2>&1)"; echo "$out" | tail -3
check "A exit 3 + FATAL names the missing mount" "$([[ "$out" == *"exit=3"* && "$out" == *"FATAL: /runpod-volume is not mounted"* ]] && echo 1 || echo 0)"

echo "== B: EMPTY cache -> incomplete line, api_server boots, /ping 200 =="
out="$(runc "${BH[@]}" "$IMG" -c "$RUN"'run_and_ping 200 12' 2>&1)"; echo "$out"
check "B logs 'HF cache incomplete, staying ONLINE'" "$([[ "$out" == *"[start] HF cache incomplete, staying ONLINE; missing:"* ]] && echo 1 || echo 0)"
check "B names all 4 missing paths" "$([[ "$out" == *"hunyuan3d-dit-v2-0/config.yaml"* && "$out" == *"hunyuan3d-dit-v2-0/model.fp16.safetensors"* && "$out" == *"hunyuan3d-vae-v2-0-turbo/config.yaml"* && "$out" == *"hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors"* ]] && echo 1 || echo 0)"
check "B api_server launched ONLINE (HF_HUB_OFFLINE=None) and the gate answered /ping 200" "$([[ "$out" == *"[stub-api] args="*"HF_HUB_OFFLINE=None"* && "$out" == *"PING 200"* ]] && echo 1 || echo 0)"
check "B start.sh did not die on its own (exit=124 = our timeout, not 1)" "$([[ "$out" == *"exit=124"* ]] && echo 1 || echo 0)"

echo "== C: partial cache (configs present, models ZERO bytes) -> missing = the two models only =="
out="$(runc "${BH[@]}" "$IMG" -c "$MK"'mk x "" x ""; timeout 6 /app/start.sh 2>&1 | grep "^\[start\] HF"' 2>&1)"; echo "$out"
check "C missing lists exactly the two model files" "$([[ "$out" == *"incomplete"* && "$out" == *"dit-v2-0/model.fp16.safetensors"* && "$out" == *"vae-v2-0-turbo/model.fp16.safetensors"* && "$out" != *"config.yaml"* ]] && echo 1 || echo 0)"

echo "== D: fully populated cache -> offline chosen, api_server sees HF_HUB_OFFLINE=1 =="
out="$(runc "${BH[@]}" "$IMG" -c "$MK"'mk x x x x; '"$RUN"'run_and_ping 200 8' 2>&1)"; echo "$out"
check "D complete line + HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 reached the api_server" "$([[ "$out" == *"[start] HF cache complete under /tmp/hf (hunyuan3d-dit-v2-0 + hunyuan3d-vae-v2-0-turbo) -> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"* && "$out" == *"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 U2NET_HOME=/tmp/hf/u2net"* && "$out" == *"PING 200"* ]] && echo 1 || echo 0)"

echo "== E: operator HF_HUB_OFFLINE=0 -> honored =="
out="$(runc "${BH[@]}" -e HF_HUB_OFFLINE=0 "$IMG" -c "$MK"'mk x x x x; timeout 6 /app/start.sh 2>&1 | grep -E "^\[start\] HF|\[stub-api\] args"' 2>&1)"; echo "$out"
check "E 'set by operator - honoring it' and the api_server sees HF_HUB_OFFLINE=0" "$([[ "$out" == *"HF_HUB_OFFLINE=0 set by operator - honoring it"* && "$out" == *"HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=None"* ]] && echo 1 || echo 0)"

echo "== F: unknown MODEL_PATH -> ONLINE naming the unknown VAE mapping =="
out="$(runc "${BH[@]}" -e MODEL_PATH=tencent/Hunyuan3D-2.1 -e HY3D_SUBFOLDER=hunyuan3d-dit-v2-1 "$IMG" -c 'timeout 6 /app/start.sh 2>&1 | grep "^\[start\] HF"' 2>&1)"; echo "$out"
check "F incomplete + '(unknown VAE mapping for tencent/Hunyuan3D-2.1)'" "$([[ "$out" == *"incomplete"* && "$out" == *"(unknown VAE mapping for tencent/Hunyuan3D-2.1)"* ]] && echo 1 || echo 0)"

echo "== G: api_server dies with 7 -> container exits 7, loudly =="
out="$(runc "${BH[@]}" -e STUB_DIE=7 "$IMG" -c 'timeout 20 /app/start.sh; echo "exit=$?"' 2>&1)"; echo "$out" | tail -4
check "G exit=7 and '[start] api_server exited with 7'" "$([[ "$out" == *"exit=7"* && "$out" == *"[start] api_server exited with 7 - taking the container down"* ]] && echo 1 || echo 0)"

echo "== H: auth_gate dies at import (PORT=notaport) -> container exits non-zero, loudly =="
out="$(runc "${BH[@]}" -e PORT=notaport "$IMG" -c 'timeout 20 /app/start.sh; echo "exit=$?"' 2>&1)"; echo "$out" | grep -E "ValueError|\[start\] auth_gate|exit=" | tail -3
check "H exit=1 naming auth_gate + the ValueError" "$([[ "$out" == *"exit=1"* && "$out" == *"[start] auth_gate exited with 1"* && "$out" == *"invalid literal for int()"* ]] && echo 1 || echo 0)"

echo "== I: api_server reports healthy:false -> /ping 503 unhealthy =="
out="$(runc "${BH[@]}" -e STUB_UNHEALTHY=1 "$IMG" -c "$MK"'mk x x x x; '"$RUN"'run_and_ping 503 8' 2>&1)"; echo "$out" | grep -E "PING|health ->" | head -3
check "I PING 503 with status unhealthy + the upstream reason" "$([[ "$out" == *'PING 503 {"status": "unhealthy"'* && "$out" == *"stub says wedged"* ]] && echo 1 || echo 0)"

echo "== J: api_server slow to listen -> 204 while loading, then 200 =="
out="$(runc "${BH[@]}" -e STUB_LOAD_S=6 "$IMG" -c "$MK"'mk x x x x; ( timeout 14 /app/start.sh; echo "exit=$?" ) > /tmp/o 2>&1 & sleep 2; python /app/ping.py 204; for i in $(seq 1 40); do sleep 0.25; python /app/ping.py 200 > /tmp/p 2>&1 && break; done; cat /tmp/p; wait; grep -c "GET /ping" /tmp/o' 2>&1)"; echo "$out"
check "J PING 204 first, PING 200 once listening" "$([[ "$out" == *"PING 204"* && "$out" == *"PING 200"* ]] && echo 1 || echo 0)"

echo "== K: refs/main present but snapshot missing -> incomplete, boots =="
out="$(runc "${BH[@]}" "$IMG" -c 'R=/tmp/hf/hub/models--tencent--Hunyuan3D-2; mkdir -p $R/refs; echo abc > $R/refs/main; '"$RUN"'run_and_ping 200 8' 2>&1)"; echo "$out" | grep -E "^\[start\] HF|PING|exit=" | head -4
check "K incomplete line + PING 200 + exit=124 (no crash)" "$([[ "$out" == *"HF cache incomplete"* && "$out" == *"PING 200"* && "$out" == *"exit=124"* ]] && echo 1 || echo 0)"

echo "== L: HY3D_MODELS names a model the catalog does not know -> exit 3 naming it, before any weight =="
out="$(runc "${BH[@]}" -e HY3D_MODELS=dit-v2-9000 "$IMG" -c 'timeout 20 /app/start.sh; echo "exit=$?"' 2>&1)"; echo "$out" | tail -3
check "L exit 3 + FATAL names the unknown model and lists the known ones" "$([[ "$out" == *"exit=3"* && "$out" == *"unknown model(s) ['dit-v2-9000']"* && "$out" == *"hunyuan3d-dit-v2-mv"* ]] && echo 1 || echo 0)"

echo "== M: the DEFAULT served set -> the missing list covers the mv repo AND zero123, and the serving line names all of them =="
out="$(runc "${BH[@]}" -e HY3D_MODELS="$DEFAULT_MODELS" "$IMG" -c "$MK"'mk x x x x; timeout 8 /app/start.sh 2>&1 | grep -E "^\[start\] (serving|HF)"' 2>&1)"; echo "$out" | cut -c1-240
check "M the serving line names all five models" "$([[ "$out" == *'"hunyuan3d-dit-v2-0"'* && "$out" == *'"hunyuan3d-dit-v2-mv"'* && "$out" == *'"hunyuan3d-dit-v2-mv-turbo"'* && "$out" == *'"zero123-xl"'* ]] && echo 1 || echo 0)"
check "M a cache holding only the single-view model is INCOMPLETE, naming the mv weights and the zero123 components" "$([[ "$out" == *"incomplete"* && "$out" == *"Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors"* && "$out" == *"Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors"* && "$out" == *"zero123-xl/unet/diffusion_pytorch_model.bin"* && "$out" == *"zero123-xl/cc_projection/config.json"* ]] && echo 1 || echo 0)"
check "M the single-view model already on the volume is NOT re-listed as missing" "$([[ "$out" != *"Hunyuan3D-2/hunyuan3d-dit-v2-0/model.fp16.safetensors"* ]] && echo 1 || echo 0)"

echo "== N: an incomplete cache PREFETCHES before serving (endpoint black-holed: every file fails, named), then boots ONLINE =="
out="$(runc "${BH[@]}" -e HY3D_PREFETCH=1 -e HF_HUB_DOWNLOAD_TIMEOUT=1 -e HF_HUB_ETAG_TIMEOUT=1 "$IMG" -c "$RUN"'run_and_ping 200 150' 2>&1)"; echo "$out" | grep -E "^\[start\]|^\[prefetch\]|PING|exit=" | head -12
check "N start.sh says it is prefetching before serving, naming what is missing" "$([[ "$out" == *"[start] HF cache incomplete, prefetching before serving; missing:"* && "$out" == *"hunyuan3d-dit-v2-0/model.fp16.safetensors"* ]] && echo 1 || echo 0)"
check "N every prefetch attempt is named with its failure (no silent skip)" "$([[ "$out" == *"[prefetch] FAILED tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0/config.yaml"* && "$out" == *"[prefetch] done:"* ]] && echo 1 || echo 0)"
check "N a prefetch that could not fetch anything still boots ONLINE and answers /ping 200" "$([[ "$out" == *"[start] HF cache incomplete, staying ONLINE; missing:"* && "$out" == *"HF_HUB_OFFLINE=None"* && "$out" == *"PING 200"* ]] && echo 1 || echo 0)"

echo "SUMMARY: $fails failure(s) across 18 cases for $IMG overlay=$OVERLAY"
exit $(( fails > 0 ))
