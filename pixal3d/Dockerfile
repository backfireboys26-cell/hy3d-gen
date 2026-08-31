# pixal3d/Dockerfile - TencentARC Pixal3D challenger image (bake-off lane 3).
#
# TRELLIS.2 backbone + pixel-aligned projection conditioning. Shares the TRELLIS.2
# extension stack (FlexGEMM, CuMesh, o-voxel compiled here; nvdiffrast JIT at runtime)
# PLUS natten (NAF neighborhood-attention upsampler via torch.hub valeoai/NAF).
#
# WHY torch 2.7.0 + cu126, not 2.6.0 + cu124 like the trellis2 lane: natten==0.21.0
# (the version Pixal3D's README pins) publishes prebuilt wheels ONLY for
# torch2.7.0+cu126/cu128 - there is no cu124 wheel at 0.21.x, and a natten source build
# is a multi-hour gamble on a 4-core CI runner. CUDA minor-version compatibility means
# a cu126 userland runs on any 12.x-capable driver (>=525), which every RunPod host has.
#
# flash-attn deliberately absent: Pixal3D supports ATTN_BACKEND=sdpa everywhere
# (dense AND sparse), which its own README recommends when flash_attn is missing.
#
# Weights NOT baked: TencentARC/Pixal3D + camenduru dinov3 mirror + MoGe-2 land in
# $HF_HOME, the NAF torch.hub checkpoint in $TORCH_HOME - all on the RunPod volume.
# Gated-repo redirect (briaai RMBG-2.0 -> ZhengPeng7/BiRefNet) lives in server.py,
# env-overridable; the container starts with zero HF tokens.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG PIXAL3D_REF=cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af
ARG TRELLIS2_REF=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG NVDIFFRAST_REF=v0.4.0
ARG CUMESH_REF=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_REF=6dd94a859c26ee8246888502eada3dd8ad85532e
ARG MOGE_REF=74fbce054ebed49800de42d0ad0e83495065719a

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch \
    ATTN_BACKEND=sdpa \
    SPARSE_CONV_BACKEND=flex_gemm \
    PIXAL_LOW_VRAM=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/runpod-volume/flexgemm-autotune.json \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    GPU_ARCHS="sm_80;sm_86;sm_89;sm_90" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates bash ninja-build \
        libgl1 libopengl0 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0 torchvision==0.22.0

# natten prebuilt wheel matched to torch 2.7.0 + cu126 (README step 3, minus the build)
RUN pip install natten==0.21.0+torch270cu126 -f https://whl.natten.org

# Pixal3D's requirements.txt pins + the contract server's. MoGe installs --no-deps:
# its dependency list drags in a DIFFERENT FlexGEMM git pin (a CUDA source build that
# would fire inside pip build isolation, without torch), gradio 6, matplotlib and a
# 'pipeline' package - none needed for the moge.model.v2 import path this server uses;
# its vendored utils3d_moge ships alongside (also --no-deps) so MoGe never touches
# Pixal3D's utils3d. LAST, Pixal3D's exact utils3d wheel (README step 4) wins the
# `utils3d` name.
RUN pip install pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 tqdm==4.67.1 \
        easydict==1.13 opencv-python-headless==4.12.0.88 trimesh==4.10.1 \
        transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 timm==1.0.22 \
        diffusers==0.37.1 accelerate==1.13.0 plyfile==1.1.3 scipy \
        fastapi uvicorn \
    && pip install --no-deps \
        git+https://github.com/microsoft/MoGe.git@${MOGE_REF} \
        git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37 \
    && pip install --force-reinstall --no-deps \
        https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

WORKDIR /app
RUN git clone https://github.com/TencentARC/Pixal3D.git /app \
    && git -C /app checkout ${PIXAL3D_REF}

# CUDA extensions (same trio as the trellis2 lane; o-voxel comes from the TRELLIS.2
# repo, which Pixal3D's install docs defer to). GPU_ARCHS pins offload archs so the
# GPU-less CI builder can compile; nvdiffrast JIT-compiles at first use (devel base
# keeps nvcc available at runtime).
RUN mkdir -p /tmp/ext \
    && git clone -b ${NVDIFFRAST_REF} https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install --no-build-isolation /tmp/ext/nvdiffrast \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && git -C /tmp/ext/CuMesh checkout ${CUMESH_REF} \
    && pip install --no-build-isolation /tmp/ext/CuMesh \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && git -C /tmp/ext/FlexGEMM checkout ${FLEXGEMM_REF} \
    && pip install --no-build-isolation /tmp/ext/FlexGEMM \
    && git clone --recursive https://github.com/microsoft/TRELLIS.2.git /tmp/ext/TRELLIS.2 \
    && git -C /tmp/ext/TRELLIS.2 checkout ${TRELLIS2_REF} \
    && git -C /tmp/ext/TRELLIS.2 submodule update --init --recursive \
    && pip install --no-build-isolation /tmp/ext/TRELLIS.2/o-voxel \
    && rm -rf /tmp/ext

# front door + contract server + entry
COPY docker/auth_gate.py /app/auth_gate.py
COPY pixal3d/server.py /app/pixal3d_server.py
COPY pixal3d/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
