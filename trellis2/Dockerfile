# trellis2/Dockerfile - Microsoft TRELLIS.2-4B challenger image (bake-off lane 2).
#
# Unlike the Hunyuan images this one CANNOT ride on pip-wheel CUDA runtimes alone:
# TRELLIS.2's mesh/GLB path hard-imports three source-built CUDA extensions (FlexGEMM,
# CuMesh, o-voxel) and nvdiffrast JIT-compiles kernels at first use, so the image is
# based on the cuda 12.4 DEVEL image - nvcc present at build time (extensions) and at
# runtime (nvdiffrast JIT). Torch 2.6.0+cu124 per the repo's own setup.sh.
#
# flash-attn is deliberately NOT installed: dense attention runs ATTN_BACKEND=xformers
# (TRELLIS.2's sparse attention accepts only xformers/flash_attn - no sdpa - so xformers
# is the no-compile backend that satisfies both paths).
#
# Extensions are compiled WITHOUT a GPU on the runner: GPU_ARCHS pins the offload archs
# (their setup.py defaults to 'native', which needs a physical GPU). sm_80/86/89/90
# covers A100 / 3090-A6000 / 4090-L40S / H100. Pre-Ampere (V100/T4) is NOT covered.
#
# Weights are NOT baked (HF_HOME on the RunPod volume). The gated-repo redirects
# (facebook dinov3 -> camenduru mirror, briaai RMBG-2.0 -> ZhengPeng7/BiRefNet) live in
# server.py and are env-overridable; the container starts with zero HF tokens.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG TRELLIS2_REF=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG NVDIFFRAST_REF=v0.4.0
ARG CUMESH_REF=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_REF=6dd94a859c26ee8246888502eada3dd8ad85532e
ARG UTILS3D_REF=9a4eb15e4021b67b12c460c7057d642626897ec8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch \
    ATTN_BACKEND=xformers \
    SPARSE_CONV_BACKEND=flex_gemm \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    GPU_ARCHS="sm_80;sm_86;sm_89;sm_90" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates bash ninja-build \
        libgl1 libopengl0 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN pip install --upgrade pip setuptools wheel

# torch stack per TRELLIS.2 setup.sh (--new-env step), plus xformers matched to it
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3

# --basic deps (setup.sh) minus training/demo-only extras, plus the contract server's
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers==4.57.3 pandas zstandard kornia timm pillow plyfile \
        fastapi uvicorn \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_REF}

WORKDIR /app
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app \
    && git -C /app checkout ${TRELLIS2_REF} \
    && git -C /app submodule update --init --recursive

# CUDA extensions, compiled for the pinned arch list (no GPU on the CI builder).
# nvdiffrast's pip install is compile-free (kernels JIT at first use, nvcc kept at
# runtime by the devel base); CuMesh / FlexGEMM / o-voxel compile here and honor
# GPU_ARCHS via their setup.py.
RUN mkdir -p /tmp/ext \
    && git clone -b ${NVDIFFRAST_REF} https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install --no-build-isolation /tmp/ext/nvdiffrast \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && git -C /tmp/ext/CuMesh checkout ${CUMESH_REF} \
    && pip install --no-build-isolation /tmp/ext/CuMesh \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && git -C /tmp/ext/FlexGEMM checkout ${FLEXGEMM_REF} \
    && pip install --no-build-isolation /tmp/ext/FlexGEMM \
    && pip install --no-build-isolation /app/o-voxel \
    && rm -rf /tmp/ext

# front door + contract server + entry
COPY docker/auth_gate.py /app/auth_gate.py
COPY trellis2/server.py /app/trellis2_server.py
COPY trellis2/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
