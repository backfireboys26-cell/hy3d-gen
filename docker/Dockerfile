# engine/generation/docker/Dockerfile - the deployable FORGE generation image (pillar-3, Phase B).
#
# The SAME patched Hunyuan3D-2GP stack rsv4/omen run locally, packaged so a rented GPU host
# (RunPod LOAD_BALANCER serverless, linux/amd64) runs it byte-for-byte unmodified. Weights are
# NOT baked: HF_HOME points at a mounted volume (RunPod network volume at /runpod-volume) and
# the first warm run downloads them there, once ever.
#
# Base: python:3.12-slim + torch cu124 wheels, NOT an nvidia/cuda base. The pip cu124 wheels
# bundle their own CUDA runtime libraries (nvidia-* wheel deps); the only host requirement is
# the NVIDIA driver, which RunPod's container runtime injects. That maximizes host
# compatibility and keeps the image smaller than a cuda-runtime base duplicating those libs.
# Python 3.12 matches the local venv (see ../README.md), so the frozen pins carry over.
#
# Build (WSL2 Ubuntu on rsv4; context = engine/generation so patches/ and requirements/ COPY in):
#   cd /mnt/c/Users/Nickz/Documents/Brain/engine/generation
#   docker build -f docker/Dockerfile -t hy3d-gen:cu124 .
# CPU-torch variant of the SAME file (contract tests on a GPU-less builder):
#   docker build -f docker/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#       -t hy3d-gen:cpu .

FROM python:3.12-slim-bookworm

# cu124 is the pinned stack (torch 2.5.1+cu124 - see requirements/requirements-frozen.txt for
# why NEVER bare PyPI for torch). Override with .../whl/cpu for a CPU-only build.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
# the pin the whole stack is patched against - bump ONLY together with rsv4-stack.patch
ARG HY3D_REF=f2456e0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf

# runtime libs for opencv-python / pymeshlab / onnxruntime on a headless box
# (libopengl0: pymeshlab's filter plugins link libOpenGL.so.0 - without it every filter
# plugin fails to load at import, observed in the first WSL contract run)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates bash \
        libgl1 libopengl0 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/deepbeepmeep/Hunyuan3D-2GP.git /app \
    && git -C /app checkout ${HY3D_REF}

# the two tracked-file fixes rsv4 depends on: mc_algo='mc' (no diso/nvcc needed), Windows-safe
# temp files, HONEST /status error reporting (a crashed worker says 'error', not 'processing'
# forever), --subfolder selection, and the degenerate-octree guard. Source of truth: ../patches.
COPY patches/rsv4-stack.patch /tmp/rsv4-stack.patch
RUN git -C /app apply --verbose /tmp/rsv4-stack.patch

# torch stack FIRST from the chosen index, then the repo requirements (the vaulted
# pascal/container variant: upstream minus diso, plus scikit-image + mmgp) under the frozen
# constraints so no dependency can silently move torch/numpy.
COPY requirements/requirements-pascal.txt /tmp/requirements-container.txt
COPY requirements/requirements-frozen.txt /tmp/constraints.txt
RUN pip install --index-url ${TORCH_INDEX} torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    && pip install -r /tmp/requirements-container.txt -c /tmp/constraints.txt

# the front door (bearer auth + health, stdlib-only) and the two-process entry
COPY docker/auth_gate.py /app/auth_gate.py
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# the gate's default listen port; RunPod overrides via env PORT and maps it itself
EXPOSE 8080
CMD ["/app/start.sh"]
