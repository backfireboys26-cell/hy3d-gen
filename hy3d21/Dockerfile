# hy3d21/Dockerfile - Hunyuan3D-2.1 SHAPE-ONLY generation image (the bake-off's upgraded
# incumbent lane: same contract as the cu124 image, newer model).
#
# Same base philosophy as docker/Dockerfile: python slim + pip cu124 torch wheels (they
# bundle the CUDA runtime libs; host only needs the NVIDIA driver, which RunPod injects).
# Python 3.10 here, NOT 3.12: Hunyuan3D-2.1 is tested on 3.10 and pins numpy==1.24.4,
# which has no cp312 wheels. No CUDA toolchain needed - shape-only avoids the paint
# pipeline's custom_rasterizer extension AND uses the skimage 'mc' surface extractor
# (the VAE default), never diso/dmc.
#
# Weights are NOT baked. hy3dshape's smart_load_model downloads into $HY3DGEN_MODELS
# (mounted volume), rembg's u2net into $U2NET_HOME, everything HF into $HF_HOME.

FROM python:3.10-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
# pin of Tencent-Hunyuan/Hunyuan3D-2.1 this image is built against
ARG HY3D21_REF=82920d643c0dc2f7bfd7255f45f62d386edfe60c

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    HY3DGEN_MODELS=/runpod-volume/hy3dgen \
    U2NET_HOME=/runpod-volume/u2net

# same headless runtime libs the cu124 image needed for opencv/pymeshlab/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates bash \
        libgl1 libopengl0 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app \
    && git -C /app checkout ${HY3D21_REF}

# torch first from the cu124 index (README-tested stack: 2.5.1+cu124), then the
# shape-only requirement set with the repo's own version pins carried over.
COPY hy3d21/requirements.txt /tmp/requirements-hy3d21.txt
RUN pip install --index-url ${TORCH_INDEX} torch==2.5.1 torchvision==0.20.1 \
    && pip install -r /tmp/requirements-hy3d21.txt

# front door (bearer auth + honest health) + the contract server + entry
COPY docker/auth_gate.py /app/auth_gate.py
COPY hy3d21/server.py /app/hy3d21_server.py
COPY hy3d21/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
