"""hy3d21/server.py - Hunyuan3D-2.1 SHAPE-ONLY inference server, FORGE /send contract.

Same wire contract as the patched Hunyuan3D-2GP api_server (the cu124 incumbent image),
so the one FORGE client drives both:

    POST /send                 {"image": <b64 png>, "seed": int, "octree_resolution": int,
                                "num_inference_steps": int, "guidance_scale": float, ...}
                               -> {"uid": "<uuid>"}       (extra params tolerated)
    GET  /status/{uid}         -> {"status": "processing"}
                                | {"status": "completed", "model_base64": "<b64 glb>"}
                                | {"status": "error", "error": "<Type: msg>"}

Deliberately SHAPE-ONLY: the hy3dpaint texture pipeline (RealESRGAN ckpt + custom
rasterizer CUDA extension + obj2gltf) is skipped entirely - we print filament, color
comes from the slicer, and paint is the part of 2.1 that does not build cleanly.

Honest async failure reporting is built in from the start (the JOB_ERRORS pattern the
2.0 image needed a patch for): a crashed worker thread reports {status: error}
immediately instead of 'processing' forever.

Weights: tencent/Hunyuan3D-2.1 via hy3dshape's smart_load_model, which downloads into
$HY3DGEN_MODELS (set to /runpod-volume/hy3dgen in the image) - NOT baked into the image.
"""
import argparse
import base64
import os
import sys
import threading
import traceback
import uuid
from io import BytesIO

sys.path.insert(0, "/app/hy3dshape")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

SAVE_DIR = os.environ.get("HY3D_SAVE_DIR", "/tmp/hy3d21_out")
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()

JOB_ERRORS = {}  # uid(str) -> {"status": "error", "error": "<Type: msg>"}
JOB_ERRORS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()  # one generation at a time on the card

worker = None  # set in main() BEFORE uvicorn listens (health = socket = model loaded)


def _needs_rembg(image: Image.Image) -> bool:
    """True when the input carries no usable alpha matte (upstream 2.1 model_worker has
    a dead-code bug here: it converts to RGBA first, then checks mode == 'RGB', which is
    never true - so its rembg never ran. Check BEFORE converting)."""
    if image.mode != "RGBA":
        return True
    alpha = np.array(image)[:, :, 3]
    return bool(np.all(alpha == 255))


class ShapeWorker:
    def __init__(self, model_path: str, subfolder: str, device: str):
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover

        print(f"[hy3d21] loading {model_path} / {subfolder} on {device} ...", flush=True)
        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path, subfolder=subfolder, device=device,
        )
        self.device = device
        print("[hy3d21] pipeline ready", flush=True)

    @torch.inference_mode()
    def generate(self, uid, params: dict) -> str:
        if "image" not in params:
            raise ValueError("no 'image' (base64 png) in request")
        image = Image.open(BytesIO(base64.b64decode(params["image"])))
        if _needs_rembg(image):
            image = self.rembg(image.convert("RGB"))
        image = image.convert("RGBA")

        seed = int(params.get("seed", 1234))
        generator = torch.Generator(self.device).manual_seed(seed)

        call_kwargs = dict(
            image=image,
            generator=generator,
            octree_resolution=int(params.get("octree_resolution", 384)),
            num_inference_steps=int(params.get("num_inference_steps", 50)),
            guidance_scale=float(params.get("guidance_scale", 5.0)),
            output_type="trimesh",
        )
        # tolerated extras with a direct pipeline meaning
        if params.get("mc_algo"):
            call_kwargs["mc_algo"] = params["mc_algo"]  # default None = VAE default (skimage mc)
        if params.get("num_chunks"):
            call_kwargs["num_chunks"] = int(params["num_chunks"])

        with GPU_LOCK:
            mesh = self.pipeline(**call_kwargs)[0]

            # optional mesh cleanup, OFF by default (bake-off judges the raw generation,
            # matching what the 2.0 incumbent image returns)
            if params.get("apply_floater_remover"):
                from hy3dshape.postprocessors import FloaterRemover
                mesh = FloaterRemover()(mesh)
            if params.get("apply_degenerate_face_remover"):
                from hy3dshape.postprocessors import DegenerateFaceRemover
                mesh = DegenerateFaceRemover()(mesh)
            if params.get("face_count"):
                from hy3dshape.postprocessors import FaceReducer
                mesh = FaceReducer()(mesh, max_facenum=int(params["face_count"]))

            torch.cuda.empty_cache()

        save_path = os.path.join(SAVE_DIR, f"{uid}.glb")
        tmp_path = save_path + ".tmp.glb"
        mesh.export(tmp_path)
        os.replace(tmp_path, save_path)  # /status sees the file only when it is whole
        return save_path


def _run_generate(uid, params):
    try:
        worker.generate(uid, params)
    except Exception as e:
        print(f"[hy3d21] worker crashed for uid {uid}:\n{traceback.format_exc()}", flush=True)
        with JOB_ERRORS_LOCK:
            JOB_ERRORS[str(uid)] = {"status": "error", "error": f"{type(e).__name__}: {e}"}


@app.post("/send")
async def send(request: Request):
    try:
        params = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON body: {e}"}, status_code=400)
    uid = uuid.uuid4()
    threading.Thread(target=_run_generate, args=(uid, params), daemon=True).start()
    return JSONResponse({"uid": str(uid)}, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_path = os.path.join(SAVE_DIR, f"{uid}.glb")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return JSONResponse({"status": "completed", "model_base64": b64}, status_code=200)
    with JOB_ERRORS_LOCK:
        err = JOB_ERRORS.get(str(uid))
    if err is not None:
        return JSONResponse(err, status_code=200)
    return JSONResponse({"status": "processing"}, status_code=200)


def main():
    global worker
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str,
                        default=os.environ.get("MODEL_PATH", "tencent/Hunyuan3D-2.1"))
    parser.add_argument("--subfolder", type=str,
                        default=os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1"))
    parser.add_argument("--device", type=str, default=os.environ.get("HY3D_DEVICE", "cuda"))
    args = parser.parse_args()

    # load BEFORE listening: the auth_gate's health probe is 'socket accepts', which must
    # mean 'weights loaded and serving', never 'python started'
    worker = ShapeWorker(args.model_path, args.subfolder, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
