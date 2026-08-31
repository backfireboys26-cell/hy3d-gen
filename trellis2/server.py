"""trellis2/server.py - Microsoft TRELLIS.2-4B image-to-3D server, FORGE /send contract.

Same wire contract as the incumbent Hunyuan images, so one FORGE client drives all lanes:

    POST /send                 {"image": <b64 png>, "seed": int, "octree_resolution": int,
                                "num_inference_steps": int, "guidance_scale": float, ...}
                               -> {"uid": "<uuid>"}       (extra params tolerated)
    GET  /status/{uid}         -> {"status": "processing"}
                                | {"status": "completed", "model_base64": "<b64 glb>", ...}
                                | {"status": "error", "error": "<Type: msg>"}

Parameter mapping onto Trellis2ImageTo3DPipeline.run():
  seed                 -> run(seed=...)
  octree_resolution    -> pipeline_type: >=1536 -> '1536_cascade', else '1024_cascade'
                          (TRELLIS.2's native resolutions; 1024 is the default lane)
  num_inference_steps  -> 'steps' override in all three sampler param dicts
  guidance_scale       -> 'guidance_strength' override for sparse-structure + shape
                          samplers (texture guidance stays at its tuned 1.0 unless
                          'tex_guidance_scale' is sent)
  tolerated extras: pipeline_type, max_num_tokens, num_samples, decimation_target,
                    texture_size, simplify_faces, extension_webp

Gated-repo redirects (both env-overridable, defaults chosen so the container starts
with ZERO tokens):
  DINOV3_MODEL  (default camenduru/dinov3-vitl16-pretrain-lvd1689m) replaces the gated
                facebook/dinov3-vitl16-pretrain-lvd1689m the pipeline.json names -
                byte-identical mirror weights, no Meta approval wall at first run.
  REMBG_MODEL   (default ZhengPeng7/BiRefNet, MIT) replaces gated briaai/RMBG-2.0.
                Set REMBG_MODEL=briaai/RMBG-2.0 + HF_TOKEN for paper-exact rembg.
"""
import argparse
import base64
import os
import threading
import traceback
import uuid
from io import BytesIO

os.environ.setdefault("ATTN_BACKEND", "xformers")  # sparse attn: no 'sdpa' in TRELLIS.2

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

SAVE_DIR = os.environ.get("HY3D_SAVE_DIR", "/tmp/trellis2_out")
os.makedirs(SAVE_DIR, exist_ok=True)

DINOV3_GATED = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_DEFAULT_MIRROR = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
REMBG_GATED = "briaai/RMBG-2.0"
REMBG_DEFAULT = "ZhengPeng7/BiRefNet"

app = FastAPI()

JOB_ERRORS = {}
JOB_ERRORS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()

worker = None


def _apply_gated_repo_redirects():
    """Point the pipeline.json's gated HF repos at ungated equivalents (env-overridable)."""
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import rembg as rembg_mod

    dinov3 = os.environ.get("DINOV3_MODEL", DINOV3_DEFAULT_MIRROR)
    rembg_name = os.environ.get("REMBG_MODEL", REMBG_DEFAULT)

    if dinov3:
        _OrigDino = image_feature_extractor.DinoV3FeatureExtractor

        class _RedirectedDino(_OrigDino):
            def __init__(self, model_name, *a, **kw):
                if model_name == DINOV3_GATED:
                    print(f"[trellis2] DinoV3 redirect: {model_name} -> {dinov3}", flush=True)
                    model_name = dinov3
                super().__init__(model_name, *a, **kw)

        image_feature_extractor.DinoV3FeatureExtractor = _RedirectedDino

    if rembg_name:
        _OrigBiRefNet = rembg_mod.BiRefNet

        class _RedirectedBiRefNet(_OrigBiRefNet):
            def __init__(self, model_name=REMBG_DEFAULT, *a, **kw):
                if model_name == REMBG_GATED and rembg_name != REMBG_GATED:
                    print(f"[trellis2] rembg redirect: {model_name} -> {rembg_name}", flush=True)
                    model_name = rembg_name
                super().__init__(model_name, *a, **kw)

        rembg_mod.BiRefNet = _RedirectedBiRefNet


class Trellis2Worker:
    def __init__(self, model_path: str):
        _apply_gated_repo_redirects()
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        print(f"[trellis2] loading {model_path} ...", flush=True)
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        # low_vram defaults True: cuda() records the device; stages move on demand
        self.pipeline.cuda()
        print("[trellis2] pipeline ready "
              f"(low_vram={getattr(self.pipeline, 'low_vram', None)})", flush=True)

    @torch.no_grad()
    def generate(self, uid, params: dict) -> str:
        import o_voxel

        if "image" not in params:
            raise ValueError("no 'image' (base64 png) in request")
        image = Image.open(BytesIO(base64.b64decode(params["image"])))

        seed = int(params.get("seed", 1234))
        octree_resolution = int(params.get("octree_resolution", 1024))
        pipeline_type = params.get("pipeline_type") or (
            "1536_cascade" if octree_resolution >= 1536 else "1024_cascade")

        sampler_override = {}
        if params.get("num_inference_steps"):
            sampler_override["steps"] = int(params["num_inference_steps"])
        guided_override = dict(sampler_override)
        if params.get("guidance_scale") is not None:
            guided_override["guidance_strength"] = float(params["guidance_scale"])
        tex_override = dict(sampler_override)
        if params.get("tex_guidance_scale") is not None:
            tex_override["guidance_strength"] = float(params["tex_guidance_scale"])

        run_kwargs = dict(
            seed=seed,
            sparse_structure_sampler_params=guided_override,
            shape_slat_sampler_params=guided_override,
            tex_slat_sampler_params=tex_override,
            pipeline_type=pipeline_type,
        )
        if params.get("max_num_tokens"):
            run_kwargs["max_num_tokens"] = int(params["max_num_tokens"])
        if params.get("num_samples"):
            run_kwargs["num_samples"] = int(params["num_samples"])

        with GPU_LOCK:
            mesh = self.pipeline.run(image, **run_kwargs)[0]
            mesh.simplify(int(params.get("simplify_faces", 16777216)))

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(params.get("decimation_target", 1000000)),
                texture_size=int(params.get("texture_size", 2048)),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            torch.cuda.empty_cache()

        save_path = os.path.join(SAVE_DIR, f"{uid}.glb")
        tmp_path = save_path + ".tmp.glb"
        try:
            glb.export(tmp_path, extension_webp=bool(params.get("extension_webp", False)))
        except TypeError:
            glb.export(tmp_path)
        os.replace(tmp_path, save_path)  # /status sees the file only when it is whole
        return save_path


def _run_generate(uid, params):
    try:
        worker.generate(uid, params)
    except Exception as e:
        print(f"[trellis2] worker crashed for uid {uid}:\n{traceback.format_exc()}", flush=True)
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
                        default=os.environ.get("MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    args = parser.parse_args()

    # load BEFORE listening: gate health = socket accepts = weights loaded
    worker = Trellis2Worker(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
