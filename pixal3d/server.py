"""pixal3d/server.py - TencentARC Pixal3D image-to-3D server, FORGE /send contract.

Same wire contract as the other lanes:

    POST /send                 {"image": <b64 png>, "seed": int, "octree_resolution": int,
                                "num_inference_steps": int, "guidance_scale": float,
                                "views": [<b64>, <b64>]? | "image_back": <b64>?, ...}
                               -> {"uid": "<uuid>"}       (extra params tolerated)
    GET  /status/{uid}         -> {"status": "processing"}
                                | {"status": "completed", "model_base64": "<b64 glb>",
                                   "warnings": [...]?}
                                | {"status": "error", "error": "<Type: msg>"}

Parameter mapping onto Pixal3DImageTo3DPipeline.run() (mirrors upstream inference.py):
  seed                 -> run(seed=...)
  octree_resolution    -> pipeline_type: >=1536 -> '1536_cascade', else '1024_cascade'
  num_inference_steps  -> 'steps' override in all three sampler param dicts
  guidance_scale       -> 'guidance_strength' for sparse-structure + shape samplers
                          (texture guidance stays at its tuned 1.0 unless
                          'tex_guidance_scale' is sent)
  manual_fov           -> camera FOV in radians (skips MoGe-2 estimation)
  tolerated extras: max_num_tokens, decimation_target, texture_size, mesh_scale

MULTI-VIEW HONESTY: Pixal3D's HF repo ships _mv checkpoints + pipeline_mv.json declaring
a `Pixal3DMVImageTo3DPipeline`, but the released GitHub code (master @ our pin) contains
NO such class - and its inference ProjGrid hard-asserts `transform_matrix is None`, so a
genuine second-view camera cannot even be injected from outside. This shim therefore
ACCEPTS `views`/`image_back` per the contract, probes for the MV pipeline class at
startup (auto-lights-up if TencentARC releases it), and until then generates from the
FRONT view and returns an explicit warning instead of crashing or silently pretending.

Low-VRAM mode (upstream's on-demand loading, ~10-12GB peak) is ON by default via
PIXAL_LOW_VRAM=1; set PIXAL_LOW_VRAM=0 on 48GB+ cards for speed.
"""
import argparse
import base64
import math
import os
import threading
import traceback
import uuid
from io import BytesIO

os.environ.setdefault("ATTN_BACKEND", "sdpa")  # Pixal3D supports sdpa everywhere
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

SAVE_DIR = os.environ.get("HY3D_SAVE_DIR", "/tmp/pixal3d_out")
os.makedirs(SAVE_DIR, exist_ok=True)

MOGE_MODEL_NAME = os.environ.get("MOGE_MODEL", "Ruicheng/moge-2-vitl")
DINOV3_MODEL = os.environ.get("DINOV3_MODEL", "camenduru/dinov3-vitl16-pretrain-lvd1689m")
REMBG_GATED = "briaai/RMBG-2.0"
REMBG_DEFAULT = os.environ.get("REMBG_MODEL", "ZhengPeng7/BiRefNet")
LOW_VRAM = os.environ.get("PIXAL_LOW_VRAM", "1") not in ("0", "false", "no", "")

# upstream inference.py constants
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

IMAGE_COND_CONFIGS = {
    "ss": {"model_name": DINOV3_MODEL, "image_size": 512, "grid_resolution": 16},
    "shape_512": {"model_name": DINOV3_MODEL, "image_size": 512, "grid_resolution": 32,
                  "use_naf_upsample": True, "naf_target_size": 512},
    "shape_1024": {"model_name": DINOV3_MODEL, "image_size": 1024, "grid_resolution": 64,
                   "use_naf_upsample": True, "naf_target_size": 512},
    "tex_1024": {"model_name": DINOV3_MODEL, "image_size": 1024, "grid_resolution": 64,
                 "use_naf_upsample": True, "naf_target_size": 1024},
}

app = FastAPI()

JOB_ERRORS = {}
JOB_WARNINGS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()

worker = None


# ---------------------------------------------------------------------------
# camera estimation (adapted from upstream inference.py, PIL-in instead of path-in)
# ---------------------------------------------------------------------------

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    return float((focal_length * resolution / 32.0).item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, _ = gp[0].item(), gp[1].item(), gp[2].item()
    xt, _ = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return float(distance_x)


def camera_params_from_fov(camera_angle_x: float, mesh_scale: float) -> dict:
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - WILD_EXTEND_PIXEL, WILD_IMAGE_RESOLUTION - 1 + WILD_EXTEND_PIXEL]),
        mesh_scale, WILD_IMAGE_RESOLUTION,
    )
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _redirect_gated_rembg():
    from pixal3d.pipelines import rembg as rembg_mod
    _Orig = rembg_mod.BiRefNet

    class _Redirected(_Orig):
        def __init__(self, model_name="ZhengPeng7/BiRefNet", *a, **kw):
            if model_name == REMBG_GATED and REMBG_DEFAULT != REMBG_GATED:
                print(f"[pixal3d] rembg redirect: {model_name} -> {REMBG_DEFAULT}", flush=True)
                model_name = REMBG_DEFAULT
            super().__init__(model_name, *a, **kw)

    rembg_mod.BiRefNet = _Redirected


class Pixal3DWorker:
    def __init__(self, model_path: str, device: str = "cuda"):
        _redirect_gated_rembg()
        import pixal3d.pipelines as pipelines_mod
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )

        # probe for the (not-yet-released) native multi-view pipeline so this container
        # lights it up automatically once TencentARC publishes the class
        self.mv_pipeline_cls = getattr(pipelines_mod, "Pixal3DMVImageTo3DPipeline", None) \
            if hasattr(pipelines_mod, "__attributes") and \
            "Pixal3DMVImageTo3DPipeline" in getattr(pipelines_mod, "__attributes", {}) else None

        print(f"[pixal3d] loading {model_path} (low_vram={LOW_VRAM}) ...", flush=True)
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
        for stage, cfg in IMAGE_COND_CONFIGS.items():
            m = DinoV3ProjFeatureExtractor(**cfg)
            m.eval()
            setattr(pipeline, f"image_cond_model_{stage}", m)

        cond_attrs = ["image_cond_model_ss", "image_cond_model_shape_512",
                      "image_cond_model_shape_1024", "image_cond_model_tex_1024"]
        if LOW_VRAM:
            for attr in cond_attrs:
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, "use_naf_upsample", False):
                    m._load_naf()  # pre-download NAF weights, CPU only
            pipeline._device = torch.device(device)
            pipeline.low_vram = True
        else:
            pipeline.low_vram = False
            pipeline.cuda()
            for attr in cond_attrs:
                m = getattr(pipeline, attr)
                m.cuda()
                if getattr(m, "use_naf_upsample", False):
                    m._load_naf()
        self.pipeline = pipeline
        self.device = device

        # MoGe-2 camera estimator: kept on CPU, visits the GPU per request
        print("[pixal3d] loading MoGe-2 (CPU-resident) ...", flush=True)
        from moge.model.v2 import MoGeModel
        self.moge = MoGeModel.from_pretrained(MOGE_MODEL_NAME)
        self.moge.eval()
        print("[pixal3d] pipeline ready", flush=True)

    def _estimate_camera(self, pil_image: Image.Image, mesh_scale: float) -> dict:
        pil_image = pil_image.convert("RGB")
        width, _ = pil_image.size
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        self.moge.to(self.device)
        try:
            with torch.no_grad():
                output = self.moge.infer(image_tensor)
        finally:
            self.moge.cpu()
            torch.cuda.empty_cache()
        intrinsics = output["intrinsics"].squeeze().cpu().numpy()
        fx = intrinsics[0, 0] * width
        camera_angle_x = 2 * math.atan(width / (2 * fx))
        return camera_params_from_fov(camera_angle_x, mesh_scale)

    @torch.no_grad()
    def generate(self, uid, params: dict) -> str:
        import o_voxel

        warnings = []

        # ---- gather views (contract: image, or views[], or image + image_back) ----
        views_b64 = []
        if params.get("views"):
            views_b64 = list(params["views"])
        elif params.get("image"):
            views_b64 = [params["image"]]
            if params.get("image_back"):
                views_b64.append(params["image_back"])
        if not views_b64:
            raise ValueError("no input image: send 'image' (b64 png), or 'views': [b64, ...]")
        views = [Image.open(BytesIO(base64.b64decode(v))) for v in views_b64]

        if len(views) > 1 and self.mv_pipeline_cls is None:
            warnings.append(
                f"{len(views)} views received but only the FRONT view was used: Pixal3D's "
                "released inference code has no multi-view pipeline class (HF ships "
                "pipeline_mv.json/_mv checkpoints, but Pixal3DMVImageTo3DPipeline is absent "
                "from the GitHub code at our pin, and its ProjGrid asserts "
                "transform_matrix is None). Rebuild this image when upstream releases it.")

        image = views[0]

        seed = int(params.get("seed", 1234))
        octree_resolution = int(params.get("octree_resolution", 0) or 0)
        if octree_resolution >= 1536:
            resolution = 1536
        elif octree_resolution > 0:
            resolution = 1024
        else:
            resolution = 1024  # default lane (also the low-VRAM-safe one)
        pipeline_type = f"{resolution}_cascade"

        sampler_override = {}
        if params.get("num_inference_steps"):
            sampler_override["steps"] = int(params["num_inference_steps"])
        guided_override = dict(sampler_override)
        if params.get("guidance_scale") is not None:
            guided_override["guidance_strength"] = float(params["guidance_scale"])
        tex_override = dict(sampler_override)
        if params.get("tex_guidance_scale") is not None:
            tex_override["guidance_strength"] = float(params["tex_guidance_scale"])

        mesh_scale = float(params.get("mesh_scale", WILD_MESH_SCALE))

        with GPU_LOCK:
            image_preprocessed = self.pipeline.preprocess_image(image)

            manual_fov = float(params.get("manual_fov", -1.0))
            if manual_fov > 0:
                camera_params = camera_params_from_fov(manual_fov, mesh_scale)
            else:
                camera_params = self._estimate_camera(image_preprocessed, mesh_scale)
            print(f"[pixal3d] uid {uid}: camera {camera_params}", flush=True)

            torch.manual_seed(seed)
            mesh_list, (_shape_slat, _tex_slat, res) = self.pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=seed,
                sparse_structure_sampler_params=guided_override,
                shape_slat_sampler_params=guided_override,
                tex_slat_sampler_params=tex_override,
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=int(params.get("max_num_tokens", 49152)),
            )
            mesh = mesh_list[0]

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=self.pipeline.pbr_attr_layout,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(params.get("decimation_target", 1000000)),
                texture_size=int(params.get("texture_size", 2048)),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=False,
            )
            torch.cuda.empty_cache()

        # upstream inference.py's canonical orientation fix
        rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                       dtype=np.float64)
        glb.apply_transform(rot)

        save_path = os.path.join(SAVE_DIR, f"{uid}.glb")
        tmp_path = save_path + ".tmp.glb"
        try:
            glb.export(tmp_path, extension_webp=bool(params.get("extension_webp", False)))
        except TypeError:
            glb.export(tmp_path)
        if warnings:
            with JOBS_LOCK:
                JOB_WARNINGS[str(uid)] = warnings
        os.replace(tmp_path, save_path)  # /status sees the file only when it is whole
        return save_path


def _run_generate(uid, params):
    try:
        worker.generate(uid, params)
    except Exception as e:
        print(f"[pixal3d] worker crashed for uid {uid}:\n{traceback.format_exc()}", flush=True)
        with JOBS_LOCK:
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
        payload = {"status": "completed", "model_base64": b64}
        with JOBS_LOCK:
            if str(uid) in JOB_WARNINGS:
                payload["warnings"] = JOB_WARNINGS[str(uid)]
        return JSONResponse(payload, status_code=200)
    with JOBS_LOCK:
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
                        default=os.environ.get("MODEL_PATH", "TencentARC/Pixal3D"))
    args = parser.parse_args()

    # load BEFORE listening: gate health = socket accepts = weights loaded
    worker = Pixal3DWorker(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
