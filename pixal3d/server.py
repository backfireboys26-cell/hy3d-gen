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

MULTI-VIEW: implemented HERE, because upstream never released it. Ground truth
established by reading their code: the HF repo ships _mv checkpoints + pipeline_mv.json
naming a `Pixal3DMVImageTo3DPipeline`, but no such class exists in the GitHub code, and
training itself picks ONE view per sample (datasets: `view_idx = randint(0, num_views)`)
- "2 views by default" is the DATA (two view-aligned latent/image pairs per object),
never per-sample fusion. What training DOES establish: the model is conditioned on one
view from an ARBITRARY camera (per-frame `transform_matrix` from transforms.json), in a
FIXED world-frame voxel grid.

So this server does principled 2-view inference as MULTI-COND GUIDANCE: build a full
conditioning set per view (front camera from MoGe / manual_fov; view k's camera =
Rz(azimuth_k) @ front, default azimuths [0, 180]), then at every flow-matching Euler
step average the model's velocity prediction across the per-view conds (classic
multi-condition diffusion guidance - every forward pass sees a valid single-view cond,
the sample follows the geometric mean of the conditionals). Requires only lifting
upstream ProjGrid's `assert transform_matrix is None` (its projection math fully
supports arbitrary cameras - the assert is the only blocker; we re-implement forward()
verbatim minus the assert). Set PIXAL_PIPELINE_CONFIG=pipeline_mv.json to load the _mv
weights (trained on 2-view-aligned data, more robust to non-front cond views).

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
# multi-view machinery
# ---------------------------------------------------------------------------

def _patch_projgrid_allow_transform():
    """Re-implement ProjGrid.forward verbatim (source: pixal3d image_conditioned_proj)
    minus the `assert transform_matrix is None` line - the projection math below it
    already handles arbitrary [B,4,4] cameras; the assert is the only blocker."""
    import torch.nn.functional as _F
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as icp

    def forward(self, features_map, camera_angle_x, distance, mesh_scale,
                transform_matrix=None, BHWC=True):
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape
        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
        else:
            transform_matrix = transform_matrix.to(grid_points.device, grid_points.dtype)
        image_points, depth, valid_mask = icp.project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution)
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)
        x = icp.sample_features(features_map, image_points_norm)
        x = x.permute(0, 2, 1)
        return x

    icp.ProjGrid.forward = forward
    print("[pixal3d] ProjGrid patched: arbitrary camera transform_matrix enabled", flush=True)


def view_transform_matrix(distance: float, azimuth_deg: float) -> torch.Tensor:
    """Camera-to-world matrix for a view at `azimuth_deg` about the world up (Z) axis.
    azimuth 0 reproduces upstream's front matrix ([1,3] = -distance) exactly."""
    front = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -float(distance)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    a = math.radians(azimuth_deg)
    rz = torch.tensor([
        [math.cos(a), -math.sin(a), 0.0, 0.0],
        [math.sin(a), math.cos(a), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rz @ front


class MultiCond(list):
    """A list of per-view cond dicts; recognized by MultiCondModel."""


class MultiCondModel(torch.nn.Module):
    """Wraps a flow model: velocity = mean over per-view conds. A real nn.Module
    (upstream sample_tex_slat isinstance-checks and .to()/.cpu() must cascade);
    every other attribute (in_channels, resolution, ...) delegates to the model."""

    def __init__(self, model):
        super().__init__()
        self._m = model

    def forward(self, x, t, cond, **kwargs):
        if isinstance(cond, MultiCond):
            pred = None
            for c in cond:
                p = self._m(x, t, c, **kwargs)
                pred = p if pred is None else pred + p
            return pred / len(cond)
        return self._m(x, t, cond, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("_m"), name)


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
        _patch_projgrid_allow_transform()
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )

        config_file = os.environ.get("PIXAL_PIPELINE_CONFIG", "pipeline.json")
        print(f"[pixal3d] loading {model_path} / {config_file} (low_vram={LOW_VRAM}) ...",
              flush=True)
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path, config_file=config_file)
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

    # -- multi-view cond builders: mirror pipeline.get_proj_cond_ss/_shape exactly,
    # -- but with an explicit per-view camera transform_matrix ------------------

    @torch.no_grad()
    def _view_cond_ss(self, image, camera_angle_x, distance, mesh_scale, transform):
        p = self.pipeline
        device = p.device
        m = p.image_cond_model_ss
        if p.low_vram:
            m.to(device)
        z_global, z_proj = m(
            [image],
            camera_angle_x=torch.tensor([camera_angle_x], device=device),
            distance=torch.tensor([distance], device=device),
            mesh_scale=torch.tensor([mesh_scale], device=device),
            transform_matrix=transform.unsqueeze(0).to(device),
        )
        if p.low_vram:
            m.cpu()
        return {"global": z_global, "proj": z_proj}

    @torch.no_grad()
    def _view_cond_stage(self, cond_model, image, coords, camera_angle_x, distance,
                         mesh_scale, transform, grid_resolution_override=None):
        from pixal3d.modules.sparse import SparseTensor
        p = self.pipeline
        device = p.device
        if p.low_vram:
            cond_model.to(device)
        orig_res = cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_res:
            cond_model.grid_resolution = grid_resolution_override
            cond_model.proj_grid = cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=cond_model.proj_grid.image_resolution).to(device)
        z_global, z_proj = cond_model(
            [image],
            camera_angle_x=torch.tensor([camera_angle_x], device=device),
            distance=torch.tensor([distance], device=device),
            mesh_scale=torch.tensor([mesh_scale], device=device),
            transform_matrix=transform.unsqueeze(0).to(device),
        )
        grid_res = cond_model.grid_resolution
        z_grid = z_proj.reshape(1, grid_res, grid_res, grid_res, -1)
        z_sparse = z_grid[coords[:, 0].long(), coords[:, 1].long(),
                          coords[:, 2].long(), coords[:, 3].long()]
        st = SparseTensor(feats=z_sparse, coords=coords)
        if grid_resolution_override is not None and grid_resolution_override != orig_res:
            cond_model.grid_resolution = orig_res
            cond_model.proj_grid = cond_model.proj_grid.__class__(
                grid_resolution=orig_res,
                image_resolution=cond_model.proj_grid.image_resolution).to(device)
        if p.low_vram:
            cond_model.cpu()
        return {"global": z_global, "proj": st}, z_sparse

    @torch.no_grad()
    def _run_multiview(self, views, camera_params, azimuths, seed, guided_override,
                       tex_override, pipeline_type, max_num_tokens):
        """Mirror of Pixal3DImageTo3DPipeline.run(), with multi-cond guidance: every
        sampling stage averages the flow model's velocity across per-view conds."""
        from pixal3d.modules.sparse import SparseTensor
        p = self.pipeline
        device = p.device
        hr_resolution = 1536 if pipeline_type == "1536_cascade" else 1024
        cax = camera_params["camera_angle_x"]
        dist = camera_params["distance"]
        scale = camera_params.get("mesh_scale", 1.0)
        transforms = [view_transform_matrix(dist, az) for az in azimuths]

        torch.manual_seed(seed)

        # ---- Stage 1: sparse structure, multi-cond ----
        per_view = [self._view_cond_ss(v, cax, dist, scale, T)
                    for v, T in zip(views, transforms)]
        neg = {"global": torch.zeros_like(per_view[0]["global"]),
               "proj": torch.zeros_like(per_view[0]["proj"])}
        cond_ss = {"cond": MultiCond(per_view), "neg_cond": neg}
        flow_ss = p.models["sparse_structure_flow_model"]
        try:
            p.models["sparse_structure_flow_model"] = MultiCondModel(flow_ss)
            coords = p.sample_sparse_structure(cond_ss, 32, 1, guided_override)
        finally:
            p.models["sparse_structure_flow_model"] = flow_ss
        del cond_ss, per_view
        torch.cuda.empty_cache()

        # ---- Stage 2: shape LR 512, multi-cond ----
        pv = [self._view_cond_stage(p.image_cond_model_shape_512, v, coords,
                                    cax, dist, scale, T)[0]
              for v, T in zip(views, transforms)]
        neg = {"global": torch.zeros_like(pv[0]["global"]),
               "proj": SparseTensor(feats=torch.zeros_like(pv[0]["proj"].feats),
                                    coords=coords)}
        cond_lr = {"cond": MultiCond(pv), "neg_cond": neg}
        lr_slat = p.sample_shape_slat(
            cond_lr, MultiCondModel(p.models["shape_slat_flow_model_512"]),
            coords, guided_override)
        del cond_lr, pv
        torch.cuda.empty_cache()

        # ---- Stage 3a: upsample LR -> HR, token-limit loop (verbatim run() logic) ----
        if p.low_vram:
            p.models["shape_slat_decoder"].to(device)
            p.models["shape_slat_decoder"].low_vram = True
        hr_coords = p.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        if p.low_vram:
            p.models["shape_slat_decoder"].cpu()
            p.models["shape_slat_decoder"].low_vram = False
        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            if hr_coords_unique.shape[0] < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128
        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: shape HR, multi-cond ----
        pv = [self._view_cond_stage(p.image_cond_model_shape_1024, v, hr_coords_unique,
                                    cax, dist, scale, T,
                                    grid_resolution_override=actual_grid_res)[0]
              for v, T in zip(views, transforms)]
        neg = {"global": torch.zeros_like(pv[0]["global"]),
               "proj": SparseTensor(feats=torch.zeros_like(pv[0]["proj"].feats),
                                    coords=hr_coords_unique)}
        flow_hr = p.models["shape_slat_flow_model_1024"]
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], flow_hr.in_channels).to(device),
            coords=hr_coords_unique)
        sampler_params_hr = {**p.shape_slat_sampler_params, **guided_override}
        if p.low_vram:
            flow_hr.to(device)
        hr_slat = p.shape_slat_sampler.sample(
            MultiCondModel(flow_hr), noise_hr,
            cond=MultiCond(pv), neg_cond=neg,
            **sampler_params_hr, verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (multi-view, {actual_hr_resolution})",
        ).samples
        if p.low_vram:
            flow_hr.cpu()
        std = torch.tensor(p.shape_slat_normalization["std"])[None].to(hr_slat.device)
        mean = torch.tensor(p.shape_slat_normalization["mean"])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        del pv, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: texture, multi-cond ----
        tex_grid_res = actual_hr_resolution // 16
        pv = [self._view_cond_stage(p.image_cond_model_tex_1024, v, shape_slat.coords,
                                    cax, dist, scale, T,
                                    grid_resolution_override=tex_grid_res)[0]
              for v, T in zip(views, transforms)]
        neg = {"global": torch.zeros_like(pv[0]["global"]),
               "proj": SparseTensor(feats=torch.zeros_like(pv[0]["proj"].feats),
                                    coords=shape_slat.coords)}
        cond_tex = {"cond": MultiCond(pv), "neg_cond": neg}
        tex_slat = p.sample_tex_slat(
            cond_tex, MultiCondModel(p.models["tex_slat_flow_model_1024"]),
            shape_slat, tex_override)
        del cond_tex, pv
        torch.cuda.empty_cache()

        # ---- Stage 5: decode ----
        res = actual_hr_resolution
        out_mesh = p.decode_latent(shape_slat, tex_slat, res)
        return out_mesh, res

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

        azimuths = params.get("view_azimuths_deg")
        if azimuths is None:
            azimuths = [0.0, 180.0, 90.0, 270.0][:len(views)]
        if len(azimuths) != len(views):
            raise ValueError(
                f"view_azimuths_deg has {len(azimuths)} entries for {len(views)} views - "
                "provide one azimuth (degrees about the up axis, 0=front) per view")

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
            views_pre = [self.pipeline.preprocess_image(v) for v in views]
            image_preprocessed = views_pre[0]

            manual_fov = float(params.get("manual_fov", -1.0))
            if manual_fov > 0:
                camera_params = camera_params_from_fov(manual_fov, mesh_scale)
            else:
                camera_params = self._estimate_camera(image_preprocessed, mesh_scale)
            print(f"[pixal3d] uid {uid}: camera {camera_params} views={len(views_pre)} "
                  f"azimuths={azimuths[:len(views_pre)]}", flush=True)

            if len(views_pre) > 1:
                mesh_list, res = self._run_multiview(
                    views_pre, camera_params, azimuths, seed,
                    guided_override, tex_override, pipeline_type,
                    int(params.get("max_num_tokens", 49152)))
                warnings.append(
                    f"multi-view mode: multi-cond guidance over {len(views_pre)} views "
                    f"at azimuths {azimuths} (per-step velocity averaging; implemented in "
                    "this container - upstream Pixal3D releases no MV inference code). "
                    "Set env PIXAL_PIPELINE_CONFIG=pipeline_mv.json for the _mv weights.")
            else:
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
