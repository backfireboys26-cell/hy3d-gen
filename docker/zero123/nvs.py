"""nvs.py - Zero123-XL novel-view synthesis for the hy3d-gen container (POST /imagine): one photo
in, the object's left / right / back views at elevation 0 out, to feed the multiview dit.

The recipe is the one proven on rsv4 (engine/generation/servers/zero123/zero123_views.py,
2026-08-25): MANUAL COMPONENT ASSEMBLY from the kxic/zero123-xl snapshot - diffusers 0.32.2 cannot
resolve the repo's own `cc_projection/pipeline_zero1to3.py` reference, so the community pipeline
file version-matched to 0.32.2 sits beside this module and is loaded from that local path, every
component is loaded fp16 (a single fp32 component ends in a mat1/mat2 dtype crash at
cc_projection), attention slicing on, no xformers/flash (Pascal-safe; a 4090 does not need them).
`kornia` is the two-function torch shim beside this file (kornia is not in the image's pins).

Only the files hy3d_models.ZERO123_FILES lists are opened; the snapshot is resolved through
snapshot_download with exactly that allow list, so an offline boot (HF_HUB_OFFLINE=1) finds the
prefetched cache and an online one downloads nothing more than those files.
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Zero123 pose = [elevation delta, azimuth, radius delta]; the input photo is the front (0, 0, 0)
POSES = {"right": (0.0, 90.0, 0.0), "back": (0.0, 180.0, 0.0), "left": (0.0, 270.0, 0.0)}
VIEWS = tuple(POSES)


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pipeline_module():
    try:
        import kornia  # noqa: F401 - a real install wins if one ever appears in the pins
    except ImportError:
        _load_module("kornia", os.path.join(HERE, "kornia.py"))
    return _load_module("pipeline_zero1to3", os.path.join(HERE, "pipeline_zero1to3.py"))


class Zero123Views:
    def __init__(self, repo="kxic/zero123-xl", allow_patterns=None, device="cuda"):
        self.repo = repo
        self.allow_patterns = list(allow_patterns) if allow_patterns else None
        self.device = device
        self.pipe = None
        self.snapshot = None

    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        pz = _pipeline_module()
        t0 = time.time()
        self.snapshot = snapshot_download(self.repo, allow_patterns=self.allow_patterns)
        dt = torch.float16
        vae = AutoencoderKL.from_pretrained(self.snapshot, subfolder="vae", torch_dtype=dt)
        unet = UNet2DConditionModel.from_pretrained(self.snapshot, subfolder="unet", torch_dtype=dt)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.snapshot, subfolder="image_encoder", torch_dtype=dt)
        feature_extractor = CLIPImageProcessor.from_pretrained(self.snapshot, subfolder="feature_extractor")
        scheduler = DDIMScheduler.from_pretrained(self.snapshot, subfolder="scheduler")
        cc_projection = pz.CCProjection.from_pretrained(self.snapshot, subfolder="cc_projection", torch_dtype=dt)
        for m in (vae, unet, image_encoder, cc_projection):
            m.to(dtype=dt)
        pipe = pz.Zero1to3StableDiffusionPipeline(
            vae=vae, image_encoder=image_encoder, unet=unet, scheduler=scheduler, safety_checker=None,
            feature_extractor=feature_extractor, cc_projection=cc_projection, requires_safety_checker=False)
        pipe.to(self.device)
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        return time.time() - t0

    def unload(self):
        self.pipe = None

    @staticmethod
    def condition(image, size):
        """Zero123 wants the object centred on WHITE: alpha (rembg's cut-out) composited on white,
        RGB, resized to the working square."""
        from PIL import Image
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            image = bg
        else:
            image = image.convert("RGB")
        return image.resize((size, size), Image.LANCZOS)

    def imagine(self, image, views, size=256, steps=75, guidance_scale=3.0, seed=0, elevation=0.0):
        """{view: PIL RGB image} for each requested view, generated in order with ONE generator
        seeded once (so a run is reproducible per seed and the views differ from each other)."""
        import torch
        if self.pipe is None:
            raise RuntimeError("zero123 pipeline not loaded")
        cond = self.condition(image, size)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = {}
        for name in views:
            el, az, r = POSES[name]
            result = self.pipe(input_imgs=cond, prompt_imgs=cond, poses=[[float(elevation) + el, az, r]],
                               height=size, width=size, num_inference_steps=int(steps),
                               guidance_scale=float(guidance_scale), num_images_per_prompt=1, generator=gen)
            out[name] = result.images[0].convert("RGB")
        return out
