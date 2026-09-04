"""Torch-only stand-in for the two kornia calls pipeline_zero1to3.py makes (CLIP preprocessing):
kornia.geometry.resize(x, size, interpolation=, align_corners=, antialias=) and
kornia.enhance.normalize(x, mean, std). Real kornia is not installed in C:\ai3d\venv and must not
be added (pinned diffusers 0.32.2 env shared with the shape server)."""
import types
import torch
import torch.nn.functional as F


def _resize(x, size, interpolation="bilinear", align_corners=None, antialias=False, **_):
    if isinstance(size, int):
        size = (size, size)
    kw = {"mode": interpolation, "antialias": bool(antialias)}
    if interpolation in ("bilinear", "bicubic") and align_corners is not None:
        kw["align_corners"] = bool(align_corners)
    return F.interpolate(x, size=tuple(size), **kw)


def _normalize(x, mean, std):
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


geometry = types.SimpleNamespace(resize=_resize)
enhance = types.SimpleNamespace(normalize=_normalize)
