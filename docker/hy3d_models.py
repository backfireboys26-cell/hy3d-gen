"""hy3d_models.py - the ONE catalog of models the hy3d-gen container serves, plus the weight-cache
check / prefetch CLI start.sh runs before the api_server boots.

Two holders of "which repo / subfolder / files is model X" used to be a bug class (start.sh's
case-statement VAE mapping vs the api_server's argparse default); this module is the single
holder: api_server.py imports it (model selection, /ping's model list, VRAM estimates) and
start.sh shells out to its CLI (cache completeness for the offline decision, the prefetch).

stdlib-only at import, so start.sh can run `check` before the ML stack is touched and
tests/test-start-sh.sh can drive it against a fake cache; `prefetch` imports huggingface_hub
lazily.

Served set = the single-image model named by MODEL_PATH + HY3D_SUBFOLDER (always served - it is
the default for an {"image"} body) + every name in HY3D_MODELS (comma list of catalog names or
their short aliases; default DEFAULT_MODELS). An unknown name is a ValueError: start.sh exits 3
naming it rather than booting with fewer models than the operator asked for.

CLI (env: HF_HOME, MODEL_PATH, HY3D_SUBFOLDER, HY3D_MODELS):
  python hy3d_models.py list      -> JSON {models, defaults, labels, components}
  python hy3d_models.py check     -> one line: "COMPLETE <components>" or "MISSING <path> ..."
  python hy3d_models.py prefetch  -> hf_hub_download of every missing file (exit 0 even when
                                     some fail - start.sh re-checks and stays online then)
"""
import json
import os
import sys
from collections import OrderedDict

# vram_mb = resident fp16 weights (the dit's safetensors carries model + conditioner + VAE; the
# turbo VAE enable_flashvdm swaps in is ~0.4 GB more) - what the LRU needs free before loading
CATALOG = OrderedDict([
    ("hunyuan3d-dit-v2-0", {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-dit-v2-0",
                            "kind": "single", "vram_mb": 5400}),
    ("hunyuan3d-dit-v2-mini-turbo", {"repo": "tencent/Hunyuan3D-2mini", "subfolder": "hunyuan3d-dit-v2-mini-turbo",
                                     "kind": "single", "vram_mb": 4300}),
    ("hunyuan3d-dit-v2-mini", {"repo": "tencent/Hunyuan3D-2mini", "subfolder": "hunyuan3d-dit-v2-mini",
                               "kind": "single", "vram_mb": 4300}),
    ("hunyuan3d-dit-v2-mv", {"repo": "tencent/Hunyuan3D-2mv", "subfolder": "hunyuan3d-dit-v2-mv",
                             "kind": "mv", "vram_mb": 5400}),
    ("hunyuan3d-dit-v2-mv-turbo", {"repo": "tencent/Hunyuan3D-2mv", "subfolder": "hunyuan3d-dit-v2-mv-turbo",
                                   "kind": "mv", "vram_mb": 5400}),
    # Zero123-XL view synthesis (POST /imagine): fp32 .bin on the hub, cast to fp16 at load;
    # measured peak 2841 MB on a GTX 1070 at 256 px
    ("zero123-xl", {"repo": "kxic/zero123-xl", "subfolder": None, "kind": "nvs", "vram_mb": 3000}),
])
DEFAULT_MODELS = "dit-v2-0,dit-v2-mv,dit-v2-mv-turbo,zero123-xl"
_ALIASES = {"zero123": "zero123-xl", "kxic/zero123-xl": "zero123-xl"}
# the turbo VAE hy3dgen's enable_flashvdm swaps in per dit repo (pipelines.py turbo_vae_mapping)
TURBO_VAE = {
    "Hunyuan3D-2": ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"),
    "Hunyuan3D-2mv": ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"),
    "Hunyuan3D-2mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-vae-v2-mini-turbo"),
}
DIT_FILES = ("config.yaml", "model.fp16.safetensors")
# exactly what the manual component assembly in zero123/nvs.py opens (no safety_checker: 1.2 GB
# the pipeline is built without)
ZERO123_FILES = (
    "model_index.json",
    "cc_projection/config.json", "cc_projection/diffusion_pytorch_model.bin",
    "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json", "image_encoder/pytorch_model.bin",
    "scheduler/scheduler_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.bin",
    "vae/config.json", "vae/diffusion_pytorch_model.bin",
)


def canonical(name):
    """Catalog key for a model name or alias ('dit-v2-mv' -> 'hunyuan3d-dit-v2-mv'), else None."""
    t = (name or "").strip().lower()
    if not t:
        return None
    t = _ALIASES.get(t, t)
    if t in CATALOG:
        return t
    if "hunyuan3d-" + t in CATALOG:
        return "hunyuan3d-" + t
    return None


def served(model_path=None, subfolder=None, models=None):
    """OrderedDict name -> spec for this process, from the arguments or the environment. The
    single-image model named by model_path/subfolder comes first (the {"image"} default); an
    unknown HY3D_MODELS entry raises ValueError naming it."""
    model_path = model_path or os.environ.get("MODEL_PATH") or "tencent/Hunyuan3D-2mini"
    subfolder = subfolder or os.environ.get("HY3D_SUBFOLDER") or "hunyuan3d-dit-v2-mini-turbo"
    models = os.environ.get("HY3D_MODELS", DEFAULT_MODELS) if models is None else models
    out = OrderedDict()
    single = canonical(subfolder)
    if single is not None and CATALOG[single]["repo"] == model_path:
        out[single] = dict(CATALOG[single])
    else:
        # a dit the catalog does not know (an operator's own weights): served as a single-image
        # model under its subfolder name; its VAE mapping may be unknown (the check says so)
        out[subfolder] = {"repo": model_path, "subfolder": subfolder, "kind": "single", "vram_mb": 5400}
    unknown = []
    for raw in [m.strip() for m in models.split(",") if m.strip()]:
        name = canonical(raw)
        if name is None:
            unknown.append(raw)
        elif name not in out:
            out[name] = dict(CATALOG[name])
    if unknown:
        raise ValueError(f"HY3D_MODELS names unknown model(s) {unknown}; known: {', '.join(CATALOG)}")
    return out


def defaults(served_models):
    """{"image": <single>, "views": <mv or None>, "imagine": <nvs or None>} - the model a body
    without 'model' gets: the first single, the first NON-turbo mv (quality by default; turbo is
    an explicit choice), the first nvs."""
    names = list(served_models)
    single = next((n for n in names if served_models[n]["kind"] == "single"), None)
    mvs = [n for n in names if served_models[n]["kind"] == "mv"]
    mv = next((n for n in mvs if not n.endswith("-turbo")), mvs[0] if mvs else None)
    nvs = next((n for n in names if served_models[n]["kind"] == "nvs"), None)
    return {"image": single, "views": mv, "imagine": nvs}


def labels(served_models):
    """The 'model' / 'subfolder' strings /ping keeps for older clients: every served dit repo and
    subfolder, comma-joined. Ordered for generate3d.py's readers (2026-09-03 client): its
    positive multiview guard takes the LAST '/'-segment of 'model' (so the mv repo goes last:
    'tencent/Hunyuan3D-2, tencent/Hunyuan3D-2mv' -> 'hunyuan3d-2mv', a known mv id) and its step
    ladder follows whether 'subfolder' ENDS with '-turbo' (so a turbo subfolder never goes
    last when a non-turbo one is served: the quality ladder is the safe default)."""
    dits = [n for n, s in served_models.items() if s["kind"] != "nvs"]
    repos = []
    for n in dits:
        r = served_models[n]["repo"]
        if r not in repos:
            repos.append(r)
    # the index is captured BEFORE the sort: repos.index(r) inside the key read a list that
    # sort() was already permuting ("'tencent/Hunyuan3D-2mini' is not in list")
    order = {r: i for i, r in enumerate(repos)}
    repos.sort(key=lambda r: ("mv" in r.rsplit("/", 1)[-1].lower(), order[r]))
    subs = [served_models[n]["subfolder"] for n in dits]
    ordered = subs[:1] + sorted(subs[1:], key=lambda s: (not s.endswith("-turbo"), s))
    return {"model": ", ".join(repos), "subfolder": ", ".join(ordered)}


def required_files(served_models):
    """[(repo, relpath)] every served model opens at load, plus the turbo VAE per dit repo, in a
    stable order; ('', '(unknown VAE mapping for <repo>)') marks a dit repo without a mapping."""
    files, seen = [], set()

    def add(repo, rel):
        if (repo, rel) not in seen:
            seen.add((repo, rel))
            files.append((repo, rel))
    for name, spec in served_models.items():
        if spec["kind"] == "nvs":
            for rel in ZERO123_FILES:
                add(spec["repo"], rel)
            continue
        for f in DIT_FILES:
            add(spec["repo"], f"{spec['subfolder']}/{f}")
    for name, spec in served_models.items():
        if spec["kind"] == "nvs":
            continue
        vae = TURBO_VAE.get(spec["repo"].rsplit("/", 1)[-1])
        if vae is None:
            add("", f"(unknown VAE mapping for {spec['repo']})")
        else:
            for f in DIT_FILES:
                add(vae[0], f"{vae[1]}/{f}")
    return files


def components(served_models):
    """'<dit subfolders> + <vae subfolders> + <nvs names>' for the start.sh 'complete' line."""
    out = []
    for name, spec in served_models.items():
        if spec["kind"] != "nvs" and spec["subfolder"] not in out:
            out.append(spec["subfolder"])
    for name, spec in served_models.items():
        if spec["kind"] != "nvs":
            vae = TURBO_VAE.get(spec["repo"].rsplit("/", 1)[-1])
            if vae is not None and vae[1] not in out:
                out.append(vae[1])
    for name, spec in served_models.items():
        if spec["kind"] == "nvs" and name not in out:
            out.append(name)
    return " + ".join(out)


def snapshot_dir(hf_root, repo):
    """The refs/main snapshot dir of a cached repo, or None (a dangling refs/main is None too)."""
    repo_dir = os.path.join(hf_root, "hub", "models--" + repo.replace("/", "--"))
    try:
        with open(os.path.join(repo_dir, "refs", "main"), encoding="utf-8") as f:
            rev = f.read().strip()
    except OSError:
        return None
    snap = os.path.join(repo_dir, "snapshots", rev)
    return snap if rev and os.path.isdir(snap) else None


def cache_status(hf_root, served_models):
    """(present, missing) as 'repo/relpath' strings; a file counts as present when it is in the
    refs/main snapshot with a non-zero size (presence + size, not content - as start.sh always
    judged it)."""
    present, missing, snaps = [], [], {}
    for repo, rel in required_files(served_models):
        if not repo:
            missing.append(rel)
            continue
        if repo not in snaps:
            snaps[repo] = snapshot_dir(hf_root, repo)
        snap = snaps[repo]
        path = os.path.join(snap, rel) if snap else None
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            present.append(f"{repo}/{rel}")
        else:
            missing.append(f"{repo}/{rel}")
    return present, missing


def prefetch(served_models, hf_root):
    """hf_hub_download every missing file, one at a time, each failure logged and skipped:
    returns the count that still fail. Downloads exactly the files the loaders open (config +
    fp16 safetensors per dit subfolder, the turbo VAE, the zero123 components) - never the whole
    subfolder hy3dgen's snapshot_download(allow_patterns=subfolder/*) would pull (the .ckpt
    twins: 4.9 GB x 3 extra per dit)."""
    from huggingface_hub import hf_hub_download
    _, missing = cache_status(hf_root, served_models)
    failed = 0
    for repo, rel in required_files(served_models):
        if not repo or f"{repo}/{rel}" not in missing:
            continue
        print(f"[prefetch] {repo}/{rel} ...", flush=True)
        try:
            path = hf_hub_download(repo_id=repo, filename=rel)
            print(f"[prefetch] ok {repo}/{rel} ({os.path.getsize(path)} B)", flush=True)
        except Exception as e:      # noqa: BLE001 - every failure is reported, none is fatal here
            failed += 1
            print(f"[prefetch] FAILED {repo}/{rel}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return failed


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    hf_root = os.environ.get("HF_HOME") or "/runpod-volume/hf"
    try:
        s = served()
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        return 3
    if cmd == "list":
        print(json.dumps({"models": list(s), "defaults": defaults(s), "labels": labels(s),
                          "components": components(s), "hf_root": hf_root}))
        return 0
    if cmd == "check":
        present, missing = cache_status(hf_root, s)
        print(("COMPLETE " + components(s)) if not missing else ("MISSING " + " ".join(missing)))
        return 0
    if cmd == "prefetch":
        failed = prefetch(s, hf_root)
        _, missing = cache_status(hf_root, s)
        print(f"[prefetch] done: {failed} failed, {len(missing)} still missing", flush=True)
        return 0
    print(f"usage: {argv[0]} list|check|prefetch", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
