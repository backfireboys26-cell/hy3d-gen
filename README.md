# hy3d-gen

Self-hosted Hunyuan3D-2 generation container (open weights, our patches, bearer-gated HTTP API).
Built from the open-source Hunyuan3D-2GP codebase pinned @ f2456e0, with local patches
(--subfolder model selection, honest async errors, octree guard). Weights are NOT baked;
mount a volume and set HF_HOME. See docker/README.md for run details.
Published images (GHCR): floating lane tags `cu124` / `hy3d21` / `trellis2` / `pixal3d` plus an
IMMUTABLE `<lane>-<utc date>-<short sha>` per build (CI: .github/workflows/build.yml). The RunPod
endpoint is pinned to an immutable `cu124-*` tag only. Python deps are pinned by docker/constraints.txt.
Contract tests (no GPU) live in tests/ and are baked into every image at /app/tests - see tests/README.md.
