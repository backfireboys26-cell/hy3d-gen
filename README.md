# hy3d-gen

Self-hosted Hunyuan3D-2 generation container (open weights, our patches, bearer-gated HTTP API).
Built from the open-source Hunyuan3D-2GP codebase pinned @ f2456e0, with local patches
(--subfolder model selection, honest async errors, octree guard). Weights are NOT baked;
mount a volume and set HF_HOME. See docker/README.md for run details.
Published image: ghcr.io/backfireboys26-cell/hy3d-gen:cu124
