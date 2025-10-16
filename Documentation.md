# SAMplify_SuGaR — Detailed Technical Documentation

## Overview

SAMplify_SuGaR is a two-stage image-to-3D pipeline that combines interactive 2D segmentation (SAM2) with GPU-accelerated Gaussian-splatting reconstruction (SuGaR). The goal is to produce clean, background-free 3D geometry from photographic datasets with minimal manual effort: precise foreground masks are generated with SAM2 and are converted to dense, continuous 3D meshes by SuGaR.

This document describes architecture, data flow, installation, configuration, usage examples, internals, recommended parameters, troubleshooting, and future improvements.

---

## Table of Contents

1. Overview
2. Architecture and Data Flow
3. Data Formats and Directory Layout
4. Installation
   - Requirements
   - Repositories and dependencies
   - Docker + GPU setup
5. Configuration & Environment
6. Pipeline Workflow (detailed)
7. Commands & Examples
8. Implementation Details
   - SAM2: annotation and mask generation
   - SuGaR: pointcloud, splats, and surface extraction
9. Performance & Tuning
10. Troubleshooting
11. Limitations and Known Issues
12. Roadmap & Future Work
13. References

---

## Quick links

- [Overview](#overview)
- [Architecture & Data Flow](#architecture-and-data-flow)
- [Data Formats & Layout](#data-formats-and-directory-layout)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow-detailed)
- [Commands & Examples](#commands--examples)
- [Troubleshooting](#troubleshooting)

---

## 2. Architecture and Data Flow

High-level components:

- Input: photographic images (single or multi-view).
- SAM2: interactive segmentation (user-provided positive/negative points) → binary masks.
- Mask-to-pointcloud: convert mask pixels into 3D samples (single-view heuristics or multi-view triangulation if camera poses available).
- SuGaR: represent points as Gaussian splats, aggregate density field, extract surface (Poisson / Marching Cubes).
- Postprocessing: smoothing, decimation, normals, optional texture baking.

Data flow diagram (logical):

images → preprocessing → SAM2 annotation → masks → point sampling → splatting (SuGaR) → density field → surface extraction → mesh → postprocessing

---

## 3. Data Formats and Directory Layout

Recommended repository layout (root = $SAMPLIFY_SUGAR_PATH):

SAMplify_SuGaR/
- data/
  - datasets/
    - <dataset_name>/
      - images/            # **.jpg** images
      - cameras/           # optional camera intrinsics / extrinsics (JSON, COLMAP files, etc.)
- externals/
  - SAM2/                 # optional clone/submodule for SAM integration
  - SuGaR/                # optional clone/submodule for SuGaR
- colmap/                # COLMAP wrapper for SfM and dense stereo (included in this repo)
- scripts/                # glue scripts and CLI wrappers (e.g. annotation, conversion, reconstruction)
- results/
  - <dataset_name>/
    - masks/
    - pointclouds/
    - splats/
    - meshes/
- docs/
  - README.md (this file)

File conventions (brief):
- **Images**: `.jpg` (RGB). Use unique, zero-padded filenames for ordering.
- **Masks**: binary PNG or `.npy` arrays with same HxW as source image (`0` background, `1` foreground).
- **Pointclouds**: PLY or `.npy` arrays `(x,y,z[,r,g,b])`.
- **Meshes**: `.ply` or `.obj` with normals; UVs/textures optional.

> Note: the `scripts/` directory contains _tool wrappers_. Avoid hard-coded references to non-existent filenames — adapt commands below to the tools available in your copy of the repo.

---

## 4. Installation

### Requirements

- **OS**: Linux (Ubuntu tested).
- **GPU**: NVIDIA with compatible drivers (recommended).
- **Containers**: Docker + NVIDIA Container Toolkit for GPU passthrough.
- **Python**: 3.8+ (for local utilities).
- **Hardware**: 8GB+ RAM (16GB recommended), disk for data.

### Clone the project

Replace `<repo-url>` with the URL for this repository and clone it to your machine:

```bash
git clone <repo-url> $SAMPLIFY_SUGAR_PATH
cd $SAMPLIFY_SUGAR_PATH
```

### Python environment (optional)

Create a virtualenv if you will run local scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present
```

### Docker & NVIDIA toolkit (Ubuntu example)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Verify GPU inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

If using GUI-based SAM2 UIs, ensure X11 socket and DISPLAY forwarded and that host allows the container to connect to X server (e.g. xhost +local:root).

---

## 5. Configuration & Environment

Set environment variables used by scripts/wrappers:

- SAM_FIT_SUGAR_PATH — path to the project root (absolute).
- DATASET — default dataset name to run on.
- DOCKER_IMAGE_SAM2 — (optional) override SAM2 docker image tag.
- DOCKER_IMAGE_SUGAR — (optional) override SuGaR docker image tag.

Example:

```bash
export SAM_FIT_SUGAR_PATH="/home/you/SAMplify_SuGaR"
export DATASET="my_dataset"
export DOCKER_IMAGE_SAM2="peasant98/sam2:cuda-12.1"
export DOCKER_IMAGE_SUGAR="ant_two/sugar:latest"
```

---

## 6. Pipeline Workflow (detailed)

1. Prepare dataset: place images under data/datasets/<dataset_name>/.
2. Preprocess images (optional): resizing, color normalization, or lens correction.
3. Annotate using SAM2 UI (interactive): add positive and negative points. Save masks to results/<dataset>/masks/.
   - Positive points: indicate foreground (object).
   - Negative points: indicate background/holes.
4. Convert masks to sampled pointclouds.
   - Single-view basic approach: backproject silhouette to approximate depth surface.
   - Recommended for accurate geometry: use calibrated multi-view images and triangulate mask outlines or dense stereo.
5. Run SuGaR reconstruction: create Gaussian splats per sample, aggregate, then extract surface.
6. Postprocess mesh: smoothing, decimation, normal recomputation, and optional texture baking.

---

## 7. Commands & Examples

### Run SAM2 (Docker) — interactive annotation

```bash
# mount project and X socket, run SAM2 docker image
docker run -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $SAM_FIT_SUGAR_PATH/data:/workspace/data \
  -v $SAM_FIT_SUGAR_PATH/results:/workspace/results \
  -e DISPLAY=$DISPLAY \
  --gpus all \
  ${DOCKER_IMAGE_SAM2:-peasant98/sam2:cuda-12.1} bash
# inside container run SAM2 UI pointing to /workspace/data/datasets/<dataset>/images
```

When GUI appears, annotate images and save masks to /workspace/results/<dataset>/masks/.

### Convert masks to pointcloud (example script)

```bash
python scripts/convert_masks_to_pointcloud.py \
  --masks results/$DATASET/masks \
  --out results/$DATASET/pointclouds \
  --sampling 1.0
```

Options:
- --sampling: pixel subsampling factor (1.0 = every pixel; 2.0 = every 2nd pixel, etc.).

### Run SuGaR reconstruction (Docker)

```bash
docker run -it \
  -v $SAM_FIT_SUGAR_PATH/data:/workspace/data \
  -v $SAM_FIT_SUGAR_PATH/results:/workspace/results \
  --gpus all \
  ${DOCKER_IMAGE_SUGAR:-ant_two/sugar:latest} bash
# inside container
python sugar_reconstruct.py --input results/$DATASET/pointclouds --out results/$DATASET/meshes --poisson-depth 10
```

Key SuGaR reconstruction parameters:
- poisson-depth: controls Poisson reconstruction resolution (higher = more detail, more memory).
- splat-radius: initial radius of Gaussian splats.
- splat-density-threshold: threshold when extracting surface from density field.

---

## 8. Implementation Details

### SAM2: annotation and mask generation

- Interaction: left-click = positive point (foreground), right-click = negative point (background). Each annotation is a (x, y) coordinate with label.
- Seed-based region growing: SAM2 uses annotated seeds to initialize a foreground region. A flood-fill/region-growing algorithm expands the region using color/intensity similarity and edge cues.
- Contour refinement: after initial region growth, boundary refinement (e.g., active contours or edge alignment) improves mask accuracy.
- Output mask: binary HxW array saved as PNG or .npy (0 = background, 1 = foreground).

Additional notes on video / sequence workflows

- SAM Video Prediction (mask propagation): when input is a video or ordered image sequence, use a propagation tool to automatically extend annotations across frames. Typical workflow:
  1. Annotate key frames (first frame or several keyframes).
  2. Run mask propagation to predict masks for intermediate frames.
  3. Inspect and correct poor predictions by adding corrective points and re-propagating if necessary.
  4. Save resulting masks into `results/<dataset>/video_masks/` as a time-ordered set of PNGs or .npy files.

- Integration with 3D pipeline: per-frame masks from video propagation can be converted to per-frame point samples and then fused in SuGaR (temporal fusion or multi-view triangulation) to improve geometry stability and coverage.

References and resources

- Segment Anything Model (official): https://github.com/facebookresearch/segment-anything
- Example community projects (search GitHub): "Video-SAM", "VideoSAM", "sam-video" — these provide mask propagation implementations and GUI wrappers for temporal annotation.

### SuGaR: Gaussian Splatting and Surface Extraction

- For each 3D sample, create a Gaussian splat defined by position, covariance (or scalar radius), color, and weight.
- Aggregate splats into a volumetric density field by evaluating contributions on a grid (or using an adaptive octree).
- Extract an isosurface from the density field using Marching Cubes or convert aggregated splats to a point set and use Poisson Surface Reconstruction.
- Post-extraction: clean small components, recompute normals, smooth, decimate.

Performance considerations:
- Splats per sample: large numbers of splats increase memory and compute; subsample if necessary.
- Use GPU-accelerated kernels for splat rasterization and density aggregation where available.

---

## 9. Performance & Tuning

Tips to balance quality vs resource usage:
- Image resolution: reduce resolution when memory is constrained. Try 720p if 1080p is too large.
- Sampling: subsample mask pixels during pointcloud conversion (e.g., keep 1/2 or 1/4 pixels).
- Poisson depth: lower values reduce memory and runtime.
- Splat radius: increase to smooth noisy pointclouds; decrease to capture fine detail.
- Run reconstruction on machines with high VRAM GPUs for best performance.

---

## 10. Troubleshooting

Problem: GPU not visible in container
- Verify host has drivers: run `nvidia-smi` on host.
- Ensure NVIDIA Container Toolkit installed and Docker restarted.
- Run container with `--gpus all` flag.

Problem: SAM2 GUI does not open from Docker
- Ensure X11 socket mounted: `-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`.
- Allow container to access X server: on host run `xhost +local:root` (security tradeoff).
- Alternatively, use VNC or X forwarding.

Problem: Masks are noisy or leaking
- Add more negative points around suspected leak areas.
- Use contour refinement settings in SAM2 (if exposed) to tighten boundaries.

Problem: Output mesh has holes or disconnected components
- Increase sampling density or triangulation accuracy.
- Run gap-filling postprocessing or use Poisson reconstruction with higher depth.

---

## 11. Limitations and Known Issues

- Single-view reconstruction is inherently ambiguous; multi-view data yields significantly better results.
- Very complex backgrounds or low contrast between object and background reduces mask accuracy.
- Reconstruction quality depends on annotation accuracy and image coverage.
- Large-scale datasets require significant GPU memory and storage.

---

## 12. Roadmap & Future Work

Planned improvements:
- Semi-automated point suggestion: use object detectors to propose initial positive/negative points.
- Native multi-view integration: **IMPLEMENTED** — a COLMAP wrapper is included in the repository under the `colmap/` subfolder. Use this wrapper to compute camera poses, run sparse and dense reconstruction, and export camera parameters / depth maps for downstream mask→pointcloud conversion.
- Texture baking and PBR export workflow.
- Adaptive splat subsampling and streaming reconstruction for very large scenes.

---

## 13. References

- SAM projects and interactive segmentation literature.
- Gaussian splatting and surface reconstruction (Poisson, Marching Cubes).
- External repos used as references: https://github.com/peasant98/SAM2-Docker, https://github.com/Anttwo/SuGaR

---

If you want, I can also:
- add a runnable `scripts/run_pipeline.sh` and example `docker-compose.yml`,
- add default config files under `configs/`,
- or create a concise `README.md` from this document.

