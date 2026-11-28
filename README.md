# WhiteMatterAnalysis GPU (`whitematteranalysis_gpu`)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pyproject.toml)
[![CUDA 12.4 / 12.8](https://img.shields.io/badge/CUDA-12.4%20%7C%2012.8-76B900.svg)](Dockerfile)
[![PyTorch + Triton](https://img.shields.io/badge/backend-PyTorch%20%2B%20Triton%20%7C%20CPU-ee4c2c.svg)](doc/acceleration.md)
[![Docker GHCR](https://img.shields.io/badge/GHCR-docker%20%7C%20apptainer-2496ED.svg)](.github/workflows/docker-publish.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-lightgrey.svg)](License.txt)

**WhiteMatterAnalysis GPU** is a high-performance Python toolkit for **diffusion MRI (dMRI) whole-brain streamline tractography registration, spectral fiber clustering, outlier removal, and automated anatomical white matter parcellation** (ORG, East-West, and NABA atlases).

Powered by **PyTorch**, custom **OpenAI Triton** GPU kernels, a vectorized multi-threaded **CPU** backend, and pure-Python/VTK transform hardening, it executes complete **rigid and nonrigid white matter parcellation in under 30 seconds per whole-brain tractogram—without requiring 3D Slicer or SlicerDMRI**.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Performance Benchmarks](#performance-benchmarks)
- [Three Ways to Use WhiteMatterAnalysis GPU](#three-ways-to-use-whitematteranalysis-gpu)
  - [1. Command-Line Interface (`pip` / Conda / `uv`)](#1-command-line-interface-pip--conda--uv)
  - [2. Python SDK (`whitematteranalysis.pipelines`)](#2-python-sdk-whitematteranalysispipelines)
  - [3. Docker & Apptainer / Singularity (HPC)](#3-docker--apptainer--singularity-hpc)
- [Supported White Matter Atlases](#supported-white-matter-atlases)
- [Environment Configuration Reference](#environment-configuration-reference)
- [Optional 3D Slicer Extension](#optional-3d-slicer-extension)
- [Documentation](#documentation)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [References & License](#references--license)

---

## Key Capabilities

- **Slicer-Free Rigid & Nonrigid ORG Parcellation**: End-to-end single-process pipeline (`registration -> spectral clustering -> outlier removal -> affine/BSpline transform hardening -> hemisphere separation -> anatomical tract assembly`) using pure Python and VTK, with no 3D Slicer or Xvfb dependency for tensor-free tractograms.
- **Custom CUDA & Triton Kernels (`float64` default, opt-in `float32`)**:
  - Fused 2D fiber distance, probability, spectral embedding, and split-C low-VRAM nearest-centroid kernels supporting arbitrary `1 <= P <= 64` fiber point counts.
  - Fused `O(N + R^3)` Triton nonrigid B-spline analytic gradient backward kernel reducing GPU VRAM by **64x–139x** at high resolution.
  - Multi-GPU dispatch for spectral labeling and cluster outlier removal, plus GPU-batched `kmeans2` and PSD-skip optimization for spectral atlas construction.
- **Zero-Intermediate Fast Path & Binary Atlas Caching**:
  - `--keep-intermediates 0` (default) operates on indexed streamline partitions without materializing 800 intermediate cluster VTP files.
  - Automatic `1.2 MiB` `.npz` spectral atlas inference cache and outlier reference cache under `WMA_ATLAS_CACHE`.
  - Persistent multi-subject batch engine reusing registration atlases, spectral atlases, GPU-resident tensors, and outlier statistics across subjects.
- **Zero-Compilation Pure-Python Packaging**: Standard PEP 621 [pyproject.toml](pyproject.toml) that preserves existing Conda/venv CUDA PyTorch installations and requires no PyTorch in CPU-only mode.
- **Built-In Diagnostics & Strict Parity**: `wma doctor` inspects Python, VTK, CUDA/PyTorch, Triton, `h5py`, and cache writability; `float64` mode matches reference outputs with `0.000e+00` coordinate deviation against full intermediate runs.

---

## Performance Benchmarks

Measured on dual NVIDIA RTX 3090 GPUs (PyTorch `2.8.0+cu128`, Triton `3.4.0`, `ORG-RegAtlas-100HCP` + `ORG-800FC-100HCP`). Complete Benchmarks A–H are documented in [doc/acceleration.md](doc/acceleration.md).

### End-to-End Whole-Brain ORG Parcellation (`sub-13086`, 340,795 streamlines, `-r rig -j 4 -gpu on`)

| Configuration | End-to-End Wall Time | Clustering Stage | Output Parity |
| --- | ---: | ---: | --- |
| Baseline single-process GPU (`--keep-intermediates 2`, cold) | 125.21 s | 90.24 s | 76 `T_*.vtp` bundles |
| Zero-intermediate (`--keep-intermediates 0`, cold cache, `float64`) | **39.76 s** | 13.73 s | Exact match (`0.000e+00` max diff) |
| Zero-intermediate (`--keep-intermediates 0`, warm cache, `float64`) | **29.78 s** (`4.20x`) | **3.17 s** (`28.5x`) | Exact match (`0.000e+00` max diff) |
| Zero-intermediate (`--keep-intermediates 0`, warm cache, `-gpu_float32`) | **25.12 s** (`4.98x`) | **1.00 s** | 76 `T_*.vtp` bundles |
| Persistent 2-subject batch (`sub-13001` 326k + `sub-13086` 341k, `float64`) | **55.30 s total** (**21.19 s** for `sub-13086`) | GPU-resident reuse | 76 `T_*.vtp` bundles / subject |

### Nonrigid B-Spline, Hardening & Atlas Construction Highlights

| Workload | Baseline | Optimized | Speedup / Savings |
| --- | ---: | ---: | --- |
| Slicer-free `-r nonrig` ORG parcellation (`5,000` fibers, `--keep-intermediates 0` vs `2`) | 186.73 s | **54.26 s** | **`3.44x`**, `0.000e+00` coord diff |
| Standalone pure-Python/VTK `105^3` BSpline hardening (`800` clusters, `-j 4`) | — | **3.34 s** | `0.000e+00` diff vs VTK filter |
| `NonrigidObjective.value_and_gradient` (`8000 x 8000`, `float64`, RTX 3090) | 303.88 ms (`4,110 MiB`) | **134.05 ms** (**`36.4 MiB`**) | **`2.27x`** faster, **`112.8x`** less VRAM |
| `NonrigidObjective.value_and_gradient` (`8000 x 8000`, `float32`, RTX 3090) | 131.58 ms (`2,862 MiB`) | **16.21 ms** (**`20.6 MiB`**) | **`8.12x`** faster, **`139.0x`** less VRAM |
| GPU batched `kmeans2` (`N = 50,000`, `k = 800`, `D = 10`) | 1.411 s (CPU) | **0.172 s** (GPU) | **`8.22x`**, `0` label differences |

---

## Three Ways to Use WhiteMatterAnalysis GPU

Choose the deployment workflow that matches your environment:

| Mode | Best For | Requirements |
| --- | --- | --- |
| **[1. CLI (`pip` / Conda / `uv`)](#1-command-line-interface-pip--conda--uv)** | Running batch tractography pipelines directly on a workstation or GPU server | Python 3.8+, optional CUDA PyTorch |
| **[2. Python SDK (`whitematteranalysis.pipelines`)](#2-python-sdk-whitematteranalysispipelines)** | Embedding ORG parcellation inside Python scripts, Jupyter notebooks, or PyTorch/dMRI pipelines | Same Python environment as Mode 1 |
| **[3. Docker & Apptainer (HPC)](#3-docker--apptainer--singularity-hpc)** | Reproducible multi-machine deployments, Slurm HPC clusters, and zero-setup collaborators | Docker + NVIDIA Container Toolkit, or Apptainer/Singularity |

---

### 1. Command-Line Interface (`pip` / Conda / `uv`)

The package installs cleanly as pure Python (`pyproject.toml`) without overwriting existing CUDA PyTorch or Conda scientific libraries.

#### Install in an Existing Conda GPU Environment (Recommended)

```shell
conda activate <your-gpu-env>

# Install directly from GitHub (preserves existing CUDA PyTorch, Triton, and Conda packages)
pip install --no-build-isolation "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"

# Optional: include h5py for ITK .h5 BSpline/Composite transform support
# pip install --no-build-isolation "whitematteranalysis[h5] @ git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"

# Verify Python, VTK, CUDA/PyTorch, Triton, and atlas cache status
wma doctor
```

#### Install in a Fresh `uv` or `venv` Environment

```shell
# 1. Create a virtual environment and install CUDA PyTorch matching your NVIDIA driver
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install WhiteMatterAnalysis GPU (or CPU-only by skipping step 1)
uv pip install "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"
# Editable local development install:
# pip install --no-build-isolation -e ".[h5,dev]"

wma doctor
```

#### Common CLI Commands

Installing the package provides the unified `wma` CLI alongside all 43 traditional `wm_*.py` scripts:

```shell
# Environment diagnostics (supports --json for automation)
wma doctor
wma list

# Single-subject Slicer-free ORG parcellation (rigid+affine, zero-intermediate default)
wma org -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r rig -j 4 -gpu on

# Nonrigid B-spline ORG parcellation without 3D Slicer (analytic GPU gradients)
WMA_NONRIGID_ANALYTIC_GRAD=1 wma org -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r nonrig -j 4 -gpu on

# Persistent multi-subject batch mode (reuses spectral atlas, GPU tensors & outlier cache)
wma org --input-dir subjects/ -o batch_out -a /path/to/ORG-Atlases -r rig -j 4 -gpu on --continue-on-error

# Standalone pure-Python/VTK transform hardening (affine, BSpline, or Composite .tfm/.h5)
wma harden --pure-python -i -j 4 input_dir output_dir transform.tfm

# Spectral clustering atlas construction
wma build-atlas input_dir atlas_out -k 800 -j 4 -gpu on
```

> **Tip**: All legacy script names (`wm_apply_ORG_atlas.py`, `wm_harden_transform.py`, `wm_cluster_atlas.py`, `wm_register_to_atlas_new.py`, etc.) remain directly executable on `PATH`. See [doc/command-line-reference.md](doc/command-line-reference.md) for the full 28-script reference.

---

### 2. Python SDK (`whitematteranalysis.pipelines`)

Embed the single-process parcellation engine directly in Python applications, Jupyter notebooks, or deep learning pipelines without spawning subprocesses. `ORGParcellationPipeline` keeps registration atlases, spectral atlases, GPU-resident tensors, and outlier reference caches warm across subjects while scoping per-pipeline GPU settings cleanly.

#### One-Shot Single-Subject Parcellation

```python
from whitematteranalysis.pipelines import parcellate_subject

result = parcellate_subject(
    "sub-01.vtp",
    "output_dir",
    "/path/to/ORG-Atlases",
    reg_mode="rig",     # "rig" (rigid+affine) or "nonrig" (affine+BSpline)
    jobs=4,
    gpu=True,
    keep_intermediates=0,
)

print(result.case_id, result.qc_summary["retained_clustered_fibers"])
af_left = result.load_tract("T_AF_left")               # Returns vtkPolyData
nonempty_tracts = result.load_all_tracts(nonempty_only=True)
```

#### Persistent Multi-Subject Session & Batch Processing

```python
from whitematteranalysis.pipelines import ORGParcellationPipeline

with ORGParcellationPipeline(
    "/path/to/ORG-Atlases",
    atlas_family="auto",   # "auto" | "org" | "east-west" | "naba"
    reg_mode="rig",
    jobs=4,
    gpu=True,
    gpu_float32=False,     # True for faster FP32 mode
    keep_resident=True,
) as pipe:
    # Option A: process a directory, text manifest, or explicit list of paths / (case_id, path) tuples
    batch = pipe.run_batch("subjects_dir/", "batch_out/", continue_on_error=True)
    print(f"Succeeded: {batch.succeeded}/{batch.total}, summary: {batch.summary_path}")

    # Option B: stream subjects interactively while reusing GPU-resident atlas state
    for subject_vtp in ["sub-01.vtp", "sub-02.vtp"]:
        res = pipe.run(subject_vtp, "stream_out/")
        print(res.case_id, res.qc_summary["stage_timings_s"]["total"])
```

---

### 3. Docker & Apptainer / Singularity (HPC)

Prebuilt multi-variant images are published automatically by [GitHub Actions](.github/workflows/docker-publish.yml) to GitHub Container Registry (`ghcr.io/xiaji/whitematteranalysis_gpu`).

#### Available Container Image Tags

| Tag | Base Image | Target Hardware & Driver Floor |
| --- | --- | --- |
| `:latest`, `:cu124` *(default)* | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | NVIDIA Linux drivers **≥ 550** (Triton PTX 8.4): V100, T4, RTX 20xx–40xx, A100, H100 |
| `:cu128` | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` | NVIDIA Linux drivers **≥ 570** (Triton PTX 8.7): RTX 30xx–50xx, A100, H100, **Blackwell** |
| `:cpu` | `python:3.11-slim-bookworm` | CPU-only servers, CI runners, and GPU-free compute nodes |

All images include `whitematteranalysis[h5]`, minimal headless VTK shared libraries, and (for GPU images) `gcc` + `libc6-dev` for Triton JIT compilation. Atlases are mounted at runtime rather than baked into the image, and caches default to `/tmp/wma-cache` for seamless non-root `--user` and read-only Apptainer execution.

#### Run with Docker

```shell
# Pull prebuilt image (or build locally: `docker build -t wma-gpu:0.4.0 .`)
docker pull ghcr.io/xiaji/whitematteranalysis_gpu:latest

# 1. Check CUDA, Triton, VTK, and cache status
docker run --rm --gpus all ghcr.io/xiaji/whitematteranalysis_gpu:latest doctor

# 2. Single-subject parcellation (non-root user + persistent cache mount)
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /data/subjects:/input:ro \
  -v /data/ORG-Atlases:/atlas:ro \
  -v /data/output:/output \
  -v /data/wma_cache:/tmp/wma-cache \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest \
  org -i /input/sub-01.vtp -o /output -a /atlas -r rig -j 4 -gpu on

# 3. Multi-subject persistent batch parcellation
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /data/subjects:/input:ro \
  -v /data/ORG-Atlases:/atlas:ro \
  -v /data/output:/output \
  -v /data/wma_cache:/tmp/wma-cache \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest \
  org --input-dir /input -o /output -a /atlas -r rig -j 4 -gpu on --continue-on-error

# 4. Run a custom Python SDK script inside the container
docker run --rm --gpus all --entrypoint python \
  -v /data:/data -v /work/run.py:/run.py:ro \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest /run.py
```

#### Local Docker Build Options

```shell
# Default GPU image (CUDA 12.4)
docker build -t wma-gpu:cu124 .

# Blackwell / CUDA 12.8 GPU image
docker build --build-arg GPU_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t wma-gpu:cu128 .

# Lightweight CPU-only image
docker build --target cpu -t wma-cpu:0.4.0 .
```

#### Run on Slurm / HPC Clusters with Apptainer (Singularity)

```shell
# Pull directly from GHCR into a single portable .sif file
apptainer pull wma-gpu-cu124.sif docker://ghcr.io/xiaji/whitematteranalysis_gpu:latest

# Run GPU parcellation without root privileges
apptainer exec --nv \
  -B /data/subjects:/input:ro,/data/ORG-Atlases:/atlas:ro,/data/output:/output,/scratch/wma_cache:/tmp/wma-cache \
  wma-gpu-cu124.sif \
  wma org --input-dir /input -o /output -a /atlas -r rig -j 4 -gpu on --continue-on-error
```

---

## Supported White Matter Atlases

WhiteMatterAnalysis GPU auto-detects and validates atlas layouts via `--atlas-family {auto,org,east-west,naba}` (or `atlas_family=` in the Python SDK):

| Atlas Family | `--atlas-family` | Output Bundles | Notes |
| --- | --- | --- | --- |
| **ORG** (O'Donnell Research Group 800-cluster atlas) | `org` (or `auto`) | 76 `T_*.vtp` bundles | Download via `wma download-atlas` |
| **East-West** | `east-west` (or `auto`) | 76 `T_*.vtp` bundles | Validated via `ATLAS.json` or layout |
| **NABA** | `naba` (or `auto`) | 78 `T_*.vtp` bundles | Validated via `ATLAS.json` or layout |

See [doc/bundles.md](doc/bundles.md) for the complete list of anatomical white matter tracts.

---

## Environment Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `WMA_GPU` | `auto` | Backend mode: `auto` (use CUDA if available), `on` (require CUDA), or `off` (CPU only) |
| `WMA_GPU_DEVICES` | all visible | Comma-separated CUDA device indices for multi-GPU operations (e.g., `0,1`) |
| `WMA_GPU_FLOAT32` | `0` | Set `1` to enable `float32` fast mode (`float64` is default) |
| `WMA_GPU_STRICT` | `0` | Set `1` to force non-fused `float64` registration reductions for strict parity |
| `WMA_GPU_MPS` | `0` | Set `1` (with `WMA_GPU_FLOAT32=1`) for experimental Apple Silicon MPS backend |
| `WMA_GPU_MAX_VRAM_GIB` | unset | Optional free-VRAM cap (GiB) used for adaptive GPU tile sizing on shared GPUs |
| `WMA_GPU_MIN_PAIR_WORK` | `4096` | Route tiny outlier clusters (`n_queries * n_fixed < threshold`) to CPU |
| `WMA_NONRIGID_ANALYTIC_GRAD` | `0` | Set `1` to use fused PyTorch/Triton analytic gradients for nonrigid `BFGS` registration |
| `WMA_GPU_PROCESS_POOL` | `0` | Set `1` to use spawn `ProcessPoolExecutor` for multi-GPU linear groupwise registration |
| `WMA_ATLAS_CACHE` | `~/.cache/whitematteranalysis` | Directory for `.npz` spectral atlas & outlier reference caches (`off` disables caching) |
| `WMA_CPU_FAST` | `1` | Set `0` (with `WMA_GPU=off`) to restore legacy per-fiber `joblib` loops |
| `WMA_CPU_THREADS` | affinity-aware | Worker thread budget for the vectorized CPU backend |
| `WMA_CPU_BLAS_THREADS` | auto | Ceiling on BLAS threads during atlas labeling (`0` disables automatic capping) |
| `WMA_VTP_COMPRESSION` | `zlib` (`lz4` in ORG CLI) | Lossless VTP compression codec (`lz4` or `zlib`) |
| `WMA_IO_PREFETCH` | `0` | Set `1` to enable POSIX readahead for large tractography files |

> **CLI Precedence**: Individual CLI tools that expose `-gpu auto|on|off`, `-gpu_float32`, or `-gpu_devices` override `WMA_GPU` when passed explicitly.

---

## Optional 3D Slicer Extension

A five-module 3D Slicer extension (Pipeline, Atlas Registration, Groupwise Registration, Atlas Building, and Tract Merging) is also included under [slicer/WMAGPU](slicer/WMAGPU/README.md) with a [universal installer](slicer/WMAGPU/dist/WMAGPU-Installer.zip). Note that **3D Slicer is not required** for CLI, Python SDK, or Docker/Apptainer execution on tensor-free tractograms.

---

## Documentation

- [Acceleration Architecture & Benchmarks A–H](doc/acceleration.md)
- [Command-Line Reference (All 28 `bin/` Scripts)](doc/command-line-reference.md)
- [Subject-Specific Tractography Parcellation Tutorial](doc/subject-specific-tractography-parcellation.md)
- [Anatomical White Matter Bundles](doc/bundles.md)
- [3D Slicer Extension Guide](slicer/WMAGPU/README.md)

---

## Frequently Asked Questions (FAQ)

**Does ORG white matter parcellation require 3D Slicer or SlicerDMRI?**  
No. Both rigid (`-r rig`) and nonrigid (`-r nonrig`) parcellation run in a single Python process with pure-Python/VTK transform hardening for tensor-free `.vtk`/`.vtp` tractograms. 3D Slicer (`-s /path/to/Slicer`) is only required if your input tractography carries per-point diffusion tensor arrays that must be reoriented during hardening.

**Which CUDA / Docker tag should I use?**  
Use `:latest` (`:cu124`, PyTorch 2.6 + CUDA 12.4) by default—its Triton PTX 8.4 kernels work on any Linux NVIDIA driver `>= 550` across V100, T4, RTX 20xx–40xx, A100, and H100 GPUs. Use `:cu128` (PyTorch 2.8 + CUDA 12.8) on Blackwell GPUs or hosts running driver `>= 570`.

**Can I run WhiteMatterAnalysis GPU without a GPU or without PyTorch?**  
Yes. PyTorch is optional; without CUDA or PyTorch, the package automatically uses the vectorized multi-threaded CPU backend (`WMA_GPU=off` or `:cpu` Docker image).

**Will `pip install` overwrite my Conda CUDA PyTorch installation?**  
No. `torch` and `triton` are intentionally excluded from default package dependencies in [pyproject.toml](pyproject.toml), and Python 3.10+ dependencies carry no restrictive upper bounds on `numpy`, `scipy`, `vtk`, `pandas`, or `matplotlib`. Run `wma doctor` after installing to verify your environment.

**Can I run it on a Slurm HPC cluster without root privileges?**  
Yes. Pull the GHCR container with `apptainer pull` and run `apptainer exec --nv` with bind-mounted input, atlas, output, and `/tmp/wma-cache` directories.

---

## References & License

Distributed under the BSD-3-Clause license ([License.txt](License.txt)). If you use WhiteMatterAnalysis or the ORG white matter atlas in academic publications, please cite:

- Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, L. J. (2018). *An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan.* **NeuroImage**, 179, 429–447.
- O'Donnell, L. J., Wells III, W. M., Golby, A. J., Westin, C.-F. (2012). *Unbiased groupwise registration of white matter tractography.* **MICCAI**, pp. 123–130.
- O'Donnell, L. J., & Westin, C.-F. (2007). *Automatic tractography segmentation using a high-dimensional white matter atlas.* **IEEE Transactions on Medical Imaging**, 26(11), 1562–1575.

If your workflow also uses 3D Slicer and SlicerDMRI:

- Zhang, F., Noh, T., Juvekar, P., et al. (2020). *SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization.* **JCO Clinical Cancer Informatics**, 4, e299–e309.
- Norton, I., Essayed, W. I., Zhang, F., et al. (2017). *SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research.* **Cancer Research**, 77(21), e101–e103.
