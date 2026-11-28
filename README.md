# WhiteMatterAnalysis GPU

[![CI](https://github.com/xiaji/whitematteranalysis_gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaji/whitematteranalysis_gpu/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776AB.svg)](pyproject.toml)
[![CUDA 12.4 | 12.8](https://img.shields.io/badge/CUDA-12.4%20%7C%2012.8-76B900.svg)](Dockerfile)
[![PyTorch + Triton](https://img.shields.io/badge/backend-PyTorch%20%2B%20Triton%20%7C%20CPU-EE4C2C.svg)](doc/acceleration.md)
[![Container](https://img.shields.io/badge/GHCR-Docker%20%7C%20Apptainer-2496ED.svg)](.github/workflows/docker-publish.yml)
[![License: 3D Slicer](https://img.shields.io/badge/license-3D%20Slicer%20(BSD--style)-blue.svg)](License.txt)

**Headless, GPU-accelerated diffusion MRI streamline tractography registration, spectral fiber clustering, and Slicer-free white matter parcellation—built for CLI, Python SDK, Docker/HPC, and AI agents.**

[Overview](#overview) · [Benchmarks](#benchmarks) · [CLI](#1-command-line-interface-conda--pip--uv) · [Python SDK](#2-python-sdk-whitematteranalysispipelines) · [Docker & HPC](#3-docker--apptainer-hpc) · [Agents](#ai-agent--automation-interface) · [Atlases](#supported-atlases) · [Configuration](#configuration) · [FAQ](#faq) · [Citation](#citation--attribution)

> [!NOTE]
> **Upstream Provenance**: Built upon **[SlicerDMRI/whitematteranalysis](https://github.com/SlicerDMRI/whitematteranalysis)** by Lauren J. O'Donnell, Fan Zhang, Isaiah Norton, et al. (Brigham and Women's Hospital, [3D Slicer License v1.0](License.txt)) and initial GPU/CPU acceleration from **[NeuroDMRI/whitematteranalysis_gpu](https://github.com/NeuroDMRI/whitematteranalysis_gpu)** by Yijie Li. See [NOTICE](NOTICE) for full attribution.

---

## Overview

**WhiteMatterAnalysis GPU** (`whitematteranalysis`) accelerates whole-brain diffusion MRI (dMRI) streamline tractography registration, Nyström spectral fiber clustering, anatomical outlier removal, and automated white matter tract parcellation across the **ORG** (O'Donnell Research Group), **East-West**, and **NABA** atlases.

- **Headless & Slicer-Free**: Runs rigid, affine, and 3D cubic B-spline nonrigid parcellation end to end in a single Python process via pure-Python/VTK transform hardening—no GUI, 3D Slicer, or Xvfb required for standard geometry/scalar `.vtk` and `.vtp` tractograms.
- **CUDA & Triton Acceleration**: Custom 2D Triton kernels for pairwise streamline distances, spectral embedding, split-C nearest centroids, `O(N + R^3)` nonrigid B-spline analytic gradients, and batched `kmeans2`, with multi-GPU dispatch and a vectorized multi-threaded CPU fallback.
- **Zero-Intermediate Execution & Caching**: In-memory tract assembly (`--keep-intermediates 0`) avoids writing 800 intermediate cluster VTP files, paired with `.npz` spectral/outlier atlas caches and resident multi-subject GPU state reuse.
- **Agent-First SDK, Telemetry & Containers**: Ships JSON runtime introspection (`wma doctor --json`, `wma list --json`), structured `qc_summary.json` outputs, atomic directory publishing, an in-process `whitematteranalysis.pipelines` Python SDK, and prebuilt CUDA 12.4 / 12.8 / CPU container images on GHCR.
- **Verified Numerical Fidelity**: Defaults to `float64` (`0.000e+00` max coordinate difference on 5,000-fiber parity checks; exact fiber-count parity on 340k whole-brain tractograms), with opt-in `float32` for maximum throughput.

---

## Benchmarks

Measured on dual NVIDIA RTX 3090 GPUs (PyTorch `2.8.0+cu128`, Triton `3.4.0`, `ORG-RegAtlas-100HCP` + `ORG-800FC-100HCP`). Complete methodology and tables A–H are documented in [doc/acceleration.md](doc/acceleration.md).

### Whole-Brain ORG Parcellation (`sub-13086`, 340,795 streamlines, `-r rig -j 4 -gpu on`)

| Pipeline Configuration | Wall Time | Clustering Stage | 340k Output Verification |
| --- | ---: | ---: | --- |
| Baseline GPU (`--keep-intermediates 2`, warm `f64` outlier cache) | 125.21 s | 90.24 s | `37,323` fibers / 76 `T_*.vtp` |
| Zero-intermediate (`--keep-intermediates 0`, cold cache, `float64`) | **39.76 s** (`3.15x`) | **3.57 s** (`25.3x`) | Exact fiber-count match (`37,323` / 76 tracts) |
| Zero-intermediate (`--keep-intermediates 0`, warm cache, `float64`) | **29.78 s** (`4.20x`) | **3.17 s** (`28.5x`) | Exact fiber-count match (`37,323` / 76 tracts) |
| Zero-intermediate (`--keep-intermediates 0`, warm cache, `float32`) | **25.12 s** (`4.98x`) | **1.00 s** | `37,007` fibers / 76 `T_*.vtp` |
| Persistent 2-subject batch (326k + 341k streamlines, warm `float64`) | **55.30 s** (**21.19 s** on `sub-13086`) | Resident GPU reuse | 76 `T_*.vtp` tracts / subject |

*Parity notes: `--keep-intermediates 0` vs `2` achieved `0.000e+00` max point coordinate difference across all 76 tracts on the 5,000-fiber `sub-13086` check (Benchmark G1). On the full 340,795-streamline run, each precision matched the Phase 2 baseline at the same precision (`48,906` retained / `37,323` tract fibers in `float64`; `48,497` retained / `37,007` tract fibers in `float32`, Benchmark G2).*

### Nonrigid Registration, Transform Hardening & Atlas Construction

| Workload | Baseline | Optimized | Speedup & Memory |
| --- | ---: | ---: | --- |
| Slicer-free `-r nonrig` ORG pipeline (`5,000` fibers, `keep=0` vs `2`) | 186.73 s | **54.26 s** | **`3.44x`**, `0.000e+00` coord diff |
| Pure-Python/VTK `105^3` BSpline hardening (`800` clusters, `-j 4`) | — | **3.34 s** | `0.000e+00` diff vs VTK filter |
| Nonrigid `value_and_gradient` (`8000 x 8000`, `float64`) | 303.88 ms (`4,110 MiB`) | **134.05 ms** (`36.4 MiB`) | **`2.27x`** faster, **`112.8x`** less VRAM |
| Nonrigid `value_and_gradient` (`8000 x 8000`, `float32`) | 131.58 ms (`2,862 MiB`) | **16.21 ms** (`20.6 MiB`) | **`8.12x`** faster, **`139.0x`** less VRAM |
| GPU batched `kmeans2` (`N = 50,000`, `k = 800`, `D = 10`) | 1.411 s (CPU) | **0.172 s** (GPU) | **`8.22x`**, `0` label diffs |

---

## Quick Start

| Interface | Best For |
| --- | --- |
| **[1. CLI (`pip` / Conda / `uv`)](#1-command-line-interface-conda--pip--uv)** | Workstation or server execution of parcellation, registration, hardening, and atlas building |
| **[2. Python SDK (`whitematteranalysis.pipelines`)](#2-python-sdk-whitematteranalysispipelines)** | In-memory multi-subject batch pipelines, notebooks, and Python neuroimaging workflows |
| **[3. Docker & Apptainer (HPC)](#3-docker--apptainer-hpc)** | Reproducible deployment on multi-GPU servers and rootless Slurm HPC clusters |

---

### 1. Command-Line Interface (Conda / `pip` / `uv`)

Pure-Python PEP 621 packaging ([pyproject.toml](pyproject.toml)) requires no compilation and never overwrites an existing CUDA PyTorch installation.

#### Install

```shell
# Option A: Existing Conda GPU environment (recommended)
conda activate <your-gpu-env>
pip install --no-build-isolation "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"
# Optional ITK .h5 transform support:
# pip install --no-build-isolation "whitematteranalysis[h5] @ git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"

# Option B: Fresh virtual environment with uv (omit torch install for CPU-only)
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # Driver >= 550
# For Blackwell GPUs (driver >= 570), use CUDA 12.8 instead:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"

# Verify active backend, CUDA/Triton availability, and atlas cache path
wma doctor
```

#### Run

```shell
# Single-subject rigid+affine ORG parcellation
wma org -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r rig -j 4 -gpu on

# Nonrigid B-spline ORG parcellation with fused analytic GPU gradients
WMA_NONRIGID_ANALYTIC_GRAD=1 wma org -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r nonrig -j 4 -gpu on

# Persistent multi-subject batch parcellation
wma org --input-dir subjects/ -o batch_out -a /path/to/ORG-Atlases -r rig -j 4 -gpu on --continue-on-error

# Standalone affine / BSpline / Composite transform hardening (.tfm or .h5; add -i to invert)
wma harden --pure-python -j 4 input_dir output_dir -t transform.tfm

# Spectral fiber clustering atlas construction (omit -norender when a display/offscreen VTK backend is available)
wma build-atlas input_dir atlas_out -k 800 -j 4 -norender -gpu on
```

Outputs are published atomically to `<output_dir>/<case_id>/AnatomicalTracts/T_*.vtp` alongside `<output_dir>/<case_id>/qc_summary.json` (and `<output_dir>/batch_qc_summary.json` in batch mode). All traditional `wm_*.py` scripts are also installed on `PATH`; run `wma list` or see [doc/command-line-reference.md](doc/command-line-reference.md).

---

### 2. Python SDK (`whitematteranalysis.pipelines`)

Call single-subject or persistent multi-subject parcellation directly from Python with zero subprocess overhead. `ORGParcellationPipeline` scopes GPU settings per call and keeps registration atlases, GPU fiber tensors, and outlier reference statistics resident across subjects.

```python
from whitematteranalysis.pipelines import ORGParcellationPipeline, parcellate_subject

# Single subject
res = parcellate_subject(
    input_file="sub-01.vtp",
    output_dir="output_dir",
    atlas_dir="/path/to/ORG-Atlases",
    reg_mode="rig",         # "rig" (rigid+affine) or "nonrig" (affine+BSpline)
    jobs=4,
    gpu=True,               # True | False | None (inherit environment)
)
print(res.qc_summary["retained_clustered_fibers"])
af_left = res.load_tract("T_AF_left")               # vtkPolyData
tracts = res.load_all_tracts(nonempty_only=True)    # dict[str, vtkPolyData]

# Multi-subject session with resident GPU & atlas reuse
with ORGParcellationPipeline(
    "/path/to/ORG-Atlases",
    atlas_family="auto",    # "auto" | "org" | "east-west" | "naba"
    reg_mode="rig",
    jobs=4,
    gpu=True,
    gpu_float32=False,
) as pipe:
    batch = pipe.run_batch("subjects_dir/", "batch_out/", continue_on_error=True)
    print(f"{batch.succeeded}/{batch.total} succeeded -> {batch.summary_path}")
```

---

### 3. Docker & Apptainer (HPC)

Prebuilt images are published to `ghcr.io/xiaji/whitematteranalysis_gpu` via [GitHub Actions](.github/workflows/docker-publish.yml). Atlases are mounted at runtime, and caches default to `/tmp/wma-cache` for rootless `--user` and read-only Apptainer/Singularity execution.

| Image Tag | Base Runtime | Driver & Architecture Support |
| --- | --- | --- |
| `:latest`, `:cu124` *(default)* | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | Linux driver **≥ 550** (PTX 8.4): V100, T4, RTX 20xx–40xx, A100, H100 |
| `:cu128` | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` | Linux driver **≥ 570** (PTX 8.7): RTX 30xx–50xx, A100, H100, Blackwell |
| `:cpu` | `python:3.11-slim-bookworm` | CPU-only hosts and GPU-free compute nodes |

#### Docker

```shell
docker pull ghcr.io/xiaji/whitematteranalysis_gpu:latest

# Probe container runtime in JSON
docker run --rm --gpus all ghcr.io/xiaji/whitematteranalysis_gpu:latest doctor --json

# Batch ORG parcellation (non-root user + persistent host cache)
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /data/subjects:/input:ro \
  -v /data/ORG-Atlases:/atlas:ro \
  -v /data/output:/output \
  -v /data/wma_cache:/tmp/wma-cache \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest \
  org --input-dir /input -o /output -a /atlas -r rig -j 4 -gpu on --continue-on-error

# Run a custom Python SDK script inside the container
docker run --rm --gpus all --entrypoint python \
  -v /data:/data -v /work/run.py:/run.py:ro \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest /run.py
```

<details>
<summary><strong>Build container images locally</strong></summary>

```shell
# Default CUDA 12.4 GPU image
docker build -t wma-gpu:cu124 .

# CUDA 12.8 / Blackwell GPU image
docker build --build-arg GPU_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t wma-gpu:cu128 .

# Lightweight CPU image
docker build --target cpu -t wma-cpu:latest .
```

</details>

#### Apptainer / Singularity (Slurm HPC)

```shell
apptainer pull wma-gpu-cu124.sif docker://ghcr.io/xiaji/whitematteranalysis_gpu:latest

apptainer exec --nv \
  -B /data/subjects:/input:ro,/data/ORG-Atlases:/atlas:ro,/data/output:/output,/scratch/wma_cache:/tmp/wma-cache \
  wma-gpu-cu124.sif \
  wma org --input-dir /input -o /output -a /atlas -r rig -j 4 -gpu on --continue-on-error
```

---

## AI Agent & Automation Interface

This repository is structured for autonomous coding agents, workflow engines, and LLM tool use:

- **JSON Runtime & Command Discovery**: Run `wma doctor --json` (probes backend, GPU VRAM, Triton, `h5py`, and cache writability) and `wma list --json` before selecting execution flags.
- **Deterministic Contracts**: `wma org`, `wma doctor`, `wma list`, and `wma harden` enforce exit codes (`0` success, `1` runtime/batch failure, `2` invalid CLI usage), stage subject outputs in `.staging_<case_id>_<pid>` before atomic rename, and emit machine-readable `qc_summary.json` / `batch_qc_summary.json` reports.
- **Context Files**: See [`AGENTS.md`](AGENTS.md) for the full operational contract and [`llms.txt`](llms.txt) for a compact codebase map.

---

## Supported Atlases

Atlas directory structures and MRML bundle scenes are validated automatically pre-flight (`--atlas-family {auto,org,east-west,naba}`):

| Atlas Family | CLI / SDK Value | Output Tracts | Notes |
| --- | --- | --- | --- |
| **ORG** (O'Donnell Research Group, 800 clusters) | `org` / `auto` | 76 `T_*.vtp` | Downloadable via `wma download-atlas` |
| **East-West** | `east-west` / `auto` | 76 `T_*.vtp` | Auto-detected via `ATLAS.json` or folder layout |
| **NABA** | `naba` / `auto` | 78 `T_*.vtp` | Auto-detected via `ATLAS.json` or folder layout |

Anatomical bundle definitions are catalogued in [doc/bundles.md](doc/bundles.md).

---

## Configuration

Explicit CLI flags (`-gpu`, `-gpu_float32`, `-gpu_devices`) take precedence over environment variables. The most common runtime controls are `WMA_GPU`, `WMA_GPU_DEVICES`, `WMA_NONRIGID_ANALYTIC_GRAD=1`, and `WMA_ATLAS_CACHE`.

<details>
<summary><strong>Full environment variable reference (click to expand)</strong></summary>

| Variable | Default | Description |
| --- | --- | --- |
| `WMA_GPU` | `auto` | `auto` (use CUDA when available), `on` (require CUDA), or `off` (CPU backend) |
| `WMA_GPU_DEVICES` | all visible | Comma-separated CUDA device IDs for multi-GPU execution (e.g., `0,1`) |
| `WMA_GPU_FLOAT32` | `0` | Enable `float32` computation (`1`) instead of default `float64` (`0`) |
| `WMA_GPU_STRICT` | `0` | Use non-fused `float64` registration reductions (`1`) for strict parity |
| `WMA_GPU_MPS` | `0` | Enable experimental Apple Silicon MPS backend (`1`, requires `WMA_GPU_FLOAT32=1`) |
| `WMA_GPU_MAX_VRAM_GIB` | unset | Cap free VRAM (GiB) assumed by adaptive GPU tile sizing |
| `WMA_GPU_MIN_PAIR_WORK` | `4096` | Minimum `n_queries * n_fixed` pair count before dispatching outlier scoring to CUDA |
| `WMA_NONRIGID_ANALYTIC_GRAD` | `0` | Enable fused PyTorch/Triton analytic gradients (`1`) for nonrigid `BFGS` |
| `WMA_GPU_PROCESS_POOL` | `0` | Use `spawn` process pool (`1`) for multi-GPU linear groupwise registration |
| `WMA_ATLAS_CACHE` | `~/.cache/whitematteranalysis` | Cache root for `.npz` spectral & outlier references (`off` disables caching) |
| `WMA_CPU_FAST` | `1` | Set `0` (with `WMA_GPU=off`) to use legacy per-fiber `joblib` loops |
| `WMA_CPU_THREADS` | affinity-aware | Thread count for the vectorized CPU backend |
| `WMA_CPU_BLAS_THREADS` | auto | BLAS thread cap during spectral labeling (`0` disables capping) |
| `WMA_VTP_COMPRESSION` | `zlib` (`lz4` in ORG) | Lossless VTP compression codec (`lz4` or `zlib`) |
| `WMA_IO_PREFETCH` | `0` | Enable whole-file POSIX readahead (`1`) |

</details>

---

## Documentation

- [Agent Integration Guide (`AGENTS.md`)](AGENTS.md) & [`llms.txt`](llms.txt)
- [Acceleration Architecture & Benchmarks A–H](doc/acceleration.md)
- [Command-Line Reference (`bin/` Scripts)](doc/command-line-reference.md)
- [Anatomical White Matter Bundles](doc/bundles.md)
- [Provenance & Attribution Notice (`NOTICE`)](NOTICE)

---

## FAQ

**Is 3D Slicer required for white matter parcellation?**  
No, for standard geometry/scalar `.vtk` and `.vtp` tractograms—both rigid (`-r rig`) and nonrigid (`-r nonrig`) modes run end to end in pure Python and VTK. 3D Slicer (`-s`) is only required when point or cell data contains active vectors, normals, tensors, or any 9-component numeric array (`transform.has_tensor_arrays()`), or when an initial pre-transform passed via `-t` cannot be parsed in both directions by `read_itk_transform_ras()`.

**Which container tag should I pull?**  
Pull `:latest` (`:cu124`, CUDA 12.4 / PTX 8.4) for broad compatibility on Linux NVIDIA drivers `>= 550`. Pull `:cu128` (CUDA 12.8 / PTX 8.7) for Blackwell GPUs or drivers `>= 570`, or `:cpu` on hosts without GPUs.

**Will `pip install` overwrite my Conda PyTorch installation?**  
No. `torch` and `triton` are intentionally excluded from default dependencies in [pyproject.toml](pyproject.toml), leaving existing Conda CUDA packages untouched.

**Does it run on CPU-only machines?**  
Yes. When CUDA or PyTorch is absent, CLI commands use the multi-threaded vectorized CPU backend under `-gpu auto` (which overrides `WMA_GPU=on`) or `-gpu off`. In the Python SDK, `gpu=None` inherits the current process/environment setting—pass `gpu=False` when CPU execution is required regardless of `WMA_GPU`. Passing `-gpu on` or `gpu=True` explicitly requires a GPU and raises an error if none is available.

---

## Citation & Attribution

Distributed under the [3D Slicer Contribution and Software License Agreement v1.0](License.txt) (BSD-style; redistributions must include Part B). Full codebase provenance—crediting upstream `SlicerDMRI/whitematteranalysis`, initial GPU/CPU acceleration by Yijie Li, and Triton/SDK/container engineering by `xiaji`—is recorded in [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).

If you use WhiteMatterAnalysis or the ORG white matter atlas in your research, please cite:

1. Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., & O'Donnell, L. J. (2018). An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. *NeuroImage*, 179, 429–447.
2. O'Donnell, L. J., Wells III, W. M., Golby, A. J., & Westin, C.-F. (2012). Unbiased groupwise registration of white matter tractography. *MICCAI*, 123–130.
3. O'Donnell, L. J., & Westin, C.-F. (2007). Automatic tractography segmentation using a high-dimensional white matter atlas. *IEEE Transactions on Medical Imaging*, 26(11), 1562–1575.

When using 3D Slicer or SlicerDMRI, please also cite Zhang et al., *JCO Clinical Cancer Informatics* (2020) and Norton et al., *Cancer Research* (2017).
