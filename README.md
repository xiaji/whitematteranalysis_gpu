# WhiteMatterAnalysis

WhiteMatterAnalysis (WMA) provides fiber clustering and tractography analysis tools.
This fork adds CUDA GPU acceleration, a vectorized CPU backend, and a five-module
3D Slicer extension. It is based on the [upstream WMA project](https://github.com/SlicerDMRI/whitematteranalysis).

## Full-pipeline performance

C028 benchmark (2026-10-30): **~330,000 streamlines, 3.6 GB**, ORG atlases,
rigid + affine registration, float64 and four CPU cores/workers. Hardware:
Intel Xeon Gold 6330, RTX 4090 (GPU 0), Slicer 5.2.2 + SlicerDMRI.

| Step | Joblib (4 workers), s | CPU matrix, s | CPU speedup | GPU first run, s | GPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Registration (rigid + affine) | 371.6 | 83.0 | 4.48x | 28.9 | 12.84x |
| Atlas labeling and cluster saving | 897.4 | 94.4 | 9.51x | 24.7 | 36.30x |
| Outlier removal + hemisphere assessment | 209.5 | 82.4 | 2.54x | 28.1 | 7.46x |
| Inverse transform + hemisphere separation* | 13.9 | 13.9 | 1.00x | 14.9 | 0.93x |
| Separation verification | 0.4 | 0.4 | 1.04x | 0.4 | 1.04x |
| Anatomical tract assembly | 15.2 | 15.0 | 1.02x | 14.9 | 1.02x |
| **Total** | **1509.3** | **290.1** | **5.20x** | **113.8** | **13.26x** |

\* Slicer initializes during outlier removal; this row counts only the additional
wall time. Totals include startup, I/O and pipeline overhead. Speedups are relative
to joblib.

Runs used fresh outputs and empty atlas caches. File/compilation caches were not
reset; GPU repeat: **111.3 s**, separate cold-data run: **155.1 s**. I/O waits vary.

Full validation covered **4,874 VTPs**, the registered tractogram and **11 transforms**.
CPU outputs were byte-identical to joblib. GPU membership, connectivity and
attributes matched; maximum coordinate difference: **1.91e-6 mm**.
See the [acceleration overview](doc/acceleration.md) for details.

### Single-process ORG CLI & Phase 2–5 GPU updates (`fan93`, 2x RTX 3090)

For tensor-free rigid or nonrigid ORG parcellation (single-subject or persistent multi-subject batch),
[bin/wm_apply_ORG_atlas.py](bin/wm_apply_ORG_atlas.py) runs registration, spectral clustering,
outlier removal, pure-Python/VTK affine or BSpline hardening, hemisphere separation and anatomical tract
assembly in a single Python process with no 3D Slicer or SlicerDMRI subprocesses required
(recording `qc_summary.json` / `batch_qc_summary.json` and publishing atomically by default):

```shell
# Single subject (zero-intermediate Tier-1 fast path is enabled by default)
wm_apply_ORG_atlas.py -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r rig -j 4 -gpu on

# Persistent multi-subject batch (reuses spectral atlas + GPU-resident tensors + outlier cache across subjects)
wm_apply_ORG_atlas.py --input-dir subjects/ -o batch_out -a /path/to/ORG-Atlases -r rig -j 4 -gpu on --continue-on-error
```

Phase 4 adds three P0 end-to-end optimizations—**zero-intermediate indexed ORG parcellation** (`--keep-intermediates 0` skipping 800-cluster VTK materialization and per-cluster affine/hemisphere copies), a **`1.2 MiB` binary `.npz` spectral atlas inference cache** (`WMA_ATLAS_CACHE/spectral-atlas/`) with **GPU-resident atlas prep**, and **persistent multi-subject batch mode** (`-i <dir>` / `--input-dir` / `--input-list`). On `sub-13086` (**340,795 streamlines**, `ORG-RegAtlas-100HCP` + `ORG-800FC-100HCP`, dual NVIDIA RTX 3090, `-r rig -j 4 -gpu on`), end-to-end wall time drops from Phase 2's **2:05.21 (125.21 s)** (`90.237 s` `clustering`) to **39.76 s** (cold cache, `float64`), **29.78 s** (warm cache, `float64`, **`4.20x` speedup** with **`28.5x` faster clustering** at `3.171 s`), and **25.12 s** (warm cache, `-gpu_float32`, **`0.996 s` clustering**) with identical `76` `AnatomicalTracts/T_*.vtp` bundles (`max_coord_diff = 0.000e+00` vs `--keep-intermediates 2` on real 5k parity test). In persistent 2-subject batch mode (`sub-13001` 326k + `sub-13086` 341k fibers, warm `float64`), both whole-brain subjects complete in **55.30 s** total (`21.19 s` for `sub-13086`).

In earlier Phase 2–3 updates on RTX 3090, WMA also added FP32 + arbitrary `1 <= P <= 64` Triton kernels, split-C low-VRAM `nearest_centroids` (`88x–111x` vs `torch.cdist` in `float32`), multi-GPU labeling/outlier dispatch, cached 1D separable GPU B-spline prefiltering + 1-thread-per-point Triton kernel + `WMA_NONRIGID_ANALYTIC_GRAD=1` L-BFGS-B gradients (`10.89x` faster at `R=4`) and vectorized `105^3` `.tfm` export, GPU `pinv_A` reuse + `O(N)` LOO similarity (`4.82x` faster), and validated multi-atlas support (`--atlas-family {auto,org,east-west,naba}` yielding `76` / `76` / `78` `T_*.vtp` bundles). Phase 5 extends Slicer-free execution and high-resolution nonrigid / atlas-construction scaling (**P1**): **Slicer-free `-r nonrig` hardening** (`read_itk_transform_ras` for ITK `BSplineTransform`/`CompositeTransform` `.tfm`/`.h5` plus `apply_transform_to_polydata`, completing `5,000`-fiber `-r nonrig` without `-s` in **54.26 s** with `0.000e+00` coordinate delta vs `--keep-intermediates 2`, and standalone `800`-cluster `105^3` BSpline hardening in **3.337 s**), a **fused `O(N + R^3)` Triton backward kernel** for `NonrigidObjective.value_and_gradient` (`4000 x 4000` / `8000 x 8000` at **2.27x–2.30x** speedup with **74x–113x** lower VRAM in `float64`, and **5.42x–8.12x** speedup with **64x–139x** lower VRAM in `float32`), and **PSD-skip + GPU batched `kmeans2`** in `cluster.spectral` (**8.22x** faster at `N=50,000, k=800` with `0` label diffs).

See [doc/acceleration.md](doc/acceleration.md#phase-2-benchmarks-fan93-rtx-3090), [Phase 3 benchmarks](doc/acceleration.md#phase-3-benchmarks-fan93-rtx-3090), [Phase 4 (P0) benchmarks](doc/acceleration.md#phase-4-p0-benchmarks-fan93-rtx-3090), and [Phase 5 (P1) benchmarks](doc/acceleration.md#phase-5-p1-benchmarks-fan93-rtx-3090) for complete Benchmarks A–H tables.

# GPU and 3D Slicer quick start

## Python / command line

This package uses a modern PEP 621 [pyproject.toml](pyproject.toml) (pure Python — no C/C++ compilation required).

### Recommended installation (Conda GPU environment or `uv` / `venv`)

In an existing Conda GPU environment with CUDA PyTorch already installed:

```shell
conda activate <your-gpu-env>

# Install directly from GitHub (preserves existing CUDA PyTorch and Conda packages)
pip install --no-build-isolation "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"
# Optional extras: [h5] for ITK .h5 transforms, [dev] for pytest, [doc] for Sphinx
# pip install --no-build-isolation "whitematteranalysis[h5] @ git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"

# Verify Python, VTK, PyTorch/CUDA, Triton, and atlas cache status
wma doctor
```

Or in a fresh `uv` / `venv` environment:

```shell
# 1. (GPU only) Install the CUDA-enabled PyTorch build matching your driver first
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Install from GitHub (or locally with `pip install --no-build-isolation -e ".[h5]"`)
uv pip install "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"
wma doctor
```

Or install from a local checkout with standard `pip`:

```shell
git clone https://github.com/xiaji/whitematteranalysis_gpu.git
cd whitematteranalysis_gpu
python -m pip install --no-build-isolation .
# Or editable mode: python -m pip install --no-build-isolation -e ".[h5]"
```

The base package installs the CPU runtime profile (no PyTorch required), including
`threadpoolctl>=3.1,<3.7` for scoped BLAS control. Existing compatible CUDA PyTorch
installations are preserved; `requirements_cpu.txt` and `requirements_gpu.txt` remain
available for requirement-file workflows.

### Unified `wma` CLI

Installing the package exposes both the unified `wma` command and all legacy `wm_*.py` scripts:

```shell
wma doctor                 # Check Python, VTK, CUDA/PyTorch, Triton, h5py, Slicer, and cache paths
wma list                   # List subcommands and resolved script locations
wma org -i subject.vtp -o out_dir -a /path/to/ORG-Atlases -r rig -j 4 -gpu on
wma harden --pure-python -i -j 4 input_dir out_dir transform.tfm
wma build-atlas ...        # Wraps wm_cluster_atlas.py
```

### Python SDK (`whitematteranalysis.pipelines`)

Embed the single-process ORG / multi-atlas pipeline directly in Python scripts, Jupyter notebooks,
or training/inference workflows while reusing resident atlas and GPU state across subjects:

```python
from whitematteranalysis.pipelines import ORGParcellationPipeline, parcellate_subject

# One-shot single subject
res = parcellate_subject(
    "sub-01.vtp",
    "output_dir",
    "/path/to/ORG-Atlases",
    reg_mode="rig",   # or "nonrig"
    jobs=4,
    gpu=True,
)
af_left_pd = res.load_tract("T_AF_left")

# Persistent multi-subject session (reuses registration atlas, spectral atlas, GPU tensors & outlier cache)
with ORGParcellationPipeline("/path/to/ORG-Atlases", reg_mode="rig", jobs=4, gpu=True) as pipe:
    batch = pipe.run_batch("subjects_dir/", "batch_out/", continue_on_error=True)
    print(batch.succeeded, batch.failed, batch.summary_path)
```

### Docker & Apptainer (HPC)

The repository includes a multi-target [Dockerfile](Dockerfile) (`gpu` default + `cpu` lightweight
target) and a GitHub Actions workflow ([.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml))
that publishes prebuilt images to GitHub Container Registry (`ghcr.io/xiaji/whitematteranalysis_gpu`).

The default GPU image uses **`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`** (Triton PTX 8.4,
compatible with NVIDIA Linux drivers `>= 550` across V100 / T4 / RTX 20xx–40xx / A100 / H100).
A `cu128` variant (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, for Blackwell / R570+ drivers)
and a lightweight `cpu` variant are also published. Atlases are mounted at runtime rather than baked
into the image, and caches default under `/tmp/wma-cache` so non-root `--user` runs and read-only
Apptainer/Singularity containers work out of the box.

```shell
# Pull prebuilt image from GHCR (default :latest is cu124; also available: :cu128, :cpu)
docker pull ghcr.io/xiaji/whitematteranalysis_gpu:latest

# Or build locally (default: CUDA 12.4; override GPU_BASE_IMAGE for CUDA 12.8 / Blackwell)
docker build -t wma-gpu:0.4.0 .
# docker build --build-arg GPU_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t wma-gpu:cu128 .
# docker build --target cpu -t wma-cpu:0.4.0 .

# 1. Inspect CUDA, Triton, VTK, and cache status inside the container
docker run --rm --gpus all ghcr.io/xiaji/whitematteranalysis_gpu:latest doctor

# 2. Single-subject ORG parcellation (mount data, atlas, output, and persistent cache)
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /data/subjects:/input:ro \
  -v /data/ORG-Atlases:/atlas:ro \
  -v /data/output:/output \
  -v /data/wma_cache:/tmp/wma-cache \
  ghcr.io/xiaji/whitematteranalysis_gpu:latest \
  org -i /input/sub-01.vtp -o /output -a /atlas -r rig -j 4 -gpu on

# 3. Persistent multi-subject batch parcellation
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

For Slurm / HPC clusters using Apptainer (Singularity), pull directly from GHCR and run with `--nv`:

```shell
apptainer pull wma-gpu-cu124.sif docker://ghcr.io/xiaji/whitematteranalysis_gpu:latest
apptainer exec --nv \
  -B /data/subjects:/input:ro,/data/ORG-Atlases:/atlas:ro,/data/output:/output,/scratch/wma_cache:/tmp/wma-cache \
  wma-gpu-cu124.sif wma org --input-dir /input -o /output -a /atlas -r rig -j 4 -gpu on
```

Install the CUDA-enabled PyTorch build matching your Python, GPU and NVIDIA
driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/)
or [version-specific commands](https://pytorch.org/get-started/previous-versions/).
WMA is not restricted to CUDA 12.1. A CPU-only PyTorch build cannot run CUDA;
check `wma doctor` or `torch.cuda.is_available()` after installation. Triton kernels are used when
Triton is available; the torch CUDA path remains available without Triton.
Linux / Python 3.8–3.12 and Slicer 5.2.2 / Python 3.9 were tested; Windows has not been tested.

| Setting | Behavior |
|---|---|
| `WMA_GPU=auto` (default) | Use available CUDA; otherwise use accelerated CPU |
| `WMA_GPU=on` | Require CUDA; raise an error if unavailable |
| `WMA_GPU=off` | Use CPU |
| `WMA_GPU=off`, `WMA_CPU_FAST=0` | Use the original per-fiber CPU implementation |
| `WMA_GPU_DEVICES=0,1` | Select visible CUDA device indices for supported multi-GPU operations |
| `WMA_GPU_FLOAT32=1` | Opt into float32; default is float64 |
| `WMA_GPU_STRICT=1` | Use strict non-fused float64 registration reductions for bit-exact parity |
| `WMA_GPU_MPS=1` | Opt into experimental Apple Silicon MPS when `WMA_GPU_FLOAT32=1` (`-gpu_float32`) and CUDA is absent |
| `WMA_ATLAS_CACHE=/path/to/cache` | Override the automatic atlas cache directory (outlier reference caches and `.npz` spectral atlas inference caches); `off` disables both |

These environment variables apply to the Python/CLI backend. Slicer pages provide
their own Backend selection. Some CLI tools also expose `-gpu auto|on|off`,
`-gpu_float32`, and `-gpu_devices`; check the individual tool's `--help`.
For those tools, use `-gpu on` explicitly: their default `-gpu auto` overrides
`WMA_GPU`. An explicit GPU request fails before processing if CUDA is unavailable.

The accelerated CPU backend streams distance/similarity tiles and honors CPU
affinity. CUDA performs distance, probability, embedding and centroid operations
on the selected device, using optional Triton kernels with a PyTorch fallback.
Float64 is the default; float32 is an explicit accuracy/performance tradeoff.
Outlier removal shares cached atlas reference statistics across subjects.

| Additional setting | Behavior |
| --- | --- |
| `WMA_CPU_THREADS=4` | Set the accelerated CPU worker budget; otherwise use an affinity-aware default |
| `WMA_CPU_BLAS_THREADS=0` | Disable automatic atlas-labeling BLAS caps; a positive value sets a ceiling |
| `WMA_VTP_COMPRESSION=lz4` | Select faster lossless VTP writing; library/Slicer defaults use compact zlib |
| `WMA_IO_PREFETCH=1` | Opt into whole-file POSIX readahead; default off, benefit depends on storage |
| `WMA_GPU_MIN_PAIR_WORK=4096` | Route tiny outlier clusters (`n_queries * n_fixed < threshold`) to the CPU backend |
| `WMA_GPU_MAX_VRAM_GIB=3.5` | Cap the GPU free-VRAM budget used for tile sizing (useful on shared or low-VRAM GPUs) |
| `WMA_NONRIGID_ANALYTIC_GRAD=1` | Opt into PyTorch analytic gradient for nonrigid `BFGS` (`RegisterTractographyNonrigid`) instead of finite differences |
| `WMA_GPU_PROCESS_POOL=1` | Opt into a spawn `ProcessPoolExecutor` for multi-GPU linear `MultiSubjectRegistration` to bypass SciPy COBYLA's thread lock |

## 3D Slicer extension

Use the [universal installer](slicer/WMAGPU/dist/WMAGPU-Installer.zip).
Extract it, run `InstallWMAGPU.py` inside the target Slicer's Python Interactor,
and click **Install WMAGPU and required dependencies**. The installer detects
Slicer's version and platform, installs missing scientific dependencies through
Slicer's Python while preserving its NumPy/VTK versions, and installs the matching
extension. This ZIP is not an Extensions Manager "Install from file" archive.
Remove an older WMAGPU release and restart before installing.
See [installation instructions](slicer/WMAGPU/README.md#installation-and-use).

The five workflows cover Pipeline, moving-to-fixed/atlas registration, groupwise
registration, atlas building, and selected-node merging. CPU mode needs no
PyTorch; optional CUDA installation targets Slicer's own Python. Registration
retains Transform nodes, and Pipeline can display anatomical tracts or save
results without loading them.

The installer adapts to Slicer's version and platform. Native validation is
Linux Slicer 5.2.2; Windows/macOS and newer Slicer versions require native checks.
macOS/ARM use CPU mode; automatic CUDA setup targets Linux/Windows x86_64.
See the [extension guide](slicer/WMAGPU/README.md) for installation, GPU selection,
registration methods, output visualization, dependencies and packaging.

# Documentation and command-line use

The [command-line reference](doc/command-line-reference.md) covers all **28 scripts
in `bin/`**, with their purpose, every argument and default, input/output layouts,
and example commands. It includes the full ORG pipeline, registration presets,
atlas building and labeling, preprocessing, merging, measurements and quality
control, plus CPU/joblib/GPU backend selection and the internal Slicer helper.

Run `wm_quality_control_tractography.py --help` to check the CLI installation.

The upstream `whitematteranalysis` documentation can be found at
https://whitematteranalysis.readthedocs.io/en/latest/.
Use the guides above for acceleration and Slicer behavior specific to this fork.

* For automated subject-specific ORG parcellation without SlicerDMRI in tensor-free rigid or nonrigid mode (falling back to 3D Slicer automatically wrapped with `xvfb-run -a` on headless Linux only for tensor-bearing inputs or unsupported non-linear pre-transforms), use the single-process CLI [bin/wm_apply_ORG_atlas.py](bin/wm_apply_ORG_atlas.py):

```shell
$ wm_apply_ORG_atlas.py \
  -i input_tractography.vtp \
  -o output_dir \
  -a path_to_atlas/ORG-Atlases-v1.x \
  --atlas-family auto \
  -r rig \
  -j 4 \
  -gpu on
```

`--atlas-family` accepts `{auto,org,east-west,naba}` (default `auto`, reading `ATLAS.json` or atlas subdirectory layout and validating cluster/MRML consistency before running). For standalone linear, BSpline, or composite `.tfm` hardening without 3D Slicer (on tensor-free files), `wm_harden_transform.py` also accepts `--pure-python` or `Slicer=none`/`auto`.

* The legacy master shell script `wm_apply_ORG_atlas_to_subject.sh` (see code [here](bin/wm_apply_ORG_atlas_to_subject.sh)) remains available for SlicerDMRI-based multi-stage workflows and `FiberTractMeasurements`:

```shell
$ wm_apply_ORG_atlas_to_subject.sh \
  -i input_tractography.vtk \
  -o output_dir \
  -a path_to_atlas/ORG-Atlases-v1.x \
  -s /Applications/Slicer5.2.2.app/Contents/MacOS/Slicer \
  -d 1 \
  -m /Applications/Slicer5.2.2.app/Contents/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules/FiberTractMeasurements
```

* A step-by-step tutorial to explain the detailed computational process within the pipeline is provided [here](doc/subject-specific-tractography-parcellation.md).

* The names of the anatomical bundles WMA extracts can be found in the [bundles](doc/bundles.md) page.

# References

**Please cite the following papers for using the code and/or the training data :**
 
    Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ. 
    An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
    NeuroImage, 2018 (179): 429-447

    O'Donnell LJ, Wells III WM, Golby AJ, Westin CF. 
    Unbiased groupwise registration of white matter tractography.
    In MICCAI, 2012, pp. 123-130.

    O'Donnell, LJ., and Westin, CF. Automatic tractography segmentation
    using a high-dimensional white matter atlas. Medical Imaging,
    IEEE Transactions on 26.11 (2007): 1562-1575.

**For projects using Slicer and SlicerDMRI please also include the following text (or similar) and citations:**

* How to cite the [Slicer platform](https://slicer.readthedocs.io/en/latest/user_guide/about.html#how-to-cite)
* An example of how to cite SlicerDMRI (modify the first part of the sentence according to your use case):

    "We performed diffusion MRI tractography and/or analysis and/or visualization in 3D Slicer (https://www.slicer.org/) via the SlicerDMRI project (https://dmri.slicer.org/) (Norton et al. 2017)."
    
    Fan Zhang, Thomas Noh, Parikshit Juvekar, Sarah F Frisken, Laura Rigolo, Isaiah Norton, Tina Kapur, Sonia Pujol, William Wells III, Alex Yarmarkovich, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Ron Kikinis, Hans J Johnson, Carl-Fredrik Westin, Steve Pieper, Alexandra J Golby, Lauren J O’Donnell. 
    SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization. 
    JCO Clinical Cancer Informatics 4, e299-309, 2020.

    Isaiah Norton, Walid Ibn Essayed, Fan Zhang, Sonia Pujol, Alex Yarmarkovich, Alexandra J. Golby, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Steve Pieper, Ron Kikinis, Hans J. Johnson, Carl-Fredrik Westin and Lauren J. O'Donnell. 
    SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21), e101-e103, 2017.
