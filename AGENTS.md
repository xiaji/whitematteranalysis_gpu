# AGENTS.md — Machine-Readable Guide for AI Agents

This repository (`whitematteranalysis_gpu`) is a **headless, pure-Python + CUDA/Triton toolkit** for diffusion MRI (dMRI) streamline tractography registration, spectral fiber clustering, outlier removal, and white matter atlas parcellation (ORG, East-West, NABA). No GUI or 3D Slicer is required for tensor-free `.vtk`/`.vtp` tractograms (3D Slicer `-s` is only needed for per-point diffusion tensor reorientation or unsupported `-t` pre-transforms).

## 1. Environment Discovery (Run First)

Always probe the target runtime in JSON before choosing backend flags:

```shell
wma doctor --json
wma list --json
```

Key fields from `wma doctor --json`:
- `runtime.effective_backend`: `"cuda"`, `"mps"`, `"cpu_fast"`, `"cpu_legacy"`, or `"error (<reason>)"` (for example, when `WMA_GPU=on` is set on a machine with no usable GPU; `wma doctor --json` still exits `0`, so check that this value does not start with `"error"` before assuming a usable backend)
- `optional.torch.cuda_available`, `optional.torch.cuda_devices[]`: GPU count, name, compute capability, VRAM (GiB)
- `optional.triton`: installed Triton version (`null` falls back to PyTorch CUDA)
- `optional.h5py`: non-null if ITK `.h5` transform files are supported
- `runtime.atlas_cache`: `{enabled, path, exists, writable}`

## 2. Installation Rules

- **Never overwrite an existing CUDA PyTorch installation**: `torch` and `triton` are intentionally excluded from default dependencies in `pyproject.toml`.
- Existing Conda GPU environment:
  ```shell
  pip install --no-build-isolation "git+https://github.com/xiaji/whitematteranalysis_gpu.git@main"
  ```
- Optional `.h5` ITK transform support: add `[h5]` extra.
- Containerized execution (no local Python setup):
  - Default GPU (driver `>= 550`, PTX 8.4): `ghcr.io/xiaji/whitematteranalysis_gpu:latest` (`:cu124`)
  - Blackwell / driver `>= 570` (PTX 8.7): `ghcr.io/xiaji/whitematteranalysis_gpu:cu128`
  - CPU-only: `ghcr.io/xiaji/whitematteranalysis_gpu:cpu`
  - Mount caches at `/tmp/wma-cache` (atlas + Triton + matplotlib caches).

## 3. Canonical Invocation Interfaces

### A. Python SDK (Preferred for multi-subject / programmatic pipelines)

```python
from whitematteranalysis.pipelines import ORGParcellationPipeline, parcellate_subject

# Single subject
res = parcellate_subject(
    input_file="/data/sub-01.vtp",
    output_dir="/out",
    atlas_dir="/atlas/ORG-Atlases",
    atlas_family="auto",        # "auto" | "org" | "east-west" | "naba"
    reg_mode="rig",             # "rig" (rigid+affine) | "nonrig" (affine+BSpline)
    jobs=4,
    gpu=True,                   # True | False | None ("auto")
    gpu_float32=False,          # False = float64 parity; True = faster FP32
    keep_intermediates=0,       # 0 = zero-intermediate fast path (default)
)
qc = res.qc_summary             # dict (same schema as qc_summary.json)
af_left = res.load_tract("T_AF_left")

# Persistent batch (reuses spectral atlas, GPU tensors, and outlier reference cache)
with ORGParcellationPipeline("/atlas/ORG-Atlases", reg_mode="rig", jobs=4, gpu=True) as pipe:
    batch = pipe.run_batch("/data/subjects", "/out", continue_on_error=True)
    # batch.total, batch.succeeded, batch.failed, batch.results, batch.errors, batch.summary_path
```

- `ORGParcellationPipeline.run()` scopes GPU settings per call and restores global `wma.gpu.config` on exit.
- `case_id` values must be single directory names without path separators or `..` (validated before touching the filesystem).

### B. Unified CLI

```shell
# Single subject
wma org -i /data/sub-01.vtp -o /out -a /atlas/ORG-Atlases -r rig -j 4 -gpu on

# Nonrigid with fused Triton/PyTorch analytic gradients (recommended for -r nonrig)
WMA_NONRIGID_ANALYTIC_GRAD=1 wma org -i /data/sub-01.vtp -o /out -a /atlas/ORG-Atlases -r nonrig -j 4 -gpu on

# Multi-subject persistent batch
wma org --input-dir /data/subjects -o /out -a /atlas/ORG-Atlases -r rig -j 4 -gpu on --continue-on-error

# Standalone affine / BSpline / Composite transform hardening (no 3D Slicer; add -i to invert)
wma harden --pure-python -j 4 /in_dir /out_dir -t /path/to/transform.tfm
```

## 4. Input / Output Contract

- **Inputs**: Whole-brain tractography `.vtp` or `.vtk` in **RAS** coordinates, plus an atlas directory (`ORG-RegAtlas-*` + `ORG-800FC-*`, or East-West / NABA layout).
- **Atomic Publish (Default)**: Subject outputs are staged in `.staging_<case_id>_<pid>` and atomically renamed to `<output>/<case_id>`. Existing non-empty `<output>/<case_id>` raises `FileExistsError` unless `--no-atomic-publish` / `atomic_publish=False` is set.
- **Outputs (`--keep-intermediates 0`)**:
  - `<output>/<case_id>/AnatomicalTracts/T_*.vtp` (76 tracts for ORG / East-West, 78 for NABA)
  - `<output>/<case_id>/AnatomicalTracts/anatomical_tracts_complete.log`
  - `<output>/<case_id>/qc_summary.json` — includes `case_id`, `atlas_family`, `reg_mode`, `hardening_backend`, `gpu_used`, `gpu_float32`, `input_fibers`, `filtered_fibers`, `outlier_removed_fibers`, `retained_clustered_fibers`, `anatomical_tracts_written`, `tracts` (`{name: {fibers, points}}`), `stage_timings_s`.
  - `<output>/batch_qc_summary.json` (batch mode) — includes `total_subjects`, `succeeded_subjects`, `failed_subjects`, `total_wall_seconds`, `subjects[]`.
- **Exit Codes**: `wma org`, `wma doctor`, `wma list`, and `wma harden` return `0` on success, `1` on runtime failure (or any failed subject in `wma org --input-dir` batch mode), and `2` on invalid CLI arguments or unknown commands. Legacy wrapped `bin/wm_*.py` commands (`wma build-atlas`, `wma cluster`, `wma register`, `wma separate`, `wma download-atlas`) may call bare `exit()` on validation errors and return `0`; verify expected outputs or prefer `wma org` / the Python SDK for automation.

## 5. Performance & Resource Controls

- **Multi-GPU**: Set `WMA_GPU_DEVICES=0,1` (or `-gpu_devices 0,1`).
- **Shared / Low-VRAM GPU**: Cap assumed free VRAM via `WMA_GPU_MAX_VRAM_GIB=<float>` (e.g., `8.0`) to guide adaptive GPU tile sizing. Bounded CPU fallback on OOM applies only to supported host-returning operations (see `doc/acceleration.md`); GPU-tensor-returning APIs such as `gpu.distance.rectangular_similarity_tensor()` re-raise OOM if even a single-row tile does not fit.
- **Speed vs Precision**: `float64` is default (`0.000e+00` coordinate delta vs full-intermediate on 5,000-fiber parity checks); opt into `float32` via `-gpu_float32` / `WMA_GPU_FLOAT32=1` / `gpu_float32=True`.
- **Cache Persistence**: Set `WMA_ATLAS_CACHE=/persistent/dir` (and `TRITON_CACHE_DIR=/persistent/dir/triton`) across runs; set `WMA_ATLAS_CACHE=off` to disable.
- **Nonrigid Speed**: Always set `WMA_NONRIGID_ANALYTIC_GRAD=1` when using `-r nonrig` on CUDA GPUs.

## 6. Verification & Development

- Run tests: `pytest tests/test_packaging_and_cli.py tests/test_pipelines_sdk.py tests/test_org_pipeline_cli.py -v`
- Full CI suite (22 modules): see `.github/workflows/ci.yml`.
- Maintain strict no-torch CPU import compatibility (`sys.modules['torch'] = None`).
