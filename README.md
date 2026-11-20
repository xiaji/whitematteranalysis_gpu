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

### Single-process ORG CLI & Phase 2 GPU updates (`fan93`, 2x RTX 3090)

For tensor-free rigid ORG parcellation, [bin/wm_apply_ORG_atlas.py](bin/wm_apply_ORG_atlas.py) runs
registration, spectral clustering, outlier removal, pure-Python affine hardening,
hemisphere separation and anatomical tract assembly in a single Python process
with no 3D Slicer or SlicerDMRI subprocesses required (recording `qc_summary.json`
and publishing atomically by default):

```shell
wm_apply_ORG_atlas.py -i subject.vtp -o output_dir -a /path/to/ORG-Atlases -r rig -j 4 -gpu on
```

On `sub-13086` (**340,795 streamlines**, `ORG-RegAtlas-100HCP` + `ORG-800FC-100HCP`,
dual NVIDIA RTX 3090, warm `float64` atlas cache, `-r rig -j 4 -gpu on`),
`wm_apply_ORG_atlas.py` completed end-to-end in **2:05.21 (125.21 s)** with
**4,057.2 MiB** peak RSS (`hardening_backend: pure_python_affine`), writing all
**76** `AnatomicalTracts/T_*.vtp` outputs (74 curated bundles plus left/right
`T_FalsePositive`). In microbenchmarks on RTX 3090, Phase 2 adds FP32 + arbitrary
`1 <= P <= 64` points-per-fiber Triton kernels (`5.9x` faster warm `spectral_atlas_label`
and `3.2x` faster warm registration in `float32`), split-C low-VRAM `nearest_centroids`
(`88x–111x` faster than `torch.cdist` in `float32`, `22x–25x` in `float64`),
and multi-GPU scheduling (`1.63x` on 340k `float64` labeling and `1.78x` on 800-cluster
outlier removal across 2 GPUs). See [doc/acceleration.md](doc/acceleration.md#phase-2-benchmarks-fan93-rtx-3090)
for complete Benchmarks A–D tables.

# GPU and 3D Slicer quick start

## Python / command line

Install this fork from a local checkout:

```shell
git clone https://github.com/yishengpoxiao/whitematteranalysis_gpu.git
cd whitematteranalysis_gpu
python -m pip install .
```

Choose one dependency profile in the same Python environment:

```shell
# CPU: base dependencies, no PyTorch required
python -m pip install -r requirements_cpu.txt

# GPU: install a CUDA-enabled PyTorch build first, then the remaining dependencies
python -m pip install -r requirements_gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`python -m pip install .` installs the current CPU profile, including
`threadpoolctl` for scoped BLAS control. The GPU profile includes the CPU profile
and `torch>=2.4.1`, without fixing a CUDA wheel index or exact PyTorch release.
Existing compatible PyTorch installations are reused.

Install the CUDA-enabled PyTorch build matching your Python, GPU and NVIDIA
driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/)
or [version-specific commands](https://pytorch.org/get-started/previous-versions/).
WMA is not restricted to CUDA 12.1. A CPU-only PyTorch build cannot run CUDA;
check `torch.cuda.is_available()` after installation. Triton kernels are used when
Triton is available; the torch CUDA path remains available without Triton.
Linux / Python 3.8 and Slicer 5.2.2 / Python 3.9 were tested; Windows has not been tested.

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
| `WMA_ATLAS_CACHE=/path/to/cache` | Override the automatic outlier atlas cache; `off` disables it |

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

* For automated subject-specific ORG parcellation without SlicerDMRI in tensor-free rigid mode (falling back to 3D Slicer automatically wrapped with `xvfb-run -a` on headless Linux only for nonrigid mode, non-linear `.tfm` pre-transforms, or tensor-bearing inputs), use the single-process CLI [bin/wm_apply_ORG_atlas.py](bin/wm_apply_ORG_atlas.py):

```shell
$ wm_apply_ORG_atlas.py \
  -i input_tractography.vtp \
  -o output_dir \
  -a path_to_atlas/ORG-Atlases-v1.x \
  -r rig \
  -j 4 \
  -gpu on
```

For standalone linear `.tfm` hardening without 3D Slicer (on tensor-free files), `wm_harden_transform.py` also accepts `--pure-python` or `Slicer=none`/`auto`.

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
