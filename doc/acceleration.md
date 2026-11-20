# Acceleration overview

WMA accelerates the expensive tractography comparisons in registration, atlas
labeling and outlier removal. The original per-fiber joblib backend remains
available for comparison. See the [README](../README.md#full-pipeline-performance)
for measured end-to-end times and output validation, and the
[command-line reference](command-line-reference.md#shared-acceleration-options)
for backend selection.

## Registration

The optimizer, sampling schedule and registration presets stay the same. The
repeated objective evaluation is faster because:

- **CPU matrix:** NumPy processes blocks of fiber pairs rather than scheduling
  one Python/joblib task per fiber. Contiguous coordinate layouts improve cache
  access, and a reusable worker pool avoids repeated worker startup.
- **GPU:** fixed/moving samples and scratch buffers stay on the selected CUDA
  device across objective evaluations. Batched affine transforms and optional
  Triton kernels (`tl.int64` indexing across `float64`/`float32` and
  `1 <= P <= 64` points per fiber) compute forward/reversed fiber distances and
  probabilities (`pairwise_probability_sums`, including a 2D fused registration
  kernel; set `WMA_GPU_STRICT=1` to force non-fused strict reductions)
  with fewer intermediate tensors and kernel launches. PyTorch provides the
  fallback when Triton is unavailable or `P > 64`. Groupwise multi-subject
  registration (`congeal_multisubject.py`) pre-renders fixed brains on the main
  thread and schedules subjects round-robin across visible GPUs.
- Length eligibility and geometry views are reused between registration samples.
  Selected indices are sorted to preserve input fiber order and the original RNG
  sequence without building a full-tractogram selection mask each iteration.
- Slicer registration reads file-based atlas references as geometry only, omitting
  unused attributes. Subject attributes remain available for the saved output.
- **Nonrigid workflows:** B-spline evaluation has batch CPU and CUDA paths,
  avoiding a Python call to VTK for every point. These paths support nonrigid
  registration; the README's `rig` benchmark measures rigid plus affine only.

Source: [CPU backend](../whitematteranalysis/gpu/cpu_backend.py),
[GPU registration](../whitematteranalysis/gpu/registration.py),
[registration kernels](../whitematteranalysis/gpu/registration_kernel.py),
[B-splines](../whitematteranalysis/gpu/bspline.py).

## Atlas labeling and atlas construction

- **CPU matrix:** distance/similarity blocks feed the Nyström projection directly.
  Only the required eigenvector columns and compact per-fiber results are kept;
  the full subject-by-atlas similarity matrix is not retained. Scoped BLAS
  thread limits avoid competing thread pools on small embedding matrices.
- **GPU:** streamed similarity blocks remain on CUDA through in-place
  normalization, embedding and 2-stage split-C `nearest_centroids` reduction
  (`whitematteranalysis/gpu/distance_kernel.py`, supporting both `float64` and
  `float32` across `1 <= P <= 64` points per fiber without materializing full
  `N_q x N_c` distance matrices). When multiple CUDA devices are visible,
  `spectral_atlas_label` shards fiber blocks across GPUs and reassembles exact
  embeddings and cluster labels. Compact labels, embedding coordinates and
  statistics accumulate on the GPU and transfer to the host once after the block
  loop completes, eliminating per-block D2H pipeline stalls. Tile sizing accounts
  for Triton's 2D footprint and uses pinned host staging for query blocks;
  pairwise/rectangular similarity APIs fuse `exp(-d/sigma^2)` on GPU.
  Experimental Apple Silicon MPS can be enabled with `WMA_GPU_MPS=1` together
  with `WMA_GPU_FLOAT32=1` (`-gpu_float32`) when CUDA is absent (unsupported
  `eigh`/Nyström ops fall back safely to CPU).
- **Shared geometry/I/O:** bulk VTK connectivity/coordinate access replaces
  per-point Python calls, and `FiberArray` resamples lines in 4,096-line chunks
  with bounds validation to keep peak host memory bounded on whole-brain
  tractograms. Cluster splitting preserves fiber order and data
  arrays while avoiding repeated scans of the full tractogram. Cluster counts
  and color means reuse stable label groups instead of rescanning all labels for
  each cluster. On Linux, parallel VTK writers use fork workers with inherited
  read-only polydata; Windows and macOS use one writer per thread. Lossless LZ4
  intermediate files reduce output overhead. When rendering is disabled,
  labeling releases its unfiltered input after length filtering and its complete
  labeled tractogram after splitting, before the cluster writers start.
- Atlas construction reuses the accelerated distance/similarity operations.
  Sampling and clustering settings are kept consistent across backends.

Source: [CPU labeling](../whitematteranalysis/gpu/cpu_backend.py),
[GPU distances](../whitematteranalysis/gpu/distance.py),
[GPU embedding](../whitematteranalysis/gpu/linalg.py),
[cluster output](../whitematteranalysis/cluster.py),
[VTK I/O](../whitematteranalysis/io.py).

Whole-file synchronous readahead is disabled by default. Two balanced pairs of
private cold C028 reads were about 3–5 seconds slower with its advice sweep.
However, a subsequent complete cold-data pipeline pair read the input in
15.6 seconds with readahead and 20.0 seconds without it. Storage conditions vary;
set `WMA_IO_PREFETCH=1` to opt in where it helps. Normal VTK reads preserve the
same geometry and attributes either way.

The four latest complete runs also included one 34.0-second CPU input read:
only 3.7 seconds were process CPU time, with 5.9 MB of storage-attributed reads
against a 3.6 GB payload. These counters do not identify the remaining wait.
That event had no concurrent kernel/scheduler trace, so its cause remains
unresolved. The README includes the observed wait in the reported total.
The separate cold-data pair repeatedly sampled blocked page/block-layer waits
(`wait_on_page_bit_common`, `blk_mq_get_tag`) with negligible runqueue delay.
This supports an I/O/page-wait explanation for those cold reads, while leaving
the exact storage or kernel cause open.

On Linux, geometry preparation and GPU host-output buffers temporarily disable
NumPy's huge-page allocation advice to avoid kernel compaction stalls on
fragmented memory. Overlapping geometry calls share the scope, and the previous
policy is restored afterward. The hint is process-wide while a scope is active,
including concurrent allocations in other threads. Set
`NUMPY_MADVISE_HUGEPAGE=1` to keep NumPy's allocation advice.
This changes memory allocation, not values or precision.

## Outlier removal

- Independent clusters run concurrently: in GPU mode, `wm_cluster_remove_outliers.py`
  uses a single-process `ThreadPoolExecutor` with one `torch.cuda.Stream` per
  worker thread (avoiding multiple CUDA contexts and preserving CLI
  `-gpu_float32` / `-gpu_devices` settings); CPU mode keeps `joblib.Parallel`.
- CPU/GPU blocks replace per-fiber distance loops while keeping the same
  probability calculation and rejection rule. Empty subject clusters skip
  unnecessary reference comparisons, and tiny clusters below
  `WMA_GPU_MIN_PAIR_WORK` (default `4096` fiber pairs) route directly to the
  accelerated CPU backend to avoid CUDA launch overhead.
- GPU outlier scoring uses a fused Triton row-reduction kernel
  (`similarity_row_sums`) for `Mean` distance without allocating a full
  subject-by-atlas distance matrix. In float64 mode, scores near the rejection
  cutoff are recomputed with the CPU metric and NumPy reductions to preserve
  cutoff-sensitive decisions. Explicit float32 mode remains approximate and
  skips this refinement.
- Atlas coordinates and leave-one-out reference probabilities can be cached
  across subjects. Cache identity includes atlas file metadata, numerical
  settings, backend/precision, library versions and implementation identity.
  Subject-dependent comparisons are always recomputed.
- The original CPU fallback also reads atlas clusters as geometry only; its
  keep/reject decisions are checked against full-attribute reference reads.
- The generic `filter.remove_outliers` API uses tiled GPU `topk`
  (`nearest_neighbor_distances_from_fiber_array`) so host memory stays $O(N)$
  instead of materializing an $O(N^2)$ distance matrix, finishing the compact
  three-neighbor reduction in NumPy to preserve exact IEEE division and
  summation order. For two or three fibers it uses the available neighbors;
  a singleton is rejected.

The atlas-based CLI requires identical subject/atlas cluster filenames before
writing outputs. Atlas loading requires a Boolean `bilateral` field; missing or
string-valued settings are rejected instead of silently changing reflection.

The README comparison starts with empty accelerated reference caches. Later
subjects may benefit from a warm cache; that benefit is not added to the reported
cold-cache speedup.

Source: [outlier command](../bin/wm_cluster_remove_outliers.py),
[reference cache](../whitematteranalysis/gpu/atlas_cache.py).

## Tiered hardening and single-process ORG pipeline

Hemisphere assessment, transform hardening, hemisphere separation and anatomical
tract assembly run on the CPU:

- **Tier 1 (SlicerDMRI-free pure-Python affine):** [whitematteranalysis/transform.py](../whitematteranalysis/transform.py)
  provides `has_tensor_arrays`, LPS-to-RAS linear ITK `.tfm` parsing
  (`read_itk_affine_matrix_ras`), and bounded-chunk `float64`
  `apply_affine_to_polydata` (used by `congeal_to_atlas.py`,
  `wm_harden_transform.py --pure-python` / `Slicer=none`/`auto`, and
  `bin/wm_apply_ORG_atlas.py` in tensor-free `-r rig` mode). Any tensor, vector,
  normal, or 9-component array triggers automatic fallback to
  `vtkTransformPolyDataFilter` / 3D Slicer.
- **Tier 2 (`vtkMRMLModelNode` fallback without SlicerDMRI):**
  [bin/harden_transform_with_slicer.py](../bin/harden_transform_with_slicer.py)
  falls back to `vtkMRMLModelNode` when `vtkMRMLFiberBundleNode` is unavailable,
  enabling plain 3D Slicer (without SlicerDMRI installed) to harden non-linear or
  tensor-free transforms, and [bin/wm_harden_transform.py](../bin/wm_harden_transform.py)
  automatically wraps headless Linux Slicer invocations with `xvfb-run -a` when
  `DISPLAY` is unset (disable with `-no_xvfb` / `--no-xvfb` or `WMA_NO_XVFB=1`).
- **Tier 3 (Full SlicerDMRI hardening):** Legacy Slicer `FiberBundleNode`
  hardening remains active when tensor arrays are present and SlicerDMRI is
  installed.
- **Single-process ORG pipeline:** [bin/wm_apply_ORG_atlas.py](../bin/wm_apply_ORG_atlas.py)
  orchestrates registration, 40 mm length filtering, spectral clustering,
  multi-GPU/threaded outlier removal + hemisphere assessment, tiered hardening
  + separation, and anatomical tract assembly in one Python process (using
  `ThreadPoolExecutor` / `use_threads=True` VTP writes after CUDA init to avoid
  post-CUDA `fork()` hazards, writing `qc_summary.json`, and staging outputs
  atomically).

### Legacy Slicer batching notes (shell pipeline)

Hemisphere assessment, inverse transformation with Slicer, hemisphere separation
and anatomical tract assembly run on the **CPU/Slicer in all three backend
profiles** of `wm_apply_ORG_atlas_to_subject.sh`. Changing `-g` does not select
another implementation for these stages.
Their improvements come from vectorized geometry/label access, independent
file processing, lossless intermediate compression and batching many files into
Slicer workers. They do not contain the large distance matrices that benefit
from CUDA.

The ORG shell pipeline assigns hemisphere labels while each outlier-filtered
cluster is still in memory, before its existing write. It writes the assessment
completion marker only after all clusters succeed, so the following assessment
step skips an additional full read and rewrite. Its work is included in the
outlier-removal timing. The standalone assessment command remains available.

The ORG shell pipeline now splits hemispheres inside each inverse-transform
worker while the hardened polydata is still in memory. Each worker reuses its
fiber-bundle, storage and display nodes across files; `vtkMRMLStorageNode.ReadData`
preserves Slicer's input-reading semantics, and the native transform-hardening
logic is unchanged. The worker removes those nodes when its batch finishes,
strips inherited `PYTHONPATH` and limits BLAS/OpenMP/ITK and classic VTK threads
to one. Batch processes disable GUI module discovery and import only the native
SlicerDMRI fiber-bundle and transform logic from Slicer's registered module paths.
Transforms load through the same native MRML storage reader, including nonlinear
HDF5 transforms. No Slicer version or installation path is hardcoded.

For a fresh rigid ORG run, Slicer initialization now overlaps outlier removal.
Workers wait before enumerating inputs; the parent releases them atomically only
after the producer succeeds and its filenames match the initial catalogue.
Preparation failure or interruption releases waiting workers without writing a
transform completion marker. No startup work is moved outside the pipeline clock.

With hemisphere splitting enabled, a lossless WMA writer saves complete VTP
files and the split outputs from the same in-memory polydata. Full VTP files
retain `SPACE=RAS` metadata; LZ4 is the default compression and an
explicit compression setting is honored. Without splitting, Slicer's storage
writer continues to save the transformed file, as it does for legacy VTK files.
Storage coordinate settings reset per file to match the GUI's RAS defaults.
Hemisphere masks use the same
selection algorithm as standalone separation. The pipeline retains the full
transformed files and avoids reading them again for separation. The following
separation command verifies the three output catalogues and repairs incomplete
sets. Its timing is therefore verification time; split work is included in the
inverse-transform stage.

Pass `--profile` to `wm_harden_transform.py` to retain each worker's stdout log
and JSON timings in its `wma-harden-*` directory. The profile separates startup,
native module preparation, imports, input waiting, reads, transform loading,
hardening, writing and splitting. `preparation.json` records preparation and
combined wall time; overlapping intervals must not be added twice.

Small differences between the three full-run columns do not establish a backend
regression: they also include process startup, filesystem cache state and shared
workstation load. Replacing one column with another run's faster times would not
be a measured pipeline result. Execution strategies must be compared on identical
inputs, and any updated total must come from a complete timed run.

## Memory & OOM hardening

- CUDA entry points disable TF32 (`allow_tf32 = False` for matmul and cuDNN)
  before compute to protect numerical precision.
- Tiled GPU loops detect both `torch.cuda.OutOfMemoryError` and cuBLAS/cuDNN
  OOM `RuntimeError`s (`whitematteranalysis/gpu/_memory.py`), halving
  `tile_rows` down to `1`. When a host-returning API still exhausts VRAM at
  `tile_rows == 1`, remaining rows finish in bounded 16 MiB CPU batches rather
  than crashing. GPU-tensor-returning APIs re-raise at `tile_rows == 1`.
- Set `WMA_GPU_MAX_VRAM_GIB` to cap the effective free-VRAM budget used for
  tile sizing on shared or low-VRAM GPUs.

## Numerical behavior

Float64 is the default. Fiber orientation handling, bilateral comparisons,
thresholds and cluster rejection rules are preserved. GPU centroid assignment
uses direct coordinate differences and selects the first centroid on a tie.
The README records the actual comparison against joblib, including small GPU
coordinate/transform differences; it does not claim universal byte identity.
`-gpu_float32` is an explicit speed/accuracy tradeoff and can change near-tie
labels. Filesystem and rendering work remain CPU operations.

The correctness audit uses seed `20261028` to select 2,048 fibers from the saved
C028 joblib registered tractogram. With the trained ORG atlas parameters, both
accelerated label arrays match joblib exactly. Maximum embedding differences
are `3.11e-15` (CPU) and `1.78e-14` (GPU). Estimated row sums differ by at most
`2.98e-8` and `5.54e-8`, respectively; these are large intermediate sums, with
errors relative to their maximum magnitude below `5.12e-14`. All checked
continuous outputs pass `rtol=1e-10, atol=1e-10`.

Independent comparisons against the upstream `similarity.py` use 128 query by
64 reference fibers, 15 points per fiber, all four supported distance methods,
both bilateral settings and thresholds 0 and 2. CPU distances are exact; GPU
maximum absolute error is `7.28e-12`. An affine objective with 128 moving and
64 fixed fibers matches the upstream value exactly on both backends. These
sampled checks supplement the earlier full-output comparison; they do not
establish accuracy for every input or for optional float32 mode.

Regression tests additionally exercise cutoff-sensitive masks, empty inputs,
one-to-three-fiber neighborhoods, atlas validation, explicit GPU failure, fused
and eager kernels, B-splines, file metadata and Slicer workflows. Float64 retains
unit coordinate differences even around `1e8`; float32 can erase those differences
and remains opt-in. Explicit `-gpu on` now requires usable CUDA rather than
silently falling back to CPU.

## Phase 2 benchmarks (`fan93`, RTX 3090)

Measured on `fan93` (NVIDIA GeForce RTX 3090 GPUs 5 & 6, PyTorch `2.8.0+cu128`,
Triton `3.4.0`, SciPy `1.13.1`, NumPy `2.0.2`, VTK `9.6.2`) using `sub-13086`
(**340,795 fibers**) and the `ORG-RegAtlas-100HCP` + `ORG-800FC-100HCP` atlases.

### Benchmark A: Split-C `nearest_centroids` microbenchmark (GPU 5, 31 repeats)

All shapes and dtypes matched `torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist').min(1)`
exactly (`max_diff = 0.00e+00`, `idx_eq = True`):

| Dtype | `(N_q, N_c, P)` | `torch.cdist + min` | Split-C Triton | Speedup | Peak VRAM (cdist -> Split-C) |
| --- | --- | ---: | ---: | ---: | ---: |
| `float64` | `(2048, 800, 10)` | `11.909 ms` | `0.533 ms` | **22.34x** | `12.78 MiB` -> `0.53 MiB` |
| `float64` | `(8192, 800, 10)` | `47.696 ms` | `1.932 ms` | **24.69x** | `51.00 MiB` -> `1.97 MiB` |
| `float64` | `(16384, 800, 10)` | `95.806 ms` | `3.813 ms` | **25.12x** | `102.06 MiB` -> `3.94 MiB` |
| `float64` | `(2048, 800, 20)` | `12.098 ms` | `0.863 ms` | **14.02x** | `13.50 MiB` -> `1.18 MiB` |
| `float32` | `(2048, 800, 10)` | `3.491 ms` | `0.140 ms` | **24.93x** | `6.47 MiB` -> `0.80 MiB` |
| `float32` | `(8192, 800, 10)` | `13.901 ms` | `0.158 ms` | **88.04x** | `25.58 MiB` -> `2.99 MiB` |
| `float32` | `(16384, 800, 10)` | `27.552 ms` | `0.247 ms` | **111.49x** | `51.22 MiB` -> `6.00 MiB` |
| `float32` | `(2048, 800, 20)` | `3.501 ms` | `0.147 ms` | **23.87x** | `6.89 MiB` -> `0.78 MiB` |

Versus Phase 1's `float64`/`P=10`-only 2D tiled kernel (`c92e92d`, `bench_gpu_centroid_tiles.py`),
Phase 2 generalizes to `float32` and `1 <= P <= 64` while cutting peak extra tensor
allocations by ~29x–50x (`13,139,968 B -> 262,144 B` at `(2048, 800, 10)`;
`52,559,872 B -> 1,048,576 B` at `(8192, 800, 10)`; `13,139,968 B -> 458,752 B` at
`(2048, 800, 20)`; `81,408 B -> 5,120 B` at `(257, 37, 10)`), at `0.549 -> 0.662 ms`,
`1.796 -> 2.236 ms`, `0.883 -> 1.021 ms`, and `0.138 -> 0.112 ms` respectively on
consumer RTX 3090 (`1:64` FP64 throughput).

### Benchmark B: Precision & kernel fusion (`sub-13086`, 340,795 fibers, GPU 5)

| Mode | Cold-process Reg / Label | Warm-GPU Reg / Label | Transform Max Diff | 340,795-fiber Label Match |
| --- | ---: | ---: | ---: | ---: |
| `float64 strict` (`WMA_GPU_STRICT=1`) | `6.900 s` / `7.564 s` | `2.424 s` / `4.394 s` | `0` (ref) | `100.0000%` (ref) |
| `float64 fused` (default, `WMA_GPU_STRICT=0`) | `6.335 s` / `4.496 s` | `2.376 s` / `4.305 s` | `2.681e-12` | `100.0000%` (`0` diffs) |
| `float32 fused` (`WMA_GPU_FLOAT32=1`) | `5.250 s` / `6.503 s` | `0.757 s` / `0.742 s` | `4.485e-02` | `100.0000%` (`0` diffs) |

Cold-process subprocess times include first-time CUDA/Triton JIT initialization;
in warm-GPU steady state, `float32` speeds up `rigid_affine_fast` registration by
**3.20x** and 340k `spectral_atlas_label` by **5.92x** vs `float64`.

### Benchmark C: Single-GPU (`5`) vs dual-GPU (`5, 6`) scaling & parity

| Workload | 1 GPU | 2 GPUs | Speedup | Numerical Parity (1 GPU vs 2 GPUs) |
| --- | ---: | ---: | ---: | --- |
| 340k `spectral_atlas_label` (`float64`, warm) | `4.374 s` | `2.690 s` | **1.63x** | `0` label diffs; `embed_max_diff = 0.000e+00` |
| 340k `spectral_atlas_label` (`float32`, warm) | `0.843 s` | `0.833 s` | `1.01x` | `0` label diffs; `embed_max_diff = 4.292e-06` |
| 800-cluster `wm_cluster_remove_outliers.py -j 4 -gpu on` (warm `f64` cache) | `19.365 s` | `10.870 s` | **1.78x** | Identical across all 800 clusters & Phase 1 (`230,639` in -> `43,336` kept / `187,303` rejected) |
| `MultiSubjectRegistration` (4 subj x 2,000 fibers, 1 Affine + 1 Nonrigid, `parallel_jobs=4`) | `3.043 s` (`1.516 s` + `1.527 s`) | `1.854 s` (`1.607 s` + `0.247 s`) | **1.64x** | `aff_diff = 0.000e+00`; `nr_diff = 0.000e+00` |

### Benchmark D: End-to-end single-process `wm_apply_ORG_atlas.py` (`sub-13086`, GPUs `5, 6`, `-r rig -j 4 -gpu on`)

Both runs verified `hardening_backend = "pure_python_affine"`, `has_tensor_arrays = false`,
`input_fibers = 340,795`, `registered_fibers = 340,795`, `filtered_fibers = 227,409`,
`excluded_short_fibers = 113,386`, `anatomical_tracts_written = 76 / 76` (74 curated
tracts plus left/right `T_FalsePositive`), and `anatomical_tracts_complete.log`.

| Run | Wall Time (`/usr/bin/time -v`) | Peak RSS | `clustering` Stage | `outlier_and_hemisphere` | Retained Clustered Fibers | Total Fibers Across 76 `T_*.vtp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `float64` (warm `f64` atlas cache) | **2:05.21** (`125.21 s`) | `4,154,596 KiB` (`4,057.22 MiB`) | `90.237 s` | `5.581 s` | `48,906` | `37,323` |
| `-gpu_float32` (first cold `f32` atlas cache) | **2:17.50** (`137.50 s`) | `4,153,856 KiB` (`4,056.50 MiB`) | `91.904 s` | `15.938 s` | `48,497` | `37,007` |

Notes:
- Between `float64` and `-gpu_float32`, **47 / 76** anatomical tracts had identical
  fiber counts (`median_abs_delta = 0.0`, `max_abs_delta = 74` on `T_Sup-P_left.vtp`).
- The ~90 s `clustering` stage is dominated by CPU/I/O work (`ORG-800FC-100HCP/atlas.p`
  load and single-threaded `mask_all_clusters` splitting 227,409 fibers into 800
  polydata objects); GPU similarity/embedding itself is < 3 s (`float64`) / < 1 s (`float32`).
- In the `-gpu_float32` run, `WMA_ATLAS_CACHE` only held pre-warmed `float64`
  entries, so `outlier_and_hemisphere` (`15.938 s` vs `5.581 s`) included building
  the 800 `float32` cache files on first run.
