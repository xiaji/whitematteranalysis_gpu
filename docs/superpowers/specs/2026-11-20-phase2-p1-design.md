# Phase 2 (P1) GPU Throughput, Multi-GPU & Single-Process ORG Pipeline — Design Spec

- **Date**: 2026-11-20
- **Base Commit**: `c92e92d` (`feat/phase1-p0-gpu-optimizations`)
- **Prior Phase Spec**: [`2026-11-20-gpu-optimization-design.md`](2026-11-20-gpu-optimization-design.md)
- **Status**: v1.1 (Addressed Codex v1 review findings #1–#13; ready for scoped re-review)

---

## 1. Goal & Scope

Implement **Phase 2 (Tasks 4 & 5)** from the approved optimization roadmap:
- **Task 4 (GPU Throughput & Multi-GPU)**: Unlock `torch.float32` and `1 <= P <= 64` (32-lane static chunked reduction covering `P=40, 50` in addition to `P <= 32`, with PyTorch eager fallback for `P > 64` to preserve CUDA `Reduce.cuh` vectorized reduction order) across Triton kernels; redesign `distance_kernel.nearest_centroids` with split-C two-stage reduction to saturate SMs on large `NC` (fixing Phase 1's `8192×800×10` 4.54 ms vs 1.78 ms latency regression while keeping VRAM compact); add a 2D-tiled fused cross-fiber registration probability reduction kernel `registration_kernel.pairwise_probability_sums`; wire `WMA_GPU_STRICT=1` vs default fused registration reduction; remove dead single-chunk `probability` allocation in `registration.py`; wire `visible_device_count()` / `WMA_GPU_DEVICES` into `congeal_multisubject.py` (covering `Affine`, `Rigid`, and `Nonrigid` via `gpu/bspline.py` device forwarding), `bin/wm_cluster_remove_outliers.py`, and `cluster.py:spectral_atlas_label`; add opt-in `WMA_GPU_MPS=1` eager `float32` Apple Silicon support (including MPS-safe atlas cache & result unpacking) and `WMA_GPU_MAX_VRAM_GIB` `isfinite` hardening.
- **Task 5 (Single-Process In-Memory ORG Pipeline & Tiered SlicerDMRI-Free Hardening)**: Add direct array sampling (preserving `filter.mask` `float32 -> float64` point rounding) + vectorized/VTK-exact affine transform in `congeal_to_atlas.py` and new `whitematteranalysis/transform.py`; add pure-Python single-process in-memory ORG pipeline CLI `bin/wm_apply_ORG_atlas.py` (sharing 1 CUDA context, preserving the 40 mm classification length filter, tiered `pure_python_affine` → Slicer `FiberBundleNode` / `ModelNode` hardening, `staging -> atomic publish`, content-level `qc_summary.json`); add Slicer-Core `ModelNode` fallback to `bin/harden_transform_with_slicer.py` and auto `xvfb-run -a` / optional pure-Python linear path to `bin/wm_harden_transform.py`.

---

## 2. Primary-Source Reference Facts (Verified at `c92e92d`)

| Component | Current State at `c92e92d` |
|---|---|
| `whitematteranalysis/gpu/config.py` | `GPUConfig` has `requested`, `float32`, `devices`, `tile_rows`, `max_vram_gib` (`WMA_GPU_MAX_VRAM_GIB`, missing `math.isfinite` guard at `L176`), `min_pair_work` (`4096`), `device_index`; `visible_device_count()` (`L145`) has 0 callers outside `config.py`. |
| `whitematteranalysis/gpu/distance_kernel.py` | `_values`, `_reduce_orientations`, `_distances`, `_centroid_distances`, `_nearest_centroids`, `_similarity_row_sums` hardcode `tl.float64`; `_nearest_centroids` (`L133-162`) launches `(nq,)` programs with `num_warps=4` scanning `NC` serially (4.54 ms vs 1.78 ms at `8192×800×10`). `_similarity_row_sums` (`L170-180`) uses `tl.int32` indices (`_values` `(fi * P + p) * 3` overflows 32-bit for large `P`). |
| `whitematteranalysis/gpu/distance.py` | `_fiber_block_distances` (`L106`) gates Triton on `float64` and `P <= 32`; `similarity_row_sums_from_arrays` (`L412`) and `iter_rectangular_similarity` (`L617`) do likewise and always use `get_device()` without explicit `device=` kwarg. |
| `whitematteranalysis/gpu/embedding_kernel.py` | `normalized_similarities` (`L28`) gates Triton on `B.dtype == torch.float64`; `_normalize` uses 32-bit `offset`. |
| `whitematteranalysis/gpu/linalg.py` | `_prepare_atlas` / `_atlas_embedding_block_gpu` / `atlas_label_embedding_blocks` hardcode `torch.float64` and `get_device()`, and `_unpack_atlas_results` (`L151-152`) casts `labels` to `torch.float64` on device. |
| `whitematteranalysis/cluster.py:591-611` | `spectral_atlas_label` passes `gpu_float32` to `iter_rectangular_similarity`, NOT to `atlas_label_embedding_blocks`, and ignores `visible_device_count() > 1`. |
| `whitematteranalysis/gpu/registration_kernel.py` | All kernels (`_affine_points`, `_squared_distances`, `_mean_squared`) hardcode `float64`; `pairwise_probabilities` writes full `(Nm, Nf)` matrix before `.sum(-1)`. |
| `whitematteranalysis/gpu/registration.py` | `inner_loop_objective` (`L99`) hardcodes `dtype = torch.float64`; `L137` always allocates `probability` even for single chunk (`s == 0 and e == n_moving`); `L113-116,145` don't use `_memory.py`; `register_two_subjects.py:262` and `register_two_subjects_nonrigid_bsplines.py:143,247` pass no `device`. |
| `whitematteranalysis/congeal_multisubject.py:289-301` | When `use_gpu()` is True, runs all subjects sequentially (`for args in arglist`) on device 0 (both `Affine`/`Rigid` and `Nonrigid`). |
| `bin/wm_cluster_remove_outliers.py:320-340` | GPU `ThreadPoolExecutor` pins all threads to `wma.gpu.get_device()` (device 0), ignoring `visible_device_count() > 1`, and creates `torch.cuda.Stream` unconditionally on GPU mode. |
| `whitematteranalysis/congeal_to_atlas.py:164-182,249-266` | `iterate()` constructs temporary `vtkPolyData` via `_sample_preprocessed_polydata` (which casts points through `float32` in `filter.mask:574`) then `FiberArray`; `save_transformed_polydata_to_disk()` uses single-threaded `vtkTransformPolyDataFilter`. |
| `bin/harden_transform_with_slicer.py:55` | Hard-imports `vtkSlicerTractographyDisplayModuleLogicPython` (`FiberBundleNode`), failing if SlicerDMRI is absent. |
| `bin/wm_harden_transform.py:44-69` | Requires manual `xvfb-run` (`-x 1` in `.sh`) on headless Linux and always invokes Slicer even for tensor-free affine transforms. |
| `bin/wm_apply_ORG_atlas_to_subject.sh` | Spawns 7 separate CLI Python processes and writes/reads intermediate VTPs across disk; `wm_cluster_from_atlas.py:36,141-151` filters `< 40 mm` fibers in registered space before `spectral_atlas_label`. |

---

## 3. Existing Test Invariants That Must Not Break

1. `tests/test_gpu_config.py:20-67`: `use_gpu(True/'on')` must raise `RuntimeError('...explicitly requested...CUDA is unavailable...')` when CUDA is unavailable unless `WMA_GPU_MPS=1` + `float32=True` is explicitly enabled; `use_gpu('off')` must never probe CUDA or MPS.
2. `tests/test_distance_kernel.py:42-52,97-108`: `distance._fiber_block_distances` and `_similarity_tensor_block` test `torch.equal` across `(torch.float64, torch.float32)` and `points` including `32` and `50` against `triton=None` PyTorch eager. Using Triton 32-lane static chunking for `1 <= P <= 64` and retaining PyTorch `Reduce.cuh` eager reduction for `P > 64` preserves `torch.equal` across all `P`.
3. `tests/test_centroid_kernel.py`: `distance_kernel.nearest_centroids` must match `torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist').min(1)` bit-for-bit (`torch.equal` on indices, exact/equal_nan on distances), preserve `IndexError` on `nc == 0`, and preserve current CUDA device.
4. `tests/test_embedding_kernel.py`: `normalized_similarities` and `_atlas_embedding_block` keep `in_place=False` default and never mutate caller `B`.
5. `tests/test_gpu_registration.py:120-234`: Existing exact-parity assertions (`test_point_kernel_and_optional_fallback`, `test_kernel_uses_selected_cuda_device`, `test_optimizer_reuses_data_and_refreshes_samples`) are unconditional at `c92e92d`; in Task 4 they must explicitly enable and restore `WMA_GPU_STRICT=1` (and update `stateless(fixed, moving, device=None, gpu_float32=None)`), while separate fused-mode tests verify `WMA_GPU_STRICT=0` within `rtol=1e-13, atol=1e-13`.
6. `tests/test_harden_preparation.py` & `tests/test_harden_workers.py`: Preserve worker env sanitization and update `test_run_slicer_builds_structured_command_and_isolates_environment` (`test_harden_workers.py:28-38`) to test both unwrapped (`DISPLAY` set / `WMA_NO_XVFB=1` / no `xvfb-run`) and wrapped headless Linux commands.

---

## 4. Architecture & Interfaces (Task 4 & Task 5)

### Task 4: GPU Throughput, Multi-GPU & Optional MPS
1. **Config & Memory (`whitematteranalysis/gpu/config.py`, `_memory.py`, `atlas_cache.py`)**:
   - `strict_enabled(requested=None) -> bool`, `set_strict(on)` (env `WMA_GPU_STRICT`, default `False`).
   - `mps_available() -> bool`, `allow_mps` (`WMA_GPU_MPS`, default `False`), plus `WMA_GPU_MAX_VRAM_GIB` `math.isfinite` validation.
   - `use_gpu` / `get_device` / `visible_device_count`: MPS activates ONLY if `config.allow_mps and float32_enabled() and not config.cuda_available() and mps_available()`. `_memory.py` guards CUDA-only APIs (`torch.cuda.*`) when `device.type == 'mps'`.
   - `atlas_cache._implementation(backend, float32, device_index, device_type='cuda')` and `load_reference(..., device=None)`: check `device_type` and only call `torch.cuda.get_device_name` / `get_device_capability` when `device_type == 'cuda'` (record `[torch.__version__, 'mps']` for MPS).
2. **Distance & Centroid Triton Kernels (`distance_kernel.py`, `distance.py`)**:
   - Parameterize `_values`, `_reduce_orientations`, `_distances`, `_similarity_row_sums`, `_centroid_distances`, `_nearest_centroids` with `DTYPE: tl.constexpr` (`tl.float64` or `tl.float32`, matching input tensor dtype). Promote element index arithmetic (`fi`, `qi`, `p`, `(fi * P + p) * 3`) to `tl.int64` across `_values`, `_similarity_row_sums`, and `_distances` to prevent 32-bit overflow at larger `P`. For `32 < P <= 64`, use `WIDTH=32, STEPS=5` with `for chunk in tl.static_range(1, CHUNKS):` (`P <= 64` covers `P=40, 50, 64` while leaving `P > 64` on PyTorch eager `Reduce.cuh` so vectorized >64 reductions remain bit-identical). Use `libdevice.expf` for `float32` and `libdevice.exp` for `float64` (verified on `fan93`).
   - Redesign `nearest_centroids`: when `nq * nc >= 65536 and nc >= 256`, launch a 2D split-centroid first pass `(cdiv(nq, BQ), cdiv(nc, BLOCK_C))` reducing into compact `(nq, n_splits)` `(val, idx)` buffers, then a second pass reducing the `n_splits` partials via `_argmin_combine`. Preserves bit-identical `torch.equal` results while saturating RTX 3090 SMs and keeping extra VRAM `<< (nq, nc)`.
   - Update Triton dispatch caps in `distance.py` (`_fiber_block_distances`, `similarity_row_sums_from_arrays`, `iter_rectangular_similarity`) to `dtype in (torch.float64, torch.float32)` and `1 <= points_per_fiber <= 64`, guard `device.type == 'cuda'` for pinned memory and `torch.cuda.empty_cache()`, and add optional `device=None` kwarg to `similarity_row_sums_from_arrays` and `iter_rectangular_similarity`.
3. **Embedding & Atlas Labeling (`embedding_kernel.py`, `linalg.py`, `cluster.py`)**:
   - Unlock `torch.float32` in `embedding_kernel.normalized_similarities` (`tl.int64` offsets in `_normalize`).
   - Thread `gpu_float32=None, device=None` through `linalg._prepare_atlas`, `_atlas_embedding_block_gpu`, `_atlas_embedding_block`, `_unpack_atlas_results`, `atlas_label_embedding`, `atlas_label_embedding_blocks`:
     - In `_unpack_atlas_results`, NEVER construct `torch.float64` tensors on device in `float32` or `mps` mode: pack `(embed, row_sum_2[:, None], b_col_sums[:, None], distances[:, None])` in `embed.dtype` (`float32` or `float64`) for the single `.cpu().numpy()` copy and transfer `labels` separately via `.cpu().numpy().astype(numpy.int32)`.
     - Update `cluster.spectral_atlas_label` to forward `gpu_float32` and fan out across `visible_device_count() > 1` CUDA devices for `number_fibers >= 16384`.
4. **Registration Kernels & Objective (`registration_kernel.py`, `registration.py`, `register_two_subjects.py`, `register_two_subjects_nonrigid_bsplines.py`, `gpu/bspline.py`)**:
   - Unlock `torch.float32` and `1 <= P <= 64` (32-lane static chunk loop; `P > 64` stays on PyTorch eager `Reduce.cuh`) in `registration_kernel.py`.
   - Add 2D-tiled fused kernel `registration_kernel.pairwise_probability_sums(fixed, moving, sigmasq)` (`BM=16, BN=64`) accumulating `(Nm,)` sums directly in registers without materializing `(Nm, Nf)` VRAM buffers.
   - Add `gpu_float32=None` across `registration.py` (`_transform_fiber_tensor`, `inner_loop_objective`, `objective_value`, `AffineObjective`), remove unused pre-allocation on single-chunk `_inner_loop_objective`, route `_pairwise_probability` to `pairwise_probability_sums` unless `strict_enabled()` or `P > 64`, integrate `_memory.py` OOM/TF32 helpers, guard CUDA-only pinned memory/streams on MPS, and forward `gpu_device=None` in both `RegisterTractography` (`AffineObjective`) and `RegisterTractographyNonrigid` (`NonrigidObjective` + `gpu/bspline.py` `device=None` arg without altering B-spline transform kernels).
5. **Multi-GPU & MPS Scheduling (`congeal_multisubject.py`, `bin/wm_cluster_remove_outliers.py`)**:
   - `congeal_multisubject.py`: forward `device=None` in `congeal_multisubject_inner_loop`; when `use_gpu()` and `len(arglist) > 1` and `get_device().type == 'cuda'`, fan out across `visible_device_count() > 1` GPUs (or up to 2 CUDA streams on single GPU when `parallel_jobs > 1`) via `ThreadPoolExecutor` preserving subject order; when `device.type == 'mps'`, run sequentially without creating `torch.cuda.Stream`.
   - `bin/wm_cluster_remove_outliers.py`: when `get_device().type == 'cuda'`, bind each `ThreadPoolExecutor` worker thread round-robin to `get_device(worker_id % n_gpus)` with per-thread CUDA stream and pass `device=device` into `load_reference` and `similarity_row_sums_from_arrays`; when `get_device().type == 'mps'`, omit `torch.cuda.Stream` while still supporting threaded/sequential execution.

### Task 5: Vectorized Sampling/Transform, Tiered Hardening & Single-Process ORG CLI
1. **`whitematteranalysis/transform.py` (new)**:
   - `has_tensor_arrays(polydata) -> bool`: checks PointData/CellData for 9-component arrays, `GetTensors()`, `GetVectors()`, or `GetNormals()`.
   - `read_itk_affine_matrix_ras(tfm_path, inverse=False) -> numpy.ndarray | None`: parses linear ITK `.tfm` (`AffineTransform_*` / `MatrixOffsetTransformBase_*`, returning `None` for nonrigid/BSpline/HDF5). Using `io.py:702-732`'s exact convention: parse 3×3 matrix $A$, translation $t$, and optional `FixedParameters` center $c$ (default `[0,0,0]`) giving effective LPS offset $o = t + c - A c$ and stored LPS matrix $L = \begin{pmatrix} A & o \\ 0 & 1 \end{pmatrix}$. With $C = \text{diag}(-1, -1, 1, 1)$, $L = C \cdot T_{\text{fwd}}^{-1} \cdot C$:
     - `inverse=True` returns $T_{\text{inv}} = C \cdot L \cdot C$ (atlas → subject inverse transform, matching `wm_harden_transform.py -i`).
     - `inverse=False` returns $T_{\text{fwd}} = (C \cdot L \cdot C)^{-1}$ (subject → atlas forward transform).
   - `apply_affine_to_polydata(polydata, transform_or_matrix, chunk_points=262144) -> vtk.vtkPolyData`: delegates to `vtkTransformPolyDataFilter` whenever `has_tensor_arrays(polydata)` is `True`; otherwise transforms points in bounded `float64` chunks using `vtkMatrix4x4::MultiplyPoint` order while preserving point dtype and shallow-copying cell/attribute arrays.
2. **`whitematteranalysis/congeal_to_atlas.py`**:
   - Add `_sample_preprocessed_fiber_array(polydata, role, sample_count, seed)` gathering `points[conn[offsets[line] + local_ids]]` with the exact intermediate `.astype(numpy.float32).astype(numpy.float64)` rounding of `filter.mask` (`whitematteranalysis/filter.py:558-574`) directly into `(3, K, P)` `float64` without building throwaway `vtkPolyData` objects (falling back to `_sample_preprocessed_polydata` + `FiberArray` if `geometry is None` or bounds check fails).
   - Route affine `save_transformed_polydata_to_disk` through `wma.transform.apply_affine_to_polydata`.
3. **Tiered Hardening (`bin/harden_transform_with_slicer.py`, `bin/wm_harden_transform.py`)**:
   - `harden_transform_with_slicer.py`: fall back cleanly to `vtkMRMLModelNode` + `vtkMRMLModelStorageNode` when SlicerDMRI's `vtkSlicerTractographyDisplayModuleLogicPython` raises `ImportError` (failing with an explicit message if 9-component tensors are present without SlicerDMRI).
   - `wm_harden_transform.py`: automatically prepend `xvfb-run -a` on headless Linux (`not DISPLAY`, `not WMA_NO_XVFB`, `shutil.which('xvfb-run')`); support pure-Python linear hardening when `--pure-python` is requested or `Slicer` is `'none'`/`'auto'` on tensor-free affine transforms.
4. **Single-Process ORG CLI (`bin/wm_apply_ORG_atlas.py`)**:
   - In-memory stages sharing one CUDA context: registration (`rigid_affine_fast`, or optional `nonrig`) → `apply_affine_to_polydata` → **classification-stage minimum-length preprocessing (`wma.filter.preprocess(reg_pd, fiber_length, return_indices=False, preserve_point_data=True, preserve_cell_data=True, verbose=False)`, default `fiber_length=40` matching `wm_cluster_from_atlas.py:36,141-151`)** → `spectral_atlas_label` → in-memory per-cluster split → threaded/multi-GPU outlier removal & hemisphere assessment (`cluster_hemisphere_location.txt`) → tiered hardening (Tier 1 in-memory pure-Python inverse affine + hemisphere split for `rig` without tensors; Tier 2/3 Slicer fallback for `nonrig` or tensors) → parallel 74 anatomical tract assembly → content-level `qc_summary.json` → staging `.staging_<caseID>_<pid>` + atomic publish. Supports `--keep-intermediates {0,1,2}`, `-l/--fiber-length` (default `40`), and omits `-s/--slicer` requirement for Tier 1 runs.

---

## 5. Open Questions: Settled vs. To-Be-Measured

### Settled Decisions
1. **Registration precision & strict switch**: `WMA_GPU_STRICT` defaults to `0` (`False`), enabling fused cross-fiber registration reduction (`pairwise_probability_sums`, `<1e-13` relative error vs strict sum). Setting `WMA_GPU_STRICT=1` preserves bit-exact parity (`torch.equal`) with pre-Phase-2 registration tests.
2. **Float32 gating**: `-gpu_float32` / `WMA_GPU_FLOAT32=1` propagates consistently through distance, registration, embedding, and centroid operations (`cluster.spectral_atlas_label` forwards `gpu_float32` to `atlas_label_embedding_blocks`). Default stays `float64` with TF32 disabled.
3. **MPS opt-in**: Apple Silicon MPS never runs implicitly under `-gpu auto`; it activates ONLY when `WMA_GPU_MPS=1` AND `float32` is enabled AND CUDA is unavailable.
4. **Pure-Python affine safety**: `has_tensor_arrays(polydata)` is checked before any matrix-only transform; polydatas with tensors/vectors/normals or nonrigid transforms always take `vtkTransformPolyDataFilter` / Slicer `FiberBundleNode`.
5. **CLI compatibility**: `bin/wm_apply_ORG_atlas_to_subject.sh` and existing standalone `bin/wm_*.py` scripts stay completely compatible; `bin/wm_apply_ORG_atlas.py` adds the single-process path alongside them.

### To-Be-Measured on `fan93` (RTX 3090)
1. **Triton `float32` `expf` bit-exactness**: Verify on `fan93` whether `libdevice.expf` matches PyTorch CUDA `float32` `exp` bit-for-bit (`test_distance_kernel.py:51,107`), or adjust accordingly during Task 4 validation.
2. **Optimal 2D/split-C tile parameters (`BQ`, `BLOCK_C`, `BM`, `BN`)** for `_nearest_centroids` and `_pairwise_probability_sums` on RTX 3090 (`bench_gpu_centroid_tiles.py` target: beat `1.78 ms` at `8192×800×10`).
3. **1-GPU vs 2-GPU scaling** (`CUDA_VISIBLE_DEVICES=3` vs `3,4`) on `sub-13086` outlier removal, atlas labeling, and multi-subject registration.
4. **End-to-end single-process `bin/wm_apply_ORG_atlas.py` runtime & peak RSS** on full `sub-13086` (340,795 fibers).

---

## 6. Non-Goals (Phase 2)
- No nonrigid B-spline `_transform` / analytical gradient rewrite (Phase 3, Task 6).
- No `cluster.spectral` / `wm_cluster_atlas.py` construction rewrite or multi-atlas East-West/NABA registry (Phase 3, Task 7).
