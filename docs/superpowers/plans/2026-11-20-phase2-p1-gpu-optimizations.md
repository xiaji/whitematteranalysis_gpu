# Phase 2 (P1) GPU Throughput, Multi-GPU & Single-Process ORG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release FP32/2D/multi-GPU Triton compute throughput (Task 4) and eliminate cross-process/disk/Slicer bottlenecks via a single-process in-memory ORG pipeline with tiered SlicerDMRI-free hardening (Task 5).

**Architecture:** Task 4 unlocks `float32` and `1 <= P <= 64` (keeping `P > 64` on PyTorch `Reduce.cuh`) across all Triton kernels, adds split-C two-stage tiling for `nearest_centroids` and 2D fused cross-fiber registration sums, wires multi-GPU device scheduling into multisubject registration (Affine, Rigid, and Nonrigid), outlier removal, and atlas labeling, and adds opt-in MPS eager `float32` (with MPS-safe atlas cache & result packing). Task 5 adds direct array sampling with float32-equivalent rounding in `congeal_to_atlas.py`, a tensor-aware vectorized `transform.py` module, Slicer-Core `ModelNode` + auto-`xvfb-run` hardening fallbacks, and the unified single-process `bin/wm_apply_ORG_atlas.py` CLI preserving the 40 mm classification length filter.

**Tech Stack:** Python 3, NumPy, SciPy, VTK 9 (`vtk.util.numpy_support`), PyTorch (CUDA / optional MPS), Triton, `unittest`.

**Spec:** [`docs/superpowers/specs/2026-11-20-phase2-p1-design.md`](../specs/2026-11-20-phase2-p1-design.md)

## Global Constraints

- Default precision stays `float64` with TF32 disabled (`allow_tf32 = False`); explicit `float32` (`-gpu_float32` / `WMA_GPU_FLOAT32=1`) keeps TF32 disabled, handles result packing without allocating `torch.float64` on MPS, and preserves `in_place=False` caller immutability on `normalized_similarities` and `_atlas_embedding_block`.
- `use_gpu()` remains strictly CUDA-only by default (`test_gpu_config.py`); MPS activates ONLY when `WMA_GPU_MPS=1`, `float32=True`, CUDA is absent, and `torch.backends.mps.is_available()`. CUDA-stream scheduling never runs on MPS devices.
- Setting `WMA_GPU_STRICT=1` preserves bit-exact (`torch.equal` / `assert_array_equal`) registration parity with `triton=None` (`test_gpu_registration.py`).
- `distance_kernel.nearest_centroids` preserves `IndexError` on `nc == 0`, exact tie-breaking (first centroid wins), `NaN`/`Inf` propagation, and current-device restoration (`test_centroid_kernel.py`).
- Pure-Python/matrix affine paths NEVER drop tensor/vector/normal rotation: whenever `has_tensor_arrays(polydata)` is `True`, delegate to `vtkTransformPolyDataFilter` or Slicer `FiberBundleNode`.
- Guard all new CPU-backed tests importing optional `torch` with `try: import torch except ImportError: self.skipTest('torch is required')`.

## Review Focus

1. **Multi-GPU & MPS isolation without mutating `config.device_index`**: Explicit `device` args across `distance.py`, `linalg.py`, `registration.py`, `gpu/bspline.py`, `atlas_cache.py`, `congeal_multisubject.py`, and `wm_cluster_remove_outliers.py` must restore `torch.cuda.current_device()`, cover `Nonrigid` multisubject device forwarding, avoid CUDA queries/streams on MPS, and avoid `torch.float64` tensors on MPS (`_unpack_atlas_results`). Pinned in Task 4 (`tests/test_multigpu_dispatch.py`, `tests/test_memory.py`, `tests/test_atlas_cache.py`, `tests/test_atlas_blocks.py`).
2. **`float32` + `1 <= P <= 64` Triton parity & split-C `_nearest_centroids` tie/NaN invariance**: 32-lane static chunking (`P <= 64` + `Reduce.cuh` eager fallback for `P > 64`), promoted `tl.int64` indexing in `_similarity_row_sums`, and split-C two-stage `_argmin_combine` (`nq * nc >= 65536`) must keep first-centroid ties and `NaN` propagation exact across split boundaries and partial final splits in both dtypes. Pinned in Task 4 (`tests/test_distance_kernel.py`, `tests/test_centroid_kernel.py`, `tests/test_gpu_registration.py`).
3. **ITK `.tfm` LPS↔RAS inverse math (`FixedParameters` center) & 9-component tensor safety**: `read_itk_affine_matrix_ras` must satisfy `T_fwd = (C·L·C)^(-1)` (`inverse=False`) and `T_inv = C·L·C` (`inverse=True`) including non-zero `FixedParameters` center `o = t + c - A·c`, and `apply_affine_to_polydata` must delegate when tensors/vectors/normals exist. Pinned in Task 5 (`tests/test_transform_and_sampling.py`).
4. **Sampler float32 round-trip & 40 mm classification filter parity**: `_sample_preprocessed_fiber_array` must apply `float32 -> float64` point rounding identically to `filter.mask` on non-integer `float64` inputs, and `wm_apply_ORG_atlas.py` must run `wma.filter.preprocess(reg_pd, fiber_length, ...)` before `spectral_atlas_label`. Pinned in Task 5 (`tests/test_transform_and_sampling.py`, `tests/test_org_pipeline_cli.py`).
5. **Atomic staging, `ModelNode` fallback, and `xvfb-run` worker contract**: `bin/wm_apply_ORG_atlas.py` failures inside staging must clean up without publishing partial outputs; `harden_transform_with_slicer.py` `ModelNode` fallback must reject 9-component tensors explicitly; `test_harden_workers.py` must verify both wrapped (headless Linux) and unwrapped commands cleanly. Pinned in Task 5 (`tests/test_org_pipeline_cli.py`, `tests/test_harden_workers.py`).

---

## File Structure

- **Create**:
  - `whitematteranalysis/transform.py`: `has_tensor_arrays`, `read_itk_affine_matrix_ras`, `apply_affine_to_polydata`.
  - `bin/wm_apply_ORG_atlas.py`: Unified single-process in-memory ORG pipeline CLI + tiered hardening + `qc_summary.json` + atomic publish.
  - `tests/test_multigpu_dispatch.py`, `tests/test_transform_and_sampling.py`, `tests/test_org_pipeline_cli.py`.
- **Modify**:
  - `whitematteranalysis/gpu/config.py`, `whitematteranalysis/gpu/_memory.py`, `whitematteranalysis/gpu/atlas_cache.py`
  - `whitematteranalysis/gpu/distance_kernel.py`, `whitematteranalysis/gpu/distance.py`
  - `whitematteranalysis/gpu/embedding_kernel.py`, `whitematteranalysis/gpu/linalg.py`
  - `whitematteranalysis/gpu/registration_kernel.py`, `whitematteranalysis/gpu/registration.py`, `whitematteranalysis/gpu/bspline.py`
  - `whitematteranalysis/cluster.py`, `whitematteranalysis/register_two_subjects.py`, `whitematteranalysis/register_two_subjects_nonrigid_bsplines.py`, `whitematteranalysis/congeal_multisubject.py`
  - `bin/wm_cluster_remove_outliers.py`
  - `whitematteranalysis/congeal_to_atlas.py`, `bin/harden_transform_with_slicer.py`, `bin/wm_harden_transform.py`
  - `tests/test_gpu_config.py`, `tests/test_distance_kernel.py`, `tests/test_centroid_kernel.py`, `tests/test_atlas_blocks.py`, `tests/test_atlas_cache.py`, `tests/test_gpu_registration.py`, `tests/test_memory.py`, `tests/test_harden_workers.py`

---

### Task 4: FP32 Triton Kernels, P <= 64 & Split-C Tiling, Fused Registration, Multi-GPU & Optional MPS

**Files:**
- Modify: `whitematteranalysis/gpu/config.py`, `whitematteranalysis/gpu/_memory.py`, `whitematteranalysis/gpu/atlas_cache.py`
- Modify: `whitematteranalysis/gpu/distance_kernel.py`, `whitematteranalysis/gpu/distance.py`
- Modify: `whitematteranalysis/gpu/embedding_kernel.py`, `whitematteranalysis/gpu/linalg.py`
- Modify: `whitematteranalysis/gpu/registration_kernel.py`, `whitematteranalysis/gpu/registration.py`, `whitematteranalysis/gpu/bspline.py`
- Modify: `whitematteranalysis/cluster.py`, `whitematteranalysis/register_two_subjects.py`, `whitematteranalysis/register_two_subjects_nonrigid_bsplines.py`, `whitematteranalysis/congeal_multisubject.py`, `bin/wm_cluster_remove_outliers.py`
- Create: `tests/test_multigpu_dispatch.py`
- Modify: `tests/test_gpu_config.py`, `tests/test_distance_kernel.py`, `tests/test_centroid_kernel.py`, `tests/test_atlas_blocks.py`, `tests/test_atlas_cache.py`, `tests/test_gpu_registration.py`, `tests/test_memory.py`

**Interfaces:**
- Produces:
  - `gpu.config.strict_enabled(requested=None) -> bool`, `gpu.config.set_strict(on)`, `gpu.config.mps_available() -> bool` (`WMA_GPU_STRICT`, `WMA_GPU_MPS`)
  - `registration_kernel.pairwise_probability_sums(fixed, moving, sigmasq) -> torch.Tensor` (1D `(Nm,)`)
  - `registration.inner_loop_objective(..., gpu_float32=None)`, `registration.AffineObjective(..., gpu_float32=None)`
  - `distance.iter_rectangular_similarity(..., device=None)`, `distance.similarity_row_sums_from_arrays(..., device=None)`
  - `linalg._prepare_atlas(atlas, dtype=None, device=None)`, `linalg.atlas_label_embedding_blocks(..., gpu_float32=None, device=None)`
  - `cluster.spectral_atlas_label`: multi-GPU sharding when `visible_device_count() > 1` and `number_fibers >= 16384`, plus `gpu_float32` passthrough.

- [ ] **Step 1: Write failing tests for config (`WMA_GPU_STRICT`, `WMA_GPU_MPS`, `inf` VRAM guard), MPS cache/packing/workers, split-C centroids, fused registration (`strict` vs fused, `float32`, `P=50`), `P=50`/`P>64`/`float32` distance kernels, and multi-GPU dispatch**

Add to `tests/test_gpu_config.py`:
- `test_strict_mps_and_nonfinite_max_vram_env`: verify `WMA_GPU_STRICT=1` sets `strict_enabled() == True`; `WMA_GPU_MAX_VRAM_GIB=inf` is ignored (stays `None`); `WMA_GPU_MPS=1` activates `use_gpu()` / `torch.device('mps')` ONLY when `float32=True`, `cuda_available=False`, and `mps_available=True`, and stays `False` / raises `'CUDA is unavailable'` otherwise.
Add to `tests/test_memory.py` & `tests/test_atlas_cache.py` & `tests/test_atlas_blocks.py`:
- `tests/test_memory.py`: `test_mps_device_guards_skip_cuda_calls` verifies `effective_free_vram(torch.device('mps'))` returns fallback bytes without touching `torch.cuda.mem_get_info`.
- `tests/test_atlas_cache.py`: test `atlas_cache.load_reference(..., backend='gpu')` (cache enabled) with mocked `get_device() -> torch.device('mps')` executes without calling `torch.cuda.get_device_name`/`get_device_capability`.
- `tests/test_atlas_blocks.py`: test `atlas_label_embedding_blocks(..., gpu_float32=True)` and mocked `mps` device return complete outputs without constructing any `torch.float64` tensor on device.
Create `tests/test_multigpu_dispatch.py`:
- CPU-mocked unit tests (with `try: import torch` guard) verifying:
  - `wm_cluster_remove_outliers` routes clusters round-robin across mocked 2 CUDA GPUs (`get_device(0)`, `get_device(1)`), preserving log order; and on mocked `mps` with `-j 2` runs without creating `torch.cuda.Stream`.
  - `cluster.spectral_atlas_label` shards fibers across mocked 2 GPUs when `number_fibers >= 16384`, preserving exact embedding/label order.
  - `congeal_multisubject` dispatches subjects round-robin across mocked 2 GPUs in `Affine`, `Rigid`, and `Nonrigid` modes preserving deterministic output order, and runs without `torch.cuda.Stream` on mocked `mps`.
  - `@unittest.skipUnless(wma.gpu.available() and ... device_count() >= 2)` real 2-GPU test comparing 1-GPU vs 2-GPU parity for `spectral_atlas_label`.
Extend `tests/test_centroid_kernel.py`, `tests/test_distance_kernel.py`, and `tests/test_gpu_registration.py`:
- `tests/test_centroid_kernel.py`: add explicit split-C test cases (`nq * nc >= 65536`, e.g., `nq=130, nc=600` with partial final split `600 % BLOCK_C != 0`, cross-split exact ties, and cross-split `NaN`s) across both `torch.float64` and `torch.float32`.
- `tests/test_distance_kernel.py`: test `P in (1, 15, 32, 50, 64, 128)` + `float32`/`float64` Triton (`1 <= P <= 64`) and PyTorch `Reduce.cuh` fallback (`P > 64`), plus `float32` Triton `similarity_row_sums`.
- `tests/test_gpu_registration.py`: update existing bit-exact tests (`test_point_kernel_and_optional_fallback`, `test_kernel_uses_selected_cuda_device`, `test_optimizer_reuses_data_and_refreshes_samples`) to explicitly enable/restore `WMA_GPU_STRICT=1` (`set_strict(True)`), update `stateless(fixed, moving, device=None, gpu_float32=None)` (`L223`), and add separate default-mode (`strict=False`) fused `pairwise_probability_sums` tests (`rtol=1e-13, atol=1e-13`), `gpu_float32=True`, and `P=50` Triton registration tests.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=.:tests /Volumes/WorkShop/Codes/Pointcept/.venv/bin/python3 -m unittest test_gpu_config test_memory test_multigpu_dispatch test_atlas_cache test_atlas_blocks -v`
Expected: FAIL (`strict_enabled`, `mps_available` missing; `test_multigpu_dispatch` fails).

- [ ] **Step 3: Implement `config.py`, `_memory.py`, and `atlas_cache.py`**

- `whitematteranalysis/gpu/config.py`: add `self.strict = False` (`WMA_GPU_STRICT`), `strict_enabled`, `set_strict`, `self.allow_mps = False` (`WMA_GPU_MPS`), `mps_available`, and `math.isfinite(value) and value > 0` guard on `WMA_GPU_MAX_VRAM_GIB`. Update `use_gpu()`, `get_device()`, `visible_device_count()` for strictly opt-in MPS (`allow_mps and float32 and not cuda and mps_available`).
- `whitematteranalysis/gpu/_memory.py`: return `4 << 30` (respecting `max_vram_gib` cap) when `device.type != 'cuda'`, and guard `torch.cuda.*` calls cleanly.
- `whitematteranalysis/gpu/atlas_cache.py`: accept optional `device=None` in `load_reference(..., device=None)`, inspect `dev.type`, only call `torch.cuda.get_device_name`/`get_device_capability` when `dev.type == 'cuda'` (record `[torch.__version__, 'mps']` for MPS), and forward `device=device` to `similarity_row_sums_from_arrays`.

- [ ] **Step 4: Implement FP32 & `1 <= P <= 64` Triton kernels, split-C `_nearest_centroids`, and MPS-safe packing in `distance_kernel.py`, `embedding_kernel.py`, `distance.py`, `linalg.py`, and `cluster.py`**

- `distance_kernel.py`:
  - Parameterize `DTYPE: tl.constexpr` (`tl.float32` / `tl.float64`), using `libdevice.expf` for `float32` and `libdevice.exp` for `float64`; promote element index arithmetic (`fi`, `qi`, `p`, `(fi * P + p) * 3`) to `tl.int64` across `_values`, `_similarity_row_sums`, and `_distances`; use 32-lane static chunk loop `CHUNKS = (P + 31) // 32` when `32 < P <= 64` (`WIDTH=32, STEPS=5`) while leaving `P > 64` on PyTorch eager `Reduce.cuh`.
  - Upgrade `nearest_centroids`: add `DTYPE` support + two-stage split-C kernel (`_nearest_centroids_stage1` over `(cdiv(nq, BQ), cdiv(nc, BLOCK_C))` writing `(nq, n_splits)` `(val, idx)`, followed by `_nearest_centroids_stage2` using `_argmin_combine`) when `nq * nc >= 65536 and nc >= 256`, keeping single-pass kernel for smaller inputs and preserving exact `torch.cdist().min(1)` bit parity.
- `embedding_kernel.py`: allow `B.dtype in (torch.float64, torch.float32)` and cast `offset` to `tl.int64` in `_normalize`.
- `distance.py`: unlock `dtype in (torch.float64, torch.float32)` and `1 <= points_per_fiber <= 64` for Triton in `_fiber_block_distances`, `similarity_row_sums_from_arrays`, `iter_rectangular_similarity`; guard `device.type == 'cuda'` for pinned memory and `torch.cuda.empty_cache()`; add `device=None` kwarg to `similarity_row_sums_from_arrays` and `iter_rectangular_similarity`. Also fix `iter_rectangular_similarity`'s `triton_2d` check (`distance_method in SUPPORTED_DISTANCE_METHODS and sigmasq > 0 and sigma > 0`).
- `linalg.py` & `cluster.py`: add `dtype=None`/`gpu_float32=None` & `device=None` across `linalg.py`; update `_unpack_atlas_results` to pack `(embed, row_sum_2[:, None], b_col_sums[:, None], distances[:, None])` in `embed.dtype` (`float32` or `float64`) without converting `labels` to `torch.float64` on device; update `_nearest_centroids` guard to `embed.is_cuda and centroids.is_cuda and embed.device == centroids.device and embed.dtype == centroids.dtype in (torch.float64, torch.float32)`; forward `gpu_float32` in `cluster.spectral_atlas_label`; add multi-GPU sharding (`visible_device_count() > 1 and number_fibers >= 16384`) using `ThreadPoolExecutor` with independent CUDA device workers and ordered concatenation.

- [ ] **Step 5: Implement FP32, `1 <= P <= 64`, 2D fused `pairwise_probability_sums`, and multi-GPU / MPS-safe registration & outliers**

- `registration_kernel.py`: unlock `float32` + `1 <= P <= 64` (`CHUNKS = (P + 31) // 32`; `P > 64` keeps `squared_distances` / `Reduce.cuh`); add 2D `(BM, BN)` fused kernel `_pairwise_probability_sums` / `pairwise_probability_sums(fixed, moving, sigmasq) -> torch.Tensor`.
- `registration.py`: add `gpu_float32=None`, `_memory.disable_tf32()`, `_memory.effective_free_vram(dev)`, `_memory.is_gpu_oom(exc)`, lazy `probability` allocation in `_inner_loop_objective`, MPS device guards on streams/pin_memory, and `_pairwise_probability` dispatch to `pairwise_probability_sums` unless `strict_enabled()` or `P > 64`.
- `register_two_subjects.py`, `register_two_subjects_nonrigid_bsplines.py`, `gpu/bspline.py`, & `congeal_multisubject.py`: forward optional `device=None` through `RegisterTractography.gpu_device` -> `AffineObjective(..., device=...)` AND `RegisterTractographyNonrigid.gpu_device` -> `NonrigidObjective(..., device=...)`; parallelize `congeal_multisubject` GPU loop across `visible_device_count()` GPUs (when `get_device().type == 'cuda'`) or up to `min(parallel_jobs, 2)` CUDA streams on single CUDA GPU via `ThreadPoolExecutor`, and run without `torch.cuda.Stream` on MPS.
- `bin/wm_cluster_remove_outliers.py`: when `get_device().type == 'cuda'`, route worker threads round-robin across `visible_device_count()` GPUs via `get_device(worker_id % n_gpus)` with per-thread CUDA streams and pass `device=device` into `load_reference` & `similarity_row_sums_from_arrays`; when `device.type == 'mps'`, execute without `torch.cuda.Stream`.

- [ ] **Step 6: Run local non-CUDA unit tests and commit Task 4**

Run: `PYTHONPATH=.:tests /Volumes/WorkShop/Codes/Pointcept/.venv/bin/python3 -m unittest test_gpu_config test_memory test_multigpu_dispatch test_distance_kernel test_centroid_kernel test_embedding_kernel test_atlas_blocks test_gpu_registration test_outlier_reductions test_cli_outlier_reduction -v`
Expected: All non-CUDA tests PASS (CUDA-only tests skipped cleanly).
Commit:
```bash
git add whitematteranalysis/gpu/config.py whitematteranalysis/gpu/_memory.py whitematteranalysis/gpu/atlas_cache.py \
  whitematteranalysis/gpu/distance_kernel.py whitematteranalysis/gpu/distance.py whitematteranalysis/gpu/embedding_kernel.py \
  whitematteranalysis/gpu/linalg.py whitematteranalysis/gpu/registration_kernel.py whitematteranalysis/gpu/registration.py \
  whitematteranalysis/gpu/bspline.py whitematteranalysis/cluster.py whitematteranalysis/register_two_subjects.py \
  whitematteranalysis/register_two_subjects_nonrigid_bsplines.py whitematteranalysis/congeal_multisubject.py \
  bin/wm_cluster_remove_outliers.py tests/test_gpu_config.py tests/test_memory.py tests/test_multigpu_dispatch.py \
  tests/test_centroid_kernel.py tests/test_distance_kernel.py tests/test_atlas_blocks.py tests/test_atlas_cache.py \
  tests/test_gpu_registration.py
git commit -m "perf(gpu): unlock FP32/P<=64 Triton kernels, split-C/2D tiling, fused registration, multi-GPU, and MPS"
```

---

### Task 5: Vectorized Transform/Sampling, Tiered SlicerDMRI-Free Hardening & Single-Process ORG CLI

**Files:**
- Create: `whitematteranalysis/transform.py`, `bin/wm_apply_ORG_atlas.py`, `tests/test_transform_and_sampling.py`, `tests/test_org_pipeline_cli.py`
- Modify: `whitematteranalysis/congeal_to_atlas.py`, `bin/harden_transform_with_slicer.py`, `bin/wm_harden_transform.py`, `tests/test_harden_workers.py`

**Interfaces:**
- Produces:
  - `transform.has_tensor_arrays(polydata) -> bool`
  - `transform.read_itk_affine_matrix_ras(tfm_path, inverse=False) -> numpy.ndarray | None` (`inverse=False` returns `(C·L·C)^(-1)`, `inverse=True` returns `C·L·C` matching ITK LPS inverse storage)
  - `transform.apply_affine_to_polydata(polydata, transform_or_matrix, chunk_points=262144) -> vtk.vtkPolyData`
  - `congeal_to_atlas.SubjectToAtlasRegistration._sample_preprocessed_fiber_array(polydata, role, sample_count, seed) -> numpy.ndarray` (`(3, K, P)` float64, matching `filter.mask` float32 point rounding)
  - `bin/wm_apply_ORG_atlas.py`: single-process CLI (`-i`, `-o`, `-a`, `-s [optional]`, `-r {rig,nonrig}`, `-l/--fiber-length [default 40]`, `-j`, `-gpu`, `-gpu_float32`, `-gpu_devices`, `--keep-intermediates {0,1,2}`, `--no-atomic-publish`)

- [ ] **Step 1: Write failing tests for `transform.py`, `congeal_to_atlas` direct array sampling, `ModelNode` fallback, `xvfb-run`, and `wm_apply_ORG_atlas.py`**

Create `tests/test_transform_and_sampling.py`:
- Test `has_tensor_arrays` (returns `False` for scalars/colors, `True` for 9-component point/cell array or `SetTensors`/`SetVectors`/`SetNormals`).
- Test `apply_affine_to_polydata` vs `vtkTransformPolyDataFilter` (`atol=1e-5` on float32 VTK points, exact on float64) for rigid+affine matrices, empty polydata, and delegation when 9-component tensors are attached.
- Test `read_itk_affine_matrix_ras` against BOTH `io.write_transforms_to_itk_format` (`inverse=True` and `inverse=False` matching `vtkTransformPolyDataFilter`) AND an independently constructed ITK `.tfm` fixture with non-zero `FixedParameters` center `c = [10.0, -20.0, 5.0]` verified against exact LPS `L(p) = A(p - c) + c + t`, plus `None` return on BSpline `.tfm`.
- Test `SubjectToAtlasRegistration._sample_preprocessed_fiber_array` equals `_sample_preprocessed_polydata` + `FiberArray.convert_from_polydata` **bit-for-bit**, specifically using non-integer **`float64`** VTK points (`10.123456789`) to pin intermediate `float32` rounding parity!

Create `tests/test_org_pipeline_cli.py`:
- Synthetic end-to-end test of `bin/wm_apply_ORG_atlas.py` with a minimal synthetic atlas (registration polydata, Nyström atlas `.p`/`.vtp`, cluster VTPs, `cluster_hemisphere_location.txt`, MRML tract definitions) AND input fibers both `< 40 mm` and `>= 40 mm`:
  - `--slicer` is NOT required when `reg_mode == 'rig'` and input has no tensors.
  - Short fibers (`< 40 mm` in registered space) are excluded identically to `wm_cluster_from_atlas.py`.
  - Produces `AnatomicalTracts/T_*.vtp` (all listed tracts), `anatomical_tracts_complete.log`, and valid `qc_summary.json` (`hardening_backend == 'pure_python_affine'`).
  - Atomic publish cleanup: induced error during assembly removes `.staging_*` and creates no final output directory.
- Test `bin/harden_transform_with_slicer.py`: mock `ImportError` on `vtkSlicerTractographyDisplayModuleLogicPython` and verify Slicer-Core `ModelNode` fallback succeeds on tensor-free VTPs and fails clearly if 9-component tensors are present.
- Update `tests/test_harden_workers.py:28-52` (`test_run_slicer_builds_structured_command_and_isolates_environment`) and add pure-Python hardening test: test BOTH unwrapped Slicer command (with `DISPLAY=':0'` or `WMA_NO_XVFB='1'` or `which('xvfb-run')=None`) AND wrapped headless Linux (`sys.platform='linux'`, no `DISPLAY`, `shutil.which('xvfb-run')='/usr/bin/xvfb-run'`), preserving all environment sanitization assertions, plus `--pure-python` linear hardening test.

- [ ] **Step 2: Run Task 5 tests to verify RED**

Run: `PYTHONPATH=.:tests /Volumes/WorkShop/Codes/Pointcept/.venv/bin/python3 -m unittest test_transform_and_sampling test_org_pipeline_cli -v`
Expected: FAIL (`whitematteranalysis.transform` and `bin/wm_apply_ORG_atlas.py` do not exist).

- [ ] **Step 3: Implement `whitematteranalysis/transform.py` and `congeal_to_atlas.py` direct sampling + vectorized save**

- `whitematteranalysis/transform.py`: implement `has_tensor_arrays`, `read_itk_affine_matrix_ras` (with centered offset `o = t + c - A @ c`, returning `(C·L·C)^(-1)` when `inverse=False` and `C·L·C` when `inverse=True`), and `apply_affine_to_polydata` with bounded `262144`-point chunks and exact `vtkMatrix4x4::MultiplyPoint` accumulation order.
- `whitematteranalysis/congeal_to_atlas.py`: add `_sample_preprocessed_fiber_array` gathering `points[global_ids].astype(numpy.float32).astype(numpy.float64)` (reproducing `filter.mask`'s float32 rounding) into `(3, K, P)` `float64` in `iterate()` (with fallback to `_sample_preprocessed_polydata` when `geometry is None` or bounds invalid); update `save_transformed_polydata_to_disk` to use `wma.transform.apply_affine_to_polydata` for linear affine transforms.

- [ ] **Step 4: Implement tiered hardening (`harden_transform_with_slicer.py`, `wm_harden_transform.py`) and `bin/wm_apply_ORG_atlas.py`**

- `bin/harden_transform_with_slicer.py`: add `ModelNode` fallback when SlicerDMRI logic is unavailable, keeping `split_output_dir` working on hardened `ModelNode` polydata and rejecting 9-component tensor files explicitly.
- `bin/wm_harden_transform.py`: add auto `xvfb-run -a` wrapper on headless Linux (bypassed via `WMA_NO_XVFB=1`) and `--pure-python` / `Slicer == 'none'` fast path for linear ITK `.tfm` without tensors.
- `bin/wm_apply_ORG_atlas.py`: implement the complete single-process in-memory ORG pipeline: registration (`rigid_affine_fast`) → `apply_affine_to_polydata` → **40 mm minimum-length preprocessing (`wma.filter.preprocess(..., fiber_length, preserve_point_data=True, preserve_cell_data=True)`)** → `spectral_atlas_label` → per-cluster split → threaded/multi-GPU outlier removal & hemisphere assessment → Tier 1/2/3 hardening → parallel anatomical tract assembly → `qc_summary.json` → staging → atomic publish. Clean up intermediate whole-brain objects before tract assembly.

- [ ] **Step 5: Run local Task 5 + regression tests and commit Task 5**

Run: `PYTHONPATH=.:tests /Volumes/WorkShop/Codes/Pointcept/.venv/bin/python3 -m unittest test_transform_and_sampling test_org_pipeline_cli test_harden_preparation test_harden_workers test_registration_preprocessing test_registration_output -v`
Expected: All tests PASS.
Commit:
```bash
git add whitematteranalysis/transform.py whitematteranalysis/congeal_to_atlas.py \
  bin/harden_transform_with_slicer.py bin/wm_harden_transform.py bin/wm_apply_ORG_atlas.py \
  tests/test_transform_and_sampling.py tests/test_org_pipeline_cli.py tests/test_harden_workers.py
git commit -m "feat(pipeline): add vectorized affine/sampling, tiered SlicerDMRI-free hardening, and single-process ORG CLI"
```

---

## Step 4 & Step 5: Reviews, `fan93` GPU Verification & Benchmarks, and Doc Update
1. Task 4 Reviewer + Task 5 Reviewer + Whole-Branch Final Reviewer + Codex CLI (`codex review - < brief.md`) with max 1 fix round and backlog recording in `docs/review_backlog.md`.
2. Sync to `fan93`, run full CUDA/Triton/2-GPU unittest suite (`CUDA_VISIBLE_DEVICES=3,4`), run Benchmarks A–D on real `sub-13086` + `ORG`, update `README.md` and `doc/acceleration.md` (with `WMA_GPU_STRICT`, `WMA_GPU_MPS`, `bin/wm_apply_ORG_atlas.py`, and observed real benchmark figures), commit, clean `.superpowers/sdd/`, and await `"finish"`.
