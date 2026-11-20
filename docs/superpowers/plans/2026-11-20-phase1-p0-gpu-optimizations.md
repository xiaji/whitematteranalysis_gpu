# Phase 1 (P0) GPU Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the three P0 GPU/memory bottlenecks in `whitematteranalysis_gpu` (outlier multiprocess/unfused row sums, atlas labeling blocking D2H syncs & unfused kernels, and host/VRAM memory safety) with zero float64 accuracy regression.

**Architecture:** Three independent, sequentially verifiable tasks: Task 1 fixes `wm_cluster_remove_outliers.py` concurrency/CLI config propagation and fuses `Mean` similarity row reductions in Triton; Task 2 fuses nearest-centroid reduction and in-place normalization while keeping atlas labeling outputs GPU-resident until loop end; Task 3 chunks `FiberArray` conversion, moves `filter.remove_outliers` to $O(N)$ GPU `topk`, and centralizes OOM fallback + TF32 disabling in `gpu/_memory.py`.

**Tech Stack:** Python 3, NumPy, VTK 9 (`vtk.util.numpy_support`), PyTorch (CUDA), Triton, `unittest` (`python -m unittest`).

**Spec:** [`docs/superpowers/specs/2026-11-20-gpu-optimization-design.md`](../specs/2026-11-20-gpu-optimization-design.md) (Fact Sheet: [`docs/superpowers/specs/2026-11-20-gpu-optimization-fact-sheet.md`](../specs/2026-11-20-gpu-optimization-fact-sheet.md))

## Global Constraints

- Default precision is `float64`; `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False` must be enforced before CUDA compute.
- `embedding_kernel.normalized_similarities(B, a, c)` and `linalg._atlas_embedding_block(B, atlas, prepared)` default to `in_place=False` and must not mutate caller tensor `B` (`test_embedding_kernel.py`).
- `distance_kernel.centroid_distances(queries, centroids)` remains intact; `linalg._nearest_centroids` matches `torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist').min(1)` bit-for-bit on ties (first centroid wins), `NaN`/`Inf`, strides, and empty inputs (`test_centroid_kernel.py`).
- `bin/wm_cluster_remove_outliers.py` preserves exact CLI flags and `cluster_log.txt` column layout (`test_cli_outlier_reduction.py`).
- No new runtime dependencies beyond `requirements_cpu.txt` + optional `torch`/`triton`.
- All non-CUDA tests must pass on macOS (`python -m unittest`); each task includes an explicit RTX 4090 GPU verification & benchmark command.

## Review Focus

1. **NaN/Inf and tie-breaking in fused `nearest_centroids` kernel**: When a query embedding contains `NaN` or ties two centroids at equal distance, `nearest_centroids` must match `torch.cdist().min(1)` (first centroid wins on tie; `NaN` propagates identically). Pinned in Task 2 Step 1 (`tests/test_centroid_kernel.py`).
2. **Thread-safety of `wm_cluster_remove_outliers.py` with `ThreadPoolExecutor` + `location_data` and VTK I/O**: Concurrent worker threads share `location_data` (read-only) and write separate output `.vtp` files while logging deterministically in cluster order `1..N`. Pinned in Task 1 Step 1 (`tests/test_cli_outlier_reduction.py`).
3. **Malformed or non-contiguous VTK connectivity in chunked `FiberArray` and `conn_is_arange`**: Polydata with out-of-range offsets/point IDs or shuffled connectivity must safely fall back or index `conn[idx]` without silent corruption. Pinned in Task 3 Step 1 (`tests/test_geometry_views.py`).
4. **Duplicate zero-distance fibers & `N <= 3` edge cases in GPU `filter.remove_outliers` `topk`**: When `N == 0`, `N == 1`, `N == 2`, `N == 3`, or multiple identical fibers exist (multiple `0.0` distances), GPU `topk` must drop exactly one self-distance and match `_mean_nearest_neighbor_distance` (`((d0 + d1) + d2) / 3.0`). Pinned in Task 3 Step 3 (`tests/test_filter_outliers.py`).
5. **Non-`OutOfMemoryError` cuBLAS/cuDNN OOM and `tile_rows == 1` exhaustion**: A `RuntimeError("CUDA error: out of memory")` or persistent OOM at `tile_rows == 1` must trigger tile halving and then clean `cpu_backend` fallback for remaining rows rather than crashing or returning partial uninitialized rows. Pinned in Task 3 Step 5 (`tests/test_memory.py`).

---

## File Structure

- **Create**:
  - `whitematteranalysis/gpu/_memory.py`: Centralized `is_gpu_oom(exc)`, `disable_tf32()`, `effective_free_vram(device)`, and `WMA_GPU_MAX_VRAM_GIB` cap helper.
- **Modify**:
  - `bin/wm_cluster_remove_outliers.py`: GPU-mode `ThreadPoolExecutor` + per-worker `torch.cuda.Stream`; retain `joblib.Parallel` for CPU mode.
  - `whitematteranalysis/gpu/config.py`: Add `max_vram_gib` (`WMA_GPU_MAX_VRAM_GIB`) and `min_pair_work` (`WMA_GPU_MIN_PAIR_WORK`, default `4096`) config fields + env parsing.
  - `whitematteranalysis/gpu/distance_kernel.py`: Add fused `_similarity_row_sums` / `similarity_row_sums(...)` (Task 1) and fused `_nearest_centroids` / `nearest_centroids(...)` (Task 2).
  - `whitematteranalysis/gpu/distance.py`: Wire fused `similarity_row_sums`, widened refinement band, and small-cluster CPU routing (Task 1); fix `iter_rectangular_similarity` tile sizing + pinned H2D and fuse `similarity_sigmasq` in `pairwise/rectangular_similarity_matrix` (Task 2); add `nearest_neighbor_distances_from_fiber_array`, GPU-side `total_similarity_matrix_strict` sign-tensor reflection (preserving NumPy `values.sum(1)`), and `_memory.py` bounded-batch CPU OOM fallback + TF32 guard across all 6 tiled GPU APIs (Task 3).
  - `whitematteranalysis/gpu/atlas_cache.py`: Include `gpu/_memory.py` in `_implementation()` hash list (Task 3).
  - `whitematteranalysis/gpu/embedding_kernel.py`: Support `in_place=False` and `return_col_sums=False` in `normalized_similarities` (Task 2).
  - `whitematteranalysis/gpu/linalg.py`: Wire `nearest_centroids`, GPU-resident block accumulation in `atlas_label_embedding_blocks`, and `in_place=True` on owned blocks (Task 2).
  - `whitematteranalysis/fibers.py`: Chunk `_convert_from_polydata_vectorized` (`L397`, fallback loop inline at `L368-391`) in 4096-line batches with bounds checks (Task 3).
  - `whitematteranalysis/filter.py`: Add full-value `conn_is_arange` check in `_lines_offsets_connectivity` / `_gather_conn`; route GPU `remove_outliers` through `gpu.distance.nearest_neighbor_distances_from_fiber_array` (Task 3).
- **Test & Benchmark**:
  - `tests/test_cli_outlier_reduction.py`, `tests/test_outlier_reductions.py` (Task 1)
  - `tests/test_centroid_kernel.py`, `tests/test_embedding_kernel.py`, `tests/test_atlas_blocks.py`, `tests/bench_gpu_centroid_tiles.py`, `tests/bench_gpu_embedding.py` (Task 2)
  - `tests/test_geometry_views.py`, `tests/test_filter_outliers.py`, `tests/test_laterality_streaming.py`, `tests/test_memory.py` (Task 3)

---

### Task 1: Outlier Removal Concurrency, Fused Row Reduction & CLI Bugfix

**Files:**
- Modify: `whitematteranalysis/gpu/config.py:22-58,141-153`
- Modify: `whitematteranalysis/gpu/distance_kernel.py:139-159`
- Modify: `whitematteranalysis/gpu/distance.py:251-312`
- Modify: `bin/wm_cluster_remove_outliers.py:318-336`
- Test: `tests/test_cli_outlier_reduction.py`, `tests/test_outlier_reductions.py`

**Interfaces:**
- Consumes: `gpu.config.config`, `gpu.cpu_backend._distance_matrix`, `distance_kernel._values`, `distance_kernel._reduce_orientations`.
- Produces:
  - `gpu.config.config.min_pair_work: int` (env `WMA_GPU_MIN_PAIR_WORK`, default `4096`)
  - `distance_kernel.similarity_row_sums(queries, fixed, threshold, sigmasq, bilateral=False, sqrt_distance_sums=False) -> tuple[torch.Tensor, torch.Tensor | None]`
  - `distance.similarity_row_sums_from_arrays(...)` (existing signature preserved; routes tiny clusters to `cpu_backend` and eligible CUDA `Mean` blocks to `distance_kernel.similarity_row_sums`)

- [ ] **Step 1: Write failing tests for CLI GPU threading/config preservation and deterministic log order**

Add to `tests/test_cli_outlier_reduction.py`:
```python
def test_gpu_multiworker_uses_threads_preserves_cli_flags_and_cluster_order(self):
    # Run cli.main() with 3 clusters, ['-gpu', 'on', '-gpu_float32', '-gpu_devices', '0', '-j', '2'].
    # Assert:
    # 1. joblib.Parallel is NOT called when backend == 'gpu'.
    # 2. Inside worker execution, wma.gpu.config.float32 is True and wma.gpu.config.devices == [0].
    # 3. cluster_log.txt rows are ordered ['1', '2', '3'] regardless of completion order.
```
Add to `tests/test_outlier_reductions.py`:
```python
def test_small_cluster_routes_to_cpu_backend(self):
    # With n_queries=4, n_fixed=4 (< WMA_GPU_MIN_PAIR_WORK=4096), mock torch/CUDA available
    # and assert similarity_row_sums_from_arrays matches exact NumPy values without launching CUDA kernels.

@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
def test_fused_similarity_row_sums_matches_eager(self):
    # Compare distance_kernel.similarity_row_sums against eager _fiber_block_distances + exp/sqrt/sum
    # across bilateral in (False, True), sqrt_distance_sums in (False, True), P in (10, 15, 20),
    # threshold in (0.0, 2.0). Assert rtol=1e-13, atol=1e-13 for float64.

def test_refinement_band_catches_1e_13_score_and_cutoff_perturbation(self):
    # Perturb total_subject and total_atlas near cutoff by 1e-13 at std_threshold=0.0 and 2.0
    # and assert refine_outlier_probabilities_from_arrays triggers CPU refinement and restores exact mask.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=.:tests python3 -m unittest test_cli_outlier_reduction test_outlier_reductions -v`
Expected: FAIL (`joblib.Parallel` still called on GPU path; `similarity_row_sums` not defined on `distance_kernel`; `1e-13` perturbation outside old `1.42e-14` band).

- [ ] **Step 3: Add `WMA_GPU_MIN_PAIR_WORK` in `whitematteranalysis/gpu/config.py`**

Add `self.min_pair_work = 4096` to `GPUConfig.__init__` and parse `WMA_GPU_MIN_PAIR_WORK` (int, `>= 0`) in `_from_env()`.

- [ ] **Step 4: Implement `distance_kernel.similarity_row_sums` in `whitematteranalysis/gpu/distance_kernel.py`**

Implement Triton kernel `_similarity_row_sums(F, Q, SimOut, DistOut, NF, NQ, P, WIDTH, STEPS, THRESHOLD2, BLOCK_N, BILATERAL, INV_SIGMA, COMPUTE_SQRT)`:
- One program per query fiber `qi = tl.program_id(0)` (or tile of queries); loops over `fi` in blocks of `BLOCK_N` (`64` or `128`).
- Computes `Mean` fiber distance `d` (before squaring: `mean_dist = min(fwd_mean, rev_mean)` and bilateral min) in registers using `_values` + reduction, so `d_sq = mean_dist * mean_dist` and `sqrt(d_sq)` is simply `mean_dist` (eliminating redundant square→sqrt!).
- Accumulates `sim_acc += libdevice.exp(-d_sq * INV_SIGMA)` and (if `COMPUTE_SQRT`) `dist_acc += mean_dist` in `float64` registers, writing one scalar per query fiber.
- Export `similarity_row_sums(queries, fixed, threshold, sigmasq, bilateral=False, sqrt_distance_sums=False)` preserving `torch.cuda.device(fixed.device)` and returning `(sim_sums, dist_sums_or_None)`.

- [ ] **Step 5: Wire fused kernel and small-cluster CPU routing in `whitematteranalysis/gpu/distance.py`**

In `similarity_row_sums_from_arrays`:
- If `not float32_enabled(gpu_float32)` and `config.cpu_fast_enabled()` and `X_queries.shape[0] * X_fixed.shape[0] < config.min_pair_work` and `tile_rows is None`: compute directly via `cpu_backend._distance_matrix(..., number_of_jobs=1)` + NumPy `exp`/`sqrt`/`sum` and return immediately.
- In the CUDA tile loop, when `distance_kernel.triton is not None and dtype == torch.float64 and distance_method == 'Mean' and queries.shape[1] <= 32 and sigmasq > 0 and queries.is_contiguous() and fixed.is_contiguous()`, call `distance_kernel.similarity_row_sums(...)` instead of `_fiber_block_distances` + `torch.exp` + `.sqrt()` + `.sum(dim=1)`.
- Keep eager path for other `distance_method`, `float32`, or `P > 32` (and pass `similarity_sigmasq=sigmasq` when `not sqrt_distance_sums`).
- In `refine_outlier_probabilities_from_arrays` (`distance.py:341`), widen the near-cutoff refinement band to `band = max(64. * numpy.finfo(numpy.float64).eps, 4e-13 * max(1.0, abs(mean_atlas), abs(std_atlas))) * (1. + abs(std_threshold))` so both subject-score and atlas-cutoff errors within the `1e-13` fused-kernel error budget always trigger exact CPU refinement.

- [ ] **Step 6: Switch GPU worker pool to `ThreadPoolExecutor` + per-thread CUDA streams in `bin/wm_cluster_remove_outliers.py`**

In `main()` around `L319-326`:
- When `backend == 'gpu'` and `number_of_jobs != 1`: resolve worker count `max_workers = max(1, os.cpu_count() or 1)` if `number_of_jobs < 0` else `number_of_jobs`; create a `threading.local()` holding one `torch.cuda.Stream(device=wma.gpu.get_device())` per worker thread; wrap `outlier_removal_one_cluster` in `with torch.cuda.stream(stream):` (with `stream.synchronize()` only implicitly via D2H inside the call); submit tasks via `ThreadPoolExecutor(max_workers=max_workers)` and collect `logtrs` via `executor.map(...)` to preserve strict cluster order `1..number_ac`.
- Keep `joblib.Parallel` unchanged when `backend != 'gpu'`.

- [ ] **Step 7: Run local tests to verify they pass**

Run: `PYTHONPATH=.:tests python3 -m unittest test_cli_outlier_reduction test_outlier_reductions test_atlas_cache test_gpu_config -v`
Expected: All non-CUDA tests PASS (CUDA-marked tests skipped on macOS).

- [ ] **Step 8: Commit Task 1**

```bash
git add whitematteranalysis/gpu/config.py whitematteranalysis/gpu/distance_kernel.py whitematteranalysis/gpu/distance.py bin/wm_cluster_remove_outliers.py tests/test_cli_outlier_reduction.py tests/test_outlier_reductions.py
git commit -m "perf(gpu): fuse outlier similarity row sums and use CUDA stream threads in CLI"
```

- [ ] **Step 9: RTX 4090 GPU Verification & Benchmark (Run on Linux GPU host)**

```bash
# Set dataset paths for your Linux GPU host:
# export C028_CLUSTERS=/path/to/C028_clustered_dir ORG_ATLAS=/path/to/ORG-800FC-100HCP OUT_DIR=/tmp/wma_bench
PYTHONPATH=.:tests python -m unittest test_outlier_reductions test_cli_outlier_reduction test_atlas_cache -v
PYTHONPATH=.:tests python tests/bench_outlier_cache.py --input "$C028_CLUSTERS" --atlas "$ORG_ATLAS" --output "$OUT_DIR/outlier_cache.json"
# Measure optimal -j (2/4/8) and WMA_GPU_MIN_PAIR_WORK on C028 800 clusters vs baseline (28.1 s)
```

---

### Task 2: Sync-Free Atlas Labeling & Fused Centroid/Normalization Kernels

**Files:**
- Modify: `whitematteranalysis/gpu/distance_kernel.py:102-136`
- Modify: `whitematteranalysis/gpu/embedding_kernel.py:1-37`
- Modify: `whitematteranalysis/gpu/linalg.py:117-176`
- Modify: `whitematteranalysis/gpu/distance.py:132-151,380-415,464-498`
- Modify: `tests/bench_gpu_centroid_tiles.py:28-55`, `tests/bench_gpu_embedding.py:50-58`
- Test: `tests/test_centroid_kernel.py`, `tests/test_embedding_kernel.py`, `tests/test_atlas_blocks.py`, `tests/test_distance_kernel.py`

**Interfaces:**
- Consumes: `distance_kernel._centroid_distances` pattern, `embedding_kernel._normalize`, `linalg._prepare_atlas`.
- Produces:
  - `distance_kernel.nearest_centroids(queries, centroids) -> tuple[torch.Tensor, torch.Tensor]` (returns `(min_distances, best_indices)` with `int64` indices matching `torch.min(1)`)
  - `embedding_kernel.normalized_similarities(B, a, c, in_place=False, return_col_sums=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]`
  - `linalg._atlas_embedding_block_gpu(B, prepared, in_place=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` (returns GPU tensors `(embed, row_sum_2, b_col_sums, labels, distances)`)
  - `distance._choose_tile_rows(..., triton_2d=False)` (uses `m * itemsize * 3` when `triton_2d=True`)

- [ ] **Step 1: Write failing tests for fused `nearest_centroids`, `in_place` normalization, and sync-free `atlas_label_embedding_blocks`**

Add to `tests/test_centroid_kernel.py`:
```python
def test_fused_nearest_centroids_matches_cdist_min(self):
    # Test D in (1, 2, 10, 16, 32), scales (1e-100, 1., 1e100), exact ties (first centroid wins),
    # NaN and Inf entries, and empty queries (0, D).
    # Assert torch.equal on indices and torch.testing.assert_close(rtol=0., atol=0., equal_nan=True) on distances.
```
Add to `tests/test_embedding_kernel.py`:
```python
def test_in_place_and_col_sums_parity(self):
    # Verify normalized_similarities(B.clone(), a, c, in_place=True, return_col_sums=True)
    # returns the same tensor storage (data_ptr matches) and exact values / col_sums vs eager reference,
    # while default in_place=False still leaves B untouched.
```
Add to `tests/test_atlas_blocks.py`:
```python
def test_atlas_label_embedding_blocks_single_d2h_transfer(self):
    # With multiple input blocks, verify atlas_label_embedding_blocks matches reference outputs
    # and invokes .cpu() once at the end (not once per block).
```
Add to `tests/test_distance_kernel.py`:
```python
def test_pairwise_and_rectangular_similarity_use_gpu_exp(self):
    # Mock/patch numpy.exp to raise AssertionError on CUDA path and verify
    # pairwise_similarity_matrix & rectangular_similarity_matrix compute exp on GPU via similarity_sigmasq.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=.:tests python3 -m unittest test_centroid_kernel test_embedding_kernel test_atlas_blocks test_distance_kernel -v`
Expected: FAIL (`nearest_centroids` not defined; `in_place` unexpected kwarg; `numpy.exp` still called).

- [ ] **Step 3: Implement `distance_kernel.nearest_centroids` in `whitematteranalysis/gpu/distance_kernel.py`**

Add Triton kernel `_nearest_centroids(Q, C, MinDist, BestIdx, NQ, NC, D, WIDTH, STEPS, BLOCK_C)`:
- One program per query row `qi = tl.program_id(0)` (or small tile of queries), iterating over centroids in `BLOCK_C` chunks.
- Computes Euclidean distance per `(qi, c)` with the exact same descending warp-shuffle reduction + `libdevice.sqrt_rn` and `enable_fp_fusion=False` as `_centroid_distances`.
- Maintains running `(best_dist, best_idx)` with strict `<` update (`dist < best_dist`, propagating `NaN` identically to `torch.min(1)` where a `NaN` centroid distance yields `NaN` and its first index).
- Wire `linalg._nearest_centroids` to call `distance_kernel.nearest_centroids(embed, centroids)` on eligible CUDA `float64` contiguous inputs (`1 <= D <= 32`, `centroids.shape[0] > 0`), keeping `torch.cdist(...).min(1)` fallback otherwise. Keep `distance_kernel.centroid_distances` untouched.

- [ ] **Step 4: Add `in_place` and `return_col_sums` to `embedding_kernel.py`**

Update `normalized_similarities(B, a, c, in_place=False, return_col_sums=False)`:
- When `in_place=True` on the Triton path, pass `B` as `Output` (zero allocation); on eager fallback, do `B.mul_(a[:, None] * c[None, :])` when `in_place=True` else out-of-place.
- When `return_col_sums=True`, compute `col_sums = out.sum(0)` (and keep hook ready for fused block reduction to be benchmarked on 4090 per Design Spec §4 To-Be-Measured #4) and return `(out, col_sums)`.

- [ ] **Step 5: Make `linalg.atlas_label_embedding_blocks` GPU-resident with single D2H copy**

In `whitematteranalysis/gpu/linalg.py`:
- Extract `_atlas_embedding_block_gpu(B, prepared, in_place=False)` returning GPU tensors `(embed, row_sum_2, b_col_sums, labels, distances)`.
- Keep `_atlas_embedding_block(B, atlas, prepared)` calling `_atlas_embedding_block_gpu(B, prepared, in_place=False)` and doing the existing single-block D2H + NaN/negative-row-sum checks (preserving 100% compatibility with `test_embedding_kernel.py`).
- Update `atlas_label_embedding_blocks(blocks, atlas, number_fibers)`: pre-allocate GPU tensors for all `number_fibers` (`embed`, `rows`, `columns`, `labels`, `distances`), loop over `blocks` calling `_atlas_embedding_block_gpu(block, prepared, in_place=True)` and slicing into the GPU buffers with zero `.cpu()` calls inside the loop; after the loop, transfer the compact packed result to host once, run the negative-row-sum warning/clamp and `isfinite` check on host, and return identical dtypes/shapes.

- [ ] **Step 6: Fix `iter_rectangular_similarity` tile sizing / pinned H2D, fuse `pairwise/rectangular_similarity_matrix`, and update GPU benchmarks**

- Add `triton_2d=False` parameter to `_choose_tile_rows`: when `triton_2d=True`, estimate `bytes_per_query_fiber = m * itemsize * 3` instead of `m * points_per_fiber * itemsize * 6`.
- In `iter_rectangular_similarity`: remove hardcoded `or 2048` fallback; pass `triton_2d=(distance_kernel.triton is not None and dtype == torch.float64 and points_per_fiber <= 32)` to `_choose_tile_rows`; stage query blocks via pinned host memory + `.to(device=device, non_blocking=True)` when `queries.shape[0] > 0`.
- In `pairwise_similarity_matrix` and `rectangular_similarity_matrix`: pass `similarity_sigmasq=sigma * sigma` (when `distance_method != 'StrictSimilarity'`) into `_distance_matrix` so `exp(-d/sigma^2)` executes fused on GPU instead of CPU `numpy.exp`.
- In `tests/bench_gpu_embedding.py:50-58`: yield fresh owned block copies (`similarities[start:start+2048].clone().t()`) on every invocation so `in_place=True` in `atlas_label_embedding_blocks` never mutates shared inputs across warm-up/timed repeats or baseline comparisons.
- In `tests/bench_gpu_centroid_tiles.py:28-55`: compare baseline `old.centroid_distances(q, c).min(1)` against `distance_kernel.nearest_centroids(q, c)` in the microbenchmark, and patch `linalg._nearest_centroids` in `classify()` so end-to-end classification timings measure the new fused dispatch.

- [ ] **Step 7: Run local tests to verify they pass**

Run: `PYTHONPATH=.:tests python3 -m unittest test_centroid_kernel test_embedding_kernel test_atlas_blocks test_distance_kernel test_cpu_atlas -v`
Expected: All non-CUDA tests PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add whitematteranalysis/gpu/distance_kernel.py whitematteranalysis/gpu/embedding_kernel.py whitematteranalysis/gpu/linalg.py whitematteranalysis/gpu/distance.py tests/test_centroid_kernel.py tests/test_embedding_kernel.py tests/test_atlas_blocks.py tests/test_distance_kernel.py tests/bench_gpu_centroid_tiles.py tests/bench_gpu_embedding.py
git commit -m "perf(gpu): fuse nearest centroids, in-place normalization, and sync-free atlas labeling"
```

- [ ] **Step 9: RTX 4090 GPU Verification & Benchmark (Run on Linux GPU host)**

```bash
# Set dataset paths for your Linux GPU host:
# export C028_TRACTOGRAM=/path/to/C028_reg.vtp ORG_ATLAS=/path/to/ORG-800FC-100HCP OUT_DIR=/tmp/wma_bench
PYTHONPATH=.:tests python -m unittest test_centroid_kernel test_embedding_kernel test_atlas_blocks test_distance_kernel -v
PYTHONPATH=.:tests python tests/bench_gpu_centroid_tiles.py --input "$C028_TRACTOGRAM" --atlas "$ORG_ATLAS" --output "$OUT_DIR/centroid_tiles.json"
PYTHONPATH=.:tests python tests/bench_gpu_embedding.py "$ORG_ATLAS" --input "$C028_TRACTOGRAM" --output "$OUT_DIR/embedding.json"
# Measure atlas labeling wall-clock on C028 (baseline 24.7 s) and sweep tile_rows cap (4096/8192/16384)
```

---

### Task 3: Chunked Fiber Conversion, O(N) GPU Outlier Filter & OOM/TF32 Hardening

**Files:**
- Create: `whitematteranalysis/gpu/_memory.py`
- Modify: `whitematteranalysis/gpu/config.py:22-58,141-153`
- Modify: `whitematteranalysis/gpu/atlas_cache.py:20-23`
- Modify: `whitematteranalysis/fibers.py:411-451`
- Modify: `whitematteranalysis/filter.py:42-82,797-862`
- Modify: `whitematteranalysis/gpu/distance.py:132-200,283-312,438-549`
- Test: `tests/test_geometry_views.py`, `tests/test_filter_outliers.py`, `tests/test_laterality_streaming.py`, `tests/test_memory.py`

**Interfaces:**
- Consumes: `gpu.config.config`, `gpu.cpu_backend._distance_matrix`, `filter._mean_nearest_neighbor_distance`.
- Produces:
  - `gpu._memory.is_gpu_oom(exc: BaseException) -> bool`
  - `gpu._memory.disable_tf32() -> None`
  - `gpu._memory.effective_free_vram(device) -> int`
  - `gpu.config.config.max_vram_gib: float | None` (env `WMA_GPU_MAX_VRAM_GIB`)
  - `gpu.distance.nearest_neighbor_distances_from_fiber_array(fiber_array, k=3, distance_method='Mean', gpu_float32=None, tile_rows=None) -> numpy.ndarray`

- [ ] **Step 1: Write failing tests for chunked `FiberArray` bounds safety and `conn_is_arange`**

Add to `tests/test_geometry_views.py`:
```python
def test_fiber_array_chunked_conversion_matches_vtk_loop_across_chunk_boundary(self):
    # Create polydata with 5000 lines (> 4096 chunk size). Compare default convert_from_polydata
    # (which calls _convert_from_polydata_vectorized) against convert_from_polydata with
    # _convert_from_polydata_vectorized patched to return False (exercising the inline VTK loop at fibers.py:368-391).
    # Assert bit-identical fiber_array_r/a/s.

def test_fiber_array_vectorized_rejects_malformed_offsets_and_falls_back(self):
    # Verify out-of-bounds offset or point id causes _convert_from_polydata_vectorized to return False cleanly.

def test_gather_conn_arange_and_interior_permutation(self):
    # Verify _gather_conn with contiguous arange connectivity, reversed connectivity,
    # and an interior permutation [0, 1, 3, 2, 4] (matching endpoints + first step)
    # returns the exact selected point IDs and offsets in all cases.
```

- [ ] **Step 2: Run geometry tests to verify new tests fail**

Run: `PYTHONPATH=.:tests python3 -m unittest test_geometry_views -v`
Expected: FAIL on malformed-offset fallback check.

- [ ] **Step 3: Write failing tests for $O(N)$ GPU `filter.remove_outliers` `topk` and GPU-side `total_similarity_matrix_strict` reflection**

Add to `tests/test_filter_outliers.py`:
```python
def test_remove_outliers_gpu_uses_topk_without_full_matrix(self):
    # Patch gpu.use_gpu -> True and assert filter.remove_outliers calls
    # gpu.distance.nearest_neighbor_distances_from_fiber_array and NEVER calls
    # gpu.distance.pairwise_distance_matrix_from_fiber_array.
    # Also test N in (0, 1, 2, 3, 5) and duplicate identical fibers (multiple 0.0 distances)
    # against _mean_nearest_neighbor_distance reference.
```
Add to `tests/test_laterality_streaming.py`:
```python
def test_total_similarity_strict_row_sum_matches_matrix_sum(self):
    # Verify total_similarity_matrix_strict(..., row_sum=True) matches matrix.sum(1) bit-for-bit
    # (preserving NumPy reduction order) with mixed reflect_mask (all False, all True, alternating).
```

- [ ] **Step 4: Run outlier & laterality tests to verify new tests fail**

Run: `PYTHONPATH=.:tests python3 -m unittest test_filter_outliers test_laterality_streaming -v`
Expected: FAIL (`nearest_neighbor_distances_from_fiber_array` not defined).

- [ ] **Step 5: Write failing tests for `_memory.py` (`is_gpu_oom`, `WMA_GPU_MAX_VRAM_GIB`, `disable_tf32`, bounded CPU fallback) and update obsolete OOM-raise assertions**

Add to `tests/test_memory.py`:
```python
def test_is_gpu_oom_detects_cuda_and_cublas_runtime_errors(self):
    # Assert is_gpu_oom returns True for RuntimeError("CUDA out of memory"),
    # RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"), and False for RuntimeError("invalid shape").

def test_effective_free_vram_respects_max_vram_gib_cap(self):
    # Set WMA_GPU_MAX_VRAM_GIB=3.5 and verify effective_free_vram(device) <= int(3.5 * (1 << 30)).

def test_tiled_host_apis_fall_back_in_bounded_cpu_batches_on_tile_1_oom(self):
    # Mock _fiber_block_distances to raise RuntimeError("CUDA out of memory") at tile_rows=1
    # for _distance_matrix, similarity_row_sums_from_arrays, nearest_neighbor_distances_from_fiber_array,
    # and total_similarity_matrix_strict (including reflect_mask & row_sum=True);
    # assert complete outputs match exact CPU reference values and reduction-only fallbacks
    # invoke cpu_backend._distance_matrix in bounded query batches (never allocating full remaining matrix).
```
Update `tests/test_laterality_streaming.py:128-158`:
- Keep the existing tile-halving assertion (`sizes == [5, 2, 1]`), and replace the obsolete `assertRaises(torch.cuda.OutOfMemoryError)` for `distance_matrix_from_arrays` and `total_similarity_matrix_strict` with assertions that the fallback completes and matches the CPU reference output. Keep `tests/test_atlas_blocks.py:104-106` (`iter_rectangular_similarity` re-raising `OutOfMemoryError` at `tile_rows=1`) unchanged.

- [ ] **Step 6: Run memory tests to verify they fail**

Run: `PYTHONPATH=.:tests python3 -m unittest test_memory -v`
Expected: FAIL (`whitematteranalysis.gpu._memory` does not exist).

- [ ] **Step 7: Implement `whitematteranalysis/gpu/_memory.py`, `config.py` cap, and `atlas_cache.py` hash entry**

- Create `whitematteranalysis/gpu/_memory.py` with:
  - `is_gpu_oom(exc)`: returns `True` if `torch.cuda.OutOfMemoryError` or `RuntimeError` whose lowercased message contains any of `('out of memory', 'cublas_status_alloc_failed', 'cudnn_status_alloc_failed', 'cuda error: out of memory')`.
  - `disable_tf32()`: when `torch` is importable, sets `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`.
  - `effective_free_vram(device)`: queries `torch.cuda.mem_get_info(device)` (fallback `4 << 30`), capped at `int(config.max_vram_gib * (1 << 30))` when `config.max_vram_gib` is set.
- Add `self.max_vram_gib = None` and `WMA_GPU_MAX_VRAM_GIB` parsing in `whitematteranalysis/gpu/config.py`.
- Add `'gpu/_memory.py'` to `_implementation()` file list in `whitematteranalysis/gpu/atlas_cache.py`.

- [ ] **Step 8: Implement chunked `FiberArray._convert_from_polydata_vectorized` and `filter.py` `conn_is_arange` + GPU `topk`**

- In `whitematteranalysis/fibers.py:397-451`: validate `offsets` monotonic (`0 <= offsets[0]` and `offsets[-1] <= len(conn)`), `line_lengths >= 1`, and `conn` bounds (`0 <= min`, `max < len(points)` when `len(conn) > 0`), returning `False` on violation (falling back to the inline VTK loop at `L368-391`); iterate in `4096`-line chunks (`first = 0, 4096, ...`) populating `fiber_array_r/a/s[first:last]` so peak temporary memory is bounded to ~10 MB instead of ~1.8 GB.
- In `whitematteranalysis/filter.py`:
  - In `_gather_conn`: perform a **full-value** check (`conn_is_arange = conn.size == 0 or (conn[0] == 0 and bool(numpy.all(conn[1:] == conn[:-1] + 1)))`) before bypassing `conn[idx]` with `selected[begin:end] = idx` (never rely on endpoints/stride alone).
  - In `remove_outliers` (`L818-822`): replace `gpu.distance.pairwise_distance_matrix_from_fiber_array` with `mindist = gpu.distance.nearest_neighbor_distances_from_fiber_array(fiber_array, k=3, distance_method=distance_method)` so `distances_matrix` is never allocated on host RAM.

- [ ] **Step 9: Implement `nearest_neighbor_distances_from_fiber_array`, GPU-side `total_similarity_matrix_strict` reflection, and OOM/TF32 hardening across all 6 tiled APIs in `whitematteranalysis/gpu/distance.py`**

- Call `_memory.disable_tf32()` at CUDA entry points (`_distance_matrix`, `similarity_row_sums_from_arrays`, `rectangular_similarity_tensor`, `iter_rectangular_similarity`, `total_similarity_matrix_strict`, `nearest_neighbor_distances_from_fiber_array`).
- Replace `torch.cuda.mem_get_info` in `_choose_tile_rows` with `_memory.effective_free_vram(device)`.
- Replace `except torch.cuda.OutOfMemoryError:` in all 6 tiled GPU loops with `except Exception as exc: if not _memory.is_gpu_oom(exc): raise`, halving `tile_rows` down to `1`:
  - For host-returning APIs (`_distance_matrix`, `similarity_row_sums_from_arrays`, `nearest_neighbor_distances_from_fiber_array`, `total_similarity_matrix_strict`), if `end - start == 1` still OOMs, emit a one-line warning and finish remaining rows `[s:n]` in **bounded CPU query batches** (`batch_rows = max(1, (16 << 20) // max(8, m * points_per_fiber * 8 * 12))`), reducing each batch immediately for reduction-only APIs (`similarity_row_sums_from_arrays`, `nearest_neighbor_distances_from_fiber_array`, `total_similarity_matrix_strict(..., row_sum=True)`) and preserving `reflect_mask` for `total_similarity_matrix_strict`.
  - For GPU-tensor-returning APIs (`rectangular_similarity_tensor`, `iter_rectangular_similarity`), re-raise on `end - start == 1` OOM (preserving CUDA tensor return contract and `test_atlas_blocks.py:104-106`).
- Implement `nearest_neighbor_distances_from_fiber_array(fiber_array, k=3, distance_method='Mean', gpu_float32=None, tile_rows=None)`:
  - Handles `N == 0` (empty float64) and `N == 1` (`[np.inf]`).
  - Tiles query rows on GPU, computes `(b, N)` distances via `_fiber_block_distances`, runs `torch.topk(d, k=min(k + 1, N), dim=1, largest=False, sorted=True).values`, drops index `0` (self-distance), and computes `((v[:, 0] + v[:, 1]) + v[:, 2]) / 3.0` when `count == 3` (else `.mean(dim=1)`), copying only the `(b,)` 1D slice to host.
- In `total_similarity_matrix_strict`: apply x-reflection via a GPU sign tensor (`sign = torch.where(mask_gpu, -1., 1.)`; `q[:, :, 0] *= sign[:, None]`) without numpy fancy-index H2D; for `row_sum=True`, keep `values.sum(1)` on the transferred tile so `test_laterality_streaming.py:114-118` remains bit-identical to `matrix.sum(1)` (including `float32`).

- [ ] **Step 10: Run full non-CUDA test suite to verify everything passes**

Run: `PYTHONPATH=.:tests python3 -m unittest discover -s tests -p "test_*.py" -v`
Expected: All non-CUDA unit tests PASS, zero regressions.

- [ ] **Step 11: Commit Task 3**

```bash
git add whitematteranalysis/gpu/_memory.py whitematteranalysis/gpu/config.py whitematteranalysis/gpu/atlas_cache.py whitematteranalysis/fibers.py whitematteranalysis/filter.py whitematteranalysis/gpu/distance.py tests/test_geometry_views.py tests/test_filter_outliers.py tests/test_laterality_streaming.py tests/test_memory.py
git commit -m "perf(memory): chunk FiberArray, add GPU topk outlier filter, and harden OOM/TF32"
```

- [ ] **Step 12: RTX 4090 GPU Verification & Benchmark (Run on Linux GPU host)**

```bash
# export OUT_DIR=/tmp/wma_bench
PYTHONPATH=.:tests python -m unittest discover -s tests -p "test_*.py" -v
WMA_GPU_MAX_VRAM_GIB=3.5 PYTHONPATH=.:tests python -m unittest test_atlas_blocks test_outlier_reductions test_filter_outliers test_laterality_streaming test_memory -v
PYTHONPATH=.:tests python tests/bench_laterality.py --output "$OUT_DIR/laterality.json"
```

---

## Self-Review Checklist (Completed)

1. **Spec coverage**: Every Phase 1 requirement in `2026-11-20-gpu-optimization-design.md` §3 (Tasks 1–3) maps to explicit steps; Phases 2–3 are intentionally deferred per Design Spec §1.
2. **Step scan**: Every step specifies exact file paths, function signatures, test assertions, commands, and expected outcomes; no TBDs or transcript bloat.
3. **Type consistency**: `similarity_row_sums`, `nearest_centroids`, `normalized_similarities(..., in_place, return_col_sums)`, `_atlas_embedding_block_gpu`, `is_gpu_oom`, `disable_tf32`, `effective_free_vram`, `nearest_neighbor_distances_from_fiber_array` match identically across `Interfaces` and steps.
4. **Review Focus**: All 5 high-risk edge cases are explicitly pinned by tests in their owning tasks.
5. **Proportion**: Plan stays focused on decisions, interfaces, and test pins without pasting full kernel bodies.
