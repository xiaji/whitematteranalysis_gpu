# WhiteMatterAnalysis GPU Optimization — Design Spec

- **Date**: 2026-11-20
- **Base Commit**: `0495c3a` (`main`)
- **Fact Sheet**: [`2026-11-20-gpu-optimization-fact-sheet.md`](2026-11-20-gpu-optimization-fact-sheet.md)
- **Status**: v1.1 (Addressed Codex v1 review findings #1–#11; ready for scoped re-review)

---

## 1. Goal & Scope

Systematically eliminate the remaining GPU, memory, and pipeline bottlenecks in `whitematteranalysis_gpu` across three incremental phases (7 tasks), incorporating proven engineering patterns from `NeuroDMRI/SlicerDSpace` while preserving `whitematteranalysis`'s existing float64 numerical contracts and CLI compatibility.

To keep reviews convergent and diffs minimal, **implementation plans are split by phase**:
- **Phase 1 (P0 — Zero Accuracy Risk / Immediate Wins)**: Tasks 1–3. Planned now in [`docs/superpowers/plans/2026-11-20-phase1-p0-gpu-optimizations.md`](../plans/2026-11-20-phase1-p0-gpu-optimizations.md).
- **Phase 2 (P1 — FP32/2D Throughput, Multi-GPU & Single-Process ORG Pipeline)**: Tasks 4–5. Planned after Phase 1 lands and RTX 4090 benchmarks resolve Phase 1 measurements.
- **Phase 3 (P2 — Nonrigid B-spline Autograd & Full-GPU Atlas Building / Multi-Atlas)**: Tasks 6–7. Planned after Phase 2.

---

## 2. Global Constraints

1. **Precision Contract**: Default mode remains `float64` with `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False` enforced before CUDA compute. Existing `float64` discrete outputs (cluster labels, outlier masks) must remain 100% identical to baseline unless explicitly running `-gpu_float32`.
2. **Non-Breaking Existing Public/Tested Signatures**:
   - `embedding_kernel.normalized_similarities(B, a, c)` and `linalg._atlas_embedding_block(B, atlas, prepared)` must keep `in_place=False` as default so existing callers and `test_embedding_kernel.py` (`torch.equal(B, before)`) never observe mutation.
   - `distance_kernel.centroid_distances(queries, centroids)` remains intact for `test_centroid_kernel.py`; fused nearest-centroid reduction is added alongside it (`nearest_centroids`).
   - `bin/wm_cluster_remove_outliers.py` preserves exact CLI flags and `cluster_log.txt` schema.
3. **Minimal Diff Discipline**: Each task touches only the files owning its bottleneck; no opportunistic refactors outside task scope.
4. **Cross-Machine Verification Contract (Mac Dev → Linux RTX 4090 Verify)**:
   - Every task must pass all non-CUDA unit/mock tests locally on macOS via `python -m unittest`.
   - Every task ships with an explicit, copy-pasteable RTX 4090 verification & benchmark command block so GPU correctness and speedup are empirically proved before advancing phases.
5. **Dependency Floor**: No new third-party runtime dependencies beyond existing `requirements_cpu.txt` + optional `torch`/`triton`.

---

## 3. Architecture Overview (3 Phases, 7 Tasks)

### Phase 1 (P0 — Detailed in Phase 1 Plan)
- **Task 1 — Outlier Removal Concurrency, Fused Row Reduction & CLI Bugfix**
  - Replace `joblib.Parallel` (`loky` multiprocess) in `bin/wm_cluster_remove_outliers.py` with `concurrent.futures.ThreadPoolExecutor` + per-worker `torch.cuda.Stream` when `backend == 'gpu'`, eliminating 4× CUDA context overhead and fixing silent loss of `-gpu_float32` / `-gpu_devices`. Retain `joblib.Parallel` for non-GPU paths.
  - Add fused Triton kernel `distance_kernel.similarity_row_sums(...)` that computes `Mean` distances, `exp(-d²/σ²)`, and optional `d` (without squaring then taking `sqrt`) in registers and accumulates per-query row sums without allocating an `(b, m)` matrix. Wire `distance.similarity_row_sums_from_arrays` to use it when eligible (CUDA, `float64`, `P <= 32`, `distance_method == 'Mean'`, contiguous, `sigmasq > 0`), keeping eager fallback for other methods/dtypes.
  - Add small-cluster CPU routing threshold in `similarity_row_sums_from_arrays` (`n_queries * n_fixed < threshold` → `cpu_backend`) to avoid CUDA launch overhead on tiny clusters.
  - Note on `atlas_cache.py`: `atlas_cache.load_reference` already delegates GPU self-similarity to `similarity_row_sums_from_arrays` (`atlas_cache.py:90`), so it automatically benefits from the fused row-sum kernel (zero `(N, N)` VRAM allocation). Update `_implementation()` file list if any new GPU helper module is added.
- **Task 2 — Sync-Free Atlas Labeling & Fused Centroid/Normalization Kernels**
  - Add fused Triton kernel `distance_kernel.nearest_centroids(queries, centroids)` that computes Euclidean distances in registers (matching `cdist` descending warp-shuffle reduction & `sqrt_rn`) and maintains `(min_dist, best_idx)` with strict `<` tie-breaking (first centroid wins, NaN-safe matching `torch.cdist().min(1)`). Wire `linalg._nearest_centroids` to use it on eligible CUDA `float64` inputs.
  - Extend `embedding_kernel.normalized_similarities(B, a, c, in_place=False, return_col_sums=False)`: when `in_place=True`, overwrite `B` in place (zero 40 MB copy); when `return_col_sums=True`, compute `b_col_sums` in the same or reduced pass without re-scanning full `Bg` from DRAM when possible (or fuse column reduction cleanly while preserving float64 accuracy).
  - Refactor `linalg.atlas_label_embedding_blocks`: pre-allocate compact GPU output tensors for all `N` fibers (`embed`, `row_sum_2`, `b_col_sums`, `labels`, `distances` — ~37 MB for 330k fibers) and perform **one single D2H copy** after the block loop completes, eliminating ~162 blocking `.cpu().numpy()` pipeline flushes. Keep `_atlas_embedding_block` non-mutating for direct callers; add internal GPU-resident helper used by `atlas_label_embedding_blocks` with `in_place=True`.
  - Fix `distance.iter_rectangular_similarity`: compute `tile_rows` from `_choose_tile_rows` using Triton's actual 2D `(b, m)` memory footprint when Triton is active instead of hardcoding `2048` (`distance.py:474`), and use pinned host memory + non-blocking H2D for query blocks.
  - Pass `similarity_sigmasq=sigma*sigma` in `pairwise_similarity_matrix` and `rectangular_similarity_matrix` (when `distance_method != 'StrictSimilarity'`) so `exp(-d/sigma^2)` runs fused on GPU instead of CPU `numpy.exp`.
- **Task 3 — Chunked Fiber Conversion, O(N) GPU Outlier Filter & OOM/TF32 Hardening**
  - Chunk `fibers.FiberArray._convert_from_polydata_vectorized` (`fibers.py:397`, fallback loop inline at `L368-391`) in batches of `4096` lines and add bounds validation on `offsets`/`conn` (falling back to the inline VTK loop on malformed data).
  - Add a full-value `conn_is_arange` check (`conn.size == 0 or (conn[0] == 0 and np.all(conn[1:] == conn[:-1] + 1))`) in `filter._lines_offsets_connectivity` / `_gather_conn` so `conn[idx]` indirection is skipped only when connectivity is verified to be strictly contiguous `0..N-1` (no endpoint-only shortcut).
  - Add `gpu.distance.nearest_neighbor_distances_from_fiber_array(fiber_array, k=3, ...)` using tiled GPU `torch.topk(k=min(4, N), largest=False)` so `filter.remove_outliers` uses $O(N)$ host RAM instead of $O(N^2)$ `distances_matrix`, preserving `_mean_nearest_neighbor_distance`'s exact left-to-right sum `((d0 + d1) + d2) / 3.0` and `N <= 1` edge cases.
  - Fix `distance.total_similarity_matrix_strict`: perform x-reflection using a GPU sign-tensor multiply (eliminating numpy fancy-index H2D); for `row_sum=True`, keep NumPy per-tile `values.sum(1)` so `test_laterality_streaming.py:114-118`'s bit-identical `matrix.sum(1)` invariant (including `float32`) is preserved.
  - Add `whitematteranalysis/gpu/_memory.py` + `WMA_GPU_MAX_VRAM_GIB` env var in `gpu/config.py`:
    - `is_gpu_oom(exc)`: matches `torch.cuda.OutOfMemoryError` and cuBLAS/cuDNN `RuntimeError` OOM messages.
    - `disable_tf32()`: sets `allow_tf32 = False` for matmul and cuDNN before CUDA compute.
    - `effective_free_vram(device)`: respects `WMA_GPU_MAX_VRAM_GIB` cap (enables deterministic low-VRAM testing on 24 GB GPUs).
    - **OOM recovery across all 6 tiled GPU APIs**:
      - Host-returning APIs (`_distance_matrix`, `similarity_row_sums_from_arrays`, `nearest_neighbor_distances_from_fiber_array`, `total_similarity_matrix_strict`): halve `tile_rows` down to `1`; if `tile_rows == 1` still OOMs, emit a one-line warning and finish remaining rows `[s:n]` in **bounded CPU query batches** (16 MiB batch budget with immediate per-batch reduction for reduction-only APIs, preserving `reflect_mask` and `row_sum` for `total_similarity_matrix_strict`).
      - GPU-tensor-returning APIs (`rectangular_similarity_tensor`, `iter_rectangular_similarity`): use `is_gpu_oom(exc)` + tile halving down to `1`; if a single row `(1, m)` cannot fit in VRAM, re-raise `OutOfMemoryError` (preserving their CUDA-tensor return contract and `test_atlas_blocks.py:104-106`).

### Phase 2 (P1 — Summary; Plan Deferred Until Post-Phase 1)
- **Task 4**: Unlock `float32` and arbitrary `P > 32` across Triton kernels (`distance_kernel`, `registration_kernel`, `embedding_kernel`, `linalg`); refactor to SoA + 2D `(BM, BN)` tiling with fused registration reduction; wire `visible_device_count()` / `WMA_GPU_DEVICES` into `congeal_multisubject`, outlier removal, and atlas labeling; optional `mps` eager `float32` path.
- **Task 5**: Vectorized/GPU affine transform & direct array sampling in `congeal_to_atlas.py`; single-process in-memory ORG pipeline CLI (`bin/wm_apply_ORG_atlas.py`, keeping `.sh` wrapper); tiered SlicerDMRI-free hardening (`FiberBundleNode` → `ModelNode` → pure-Python affine) + auto `xvfb-run` + atomic publish + JSON QC report.

### Phase 3 (P2 — Summary; Plan Deferred Until Post-Phase 2)
- **Task 6**: Nonrigid B-spline (`bspline.py` + `register_two_subjects_nonrigid_bsplines.py` + `io.py`): 6a equivalent vectorization/GPU coefficients/narrowed `except:`; 6b analytical/autograd gradient replacing `approx_grad=True`.
- **Task 7**: Full-GPU `cluster.spectral` (GPU Nyström `k+1` cols, symmetric `eigh` `nearPSD` reusing `pinv_A`, parallel write) + vectorized GPU LOO in `wm_cluster_atlas.py` + declarative multi-atlas registry (`org`, `east-west`, `naba`).

---

## 4. Open Questions: Settled vs. To-Be-Measured

Per `review-convergence.md` §4.2, questions resolvable from primary sources are **settled** here; questions requiring RTX 4090 hardware measurements are **to-be-measured** tasks, not guessed thresholds.

### Settled Decisions (Do Not Reopen in Review)
1. **Default precision & strictness**: Phase 1 defaults to `float64` with TF32 disabled. In Phase 1:
   - `nearest_centroids` matches `torch.cdist().min(1)` bit-for-bit (`torch.equal` on distances and indices, matching `test_centroid_kernel.py`).
   - `similarity_row_sums` matches eager GPU within `< 1e-13` relative error; `refine_outlier_probabilities_from_arrays` (`distance.py:315-345`) widens its candidate band to cover both subject-score and atlas-cutoff perturbations at that bound (`band = max(64. * eps, 4e-13 * max(1.0, abs(mean_atlas), abs(std_atlas))) * (1. + abs(std_threshold))`) so discrete outlier masks remain 100% identical to CPU reference.
   - `filter.remove_outliers` GPU `topk` preserves exact `((d0 + d1) + d2) / 3.0` reduction order.
2. **Thread safety in `wm_cluster_remove_outliers.py`**: GPU mode uses `ThreadPoolExecutor` + per-worker `torch.cuda.Stream`; CPU mode keeps `joblib.Parallel`. No changes to `cluster_log.txt` format.
3. **Caller-immutability contract**: `normalized_similarities` and `_atlas_embedding_block` keep `in_place=False` default; only `atlas_label_embedding_blocks` opts into `in_place=True` on its exclusively owned block.
4. **OOM fallback behavior**: Normal halving (`rows // 2`) runs first across all 6 tiled GPU APIs. If `tile_rows == 1` still OOMs: host-returning APIs fall back remaining rows in bounded CPU batches (with a one-line warning); GPU-tensor-returning APIs (`rectangular_similarity_tensor`, `iter_rectangular_similarity`) re-raise `OutOfMemoryError`.
5. **Plan splitting**: Only Phase 1 (Tasks 1–3) is planned to code-level detail now; Phases 2–3 are planned after Phase 1's 4090 measurements.

### To-Be-Measured on RTX 4090 (Not Hardcoded From Reasoning Alone)
1. **Small-cluster CPU routing threshold** (`WMA_GPU_MIN_PAIR_WORK`, Task 1): Initial conservative default `4096` pairs (`n_queries * n_fixed < 4096`), configurable via env var; exact break-even threshold to be measured on C028 800 clusters on RTX 4090.
2. **Optimal `tile_rows` cap for `iter_rectangular_similarity`** (Task 2): Formula-driven via Triton 2D footprint (`m * itemsize * 3`) instead of hardcoded `2048`; measure optimal cap (`4096` vs `8192` vs `16384`) on 330k × 10k atlas labeling on RTX 4090.
3. **Optimal GPU worker thread count** for `wm_cluster_remove_outliers.py` (Task 1): Measure `-j 2`, `-j 4`, `-j 8` with per-thread CUDA streams on C028 on RTX 4090.
4. **Fused column-sum reduction method** in `embedding_kernel.py` (Task 2): Measure whether a two-stage block-reduced Triton kernel beats `Bg.sum(0)` after in-place `_normalize` on `(10000, 8192)` blocks on RTX 4090 (must preserve exact/near-exact float64 sums).
5. **Phase 2 (`WMA_GPU_STRICT=1` & CUDA Graph & FP32 accuracy)**: Deferred to Phase 2 spec after Phase 1 baseline numbers are recorded.

---

## 5. Non-Goals (Phase 1)
- No `float32` Triton kernels or `P > 32` kernel tiling (Phase 2, Task 4).
- No multi-GPU scheduling changes (Phase 2, Task 4).
- No changes to `wm_apply_ORG_atlas_to_subject.sh` or SlicerDMRI hardening (Phase 2, Task 5).
- No changes to `bspline.py`, `cluster.py:spectral`, or multi-atlas registries (Phase 3, Tasks 6–7).
