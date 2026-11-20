# WhiteMatterAnalysis GPU Optimization — One-Page Fact Sheet

> Primary-source reference verified at repo HEAD `0495c3a` (`main`) on 2026-11-20. Reviewers check the Design Spec and Implementation Plans against this sheet.

## 1. Repository & Runtime Environment
- **Repo root**: `/Volumes/WorkShop/Codes/whitematteranalysis_gpu` (`whitematteranalysis` v0.3.0, `setup.py:34`)
- **Base commit**: `0495c3a` on branch `main` (clean working tree prior to `docs/superpowers/`)
- **Dev vs. Verification hosts**:
  - **Development host**: macOS (Apple Silicon, no NVIDIA CUDA; `python3` / `unittest` for CPU/VTK/mock tests)
  - **GPU verification host**: Linux with NVIDIA RTX 4090 (CUDA + Triton; FP32:FP64 throughput ratio 64:1; C028 subject benchmark)
- **Test suite**: 27 `tests/test_*.py` modules + 16 `tests/bench_*.py` + 2 `tests/verify_*.py`, written with Python `unittest` (`PYTHONPATH=.:tests python -m unittest discover -s tests` or `PYTHONPATH=.:tests python -m unittest <module>` so sibling imports like `test_cli_file_stages` resolve). CUDA tests use `@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')`.

## 2. GPU Package Inventory (`whitematteranalysis/gpu/`, 2,329 lines total)
| File | Lines | Role & Key Facts |
|---|---:|---|
| `__init__.py` | 63 | Re-exports config helpers (`use_gpu`, `available`, `apply_gpu_arguments`, etc.) |
| `config.py` | 190 | `GPUConfig` singleton; env vars `WMA_GPU`, `WMA_GPU_FLOAT32`, `WMA_GPU_DEVICES`, `WMA_GPU_TILE`, `WMA_CPU_FAST`, `WMA_CPU_THREADS`. `visible_device_count()` exists (`L134`) with 0 callers in repo. |
| `distance.py` | 549 | `_fiber_block_distances` (`L100`, Triton gate: `float64`, `P<=32`), `_choose_tile_rows` (`L132`, estimates `m*P*itemsize*6`), `_distance_matrix` (`L153`), `similarity_row_sums_from_arrays` (`L251`, allocates full `(b, m)` then `exp`/`sqrt`/`sum`), `refine_outlier_probabilities_from_arrays` (`L315`), `pairwise/rectangular_similarity_matrix` (`L380-415`, CPU `numpy.exp`), `iter_rectangular_similarity` (`L464`, hardcodes `requested = tile_rows or config.tile_rows or 2048`), `total_similarity_matrix_strict` (`L500`, CPU `.sum(1)` and numpy reflection index). |
| `distance_kernel.py` | 159 | Triton `float64` kernels `_distances` (`L63`, 1D pair, `enable_fp_fusion=False`) and `_centroid_distances` (`L103`, `1<=D<=32`, outputs full `(N, NC)` matrix). |
| `embedding_kernel.py` | 37 | Triton `float64` `_normalize` (`L8`, allocates `torch.empty_like(B)`, gate `B.numel() >= 1<<20`). |
| `linalg.py` | 176 | `_prepare_atlas` (`L106`), `_nearest_centroids` (`L117`, calls `centroid_distances` then `.min(1)`), `_atlas_embedding_block` (`L131`, `Bg.sum(0)` after `normalized_similarities`, single `.cpu().numpy()` per block), `atlas_label_embedding_blocks` (`L162`, loops blocks calling `_atlas_embedding_block`). |
| `atlas_cache.py` | 116 | Disk `.npz` cache for atlas `(N, 15, 3)` coords + LOO probabilities; hashes 10 source files (`L20-22`); calls `similarity_row_sums_from_arrays(coordinates, coordinates, ...)` (`L90`). |
| `registration.py` | 237 | GPU registration objective; ignores `WMA_GPU_FLOAT32` (`L99`); catches only `torch.cuda.OutOfMemoryError` (`L145`). |
| `registration_kernel.py` | 165 | Triton `float64` registration objective kernel writing `(Nm, Nf)` matrix before `.sum(-1)`. |
| `bspline.py` | 144 | GPU B-spline transform; calls CPU `vtkImageBSplineCoefficients` per eval; 4×4×4 loop. |
| `cpu_backend.py` | 445 | Vectorized multi-threaded NumPy fallback for distances/similarities. |
| `fiber_tensors.py` | 48 | `polydata_to_numpy`, `fiber_array_to_numpy`, `SUPPORTED_DISTANCE_METHODS`. |

## 3. Key Non-GPU Primary Sources in Scope
- `bin/wm_cluster_remove_outliers.py:319-326`: `joblib.Parallel(n_jobs=number_of_jobs)` (`loky` default) spawns worker processes; each re-initializes `gpu.config` from env vars, silently dropping CLI `-gpu_float32` / `-gpu_devices`, and opens separate CUDA contexts.
- `whitematteranalysis/fibers.py:363-450`: `_convert_from_polydata_vectorized` (`L397`, called at `L363`; inline VTK fallback loop at `L368-391`) allocates full `(n, P)` `positions`, `local_ids`, `global_ids`, and `(n, P, 3)` `pts` at once (~1.8 GB peak for 1M fibers × 50 points).
- `whitematteranalysis/filter.py:42-81, 797-861`: `_lines_offsets_connectivity` returns `(offsets, conn, points)`; `_gather_conn` always indexes `conn[idx]`; `remove_outliers` materializes full `(N, N)` `distances_matrix` on host RAM before `_mean_nearest_neighbor_distance` (mean of 3 smallest non-self distances, `L98-121`).
- `whitematteranalysis/congeal_to_atlas.py`, `congeal_multisubject.py`, `bin/wm_apply_ORG_atlas_to_subject.sh`: Phase 2 targets.
- `whitematteranalysis/register_two_subjects_nonrigid_bsplines.py`, `io.py`, `cluster.py`, `bin/wm_cluster_atlas.py`: Phase 3 targets.

## 4. Existing Test Invariants That Must Not Break
- `tests/test_centroid_kernel.py`: `distance_kernel.centroid_distances` must match `torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist')` with `torch.equal`; `linalg._nearest_centroids` must preserve tie-breaking (first centroid wins), `nan`/`inf`, `D in (0, 1, 10, 32, 33)`, non-contiguous strides, empty queries, CPU fallback, and multi-GPU device restoration.
- `tests/test_embedding_kernel.py`: `embedding_kernel.normalized_similarities(B, a, c)` and `linalg._atlas_embedding_block(B, atlas, prepared)` must **not mutate caller tensor `B`** by default (`torch.equal(B, before)`), and `_atlas_embedding_block` must match git baseline `8f83f6f` (`np.testing.assert_array_equal`).
- `tests/test_cli_outlier_reduction.py`: `wm_cluster_remove_outliers.py` GPU path must call `similarity_row_sums_from_arrays(..., sqrt_distance_sums=True)`, never call `distance_matrix_from_arrays`, and preserve exact `cluster_log.txt` columns.
- `tests/test_filter_outliers.py`: `filter._mean_nearest_neighbor_distance` keeps left-to-right addition `((d0 + d1) + d2) / 3.0` for `count == 3` and `inf` for 1-fiber input.
- `tests/test_laterality_streaming.py`: `total_similarity_matrix_strict(..., row_sum=True)` requires exact equality (`np.testing.assert_array_equal`) with `matrix.sum(1)` including `float32` (`L114-118`); `L128-158` currently expects `OutOfMemoryError` at `tile_rows == 1` (updated in Task 3 to assert CPU fallback).
- `tests/test_atlas_blocks.py:104-106`: `iter_rectangular_similarity(..., tile_rows=1)` asserts `OutOfMemoryError` on single-row GPU exhaustion.
