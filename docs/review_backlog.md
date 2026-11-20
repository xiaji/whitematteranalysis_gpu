# Review Backlog

Non-blocking (P2) items recorded per `review-convergence.md` §1.2. Items not pulled into a task by their decide-by date are closed "won't do".

| Date | Source | File:Line | Item | Decide-By |
|---|---|---|---|---|
| 2026-11-20 | Codex Phase 1 Code Re-Review | `tests/test_filter_outliers.py:166-168` | Add a CUDA-backed case (skipped when CUDA unavailable) to `test_nearest_neighbor_distances_uses_exact_numpy_division` so CUDA reciprocal-multiply regressions are directly caught on GPU runners. | 2026-12-04 |

| 2026-11-21 | Task 4 Reviewer | `whitematteranalysis/gpu/_memory.py:67` | Check `_device_type(device) == 'cuda'` instead of `!= 'mps'` in `effective_free_vram` so `'cpu'` devices avoid calling `torch.cuda.mem_get_info` before exception fallback. | 2026-12-05 |
| 2026-11-21 | Codex Phase 2 Final Review | `whitematteranalysis/congeal_multisubject.py:317-318` | Evaluate process-based isolation or a lock-free optimizer alternative for SciPy >=1.15 `fmin_cobyla` when running multi-GPU `Affine`/`Rigid` multisubject registration concurrently. | 2026-12-05 |
| 2026-11-21 | Whole-Branch Final Reviewer | `whitematteranalysis/gpu/bspline.py:117` | Wire optional `float32_enabled()` support cleanly into `NonrigidObjective` (including CUDA B-spline sampling/transform precision parity) instead of keeping CUDA nonrigid fixed at `float64`. | 2026-12-05 |
| 2026-11-21 | Task 5 Reviewer | `bin/wm_apply_ORG_atlas.py:785-793` | Preserve empty-string `WMA_VTP_COMPRESSION=''` env restoration and consider `lz4` compression for `--keep-intermediates 1/2` intermediate VTP dumps. | 2026-12-05 |
| 2026-11-21 | Final Fix Re-Reviewer | `bin/wm_apply_ORG_atlas.py:835-865` | In non-Tier-1 `-t` runs with `--keep-intermediates 2`, decide whether `TransformedClusters/` should also receive inverse pre-transformed native coordinates to mirror Tier-1. | 2026-12-05 |
| 2026-11-21 | Task 5 Reviewer | `whitematteranalysis/transform.py:102` | Surface an explicit `ValueError` context message if `numpy.linalg.inv` encounters a singular `.tfm` matrix. | 2026-12-05 |
| 2026-11-21 | Task 5 Fix Re-Reviewer | `bin/wm_harden_transform.py:330,339-342` | Include pre-fallback `--prepare-command` duration in `preparation.json` `combined_s` when `Slicer='auto'` falls back due to tensor inputs. | 2026-12-05 |
| 2026-11-21 | Whole-Branch Final Reviewer | `whitematteranalysis/gpu/__init__.py:43` | Add `'bspline'` to `_BACKENDS` in `whitematteranalysis/gpu/__init__.py` if `wma.gpu.bspline` lazy attribute access is desired alongside `distance`, `linalg`, `registration`. | 2026-12-05 |