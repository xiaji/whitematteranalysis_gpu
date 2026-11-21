# Review Backlog

All Phase 1 and Phase 2 non-blocking (P2) items recorded per `review-convergence.md` §1.2 were resolved in Phase 3 (`feat/phase3-p2-nonrigid-atlas-and-backlog`).

| Date | Source | File:Line | Item | Resolution |
|---|---|---|---|---|
| 2026-11-20 | Codex Phase 1 Code Re-Review | `tests/test_filter_outliers.py` | Add CUDA-backed exact division test for `nearest_neighbor_distances_from_fiber_array`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Task 4 Reviewer | `whitematteranalysis/gpu/_memory.py` | Guard `effective_free_vram` with `_device_type(device) == 'cuda'`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Codex Phase 2 Final Review | `whitematteranalysis/congeal_multisubject.py` | Opt-in spawn `ProcessPoolExecutor` (`WMA_GPU_PROCESS_POOL=1`) for concurrent multi-GPU `fmin_cobyla`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Whole-Branch Final Reviewer | `whitematteranalysis/gpu/bspline.py` | Support `float32_enabled()` in `NonrigidObjective` (CUDA & MPS/CPU). | Resolved (`ab89821`) |
| 2026-11-21 | Task 5 Reviewer | `bin/wm_apply_ORG_atlas.py` | Preserve empty-string `WMA_VTP_COMPRESSION=''` restoration & use `_default_lz4_compression()` on intermediate dumps. | Resolved (`41c5e8f`) |
| 2026-11-21 | Final Fix Re-Reviewer | `bin/wm_apply_ORG_atlas.py` | Single `pre_tfm` hardening pass producing native coordinates in `TransformedClusters/` for non-Tier-1 `-t` `--keep-intermediates 2`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Task 5 Reviewer | `whitematteranalysis/transform.py` | Explicit `ValueError` context on singular `.tfm` matrices in `read_itk_affine_matrix_ras`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Task 5 Fix Re-Reviewer | `bin/wm_harden_transform.py` | Include pre-fallback `--prepare-command` duration in `preparation.json` `combined_s`. | Resolved (`41c5e8f`) |
| 2026-11-21 | Whole-Branch Final Reviewer | `whitematteranalysis/gpu/__init__.py` | Export `'bspline'` in `_BACKENDS` for lazy `wma.gpu.bspline` access. | Resolved (`41c5e8f`) |