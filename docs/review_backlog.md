# Review Backlog

Non-blocking (P2) items recorded per `review-convergence.md` §1.2. Items not pulled into a task by their decide-by date are closed "won't do".

| Date | Source | File:Line | Item | Decide-By |
|---|---|---|---|---|
| 2026-11-20 | Codex Phase 1 Code Re-Review | `tests/test_filter_outliers.py:166-168` | Add a CUDA-backed case (skipped when CUDA unavailable) to `test_nearest_neighbor_distances_uses_exact_numpy_division` so CUDA reciprocal-multiply regressions are directly caught on GPU runners. | 2026-12-04 |
