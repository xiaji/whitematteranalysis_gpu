"""Regression tests for Task 7 spectral atlas construction and LOO outlier scoring."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma


def _make_bundle_polydata(seed=0, fibers_per_cluster=12, n_pts=15,
                          jitter_scale=0.6, bend_scale=0.0):
    rng = np.random.RandomState(seed)
    centers = ((-35.0, 0.0, 5.0), (35.0, 0.0, 5.0), (0.0, 35.0, 10.0))
    t = np.linspace(0.0, 1.0, n_pts)
    arch = np.sin(np.pi * t)
    all_pts = []
    offsets = [0]
    for cx, cy, cz in centers:
        for _ in range(fibers_per_cluster):
            jitter = rng.normal(scale=jitter_scale, size=3)
            bend = rng.normal(scale=bend_scale, size=2) if bend_scale > 0.0 else (0.0, 0.0)
            xs = np.linspace(cx + jitter[0], cx + jitter[0] + 15.0, n_pts)
            ys = np.full(n_pts, cy + jitter[1]) + np.linspace(-5.0, 5.0, n_pts) + bend[0] * arch
            zs = np.full(n_pts, cz + jitter[2]) + bend[1] * arch
            all_pts.append(np.column_stack((xs, ys, zs)))
            offsets.append(offsets[-1] + n_pts)
    pts_arr = np.ascontiguousarray(np.vstack(all_pts), dtype=np.float64)
    n_fibers = len(offsets) - 1
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts_arr, deep=True))
    cells = vtk.vtkCellArray()
    conn = np.arange(pts_arr.shape[0], dtype=np.int64)
    cells.SetData(
        numpy_to_vtk(np.asarray(offsets, dtype=np.int64), deep=True, array_type=vtk.VTK_ID_TYPE),
        numpy_to_vtk(conn, deep=True, array_type=vtk.VTK_ID_TYPE),
    )
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetLines(cells)
    return pd, n_fibers


def _naive_loo_similarity(pd_c, subject_ids, sigma, distance_method='Mean', bilateral=False):
    dist = wma.cluster._pairwise_distance_matrix(
        pd_c, 0.0, number_of_jobs=1, bilateral=bilateral,
        distance_method=distance_method, sigmasq=sigma * sigma, use_gpu=False)
    sim = dist if distance_method == 'StrictSimilarity' else wma.similarity.distance_to_similarity(dist, sigma * sigma)
    subj = np.asarray(subject_ids)
    out = np.empty(len(subj), dtype=np.float64)
    for i, s in enumerate(subj):
        mask = subj != s
        n_loo = int(np.sum(mask))
        out[i] = np.nan if n_loo == 0 else np.sum(sim[i, mask]) / n_loo
    return out, dist, sim


class AtlasConstructionTests(unittest.TestCase):
    def test_near_psd_symmetric_eigh_matches_legacy_reference(self):
        rng = np.random.RandomState(42)
        M = rng.normal(size=(16, 16))
        A = M @ M.T
        d = np.sqrt(np.diag(A))
        C = A / np.outer(d, d)
        # Perturb to introduce negative eigenvalues while keeping symmetry and unit diag
        C[0, 1] = C[1, 0] = 0.99
        C[0, 2] = C[2, 0] = -0.99
        C[1, 2] = C[2, 1] = 0.99

        # Legacy numpy.linalg.eig formula without matrix class deprecation issues
        eigval, eigvec = np.linalg.eig(C)
        val = np.maximum(eigval.real, 0.0)
        vec = eigvec.real
        t_diag = np.sqrt(1.0 / ((vec * vec) @ val))
        B = (t_diag[:, None] * vec) * np.sqrt(val)[None, :]
        expected = B @ B.T

        actual = wma.cluster.nearPSD(C)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual, actual.T, rtol=1e-13, atol=1e-13)
        evals = np.linalg.eigvalsh(actual)
        self.assertGreaterEqual(evals[0], -1e-12)

    def test_spectral_nystrom_pinv_and_top_k_projection_cpu(self):
        pd, n_fibers = _make_bundle_polydata(seed=7, fibers_per_cluster=40)
        nystrom_mask = np.zeros(n_fibers, dtype=bool)
        nystrom_mask[:105] = True

        np.random.seed(123)
        out_pd, idx, color, embed, _, atlas, reject_idx = wma.cluster.spectral(
            pd,
            number_of_clusters=3,
            number_of_eigenvectors=3,
            sigma=30.0,
            threshold=0.0,
            number_of_jobs=1,
            use_nystrom=True,
            nystrom_mask=nystrom_mask.copy(),
            distance_method='MeanSquared',
            outlier_std_threshold=np.inf,
            pos_def_approx=True,
            bilateral=False,
            use_gpu=False,
        )
        self.assertEqual(len(reject_idx), 0)
        self.assertEqual(out_pd.GetNumberOfLines(), n_fibers)
        self.assertEqual(atlas.e_vec_norm.shape, (atlas.e_vec.shape[1],))
        self.assertEqual(embed.shape, (n_fibers, 3))
        self.assertTrue(np.all(np.isfinite(embed)))

        # Verify spectral_atlas_label reproduces consistent assignments on sampled input
        labeled_pd, lbl_idx, _, lbl_embed = wma.cluster.spectral_atlas_label(
            pd, atlas, number_of_jobs=1, use_gpu=False)
        self.assertEqual(labeled_pd.GetNumberOfLines(), n_fibers)
        np.testing.assert_allclose(lbl_embed, embed, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(lbl_idx, idx)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
    def test_spectral_nystrom_gpu_resident_matches_cpu(self):
        pd, n_fibers = _make_bundle_polydata(
            seed=19, fibers_per_cluster=40, jitter_scale=12.0, bend_scale=6.0)
        nystrom_mask = np.zeros(n_fibers, dtype=bool)
        nystrom_mask[:105] = True

        np.random.seed(55)
        _, idx_cpu, _, embed_cpu, _, atlas_cpu, rej_cpu = wma.cluster.spectral(
            pd,
            number_of_clusters=3,
            number_of_eigenvectors=3,
            sigma=25.0,
            threshold=0.0,
            number_of_jobs=1,
            use_nystrom=True,
            nystrom_mask=nystrom_mask.copy(),
            distance_method='Mean',
            outlier_std_threshold=np.inf,
            pos_def_approx=True,
            bilateral=True,
            use_gpu=False,
        )
        np.random.seed(55)
        _, idx_gpu, _, embed_gpu, _, atlas_gpu, rej_gpu = wma.cluster.spectral(
            pd,
            number_of_clusters=3,
            number_of_eigenvectors=3,
            sigma=25.0,
            threshold=0.0,
            number_of_jobs=1,
            use_nystrom=True,
            nystrom_mask=nystrom_mask.copy(),
            distance_method='Mean',
            outlier_std_threshold=np.inf,
            pos_def_approx=True,
            bilateral=True,
            use_gpu=True,
        )
        np.testing.assert_array_equal(rej_gpu, rej_cpu)
        np.testing.assert_allclose(atlas_gpu.pinv_A, atlas_cpu.pinv_A, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(atlas_gpu.e_vec_norm[-4:], atlas_cpu.e_vec_norm[-4:], rtol=1e-10, atol=1e-10)
        aligned_embed_gpu = embed_gpu.copy()
        for col in range(aligned_embed_gpu.shape[1]):
            if np.dot(aligned_embed_gpu[:, col], embed_cpu[:, col]) < 0.0:
                aligned_embed_gpu[:, col] *= -1.0
        np.testing.assert_allclose(aligned_embed_gpu, embed_cpu, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(
            idx_gpu[:, None] == idx_gpu[None, :],
            idx_cpu[:, None] == idx_cpu[None, :],
        )

    def test_compute_cluster_loo_similarity_cpu_and_edge_cases(self):
        pd, n_fibers = _make_bundle_polydata(seed=31, fibers_per_cluster=6)
        # Include singleton subject (id=99) and multi-fiber subjects
        subj_ids = np.array([0] * 10 + [1] * 7 + [99], dtype=np.int64)
        self.assertEqual(len(subj_ids), n_fibers)

        for bilateral in (False, True):
            for method in ('Mean', 'MeanSquared'):
                with self.subTest(bilateral=bilateral, method=method):
                    expected_sim, expected_dist, expected_mat = _naive_loo_similarity(
                        pd, subj_ids, sigma=20.0, distance_method=method, bilateral=bilateral)
                    actual_sim, actual_dist, actual_mat = wma.cluster.compute_cluster_loo_similarity(
                        pd, subj_ids, sigma=20.0, distance_method=method,
                        bilateral=bilateral, use_gpu=False, return_distances=True)
                    np.testing.assert_allclose(actual_dist, expected_dist, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(actual_mat, expected_mat, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(actual_sim, expected_sim, rtol=1e-12, atol=1e-12)

                    sim_only = wma.cluster.compute_cluster_loo_similarity(
                        pd, subj_ids, sigma=20.0, distance_method=method,
                        bilateral=bilateral, use_gpu=False, return_distances=False)
                    np.testing.assert_allclose(sim_only, expected_sim, rtol=1e-12, atol=1e-12)

        # Single-subject cluster returns all NaNs without errors
        single_subj = np.zeros(n_fibers, dtype=np.int64)
        nan_out = wma.cluster.compute_cluster_loo_similarity(
            pd, single_subj, sigma=20.0, distance_method='Mean', use_gpu=False)
        self.assertTrue(np.all(np.isnan(nan_out)))

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
    def test_compute_cluster_loo_similarity_cuda(self):
        pd, n_fibers = _make_bundle_polydata(seed=41, fibers_per_cluster=10)
        subj_ids = np.array([0] * 12 + [1] * 17 + [2], dtype=np.int64)
        self.assertEqual(len(subj_ids), n_fibers)

        for bilateral in (False, True):
            expected_sim, _, _ = _naive_loo_similarity(
                pd, subj_ids, sigma=20.0, distance_method='Mean', bilateral=bilateral)
            gpu_sim = wma.cluster.compute_cluster_loo_similarity(
                pd, subj_ids, sigma=20.0, distance_method='Mean',
                bilateral=bilateral, use_gpu=True, gpu_float32=False, return_distances=False)
            np.testing.assert_allclose(gpu_sim, expected_sim, rtol=1e-12, atol=1e-12)

            gpu_f32 = wma.cluster.compute_cluster_loo_similarity(
                pd, subj_ids, sigma=20.0, distance_method='Mean',
                bilateral=bilateral, use_gpu=True, gpu_float32=True, return_distances=False)
            np.testing.assert_allclose(gpu_f32, expected_sim, rtol=1e-5, atol=1e-5)

    def test_output_and_quality_control_cluster_atlas_parallel_write(self):
        pd, n_fibers = _make_bundle_polydata(seed=52, fibers_per_cluster=36)
        nystrom_mask = np.zeros(n_fibers, dtype=bool)
        nystrom_mask[:102] = True
        np.random.seed(7)
        out_pd, idx, color, embed, _, atlas, reject_idx = wma.cluster.spectral(
            pd,
            number_of_clusters=3,
            number_of_eigenvectors=3,
            sigma=30.0,
            threshold=0.0,
            number_of_jobs=1,
            use_nystrom=True,
            nystrom_mask=nystrom_mask,
            distance_method='MeanSquared',
            outlier_std_threshold=np.inf,
            use_gpu=False,
        )
        subj_list = np.arange(out_pd.GetNumberOfLines(), dtype=np.int64) % 3
        with tempfile.TemporaryDirectory() as tmp:
            wma.cluster.output_and_quality_control_cluster_atlas(
                atlas, out_pd, subj_list, ['sub0.vtp', 'sub1.vtp', 'sub2.vtp'],
                3, tmp, idx, color, embed, 50,
                testing=False, verbose=False, render_images=False, number_of_jobs=2,
            )
            out_dir = Path(tmp)
            self.assertTrue((out_dir / 'atlas.p').is_file())
            self.assertTrue((out_dir / 'atlas.vtp').is_file())
            self.assertTrue((out_dir / 'cluster_quality_control.txt').is_file())
            self.assertTrue((out_dir / 'clustered_tracts.mrml').is_file())
            for c_idx in range(1, 4):
                c_vtp = out_dir / f'cluster_{c_idx:05d}.vtp'
                self.assertTrue(c_vtp.is_file())
                c_pd = wma.io.read_polydata(str(c_vtp))
                self.assertIsNotNone(c_pd.GetPointData().GetArray('Subject_ID'))
                self.assertIsNotNone(c_pd.GetPointData().GetArray('Hemisphere'))

    def test_spectral_skips_near_psd_when_already_psd_and_runs_near_psd_when_indefinite(self):
        from unittest.mock import patch

        pd, n_fibers = _make_bundle_polydata(seed=61, fibers_per_cluster=40)
        nystrom_mask = np.zeros(n_fibers, dtype=bool)
        nystrom_mask[:105] = True

        # Well-separated bundles yield PSD affinity A -> 2 eigh calls (A and normalized A) and 0 nearPSD calls
        with patch.object(wma.cluster, 'nearPSD', wraps=wma.cluster.nearPSD) as psd_spy, \
                patch.object(wma.gpu.linalg, 'eigh', wraps=wma.gpu.linalg.eigh) as eigh_spy:
            wma.cluster.spectral(
                pd, number_of_clusters=3, number_of_eigenvectors=3, sigma=30.0,
                threshold=0.0, number_of_jobs=1, use_nystrom=True,
                nystrom_mask=nystrom_mask.copy(), distance_method='MeanSquared',
                outlier_std_threshold=np.inf, pos_def_approx=True, bilateral=False, use_gpu=False,
            )
        self.assertEqual(psd_spy.call_count, 0)
        self.assertEqual(eigh_spy.call_count, 2)

        # Force negative eigenvalue in A while keeping diag(A)==1 -> must call nearPSD + 3rd eigh
        orig_pw = wma.cluster._pairwise_similarity_matrix

        def indefinite_pw(*args, **kwargs):
            A = orig_pw(*args, **kwargs).copy()
            A[0, 1] = A[1, 0] = 0.99
            A[0, 2] = A[2, 0] = 0.01
            A[1, 2] = A[2, 1] = 0.99
            return A

        with patch.object(wma.cluster, '_pairwise_similarity_matrix', side_effect=indefinite_pw), \
                patch.object(wma.cluster, 'nearPSD', wraps=wma.cluster.nearPSD) as psd_spy, \
                patch.object(wma.gpu.linalg, 'eigh', wraps=wma.gpu.linalg.eigh) as eigh_spy:
            wma.cluster.spectral(
                pd, number_of_clusters=3, number_of_eigenvectors=3, sigma=30.0,
                threshold=0.0, number_of_jobs=1, use_nystrom=True,
                nystrom_mask=nystrom_mask.copy(), distance_method='MeanSquared',
                outlier_std_threshold=np.inf, pos_def_approx=True, bilateral=False, use_gpu=False,
            )
        self.assertEqual(psd_spy.call_count, 1)
        self.assertEqual(eigh_spy.call_count, 3)

    def test_gpu_kmeans2_matches_scipy_on_cpu_fallback_and_cuda_triton(self):
        from unittest.mock import patch
        import warnings
        import scipy.cluster.vq
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')

        rng = np.random.RandomState(2026)
        data10 = rng.normal(size=(250, 10)).astype(np.float64)
        init10 = data10[:12].copy()
        # Append an unreachable centroid to verify empty-cluster preservation + UserWarning
        init_with_empty = np.vstack([data10[:11], np.full((1, 10), 1e6, dtype=np.float64)])
        exp_c, exp_l = scipy.cluster.vq.kmeans2(data10, init10, iter=10, minit='matrix')

        act_c, act_l = wma.gpu.linalg.kmeans2(
            data10, 12, n_iter=10, gpu_float32=False, device='cpu', init_centroids=init10)
        np.testing.assert_array_equal(act_l, exp_l)
        np.testing.assert_allclose(act_c, exp_c, rtol=1e-12, atol=1e-12)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            exp_ec, exp_el = scipy.cluster.vq.kmeans2(data10, init_with_empty, iter=5, minit='matrix')
            act_ec, act_el = wma.gpu.linalg.kmeans2(
                data10, 12, n_iter=5, gpu_float32=False, device='cpu', init_centroids=init_with_empty)
        self.assertTrue(any('empty' in str(w.message).lower() for w in caught))
        np.testing.assert_array_equal(act_el, exp_el)
        np.testing.assert_allclose(act_ec, exp_ec, rtol=1e-12, atol=1e-12)

        # High-dimensional D=40 cdist fallback
        data40 = rng.normal(size=(120, 40)).astype(np.float64)
        init40 = data40[:8].copy()
        exp40_c, exp40_l = scipy.cluster.vq.kmeans2(data40, init40, iter=5, minit='matrix')
        act40_c, act40_l = wma.gpu.linalg.kmeans2(
            data40, 8, n_iter=5, gpu_float32=False, device='cpu', init_centroids=init40)
        np.testing.assert_array_equal(act40_l, exp40_l)
        np.testing.assert_allclose(act40_c, exp40_c, rtol=1e-12, atol=1e-12)

        # Verify cluster.spectral dispatches kmeans2 when use_gpu=True
        pd, n_fibers = _make_bundle_polydata(seed=73, fibers_per_cluster=40)
        nystrom_mask = np.zeros(n_fibers, dtype=bool)
        nystrom_mask[:105] = True

        orig_pw = wma.cluster._pairwise_similarity_matrix
        orig_rect = wma.cluster._rectangular_similarity_matrix
        orig_kmeans2 = wma.gpu.linalg.kmeans2
        calls = []

        def tracked_kmeans2(data, k, **kwargs):
            calls.append((data.shape, k))
            return orig_kmeans2(data, k, n_iter=10, gpu_float32=False, device='cpu', seed=99)

        with patch.object(wma.gpu, 'use_gpu', side_effect=lambda req=None: bool(req)), \
                patch.object(wma.cluster, '_pairwise_similarity_matrix',
                             side_effect=lambda *a, **kw: orig_pw(*a, **{**kw, 'use_gpu': False})), \
                patch.object(wma.cluster, '_rectangular_similarity_matrix',
                             side_effect=lambda *a, **kw: orig_rect(*a, **{**kw, 'use_gpu': False})), \
                patch.object(wma.gpu.linalg, 'nystrom_row_sum_2',
                             side_effect=lambda B, rm: B.sum(0) + np.dot(rm, B)), \
                patch.object(wma.gpu.linalg, 'nystrom_normalize_and_embed_top',
                             side_effect=lambda B, da, db, ev, el, **kw: (
                                 ((da[:, None] * B * db[None, :]).T @ ev) / el,
                                 np.sum(np.vstack((ev, ((da[:, None] * B * db[None, :]).T @ ev) / el)) ** 2, 0),
                             )), \
                patch.object(wma.gpu.linalg, 'kmeans2', side_effect=tracked_kmeans2):
            wma.cluster.spectral(
                pd, number_of_clusters=3, number_of_eigenvectors=3, sigma=30.0,
                threshold=0.0, number_of_jobs=1, use_nystrom=True,
                nystrom_mask=nystrom_mask.copy(), distance_method='MeanSquared',
                outlier_std_threshold=np.inf, pos_def_approx=True, bilateral=False, use_gpu=True,
            )
        self.assertEqual(calls, [((n_fibers, 3), 3)])

        if wma.gpu.available():
            data_k800 = rng.normal(size=(4000, 10)).astype(np.float64)
            init_k800 = data_k800[:800].copy()
            exp800_c, exp800_l = scipy.cluster.vq.kmeans2(data_k800, init_k800, iter=10, minit='matrix')
            act800_c, act800_l = wma.gpu.linalg.kmeans2(
                data_k800, 800, n_iter=10, gpu_float32=False, init_centroids=init_k800)
            np.testing.assert_array_equal(act800_l, exp800_l)
            np.testing.assert_allclose(act800_c, exp800_c, rtol=1e-11, atol=1e-11)

            act40_gpu_c, act40_gpu_l = wma.gpu.linalg.kmeans2(
                data40, 8, n_iter=5, gpu_float32=False, init_centroids=init40)
            np.testing.assert_array_equal(act40_gpu_l, exp40_l)
            np.testing.assert_allclose(act40_gpu_c, exp40_c, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
