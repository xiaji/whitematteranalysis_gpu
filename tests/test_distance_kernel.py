"""Fused distance arithmetic retains eager PyTorch values and fallback paths."""
import unittest
from itertools import product
from unittest.mock import patch
import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, distance_kernel


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class DistanceKernelTests(unittest.TestCase):
    def test_empty_public_matrices(self):
        from test_atlas_blocks import polydata
        empty, populated = polydata(0), polydata(3)
        for method in distance.SUPPORTED_DISTANCE_METHODS:
            result = distance.pairwise_distance_matrix(empty, 0., distance_method=method)
            self.assertEqual(result.shape, (0, 0))
            for queries, fixed, shape in ((empty, populated, (0, 3)),
                                          (populated, empty, (3, 0))):
                result = distance.rectangular_distance_matrix(
                    queries, fixed, 0., distance_method=method)
                self.assertEqual(result.shape, shape)

    def test_float64_preserves_small_differences_at_large_offsets(self):
        queries = np.full((1, 15, 3), 1e8)
        fixed = queries.copy()
        fixed[:, :, 0] += 1.
        for method in ('Mean', 'MeanSquared', 'Hausdorff'):
            precise = distance.distance_matrix_from_arrays(
                queries, fixed, 0., distance_method=method, gpu_float32=False)
            np.testing.assert_array_equal(precise, [[1.]])
            approximate = distance.distance_matrix_from_arrays(
                queries, fixed, 0., distance_method=method, gpu_float32=True)
            np.testing.assert_array_equal(approximate, [[0.]])

    def test_gaussian_similarities_match_frozen_backend(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._gaussianbaseline',
                                   'gpu/distance.py', '318a392')
        torch.manual_seed(83)
        for points, dtype in product((10, 15, 50), (torch.float64, torch.float32)):
            fixed = torch.randn((37, points, 3), dtype=dtype, device='cuda')
            queries = fixed[:5].clone()
            fixed[0, 0, 0], fixed[1, 0, 0] = float('nan'), float('inf')
            before = queries.clone()
            for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                for sigma in (1e-5, 20., 1e6, 0., float('inf'), float('nan')):
                    expected = original._similarity_tensor_block(queries, fixed, .1, sigma, method, bilateral, 6400.)
                    actual = distance._similarity_tensor_block(queries, fixed, .1, sigma, method, bilateral, 6400.)
                    torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
            self.assertTrue(torch.equal(queries, before))

    def test_eager_scratch_matches_frozen_backend(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._scratchbaseline',
                                   'gpu/distance.py', '9c53829')
        rng = np.random.RandomState(78)
        with patch.object(distance_kernel, 'triton', None):
            for points, dtype, strided in product((10, 50), (torch.float32, torch.float64), (False, True)):
                fixed = torch.tensor(rng.normal(size=(37, points, 3)), dtype=dtype, device='cuda')
                moving = fixed[:5].clone()
                fixed[0, 0, 0], fixed[1, 1, 1] = float('nan'), float('inf')
                if strided:
                    fixed, moving = fixed[:, ::2], moving[:, ::2]
                before_fixed, before_moving = fixed.clone(), moving.clone()
                for method, bilateral, threshold in product(distance.SUPPORTED_DISTANCE_METHODS,
                                                            (False, True), (0., .1, float('inf'), float('nan'))):
                    sigmas = (1e-5, 100., 1e6, 0., -1., float('inf'), float('nan')) if method == 'StrictSimilarity' else (6400.,)
                    for sigma in sigmas:
                        expected = original._fiber_block_distances(moving, fixed, threshold, method, sigma, bilateral)
                        actual = distance._fiber_block_distances(moving, fixed, threshold, method, sigma, bilateral)
                        torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
                torch.testing.assert_close(fixed, before_fixed, rtol=0., atol=0., equal_nan=True)
                torch.testing.assert_close(moving, before_moving, rtol=0., atol=0., equal_nan=True)

    def test_strict_probability_scales(self):
        import torch
        rng = np.random.RandomState(48)
        for points in (1, 3, 15, 20, 32, 50, 64, 128):
            fixed = torch.tensor(rng.normal(size=(37, points, 3)), dtype=torch.float64, device='cuda')
            moving = fixed[:5].clone()
            for sigma in (1e-5, 100., 1e6, 0., -1., float('inf'), float('nan')):
                for bilateral in (False, True):
                    actual = distance._fiber_block_distances(moving, fixed, .1,
                        'StrictSimilarity', sigma, bilateral)
                    with patch.object(distance_kernel, 'triton', None):
                        expected = distance._fiber_block_distances(moving, fixed, .1,
                            'StrictSimilarity', sigma, bilateral)
                    torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)

    def test_eager_equivalence(self):
        import torch
        rng = np.random.RandomState(42)
        for points in (1, 2, 3, 7, 10, 15, 16, 20, 31, 32, 50, 64, 128):
            for dtype in (torch.float64, torch.float32):
                fixed = torch.tensor(rng.normal(size=(37, points, 3)) * 40., dtype=dtype, device='cuda')
                moving = torch.tensor(rng.normal(size=(5, points, 3)) * 40., dtype=dtype, device='cuda')
                before = moving.clone()
                for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                    for threshold in (0., .1, 2., 100.):
                        with self.subTest(points=points, dtype=dtype, method=method, threshold=threshold, bilateral=bilateral):
                            actual = distance._fiber_block_distances(moving, fixed, threshold, method, 6400., bilateral)
                            with patch.object(distance_kernel, 'triton', None):
                                expected = distance._fiber_block_distances(moving, fixed, threshold, method, 6400., bilateral)
                            self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(moving, before))

    def test_triton_p_boundary_and_float32_similarity_row_sums(self):
        import torch
        if distance_kernel.triton is None:
            self.skipTest('Triton is unavailable')
        rng = np.random.RandomState(91)
        for points in (1, 15, 32, 50, 64, 128):
            for dtype in (torch.float64, torch.float32):
                fixed = torch.tensor(rng.normal(size=(37, points, 3)) * 18., dtype=dtype, device='cuda')
                queries = torch.tensor(rng.normal(size=(13, points, 3)) * 18., dtype=dtype, device='cuda')
                if points <= 64:
                    with patch.object(distance_kernel, 'fiber_distances', wraps=distance_kernel.fiber_distances) as spy:
                        distance._fiber_block_distances(queries, fixed, 0.1, 'Mean', 400.)
                        self.assertEqual(spy.call_count, 1)
                else:
                    with patch.object(distance_kernel, 'fiber_distances',
                                      side_effect=AssertionError('P > 64 must use PyTorch eager fallback')):
                        distance._fiber_block_distances(queries, fixed, 0.1, 'Mean', 400.)

                for bilateral, sqrt_sums in product((False, True), (False, True)):
                    with patch.object(distance_kernel, 'triton', None):
                        eager_dists = distance._fiber_block_distances(
                            queries, fixed, 0.5, 'Mean', 400., bilateral=bilateral)
                    expected_sim = torch.exp(-eager_dists / 400.).sum(dim=1)
                    expected_dist = eager_dists.sqrt().sum(dim=1) if sqrt_sums else None
                    if points <= 64:
                        actual_sim, actual_dist = distance_kernel.similarity_row_sums(
                            queries, fixed, 0.5, 400., bilateral=bilateral,
                            sqrt_distance_sums=sqrt_sums)
                    else:
                        q_np = queries.cpu().numpy()
                        f_np = fixed.cpu().numpy()
                        with patch.object(distance_kernel, 'similarity_row_sums',
                                          side_effect=AssertionError('P > 64 must use PyTorch fallback')):
                            res = distance.similarity_row_sums_from_arrays(
                                q_np, f_np, 0.5, distance_method='Mean',
                                bilateral=bilateral, sigmasq=400.,
                                gpu_float32=(dtype == torch.float32),
                                tile_rows=64, sqrt_distance_sums=sqrt_sums)
                        if sqrt_sums:
                            actual_sim = torch.as_tensor(res[0])
                            actual_dist = torch.as_tensor(res[1])
                        else:
                            actual_sim = torch.as_tensor(res)
                            actual_dist = None
                    tol = 2e-5 if dtype == torch.float32 else 1e-13
                    np.testing.assert_allclose(
                        actual_sim.cpu().numpy(), expected_sim.cpu().numpy(),
                        rtol=tol, atol=tol)
                    if sqrt_sums:
                        np.testing.assert_allclose(
                            actual_dist.cpu().numpy(), expected_dist.cpu().numpy(),
                            rtol=tol, atol=tol)

    def test_special_values_and_strides(self):
        import torch
        for strided in (False, True):
            fixed = torch.ones((7, 10, 3), dtype=torch.float64, device='cuda')
            fixed[0, 0, 0], fixed[1, 1, 1] = float('nan'), float('inf')
            moving = torch.zeros((3, 10, 3), dtype=torch.float64, device='cuda')
            if strided:
                fixed, moving = fixed[:, ::2], moving[:, ::2]
            for method, bilateral in product(distance.SUPPORTED_DISTANCE_METHODS, (False, True)):
                actual = distance._fiber_block_distances(moving, fixed, .1, method, 6400., bilateral)
                with patch.object(distance_kernel, 'triton', None):
                    expected = distance._fiber_block_distances(moving, fixed, .1, method, 6400., bilateral)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)

    def test_selected_device(self):
        import torch
        if distance_kernel.triton is None or torch.cuda.device_count() < 2:
            self.skipTest('Triton and two GPUs are required')
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        fixed = torch.ones((7, 10, 3), dtype=torch.float64, device=selected)
        moving = torch.zeros((3, 10, 3), dtype=torch.float64, device=selected)
        for similarity_sigmasq in (None, 400.):
            actual = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400.,
                bilateral=True, similarity_sigmasq=similarity_sigmasq)
            with patch.object(distance_kernel, 'triton', None):
                expected = distance._fiber_block_distances(moving, fixed, .1, 'Mean', 6400.,
                    bilateral=True, similarity_sigmasq=similarity_sigmasq)
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.cuda.current_device(), previous)

    def test_pairwise_and_rectangular_similarity_use_gpu_exp(self):
        from test_atlas_blocks import polydata
        pd_n, pd_m = polydata(5), polydata(4)
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            ref_pair = distance.pairwise_distance_matrix(pd_n, 0.1, distance_method=method)
            ref_rect = distance.rectangular_distance_matrix(pd_n, pd_m, 0.1, distance_method=method)
            if method != 'StrictSimilarity':
                ref_pair = np.exp(-ref_pair / (20. * 20.))
                ref_rect = np.exp(-ref_rect / (20. * 20.))
            with patch.object(np, 'exp', side_effect=AssertionError('numpy.exp should not be called')):
                actual_pair = distance.pairwise_similarity_matrix(pd_n, 0.1, 20., distance_method=method)
                actual_rect = distance.rectangular_similarity_matrix(pd_n, pd_m, 0.1, 20., distance_method=method)
            np.testing.assert_allclose(actual_pair, ref_pair, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual_rect, ref_rect, rtol=1e-12, atol=1e-12)


class DistanceSimilarityCpuTests(unittest.TestCase):
    def _require_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        return torch

    def test_pairwise_and_rectangular_similarity_use_gpu_exp_cpu(self):
        torch = self._require_torch()
        from test_atlas_blocks import polydata
        pd_n, pd_m = polydata(5), polydata(4)
        cpu_dev = torch.device('cpu')
        with patch.object(distance, 'get_device', return_value=cpu_dev):
            for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
                ref_pair = distance.pairwise_distance_matrix(pd_n, 0.1, distance_method=method)
                ref_rect = distance.rectangular_distance_matrix(pd_n, pd_m, 0.1, distance_method=method)
                if method != 'StrictSimilarity':
                    ref_pair = np.exp(-ref_pair / (20. * 20.))
                    ref_rect = np.exp(-ref_rect / (20. * 20.))
                with patch.object(np, 'exp', side_effect=AssertionError('numpy.exp should not be called')):
                    actual_pair = distance.pairwise_similarity_matrix(pd_n, 0.1, 20., distance_method=method)
                    actual_rect = distance.rectangular_similarity_matrix(pd_n, pd_m, 0.1, 20., distance_method=method)
                np.testing.assert_allclose(actual_pair, ref_pair, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual_rect, ref_rect, rtol=1e-12, atol=1e-12)

    def test_choose_tile_rows_triton_2d(self):
        torch = self._require_torch()
        # With 4 GiB fallback budget (0.55 * 4 GiB = 2.2 GiB):
        # m = 2000, P = 15, float64 (itemsize = 8):
        # eager: bytes_per_row = 2000 * 15 * 8 * 6 = 1,440,000 -> ~1640 rows
        # triton_2d: bytes_per_row = 2000 * 8 * 3 = 48,000 -> capped at min(49198, 16384, n)
        with patch.object(torch.cuda, 'mem_get_info', side_effect=RuntimeError('no cuda')):
            eager_rows = distance._choose_tile_rows(50000, 2000, 15, torch.float64, torch.device('cpu'), triton_2d=False)
            triton_rows = distance._choose_tile_rows(50000, 2000, 15, torch.float64, torch.device('cpu'), triton_2d=True)
        self.assertLess(eager_rows, triton_rows)
        self.assertEqual(triton_rows, 16384)

    def test_explicit_device_and_iter_triton_2d_checks(self):
        torch = self._require_torch()
        from test_atlas_blocks import polydata

        pd_n, pd_m = polydata(5), polydata(4)
        cpu_dev = torch.device('cpu')
        with patch.object(distance, 'get_device', side_effect=AssertionError('explicit device must skip get_device')):
            blocks = list(distance.iter_rectangular_similarity(
                pd_n, pd_m, 0.0, 20.0, distance_method='Mean',
                gpu_float32=True, tile_rows=2, device=cpu_dev))
            self.assertEqual(sum(b.shape[1] for _, b in blocks), 5)
            self.assertEqual(blocks[0][1].dtype, torch.float32)

            rng = np.random.RandomState(55)
            q = rng.normal(size=(5, 15, 3))
            f = rng.normal(size=(4, 15, 3))
            sums = distance.similarity_row_sums_from_arrays(
                q, f, 0.0, distance_method='Mean', sigmasq=400.,
                gpu_float32=True, device=cpu_dev)
            self.assertEqual(sums.dtype, np.float32)

        # Verify iter_rectangular_similarity passes triton_2d=False when sigma<=0 or sigmasq<=0 or non-CUDA
        seen_triton_2d = []
        orig_choose = distance._choose_tile_rows
        def recording_choose(*args, **kwargs):
            seen_triton_2d.append(kwargs.get('triton_2d', False))
            return orig_choose(*args, **kwargs)
        with patch.object(distance_kernel, 'triton', object()), \
                patch.object(distance, '_choose_tile_rows', side_effect=recording_choose):
            list(distance.iter_rectangular_similarity(
                pd_n, pd_m, 0.0, 0.0, distance_method='Mean', sigmasq=400., device=cpu_dev))
            list(distance.iter_rectangular_similarity(
                pd_n, pd_m, 0.0, 20.0, distance_method='Mean', sigmasq=0.0, device=cpu_dev))
        self.assertEqual(seen_triton_2d, [False, False])

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_dev = torch.device('mps')
            mps_blocks = list(distance.iter_rectangular_similarity(
                pd_n, pd_m, 0.0, 20.0, distance_method='Mean',
                gpu_float32=True, tile_rows=3, device=mps_dev))
            self.assertEqual(mps_blocks[0][1].device.type, 'mps')
            self.assertEqual(mps_blocks[0][1].dtype, torch.float32)
