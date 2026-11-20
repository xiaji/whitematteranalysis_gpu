"""Short-vector CUDA distances preserve direct cdist values, ties and fallback."""
import unittest
from unittest.mock import patch

import whitematteranalysis as wma
from whitematteranalysis.gpu import distance_kernel, linalg


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class CentroidKernelTests(unittest.TestCase):
    def test_does_not_retain_tensors(self):
        import gc
        import torch
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        query = torch.ones((257, 10), dtype=torch.float64, device='cuda')
        centroids = torch.zeros((37, 10), dtype=torch.float64, device='cuda')
        result = linalg._nearest_centroids(query, centroids)
        del result, query, centroids
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), baseline)

    def test_exact_direct_distances(self):
        import torch
        if distance_kernel.triton is None:
            self.skipTest('Triton is unavailable')
        torch.manual_seed(42)
        for dimensions in (1, 2, 3, 4, 5, 7, 10, 15, 16, 20, 31, 32):
            for scale in (1e-100, 1., 1e100, 1e160):
                query = torch.randn((37, dimensions), dtype=torch.float64, device='cuda') * scale
                centroids = torch.randn((257, dimensions), dtype=torch.float64, device='cuda') * scale
                actual = distance_kernel.centroid_distances(query, centroids)
                expected = torch.cdist(query, centroids, compute_mode='donot_use_mm_for_euclid_dist')
                self.assertTrue(torch.equal(actual, expected), (dimensions, scale))

    def test_ties_special_values_strides_and_fallback(self):
        import torch
        for dimensions in (0, 1, 10, 32, 33):
            for dtype in (torch.float32, torch.float64):
                query = torch.zeros((7, dimensions), dtype=dtype, device='cuda')
                centroids = torch.ones((5, dimensions), dtype=dtype, device='cuda')
                query[1:] = 1.
                for inputs in ((query, centroids), (query[:, ::2], centroids[:, ::2]),
                               (query[:0], centroids), (query.cpu(), centroids.cpu())):
                    actual = linalg._nearest_centroids(*inputs)
                    with patch.object(distance_kernel, 'triton', None):
                        expected = linalg._nearest_centroids(*inputs)
                    for a, b in zip(actual, expected):
                        self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(actual[1], torch.zeros_like(actual[1])))
        query = torch.zeros((3, 10), dtype=torch.float64, device='cuda')
        centroids = torch.ones((7, 10), dtype=torch.float64, device='cuda')
        query[0, 0], query[1, 1] = float('nan'), float('inf')
        actual = linalg._nearest_centroids(query, centroids)
        with patch.object(distance_kernel, 'triton', None):
            expected = linalg._nearest_centroids(query, centroids)
        torch.testing.assert_close(actual[0], expected[0], rtol=0., atol=0., equal_nan=True)
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_selected_device(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('Two GPUs are required')
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        query = torch.ones((7, 10), dtype=torch.float64, device=selected)
        centroids = torch.zeros((257, 10), dtype=torch.float64, device=selected)
        actual = linalg._nearest_centroids(query, centroids)
        with patch.object(distance_kernel, 'triton', None):
            expected = linalg._nearest_centroids(query, centroids)
        self.assertEqual(torch.cuda.current_device(), previous)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))

    def test_fused_nearest_centroids_matches_cdist_min(self):
        import torch
        if distance_kernel.triton is None:
            self.skipTest('Triton is unavailable')
        torch.manual_seed(42)
        for dimensions in (1, 2, 10, 16, 32):
            for scale in (1e-100, 1., 1e100):
                for nc in (5, 64, 65, 257):
                    query = torch.randn((37, dimensions), dtype=torch.float64, device='cuda') * scale
                    centroids = torch.randn((nc, dimensions), dtype=torch.float64, device='cuda') * scale
                    actual_d, actual_i = distance_kernel.nearest_centroids(query, centroids)
                    expected_d, expected_i = torch.cdist(
                        query, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                    self.assertEqual(actual_i.dtype, torch.int64)
                    self.assertTrue(torch.equal(actual_i, expected_i), (dimensions, scale, nc))
                    torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
            # Exact ties across blocks (first centroid wins)
            query = torch.zeros((4, dimensions), dtype=torch.float64, device='cuda')
            centroids = torch.ones((150, dimensions), dtype=torch.float64, device='cuda')
            centroids[3] = 0.
            centroids[70] = 0.
            actual_d, actual_i = distance_kernel.nearest_centroids(query, centroids)
            expected_d, expected_i = torch.cdist(
                query, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
            self.assertTrue(torch.equal(actual_i, expected_i))
            torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
            # All-+inf distances against finite centroids (first centroid 0 wins)
            inf_q = torch.zeros((2, dimensions), dtype=torch.float64, device='cuda')
            inf_q[1, 0] = float('inf')
            finite_c = torch.ones((150, dimensions), dtype=torch.float64, device='cuda')
            actual_d, actual_i = distance_kernel.nearest_centroids(inf_q, finite_c)
            expected_d, expected_i = torch.cdist(
                inf_q, finite_c, compute_mode='donot_use_mm_for_euclid_dist').min(1)
            self.assertTrue(torch.equal(actual_i, expected_i))
            torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
            # NaN and Inf entries (in queries and in later centroid blocks)
            query = torch.zeros((5, dimensions), dtype=torch.float64, device='cuda')
            centroids = torch.ones((150, dimensions), dtype=torch.float64, device='cuda')
            query[0, 0] = float('nan')
            query[1, 0] = float('inf')
            centroids[75, 0] = float('nan')
            centroids[120, 0] = float('nan')
            actual_d, actual_i = distance_kernel.nearest_centroids(query, centroids)
            expected_d, expected_i = torch.cdist(
                query, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
            self.assertTrue(torch.equal(actual_i, expected_i))
            torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
            # Empty queries (0, D)
            empty_q = torch.empty((0, dimensions), dtype=torch.float64, device='cuda')
            actual_d, actual_i = distance_kernel.nearest_centroids(empty_q, centroids)
            expected_d, expected_i = torch.cdist(
                empty_q, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
            self.assertEqual(actual_i.dtype, torch.int64)
            self.assertTrue(torch.equal(actual_i, expected_i))
            torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
            # Empty centroids (0, D) raises IndexError matching torch.min(1)
            empty_c = torch.empty((0, dimensions), dtype=torch.float64, device='cuda')
            for q_in in (query, empty_q):
                with self.assertRaises(IndexError):
                    distance_kernel.nearest_centroids(q_in, empty_c)

    def test_split_c_nearest_centroids_matches_cdist_min(self):
        import torch
        if distance_kernel.triton is None:
            self.skipTest('Triton is unavailable')
        torch.manual_seed(2026)
        nq, nc = 130, 600  # nq * nc = 78000 >= 65536, nc % BLOCK_C != 0
        for dimensions in (1, 10, 16, 32):
            for dtype in (torch.float64, torch.float32):
                with self.subTest(dimensions=dimensions, dtype=dtype):
                    query = torch.randn((nq, dimensions), dtype=dtype, device='cuda')
                    centroids = torch.randn((nc, dimensions), dtype=dtype, device='cuda')
                    actual_d, actual_i = distance_kernel.nearest_centroids(query, centroids)
                    expected_d, expected_i = torch.cdist(
                        query, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                    self.assertEqual(actual_d.dtype, dtype)
                    self.assertEqual(actual_i.dtype, torch.int64)
                    self.assertTrue(torch.equal(actual_i, expected_i))
                    torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)

                    # Cross-split exact ties (first centroid wins across splits 0, 1, final)
                    tie_q = torch.zeros((nq, dimensions), dtype=dtype, device='cuda')
                    tie_c = torch.ones((nc, dimensions), dtype=dtype, device='cuda')
                    tie_c[3] = 0.
                    tie_c[70] = 0.
                    tie_c[580] = 0.
                    actual_d, actual_i = distance_kernel.nearest_centroids(tie_q, tie_c)
                    expected_d, expected_i = torch.cdist(
                        tie_q, tie_c, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                    self.assertTrue(torch.equal(actual_i, expected_i))
                    torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)

                    # Cross-split NaNs and Infs
                    nan_q = torch.zeros((nq, dimensions), dtype=dtype, device='cuda')
                    nan_c = torch.ones((nc, dimensions), dtype=dtype, device='cuda')
                    nan_q[0, 0] = float('nan')
                    nan_q[1, 0] = float('inf')
                    nan_c[75, 0] = float('nan')
                    nan_c[585, 0] = float('nan')
                    actual_d, actual_i = distance_kernel.nearest_centroids(nan_q, nan_c)
                    expected_d, expected_i = torch.cdist(
                        nan_q, nan_c, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                    self.assertTrue(torch.equal(actual_i, expected_i))
                    torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)


class CentroidKernelApiTests(unittest.TestCase):
    def test_nearest_centroids_symbol_and_cpu_fallback(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        self.assertTrue(hasattr(distance_kernel, 'nearest_centroids'))
        self.assertTrue(callable(distance_kernel.nearest_centroids))
        query = torch.zeros((4, 10), dtype=torch.float64)
        centroids = torch.ones((7, 10), dtype=torch.float64)
        query[0, 0], query[1, 1] = float('nan'), float('inf')
        actual_d, actual_i = linalg._nearest_centroids(query, centroids)
        expected_d, expected_i = torch.cdist(
            query, centroids, compute_mode='donot_use_mm_for_euclid_dist').min(1)
        self.assertEqual(actual_i.dtype, torch.int64)
        self.assertTrue(torch.equal(actual_i, expected_i))
        torch.testing.assert_close(actual_d, expected_d, rtol=0., atol=0., equal_nan=True)
        # Empty centroids (nc == 0) raises IndexError for both nq > 0 and nq == 0
        empty_c = torch.empty((0, 10), dtype=torch.float64)
        for q_in in (query, torch.empty((0, 10), dtype=torch.float64)):
            with self.assertRaises(IndexError):
                distance_kernel.nearest_centroids(q_in, empty_c)
            with self.assertRaises(IndexError):
                torch.cdist(q_in, empty_c, compute_mode='donot_use_mm_for_euclid_dist').min(1)


if __name__ == '__main__':
    unittest.main()
