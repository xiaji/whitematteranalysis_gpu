"""Tiled GPU outlier reductions match the previous full NumPy matrices."""
import os
import unittest
from itertools import product

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, cpu_backend, distance_kernel
from unittest.mock import patch


@unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
class OutlierReductionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(20261030)
        self.atlas = rng.normal(size=(37, 15, 3)) * 18.
        self.queries = rng.normal(size=(13, 15, 3)) * 18.
        self.sigma_sq = 400.

    def old_subject_reductions(self, queries, atlas, threshold, bilateral):
        # Reproduce the old CLI layout: transpose distances, transform the
        # complete matrix with NumPy, then reduce over atlas fibers.
        matrix = distance._distance_matrix(
            queries, atlas, threshold, distance_method='Mean',
            bilateral=bilateral, sigmasq=self.sigma_sq, tile_rows=3)
        similarities = np.exp(-matrix.T / self.sigma_sq)
        return (np.sum(similarities, axis=0),
                np.sum(np.sqrt(matrix.T), axis=0))

    def test_tiled_similarity_and_mean_distance_sums(self):
        for threshold, bilateral in product((0., 2.), (False, True)):
            with self.subTest(threshold=threshold, bilateral=bilateral):
                expected_similarity, expected_distance = self.old_subject_reductions(
                    self.queries, self.atlas, threshold, bilateral)
                actual_similarity, actual_distance = distance.similarity_row_sums_from_arrays(
                    self.queries, self.atlas, threshold, distance_method='Mean',
                    bilateral=bilateral, sigmasq=self.sigma_sq, tile_rows=3,
                    sqrt_distance_sums=True)
                np.testing.assert_allclose(actual_similarity, expected_similarity,
                                           rtol=2e-14, atol=2e-15)
                np.testing.assert_allclose(actual_distance, expected_distance,
                                           rtol=2e-14, atol=2e-13)

    def test_near_cutoff_rejection_masks_match_old_numpy_reductions(self):
        for bilateral in (False, True):
            with self.subTest(bilateral=bilateral):
                # Frozen CPU/NumPy reference, matching joblib's transposed
                # subject matrix reduction and atlas row reductions.
                atlas_distances = cpu_backend._distance_matrix(
                    self.atlas, self.atlas, 0., distance_method='Mean',
                    bilateral=bilateral, sigmasq=self.sigma_sq,
                    number_of_jobs=1, tile_rows=3)
                old_atlas = (np.sum(np.exp(-atlas_distances / self.sigma_sq), axis=1) - 1.) / len(self.atlas)
                subject_distances = cpu_backend._distance_matrix(
                    self.queries, self.atlas, 0., distance_method='Mean',
                    bilateral=bilateral, sigmasq=self.sigma_sq,
                    number_of_jobs=1, tile_rows=3).T
                old_subject = np.sum(np.exp(-subject_distances / self.sigma_sq), axis=0) / len(self.atlas)

                gpu_subject = distance.similarity_row_sums_from_arrays(
                    self.queries, self.atlas, 0., distance_method='Mean',
                    bilateral=bilateral, sigmasq=self.sigma_sq, tile_rows=3) / len(self.atlas)
                gpu_atlas = (distance.similarity_row_sums_from_arrays(
                    self.atlas, self.atlas, 0., distance_method='Mean',
                    bilateral=bilateral, sigmasq=self.sigma_sq, tile_rows=3) - 1.) / len(self.atlas)

                old_mean, old_std = np.mean(old_atlas), np.std(old_atlas)
                # Put one subject score at the computed boundary, then probe
                # adjacent representable thresholds to exercise strict '<'.
                target = int(np.argmin(old_subject))
                boundary = ((old_mean - old_subject[target]) / old_std
                            if old_std else 0.)
                thresholds = (boundary,
                              np.nextafter(boundary, -np.inf),
                              np.nextafter(boundary, np.inf))
                for std_threshold in thresholds:
                    old_cutoff = old_mean - std_threshold * old_std
                    old_rejected = old_subject < old_cutoff
                    refined_subject, refined_atlas, did_refine = \
                        distance.refine_outlier_probabilities_from_arrays(
                            self.queries, self.atlas, gpu_subject, gpu_atlas,
                            std_threshold, sigmasq=self.sigma_sq,
                            bilateral=bilateral, tile_rows=3)
                    self.assertTrue(did_refine)
                    refined_cutoff = (np.mean(refined_atlas) -
                                      std_threshold * np.std(refined_atlas))
                    np.testing.assert_array_equal(
                        refined_subject < refined_cutoff, old_rejected)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
    def test_fused_similarity_row_sums_matches_eager(self):
        if distance_kernel.triton is None:
            self.skipTest('Triton is required')
        import torch

        device = wma.gpu.get_device()
        rng = np.random.RandomState(20261120)
        for P, threshold, bilateral, sqrt_distance_sums in product(
                (10, 15, 20), (0.0, 2.0), (False, True), (False, True)):
            with self.subTest(P=P, threshold=threshold, bilateral=bilateral,
                              sqrt_distance_sums=sqrt_distance_sums):
                q = torch.as_tensor(rng.normal(size=(13, P, 3)) * 18.,
                                    dtype=torch.float64, device=device)
                f = torch.as_tensor(rng.normal(size=(37, P, 3)) * 18.,
                                    dtype=torch.float64, device=device)
                with patch.object(distance_kernel, 'triton', None):
                    eager_dists = distance._fiber_block_distances(
                        q, f, threshold, 'Mean', self.sigma_sq, bilateral=bilateral)
                expected_sim = torch.exp(-eager_dists / self.sigma_sq).sum(dim=1)
                expected_dist = eager_dists.sqrt().sum(dim=1) if sqrt_distance_sums else None

                actual_sim, actual_dist = distance_kernel.similarity_row_sums(
                    q, f, threshold, self.sigma_sq,
                    bilateral=bilateral, sqrt_distance_sums=sqrt_distance_sums)
                np.testing.assert_allclose(
                    actual_sim.cpu().numpy(), expected_sim.cpu().numpy(),
                    rtol=1e-13, atol=1e-13)
                if sqrt_distance_sums:
                    self.assertIsNotNone(actual_dist)
                    np.testing.assert_allclose(
                        actual_dist.cpu().numpy(), expected_dist.cpu().numpy(),
                        rtol=1e-13, atol=1e-13)
                else:
                    self.assertIsNone(actual_dist)


@unittest.skipIf(distance.torch is None, 'PyTorch is unavailable')
class SmallClusterRoutingTests(unittest.TestCase):
    def test_small_cluster_routes_to_cpu_backend(self):
        rng = np.random.RandomState(20261120)
        queries = rng.normal(size=(4, 15, 3)) * 18.
        fixed = rng.normal(size=(4, 15, 3)) * 18.
        sigmasq = 400.

        self.assertEqual(wma.gpu.config.min_pair_work, 4096)
        for bilateral, sqrt_distance_sums in product((False, True), (False, True)):
            with self.subTest(bilateral=bilateral, sqrt_distance_sums=sqrt_distance_sums):
                expected_d = cpu_backend._distance_matrix(
                    queries, fixed, 0., distance_method='Mean',
                    bilateral=bilateral, sigmasq=sigmasq, number_of_jobs=1)
                expected_sim = np.sum(np.exp(-expected_d / sigmasq), axis=1)
                expected_dist = np.sum(np.sqrt(expected_d), axis=1)

                with patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                        patch.object(distance, 'get_device',
                                     side_effect=AssertionError('CUDA must not be invoked for tiny clusters')), \
                        patch.object(distance, '_fiber_block_distances',
                                     side_effect=AssertionError('CUDA kernel must not launch')):
                    result = distance.similarity_row_sums_from_arrays(
                        queries, fixed, 0., distance_method='Mean',
                        bilateral=bilateral, sigmasq=sigmasq,
                        sqrt_distance_sums=sqrt_distance_sums)

                if sqrt_distance_sums:
                    actual_sim, actual_dist = result
                    np.testing.assert_allclose(actual_sim, expected_sim, rtol=1e-14, atol=1e-15)
                    np.testing.assert_allclose(actual_dist, expected_dist, rtol=1e-14, atol=1e-15)
                else:
                    np.testing.assert_allclose(result, expected_sim, rtol=1e-14, atol=1e-15)

    def test_min_pair_work_env_parsing(self):
        import importlib
        gpu_config = importlib.import_module('whitematteranalysis.gpu.config')
        saved = gpu_config.config.min_pair_work
        self.addCleanup(setattr, gpu_config.config, 'min_pair_work', saved)

        with patch.dict(os.environ, {'WMA_GPU_MIN_PAIR_WORK': '0'}):
            gpu_config._from_env()
            self.assertEqual(gpu_config.config.min_pair_work, 0)

        with patch.dict(os.environ, {'WMA_GPU_MIN_PAIR_WORK': '8192'}):
            gpu_config._from_env()
            self.assertEqual(gpu_config.config.min_pair_work, 8192)

        with patch.dict(os.environ, {'WMA_GPU_MIN_PAIR_WORK': '-5'}):
            gpu_config._from_env()
            self.assertEqual(gpu_config.config.min_pair_work, 8192)

        with patch.dict(os.environ, {'WMA_GPU_MIN_PAIR_WORK': 'invalid'}):
            gpu_config._from_env()
            self.assertEqual(gpu_config.config.min_pair_work, 8192)

@unittest.skipIf(distance.torch is None, 'PyTorch is unavailable')
class EmptyOutlierReductionTests(unittest.TestCase):
    def test_empty_queries_and_references(self):
        empty = np.empty((0, 15, 3), dtype=np.float64)
        populated = np.zeros((2, 15, 3), dtype=np.float64)
        sums, distances = distance.similarity_row_sums_from_arrays(
            empty, populated, 0., sqrt_distance_sums=True)
        self.assertEqual(sums.shape, (0,))
        self.assertEqual(distances.shape, (0,))
        sums, distances = distance.similarity_row_sums_from_arrays(
            populated, empty, 0., sqrt_distance_sums=True)
        np.testing.assert_array_equal(sums, [0., 0.])
        np.testing.assert_array_equal(distances, [0., 0.])


class BoundaryRefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(20261030)
        self.atlas = rng.normal(size=(9, 15, 3)) * 18.
        self.subject = rng.normal(size=(4, 15, 3)) * 18.
        self.sigma_sq = 400.

    def reference(self, distance_method='Mean'):
        atlas_distances = cpu_backend._distance_matrix(
            self.atlas, self.atlas, 0., distance_method=distance_method,
            sigmasq=self.sigma_sq, number_of_jobs=1, tile_rows=2)
        atlas_probs = (np.exp(-atlas_distances / self.sigma_sq).sum(1) - 1.) / len(self.atlas)
        subject_distances = cpu_backend._distance_matrix(
            self.subject, self.atlas, 0., distance_method=distance_method,
            sigmasq=self.sigma_sq, number_of_jobs=1, tile_rows=2).T
        subject_probs = np.exp(-subject_distances / self.sigma_sq).sum(0) / len(self.atlas)
        return subject_probs, atlas_probs

    def test_near_cutoff_refines_selected_metric_to_cpu_reference(self):
        for method in ('Mean', 'MeanSquared'):
            with self.subTest(method=method):
                subject, atlas = self.reference(method)
                target = int(np.argmin(subject))
                std_threshold = ((np.mean(atlas) - subject[target]) / np.std(atlas))
                approximate_subject = subject.copy()
                approximate_atlas = atlas.copy()
                approximate_subject[target] += 2e-15
                approximate_atlas += np.linspace(-1e-15, 1e-15, len(atlas))
                actual_subject, actual_atlas, refined = \
                    distance.refine_outlier_probabilities_from_arrays(
                        self.subject, self.atlas, approximate_subject,
                        approximate_atlas, std_threshold, sigmasq=self.sigma_sq,
                        tile_rows=2, distance_method=method)
                self.assertTrue(refined)
                np.testing.assert_array_equal(actual_subject, subject)
                np.testing.assert_array_equal(actual_atlas, atlas)

    def test_safe_far_path_never_calls_cpu_matrix(self):
        subject = np.array([.1, .2, .3])
        atlas = np.array([.2, .4, .6])
        with patch.object(cpu_backend, '_distance_matrix',
                          side_effect=AssertionError('CPU fallback should stay cold')):
            actual_subject, actual_atlas, refined = \
                distance.refine_outlier_probabilities_from_arrays(
                    self.subject[:3], self.atlas, subject, atlas, 0.)
        self.assertFalse(refined)
        self.assertIs(actual_subject, subject)
        self.assertIs(actual_atlas, atlas)

    def test_refinement_band_catches_1e_13_score_and_cutoff_perturbation(self):
        rng = np.random.RandomState(20261120)
        atlas = rng.normal(size=(9, 15, 3)) * 5.
        atlas_d = cpu_backend._distance_matrix(
            atlas, atlas, 0., distance_method='Mean',
            sigmasq=self.sigma_sq, number_of_jobs=1)
        exact_atlas = (np.exp(-atlas_d / self.sigma_sq).sum(1) - 1.) / len(atlas)
        anchor = atlas[int(np.argmax(exact_atlas))].copy()

        def score_for_shift(shift):
            candidate = anchor.copy()
            candidate[:, 0] += shift
            d = cpu_backend._distance_matrix(
                candidate[None, :, :], atlas, 0., distance_method='Mean',
                sigmasq=self.sigma_sq, number_of_jobs=1).T
            return float(np.exp(-d / self.sigma_sq).sum(0)[0] / len(atlas))

        for std_threshold in (0.0, 2.0):
            with self.subTest(std_threshold=std_threshold):
                exact_cutoff = np.mean(exact_atlas) - std_threshold * np.std(exact_atlas)
                self.assertGreater(exact_cutoff, 0.0)
                # Bisect a subject fiber whose exact score lies just below exact_cutoff.
                lo, hi = 0.0, 500.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if score_for_shift(mid) > exact_cutoff - 1e-14:
                        lo = mid
                    else:
                        hi = mid
                boundary_fiber = anchor.copy()
                boundary_fiber[:, 0] += hi
                subject = np.concatenate([self.subject[:3], boundary_fiber[None, :, :]], axis=0)
                subj_d = cpu_backend._distance_matrix(
                    subject, atlas, 0., distance_method='Mean',
                    sigmasq=self.sigma_sq, number_of_jobs=1).T
                exact_subject = np.exp(-subj_d / self.sigma_sq).sum(0) / len(atlas)
                exact_mask = exact_subject < exact_cutoff
                self.assertTrue(exact_mask[-1])

                # Perturb subject score and atlas scores by 1e-13 in opposite directions,
                # flipping the raw mask across cutoff by ~2e-13.
                approx_subject = exact_subject.copy()
                approx_subject[-1] += 1e-13
                approx_atlas = exact_atlas - 1e-13
                approx_cutoff = np.mean(approx_atlas) - std_threshold * np.std(approx_atlas)
                self.assertFalse(approx_subject[-1] < approx_cutoff)

                actual_subject, actual_atlas, refined = \
                    distance.refine_outlier_probabilities_from_arrays(
                        subject, atlas, approx_subject, approx_atlas,
                        std_threshold, sigmasq=self.sigma_sq)
                self.assertTrue(refined)
                refined_cutoff = np.mean(actual_atlas) - std_threshold * np.std(actual_atlas)
                np.testing.assert_array_equal(actual_subject < refined_cutoff, exact_mask)


if __name__ == '__main__':
    unittest.main()
