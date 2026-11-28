"""Atlas caching preserves probabilities and invalidates changed references."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import atlas_cache
from whitematteranalysis.gpu.config import float32_enabled
from test_cli_file_stages import tractography


class AtlasCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'atlas.vtp'
        self.cache = self.root / 'cache'
        environment = patch.dict(os.environ, WMA_ATLAS_CACHE=str(self.cache))
        environment.start()
        self.addCleanup(environment.stop)
        self.data = tractography(80)
        wma.io.write_polydata(self.data, str(self.source))

    def check_reference(self, backend, method='Mean', bilateral=True, sigmasq=400.):
        load = lambda: atlas_cache.load_reference(self.source, sigmasq, method, bilateral, backend)
        arrays, probabilities = load()
        distances = wma.cluster._pairwise_distance_matrix(self.data, 0., number_of_jobs=1,
            distance_method=method, bilateral=bilateral, sigmasq=sigmasq, use_gpu=backend == 'gpu')
        expected = (np.exp(-distances / sigmasq).sum(axis=1) - 1.) / len(arrays)
        if backend == 'gpu':
            tolerance = 2e-6 if float32_enabled() else 2e-14
            np.testing.assert_allclose(probabilities, expected, rtol=tolerance,
                                       atol=tolerance * 0.1)
        else:
            np.testing.assert_array_equal(probabilities, expected)
        with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')):
            cached_arrays, cached_probabilities = load()
        np.testing.assert_array_equal(cached_arrays, arrays)
        np.testing.assert_array_equal(cached_probabilities, probabilities)

    def test_cpu_probabilities_are_exact(self):
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                self.check_reference('cpu', method, bilateral)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_gpu_probabilities_and_precision_are_exact(self):
        for float32 in (False, True):
            with patch.object(wma.gpu.config, 'float32', float32):
                for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
                    self.check_reference('gpu', method)
        self.assertEqual(len(list(self.cache.glob('*.npz'))), 8)

    def test_parameters_reference_and_implementation_invalidate(self):
        load = lambda **kwargs: atlas_cache.load_reference(self.source, backend='cpu', **kwargs)
        load(sigmasq=400.)
        read = wma.io.read_polydata
        with patch.object(wma.io, 'read_polydata', wraps=read) as reader:
            load(sigmasq=401.)
            load(sigmasq=400., bilateral=False)
            load(sigmasq=400., method='MeanSquared')
            stat = self.source.stat()
            os.utime(str(self.source), ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            load(sigmasq=400.)
            with patch.object(atlas_cache, '_implementation', return_value=['changed implementation']):
                load(sigmasq=400.)
            self.assertEqual(reader.call_count, 5)
        self.assertEqual(len(list(self.cache.glob('*.npz'))), 6)

    def test_corrupt_and_unavailable_cache_recompute(self):
        load = lambda: atlas_cache.load_reference(self.source, 400., backend='cpu')
        expected = load()
        path, = self.cache.glob('*.npz')
        for damaged in (b'corrupt file', b'PK\x03\x04'):
            path.write_bytes(damaged)
            for actual, reference in zip(load(), expected):
                np.testing.assert_array_equal(actual, reference)
        np.savez(str(path), coordinates=np.zeros((3, 4)), probabilities=np.zeros(3))
        np.testing.assert_array_equal(load()[1], expected[1])
        with path.open('wb') as output:
            np.save(output, np.zeros(3))
        np.testing.assert_array_equal(load()[1], expected[1])
        blocked = self.root / 'not a directory'
        blocked.write_text('occupied')
        with patch.dict(os.environ, WMA_ATLAS_CACHE=str(blocked / 'cache')):
            np.testing.assert_array_equal(load()[1], expected[1])
        with patch.dict(os.environ, WMA_ATLAS_CACHE='off'), patch.object(
                wma.io, 'read_polydata', wraps=wma.io.read_polydata) as reader:
            load()
            load()
            self.assertEqual(reader.call_count, 2)

    def test_empty_and_concurrent_writers(self):
        wma.io.write_polydata(tractography(0), str(self.source))
        load = lambda _: atlas_cache.load_reference(self.source, 400., backend='cpu')
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(load, range(8)))
        for arrays, probabilities in results:
            self.assertEqual(arrays.shape, (0, 15, 3))
            self.assertEqual(probabilities.shape, (0,))
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')):
            load(0)

    def test_mps_cache_avoids_cuda_device_queries(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        mps_device = torch.device('mps')
        mps_real = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        atlas_cache._implementation.cache_clear()
        with patch.object(wma.gpu.config, 'float32', True), \
                patch.object(atlas_cache, 'get_device', return_value=mps_device), \
                patch.object(torch.cuda, 'get_device_name',
                             side_effect=AssertionError('must not query CUDA device name on MPS')), \
                patch.object(torch.cuda, 'get_device_capability',
                             side_effect=AssertionError('must not query CUDA capability on MPS')), \
                patch.object(torch.cuda, 'mem_get_info',
                             side_effect=AssertionError('must not query CUDA mem_get_info on MPS')), \
                patch.object(torch.cuda, 'empty_cache',
                             side_effect=AssertionError('must not call cuda.empty_cache on MPS')):
            if mps_real:
                arrays, probabilities = atlas_cache.load_reference(
                    self.source, 400., 'Mean', True, backend='gpu')
            else:
                fake_sums = np.ones(80, dtype=np.float32) * 10.
                with patch.object(wma.gpu.distance, 'similarity_row_sums_from_arrays',
                                  return_value=fake_sums):
                    arrays, probabilities = atlas_cache.load_reference(
                        self.source, 400., 'Mean', True, backend='gpu')
            self.assertEqual(probabilities.dtype, np.float32)
            self.assertEqual(probabilities.shape, (80,))
            self.assertEqual(len(list(self.cache.glob('*.npz'))), 1)
            with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')):
                cached_arrays, cached_probs = atlas_cache.load_reference(
                    self.source, 400., 'Mean', True, backend='gpu')
            np.testing.assert_array_equal(cached_arrays, arrays)
            np.testing.assert_array_equal(cached_probs, probabilities)


class SpectralAtlasCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.atlas_dir = self.root / 'atlas'
        self.atlas_dir.mkdir()
        self.cache = self.root / 'cache'
        environment = patch.dict(os.environ, WMA_ATLAS_CACHE=str(self.cache))
        environment.start()
        self.addCleanup(environment.stop)

        # Build a synthetic ClusterAtlas fixture (12 Nystrom fibers, 3 eigenvectors, 3 clusters)
        atlas_pd = tractography(12)
        rng = np.random.RandomState(31)
        atlas = wma.cluster.ClusterAtlas()
        atlas.nystrom_polydata = atlas_pd
        atlas.pinv_A = np.eye(12, dtype=np.float64)
        atlas.row_sum_1 = rng.uniform(5.0, 15.0, size=12)
        atlas.row_sum_matrix = rng.uniform(0.1, 1.0, size=12)
        atlas.e_vec = rng.normal(size=(12, 12))
        atlas.e_val = np.linspace(0.1, 1.0, 12)
        atlas.e_vec_norm = np.sqrt(np.sum(atlas.e_vec ** 2, axis=0) + 1.0)
        atlas.centroids = rng.normal(size=(3, 3))
        atlas.number_of_eigenvectors = 3
        atlas.sigma = 45.0
        atlas.threshold = 0.0
        atlas.bilateral = True
        atlas.distance_method = 'Mean'
        atlas.save(str(self.atlas_dir), 'atlas')
        self.atlas_p = self.atlas_dir / 'atlas.p'
        self.atlas_vtp = self.atlas_dir / 'atlas.vtp'
        self.subject = tractography(24)

    def test_spectral_atlas_cache_roundtrip_and_labeling_parity(self):
        uncached = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=False)
        first = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)

        cache_files = list((self.cache / 'spectral-atlas').glob('*.npz'))
        self.assertEqual(len(cache_files), 1)
        self.assertIsNone(first.nystrom_polydata)
        self.assertEqual(first.e_vec.shape, (12, 4))

        with patch.object(wma.io, 'read_polydata', side_effect=AssertionError('cache miss')), \
                patch.object(wma.cluster.ClusterAtlas, 'load', side_effect=AssertionError('pickle miss')):
            second = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)

        np.testing.assert_array_equal(second.nystrom_coords, first.nystrom_coords)
        np.testing.assert_array_equal(second.e_vec, first.e_vec)
        np.testing.assert_array_equal(second.e_val, first.e_val)
        np.testing.assert_array_equal(second.e_vec_norm, first.e_vec_norm)
        np.testing.assert_array_equal(second.centroids, first.centroids)

        # Verify spectral_atlas_label produces identical results across uncached, cached, and shared queries_15
        _, lbl_ref, _, emb_ref = wma.cluster.spectral_atlas_label(
            self.subject, uncached, number_of_jobs=1, use_gpu=False)
        q15 = wma.gpu.fiber_tensors.polydata_to_numpy(self.subject, 15)
        _, lbl_fast, _, emb_fast = wma.cluster.spectral_atlas_label(
            self.subject, second, number_of_jobs=1, use_gpu=False, queries_15=q15)
        np.testing.assert_array_equal(lbl_fast, lbl_ref)
        np.testing.assert_allclose(emb_fast, emb_ref, rtol=1e-13, atol=1e-13)

        # Legacy fallback path still works via lazy _ensure_nystrom_polydata
        with patch.object(wma.gpu.config, 'cpu_fast', False):
            _, lbl_leg, _, emb_leg = wma.cluster.spectral_atlas_label(
                self.subject, second, number_of_jobs=1, use_gpu=False)
        np.testing.assert_array_equal(lbl_leg, lbl_ref)
        np.testing.assert_allclose(emb_leg, emb_ref, rtol=1e-13, atol=1e-13)

    def test_spectral_atlas_cache_invalidation_recovery_and_disable(self):
        first = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
        path, = (self.cache / 'spectral-atlas').glob('*.npz')

        # Corrupt cache triggers clean recompute
        path.write_bytes(b'corrupt npz data')
        recovered = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
        np.testing.assert_array_equal(recovered.nystrom_coords, first.nystrom_coords)

        # mtime change invalidates cache key and rebuilds on read
        stat = self.atlas_vtp.stat()
        os.utime(str(self.atlas_vtp), ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with patch.object(wma.io, 'read_polydata', wraps=wma.io.read_polydata) as reader:
            wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
            self.assertGreaterEqual(reader.call_count, 1)
        self.assertEqual(len(list((self.cache / 'spectral-atlas').glob('*.npz'))), 2)

        # WMA_ATLAS_CACHE=off bypasses cache completely
        with patch.dict(os.environ, WMA_ATLAS_CACHE='off'), \
                patch.object(wma.cluster.ClusterAtlas, 'load', autospec=True,
                             side_effect=wma.cluster.ClusterAtlas.load) as loader:
            wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
            wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
            self.assertEqual(loader.call_count, 2)

        # Non-Boolean bilateral still raises ValueError
        import pickle
        with open(str(self.atlas_p), 'rb') as fh:
            obj = pickle.load(fh)
        obj.bilateral = None
        with open(str(self.atlas_p), 'wb') as fh:
            pickle.dump(obj, fh)
        with self.assertRaises(ValueError):
            wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)

    @unittest.skipUnless(wma.gpu.available(), 'CUDA is unavailable')
    def test_gpu_resident_prepared_atlas_matches_standard_gpu(self):
        cached = wma.cluster.load_atlas(str(self.atlas_dir), 'atlas', for_labeling=True)
        dev = wma.gpu.get_device()
        prepared = {dev: wma.gpu.linalg.prepare_atlas_gpu(cached, device=dev)}
        q15 = wma.gpu.fiber_tensors.polydata_to_numpy(self.subject, 15)

        _, lbl_std, _, emb_std = wma.cluster.spectral_atlas_label(
            self.subject, cached, use_gpu=True)
        _, lbl_res, _, emb_res = wma.cluster.spectral_atlas_label(
            self.subject, cached, use_gpu=True, queries_15=q15, prepared_gpu=prepared)
        np.testing.assert_array_equal(lbl_res, lbl_std)
        np.testing.assert_allclose(emb_res, emb_std, rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
