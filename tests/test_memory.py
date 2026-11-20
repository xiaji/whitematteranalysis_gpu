"""Geometry allocation policy restores state across failures and worker overlap."""
import os
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch
from joblib.externals import cloudpickle

from whitematteranalysis import _memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop('NUMPY_MADVISE_HUGEPAGE', None)
        self.platform = patch.object(_memory.sys, 'platform', 'linux')
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_nested_scopes_and_failure_restore_original_policy(self):
        for original in (False, True):
            with self.subTest(original=original):
                setter = Mock(return_value=original)
                with patch.object(_memory, '_set_hugepage', setter):
                    with self.assertRaisesRegex(ValueError, 'failed'):
                        with _memory.SmallPages():
                            with _memory.SmallPages():
                                self.assertEqual(setter.call_count, 1)
                                raise ValueError('failed')
                self.assertEqual([c.args for c in setter.call_args_list],
                                 [(False,), (original,)])

    def test_overlapping_workers_restore_only_after_last_exit(self):
        entered, release = Event(), Event()
        setter = Mock(return_value=True)

        def worker():
            with _memory.SmallPages():
                entered.set()
                release.wait(timeout=5)

        with patch.object(_memory, '_set_hugepage', setter):
            with _memory.SmallPages():
                thread = Thread(target=worker)
                thread.start()
                self.assertTrue(entered.wait(timeout=5))
            self.assertEqual([c.args for c in setter.call_args_list], [(False,)])
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual([c.args for c in setter.call_args_list], [(False,), (True,)])

    def test_explicit_hint_and_unsupported_platform_are_unchanged(self):
        with patch.object(_memory, '_set_hugepage') as setter:
            with patch.dict(os.environ, {'NUMPY_MADVISE_HUGEPAGE': '1'}):
                with _memory.SmallPages():
                    pass
            with patch.object(_memory.sys, 'platform', 'win32'):
                with _memory.SmallPages():
                    pass
            setter.assert_not_called()
        with patch.object(_memory, '_set_hugepage', None):
            with _memory.SmallPages():
                pass

    def test_decorated_cli_tasks_can_be_pickled(self):
        @_memory.SmallPages()
        def task(value):
            return value + 1

        restored = cloudpickle.loads(cloudpickle.dumps(task))
        self.assertEqual(restored(4), 5)


class GPUMemoryTests(unittest.TestCase):
    def _require_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        return torch

    def test_is_gpu_oom_detects_cuda_and_cublas_runtime_errors(self):
        torch = self._require_torch()
        from whitematteranalysis.gpu import _memory as gpu_memory

        self.assertTrue(gpu_memory.is_gpu_oom(torch.cuda.OutOfMemoryError('oom')))
        self.assertTrue(gpu_memory.is_gpu_oom(RuntimeError('CUDA out of memory')))
        self.assertTrue(gpu_memory.is_gpu_oom(RuntimeError('CUBLAS_STATUS_ALLOC_FAILED')))
        self.assertTrue(gpu_memory.is_gpu_oom(RuntimeError('CUDNN_STATUS_ALLOC_FAILED')))
        self.assertTrue(gpu_memory.is_gpu_oom(RuntimeError('CUDA error: out of memory')))
        self.assertFalse(gpu_memory.is_gpu_oom(RuntimeError('invalid shape')))
        self.assertFalse(gpu_memory.is_gpu_oom(ValueError('CUDA out of memory')))

    def test_effective_free_vram_respects_max_vram_gib_cap_and_disable_tf32(self):
        import importlib
        torch = self._require_torch()
        from whitematteranalysis.gpu import _memory as gpu_memory

        gpu_config = importlib.import_module('whitematteranalysis.gpu.config')
        prev_cap = gpu_config.config.max_vram_gib
        self.addCleanup(setattr, gpu_config.config, 'max_vram_gib', prev_cap)

        with patch.dict(os.environ, {'WMA_GPU_MAX_VRAM_GIB': '3.5'}):
            gpu_config._from_env()
            self.assertAlmostEqual(gpu_config.config.max_vram_gib, 3.5)
            with patch.object(torch.cuda, 'mem_get_info',
                              return_value=(20 << 30, 24 << 30)):
                free = gpu_memory.effective_free_vram(torch.device('cpu'))
            self.assertLessEqual(free, int(3.5 * (1 << 30)))
            self.assertEqual(free, int(3.5 * (1 << 30)))

        gpu_config.config.max_vram_gib = None
        with patch.object(torch.cuda, 'mem_get_info',
                          return_value=(2 << 30, 24 << 30)):
            self.assertEqual(gpu_memory.effective_free_vram(torch.device('cpu')), 2 << 30)

        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', prev_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', prev_cudnn)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_memory.disable_tf32()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_mps_device_guards_skip_cuda_calls(self):
        import importlib
        torch = self._require_torch()
        from whitematteranalysis.gpu import _memory as gpu_memory

        gpu_config = importlib.import_module('whitematteranalysis.gpu.config')
        prev_cap = gpu_config.config.max_vram_gib
        self.addCleanup(setattr, gpu_config.config, 'max_vram_gib', prev_cap)

        mps_device = torch.device('mps')
        gpu_config.config.max_vram_gib = None
        with patch.object(torch.cuda, 'mem_get_info',
                          side_effect=AssertionError('must not query CUDA on MPS')):
            self.assertEqual(gpu_memory.effective_free_vram(mps_device), 4 << 30)
            gpu_config.config.max_vram_gib = 2.0
            self.assertEqual(gpu_memory.effective_free_vram(mps_device), 2 << 30)

        with patch.object(torch.cuda, 'empty_cache',
                          side_effect=AssertionError('must not call cuda.empty_cache on MPS')):
            gpu_memory.empty_cache(mps_device)
            gpu_memory.empty_cache(torch.device('cpu'))

    def test_tiled_host_apis_fall_back_in_bounded_cpu_batches_on_tile_1_oom(self):
        import numpy as np
        torch = self._require_torch()
        import whitematteranalysis as wma
        from whitematteranalysis.gpu import cpu_backend, distance

        rng = np.random.RandomState(53)
        n, m, p = 7, 11, 15
        Xn = rng.normal(size=(n, p, 3))
        Xm = rng.normal(size=(m, p, 3))
        reflect_mask = np.arange(n) % 2 == 0
        cpu_device = torch.device('cpu')

        def always_oom(*args, **kwargs):
            raise RuntimeError('CUDA out of memory')

        with patch.object(distance, 'get_device', return_value=cpu_device), \
                patch.object(torch.cuda, 'empty_cache'), \
                patch.object(distance, '_fiber_block_distances', side_effect=always_oom):
            # 1. _distance_matrix
            actual_dist = distance._distance_matrix(
                Xn, Xm, 0., distance_method='Mean', tile_rows=4)
            expected_dist = cpu_backend._distance_matrix(
                Xn, Xm, 0., distance_method='Mean', number_of_jobs=1)
            np.testing.assert_allclose(actual_dist, expected_dist, rtol=1e-12, atol=1e-12)

            # 2. similarity_row_sums_from_arrays (with and without sqrt_distance_sums)
            actual_sim, actual_sqrt = distance.similarity_row_sums_from_arrays(
                Xn, Xm, 0., distance_method='Mean', sigmasq=400.,
                tile_rows=4, sqrt_distance_sums=True)
            expected_sim = np.sum(np.exp(-expected_dist / 400.), axis=1)
            expected_sqrt = np.sum(np.sqrt(expected_dist), axis=1)
            np.testing.assert_allclose(actual_sim, expected_sim, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual_sqrt, expected_sqrt, rtol=1e-12, atol=1e-12)

            # 3. nearest_neighbor_distances_from_fiber_array
            fa = wma.fibers.FiberArray()
            fa.number_of_fibers = n
            fa.points_per_fiber = p
            fa.fiber_array_r = Xn[:, :, 0].copy()
            fa.fiber_array_a = Xn[:, :, 1].copy()
            fa.fiber_array_s = Xn[:, :, 2].copy()
            actual_nn = distance.nearest_neighbor_distances_from_fiber_array(
                fa, k=3, distance_method='Mean', tile_rows=4)
            ref_self = cpu_backend._distance_matrix(
                Xn, Xn, 0., distance_method='Mean', number_of_jobs=1)
            expected_nn = wma.filter._mean_nearest_neighbor_distance(ref_self)
            np.testing.assert_allclose(actual_nn, expected_nn, rtol=1e-12, atol=1e-12)

            # 4. total_similarity_matrix_strict (full and row_sum=True with reflect_mask)
            actual_strict = distance.total_similarity_matrix_strict(
                Xn, Xm, reflect_mask, .1, 400., tile_rows=4, row_sum=False)
            expected_strict = cpu_backend.total_similarity_matrix_strict(
                Xn, Xm, reflect_mask, .1, 400., row_sum=False)
            np.testing.assert_allclose(actual_strict, expected_strict, rtol=1e-12, atol=1e-12)

            actual_strict_sum = distance.total_similarity_matrix_strict(
                Xn, Xm, reflect_mask, .1, 400., tile_rows=4, row_sum=True)
            np.testing.assert_array_equal(actual_strict_sum, expected_strict.sum(1))

        # Verify bounded query batches for reduction-only fallbacks (never full remaining matrix)
        big_n, big_m = 8, 3000
        big_q = rng.normal(size=(big_n, p, 3))
        big_f = rng.normal(size=(big_m, p, 3))
        big_mask = np.arange(big_n) % 2 == 0
        orig_cpu_dm = cpu_backend._distance_matrix

        for label, call in (
            ('similarity_row_sums', lambda: distance.similarity_row_sums_from_arrays(
                big_q, big_f, 0., distance_method='Mean', sigmasq=400., tile_rows=2)),
            ('total_similarity_strict_row_sum', lambda: distance.total_similarity_matrix_strict(
                big_q, big_f, big_mask, .1, 400., tile_rows=2, row_sum=True)),
        ):
            with self.subTest(bounded_fallback=label):
                seen_batch_rows = []
                def recording_cpu_dm(q, f, *args, **kwargs):
                    seen_batch_rows.append(len(q))
                    return orig_cpu_dm(q, f, *args, **kwargs)
                with patch.object(distance, 'get_device', return_value=cpu_device), \
                        patch.object(torch.cuda, 'empty_cache'), \
                        patch.object(distance, '_fiber_block_distances', side_effect=always_oom), \
                        patch.object(cpu_backend, '_distance_matrix', side_effect=recording_cpu_dm):
                    call()
                self.assertGreater(len(seen_batch_rows), 1)
                self.assertTrue(all(b < big_n for b in seen_batch_rows), seen_batch_rows)
                self.assertEqual(sum(seen_batch_rows), big_n)

    def test_rectangular_similarity_tensor_mps_empty_cache_and_strict_float32(self):
        import numpy as np
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        torch = self._require_torch()
        from whitematteranalysis.gpu import _memory as gpu_memory, distance

        def _make_pd(arr):
            pd = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(arr.reshape(-1, 3), deep=True))
            lines = vtk.vtkCellArray()
            for i in range(arr.shape[0]):
                lines.InsertNextCell(arr.shape[1])
                for j in range(arr.shape[1]):
                    lines.InsertCellPoint(i * arr.shape[1] + j)
            pd.SetPoints(pts)
            pd.SetLines(lines)
            return pd

        rng = np.random.RandomState(61)
        Xn = rng.normal(size=(5, 7, 3))
        Xm = rng.normal(size=(4, 7, 3))
        pd_n, pd_m = _make_pd(Xn), _make_pd(Xm)
        mps_device = torch.device('mps')
        has_real_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        target_dev = mps_device if has_real_mps else torch.device('cpu')

        orig_block = distance._fiber_block_distances
        seen_sizes = []

        def oom_above_two(q, *args, **kwargs):
            seen_sizes.append(len(q))
            if len(q) > 2:
                raise RuntimeError('MPS out of memory')
            return orig_block(q, *args, **kwargs)

        with patch.object(distance, 'get_device', return_value=mps_device if has_real_mps else target_dev), \
                patch.object(torch.cuda, 'empty_cache',
                             side_effect=AssertionError('must not call cuda.empty_cache on MPS')), \
                patch.object(gpu_memory, 'empty_cache', wraps=gpu_memory.empty_cache) as empty_spy, \
                patch.object(distance, '_fiber_block_distances', side_effect=oom_above_two):
            if not has_real_mps:
                # Simulate dev.type == 'mps' for empty_cache even when MPS backend is absent
                orig_empty = torch.empty
                orig_as_tensor_local = torch.as_tensor
                with patch.object(distance, 'get_device', return_value=mps_device), \
                        patch.object(torch, 'empty', side_effect=lambda *a, **kw: orig_empty(*a, **{**kw, 'device': torch.device('cpu')})), \
                        patch.object(torch, 'as_tensor', side_effect=lambda d, *a, **kw: orig_as_tensor_local(d, *a, **{**kw, 'device': torch.device('cpu')})):
                    tensor_out = distance.rectangular_similarity_tensor(
                        pd_n, pd_m, 0., 20., gpu_float32=True, tile_rows=5)
            else:
                tensor_out = distance.rectangular_similarity_tensor(
                    pd_n, pd_m, 0., 20., gpu_float32=True, tile_rows=5)
        self.assertEqual(seen_sizes, [5, 2, 2, 1])
        self.assertEqual(tensor_out.shape, (4, 5))
        empty_spy.assert_called_once_with(mps_device)

        # Verify total_similarity_matrix_strict(..., gpu_float32=True) never creates float64 tensors
        orig_as_tensor = torch.as_tensor

        def no_float64_as_tensor(data, *args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                raise AssertionError('total_similarity_matrix_strict(gpu_float32=True) requested float64')
            return orig_as_tensor(data, *args, **kwargs)

        with patch.object(distance, 'get_device', return_value=target_dev), \
                patch.object(torch, 'as_tensor', side_effect=no_float64_as_tensor):
            strict_out = distance.total_similarity_matrix_strict(
                Xn, Xm, np.array([True, False, True, False, True]),
                0., 400., gpu_float32=True, tile_rows=2, row_sum=True)
        self.assertEqual(strict_out.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
