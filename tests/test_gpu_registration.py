"""Compare the fused registration path with the original VTK/NumPy code.

Run: CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests
"""
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import registration as gpu
from whitematteranalysis.gpu import registration_kernel
from whitematteranalysis.register_two_subjects import (
    RegisterTractography, inner_loop_objective, transform_fiber_array_numpy)


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class RegistrationTests(unittest.TestCase):
    def test_workspace_retries_reuse_and_streams(self):
        import torch
        rng = np.random.RandomState(45)
        fixed = rng.normal(size=(3, 31, 10)) * 40.
        moving = rng.normal(size=(3, 5, 10)) * 40.
        x = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
        expected = gpu.objective_value(fixed, moving, x, [1]*4, 20.)
        real_probability = gpu._pairwise_probability
        sizes = []

        def constrained(fixed, moving, sigmasq):
            sizes.append(len(moving))
            if len(moving) > 2:
                raise torch.cuda.OutOfMemoryError('test workspace tile exceeds memory')
            return real_probability(fixed, moving, sigmasq)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            objective = gpu.AffineObjective(fixed, moving)
            objective.workspace.chunk_rows = 5
            host = objective.workspace.probability
            self.assertTrue(host.is_pinned())
            with patch.object(gpu, '_pairwise_probability', side_effect=constrained), patch.object(
                    torch.cuda, 'mem_get_info', side_effect=AssertionError('repeated memory query')) as memory:
                self.assertEqual(objective(x, [1]*4, 20.), expected)
                self.assertEqual(sizes, [5, 2, 2, 1])
                sizes.clear()
                self.assertEqual(objective(x, [1]*4, 20.), expected)
                self.assertEqual(sizes, [2, 2, 1])
                memory.assert_not_called()
            self.assertIs(objective.workspace.probability, host)

    def test_empty_moving_objective(self):
        objective = gpu.AffineObjective(np.zeros((3, 5, 10)), np.zeros((3, 0, 10)))
        x = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(objective(x, [1]*4, 20.), 0.)

    def test_resident_affine_transform_matches_vtk(self):
        import torch
        if registration_kernel.triton is None:
            self.skipTest('Triton is required')
        rng = np.random.RandomState(44)
        for points in (1, 3, 10, 15, 50):
            fixed = rng.normal(size=(3, 31, points)) * 40.
            moving = (rng.normal(size=(3, 34, points * 2)) * 40.)[:, ::2, ::2]
            before = moving.copy()
            objective = gpu.AffineObjective(fixed, moving)
            x = np.array([5., -3., 2., 4., -2., 1., 1.02, .98, 1.01,
                          1., -1., .5, -.5, 2., -2.])
            for mode in ([1]*4, [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]):
                actual = objective(x, mode, 20.)
                expected = transform_fiber_array_numpy(moving, x, mode)
                np.testing.assert_array_equal(objective.transformed.cpu().numpy(), expected.transpose(1, 2, 0))
                self.assertEqual(actual, gpu.objective_value(fixed, moving, x, mode, 20.))
                x[0] += .1  # Exercise matrix/buffer reuse between evaluations.
            np.testing.assert_array_equal(moving, before)

    def test_eager_objective_matches_committed_baseline(self):
        import torch
        from test_output_pipeline import baseline_module
        original = baseline_module('whitematteranalysis.gpu._eagerbaseline',
                                   'gpu/registration.py', '60c7b2d')
        rng = np.random.RandomState(43)
        for points in (1, 10, 15, 32, 50):
            fixed = gpu._prepare_fixed(rng.normal(size=(3, 257, points)) * 40.,
                                       'cuda', torch.float64)
            moving = gpu._prepare_fixed(rng.normal(size=(3, 17, points)) * 40.,
                                        'cuda', torch.float64)
            before_fixed, before_moving = fixed.clone(), moving.clone()
            with patch.object(registration_kernel, 'triton', None):
                for sigma in (100., 400., 1e6):
                    expected = original._pairwise_probability(fixed, moving, sigma)
                    self.assertTrue(torch.equal(gpu._pairwise_probability(fixed, moving, sigma), expected))
            self.assertTrue(torch.equal(fixed, before_fixed))
            self.assertTrue(torch.equal(moving, before_moving))

    def test_fused_objective(self):
        rng = np.random.RandomState(42)
        fixed = rng.normal(size=(3, 31, 10)) * 40
        moving = rng.normal(size=(3, 17, 10)) * 40
        transform = np.array([5., -3., 2., 4., -2., 1., 1.02, .98, 1.01,
                              1., -1., .5, -.5, 2., -2.])
        for mode in ([1]*4, [1, 0, 0, 0], [0, 1, 0, 0]):
            with self.subTest(mode=mode):
                original = transform_fiber_array_numpy(moving, transform, mode)
                tensor = gpu._transform_fiber_tensor(moving, transform, mode, None)
                np.testing.assert_array_equal(original, tensor.cpu().numpy())
                expected = inner_loop_objective(fixed, original, 400.)
                for rows in (1, 7, 17):
                    value = gpu.inner_loop_objective(fixed, tensor, 400., chunk_rows=rows)
                    np.testing.assert_allclose(value, expected, rtol=1e-12)
                # The fused objective must not call the NumPy-returning wrapper.
                with patch.object(gpu, 'transform_fiber_array', side_effect=AssertionError):
                    value = gpu.objective_value(fixed, moving, transform, mode, 20.)
                split = gpu.inner_loop_objective(fixed, original, 400.)
                self.assertEqual(value, split)
                reg = RegisterTractography()
                reg.fixed, reg.moving = fixed, moving
                reg.use_gpu, reg.mode, reg.sigma = True, mode, 20.
                self.assertEqual(reg.objective_function(transform), value)
                self.assertEqual(reg.objective_function_values, [value])

    def test_point_kernel_and_optional_fallback(self):
        import torch

        if registration_kernel.triton is None:
            self.skipTest('Triton is not installed')
        prev_strict = wma.gpu.strict_enabled()
        wma.gpu.set_strict(True)
        self.addCleanup(wma.gpu.set_strict, prev_strict)
        rng = np.random.RandomState(2026)
        for points in (1, 2, 3, 7, 8, 10, 15, 16, 20, 31, 32, 50, 64):
            with self.subTest(points=points):
                fixed = gpu._prepare_fixed(
                    rng.normal(size=(3, 257, points)) * 40., 'cuda', torch.float64)
                moving = gpu._prepare_fixed(
                    rng.normal(size=(3, 5, points)) * 40., 'cuda', torch.float64)
                forward, reverse = registration_kernel.squared_distances(fixed, moving)
                self.assertTrue(torch.equal(forward, gpu._squared_point_distances(fixed, moving)))
                self.assertTrue(torch.equal(reverse, gpu._squared_point_distances(fixed, moving.flip(1))))
                mean, reverse_mean = registration_kernel.mean_squared_distances(fixed, moving)
                self.assertTrue(torch.equal(mean, forward.sum(2) / points))
                self.assertTrue(torch.equal(reverse_mean, reverse.sum(2) / points))
                for sigmasq in (1e-5, 100., 1e6):
                    actual = gpu._pairwise_probability(fixed, moving, sigmasq)
                    with patch.object(registration_kernel, 'triton', None):
                        expected = gpu._pairwise_probability(fixed, moving, sigmasq)
                    self.assertTrue(torch.equal(actual, expected))
                # Uneven chunks must preserve each moving fiber's probability.
                actual = gpu._inner_loop_objective(fixed, moving, 100., 'cuda', 2)
                with patch.object(registration_kernel, 'triton', None):
                    expected = gpu._inner_loop_objective(fixed, moving, 100., 'cuda', 2)
                self.assertEqual(actual, expected)

    def test_kernel_uses_selected_cuda_device(self):
        import torch

        if registration_kernel.triton is None or torch.cuda.device_count() < 2:
            self.skipTest('Triton and two visible devices are required')
        prev_strict = wma.gpu.strict_enabled()
        wma.gpu.set_strict(True)
        self.addCleanup(wma.gpu.set_strict, prev_strict)
        previous = torch.cuda.current_device()
        selected = (previous + 1) % torch.cuda.device_count()
        fixed = torch.ones((7, 10, 3), dtype=torch.float64, device=selected)
        moving = torch.zeros((3, 10, 3), dtype=torch.float64, device=selected)
        actual = gpu._pairwise_probability(fixed, moving, 100.)
        with patch.object(registration_kernel, 'triton', None):
            expected = gpu._pairwise_probability(fixed, moving, 100.)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.cuda.current_device(), previous)
        objective = gpu.AffineObjective(fixed.cpu().numpy().transpose(2, 0, 1),
            moving.cpu().numpy().transpose(2, 0, 1), device=torch.device('cuda', selected))
        x = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
        objective(x, [1]*4, 10.)
        if registration_kernel.triton is not None:
            self.assertEqual(objective.transformed.device.index, selected)
        self.assertEqual(torch.cuda.current_device(), previous)

    def test_out_of_memory_retries_and_terminates(self):
        import torch

        fixed = torch.zeros((3, 10, 3), dtype=torch.float64, device='cuda')
        moving = torch.zeros((5, 10, 3), dtype=torch.float64, device='cuda')
        original = gpu._pairwise_probability
        sizes = []

        def constrained(fixed, moving, sigmasq):
            sizes.append(len(moving))
            if len(moving) > 2:
                raise torch.cuda.OutOfMemoryError('test tile exceeds memory')
            return original(fixed, moving, sigmasq)

        expected = gpu._inner_loop_objective(fixed, moving, 100., 'cuda', 2)
        for rows in (5, 8192):
            sizes.clear()
            with patch.object(gpu, '_pairwise_probability', side_effect=constrained):
                actual = gpu._inner_loop_objective(fixed, moving, 100., 'cuda', rows)
            self.assertEqual(actual, expected)
            self.assertEqual(sizes, [5, 2, 2, 1])
        for rows in (0, -1):
            with self.assertRaisesRegex(ValueError, 'chunk_rows must be positive'):
                gpu._inner_loop_objective(fixed, moving, 100., 'cuda', rows)
        with patch.object(gpu, '_pairwise_probability', side_effect=torch.cuda.OutOfMemoryError):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                gpu._inner_loop_objective(fixed, moving, 100., 'cuda', 1)

    def test_optimizer_reuses_data_and_refreshes_samples(self):
        prev_strict = wma.gpu.strict_enabled()
        wma.gpu.set_strict(True)
        self.addCleanup(wma.gpu.set_strict, prev_strict)
        rng = np.random.RandomState(123)
        reg = RegisterTractography()
        reg.fixed = rng.normal(size=(3, 31, 10)) * 20
        reg.moving = rng.normal(size=(3, 17, 10)) * 20
        reg.use_gpu = True
        reg.maxfun = 25
        reg.mode = [1, 1, 0, 0]
        for optimizer in ('Cobyla', 'BFGS', 'Powell'):
            reg.optimizer = optimizer
            for scale in range(2):
                with self.subTest(optimizer=optimizer, scale=scale):
                    # Same objects, new samples: an identity-only cache is unsafe.
                    reg.fixed += 2.
                    reg.moving *= .95
                    reg.sigma = 10. + scale * 5.
                    fixed_before, moving_before = reg.fixed.copy(), reg.moving.copy()
                    reg.objective_function_values.clear()
                    with patch.object(gpu, 'AffineObjective', wraps=gpu.AffineObjective) as factory:
                        actual = reg.compute().copy()
                    factory.assert_called_once()
                    self.assertIsNone(reg._gpu_objective)
                    actual_values = reg.objective_function_values.copy()
                    # Run the same optimizer with the preceding stateless GPU path.
                    def stateless(fixed, moving, device=None, gpu_float32=None):
                        return lambda x, mode, sigma: gpu.objective_value(
                            fixed, moving, x, mode, sigma,
                            device=device, gpu_float32=gpu_float32)
                    reg.objective_function_values.clear()
                    with patch.object(gpu, 'AffineObjective', side_effect=stateless), \
                            patch.object(registration_kernel, 'triton', None):
                        expected = reg.compute()
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_values, reg.objective_function_values)
                    np.testing.assert_array_equal(reg.fixed, fixed_before)
                    np.testing.assert_array_equal(reg.moving, moving_before)

    def test_optimizer_failure_releases_device_data(self):
        reg = RegisterTractography()
        reg.fixed = np.zeros((3, 3, 10))
        reg.moving = np.ones((3, 2, 10))
        reg.use_gpu = True

        def fail(objective, *args, **kwargs):
            objective(reg.initial_transform * reg.transform_scaling)
            raise RuntimeError('optimizer failed')

        with patch('scipy.optimize.fmin_cobyla', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'optimizer failed'):
                reg.compute()
        self.assertIsNone(reg._gpu_objective)
        # A direct call outside compute must read current arrays.
        reg.fixed += 4.
        x = reg.initial_transform * reg.transform_scaling
        self.assertEqual(reg.objective_function(x), gpu.objective_value(
            reg.fixed, reg.moving, x, reg.mode, reg.sigma))

    def test_fused_pairwise_probability_sums_float32_and_p50(self):
        import torch
        if registration_kernel.triton is None:
            self.skipTest('Triton is not installed')
        prev_strict = wma.gpu.strict_enabled()
        wma.gpu.set_strict(False)
        self.addCleanup(wma.gpu.set_strict, prev_strict)

        rng = np.random.RandomState(2027)
        for points in (1, 15, 32, 50, 64):
            for dtype in (torch.float64, torch.float32):
                with self.subTest(points=points, dtype=dtype):
                    fixed = gpu._prepare_fixed(
                        rng.normal(size=(3, 700, points)) * 40., 'cuda', dtype)
                    moving = gpu._prepare_fixed(
                        rng.normal(size=(3, 37, points)) * 40., 'cuda', dtype)
                    for sigmasq in (100., 400.):
                        actual = gpu._pairwise_probability(fixed, moving, sigmasq)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertEqual(actual.shape, (37,))
                        with patch.object(registration_kernel, 'triton', None):
                            expected = gpu._pairwise_probability(fixed, moving, sigmasq)
                        tol = 1e-5 if dtype == torch.float32 else 1e-13
                        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

        # Verify P=50 & gpu_float32=True end-to-end via AffineObjective and inner_loop_objective
        f50 = rng.normal(size=(3, 257, 50)) * 40.
        m50 = rng.normal(size=(3, 19, 50)) * 40.
        x = np.array([2., -1., 1., 1., -1., 0.5, 1.01, .99, 1.0,
                      0., 0., 0., 0., 0., 0.])
        val64 = gpu.objective_value(f50, m50, x, [1] * 4, 20., gpu_float32=False)
        with patch.object(registration_kernel, 'triton', None):
            ref64 = gpu.objective_value(f50, m50, x, [1] * 4, 20., gpu_float32=False)
        np.testing.assert_allclose(val64, ref64, rtol=1e-13, atol=1e-13)

        obj32 = gpu.AffineObjective(f50, m50, gpu_float32=True)
        val32 = obj32(x, [1] * 4, 20.)
        self.assertEqual(obj32.fixed.dtype, torch.float32)
        self.assertAlmostEqual(val32, ref64, delta=max(1e-3, abs(ref64) * 1e-4))


class CPURegistrationTests(unittest.TestCase):
    def test_optimizer_reuses_cpu_workers_and_closes_on_failure(self):
        reg = RegisterTractography()
        rng = np.random.RandomState(322)
        reg.fixed, reg.moving = rng.normal(size=(3, 512, 10)), rng.normal(size=(3, 128, 10))
        reg.use_gpu, reg.maxfun = False, 3
        cpu = wma.gpu.cpu_backend
        original = cpu.inner_loop_objective
        executors = []

        def record(*args, **kwargs):
            executors.append(kwargs['executor'])
            return original(*args, **kwargs)

        with patch.object(cpu, 'cpu_threads', return_value=4), \
                patch.object(cpu, 'inner_loop_objective', side_effect=record):
            reg.compute()
        self.assertGreater(len(executors), 1)
        self.assertTrue(all(pool is executors[0] for pool in executors))
        self.assertTrue(executors[0]._shutdown)
        self.assertIsNone(reg._cpu_executor)
        with patch.object(cpu, 'cpu_threads', return_value=4), \
                patch.object(cpu, 'inner_loop_objective', side_effect=RuntimeError('worker failed')):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                reg.compute()
        self.assertIsNone(reg._cpu_executor)

    def test_cpu_optimizer_never_prepares_cuda_data(self):
        rng = np.random.RandomState(321)
        reg = RegisterTractography()
        reg.fixed = rng.normal(size=(3, 7, 10)) * 20
        reg.moving = rng.normal(size=(3, 5, 10)) * 20
        reg.use_gpu, reg.maxfun = False, 20
        with patch.object(gpu, 'AffineObjective', side_effect=AssertionError):
            actual = reg.compute().copy()
        values = reg.objective_function_values.copy()
        reg.objective_function_values.clear()
        with patch.object(reg, '_using_cpu_fast', return_value=False):
            expected = reg.compute()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, reg.objective_function_values)
        self.assertIsNone(reg._gpu_objective)

    def test_gpu_float32_lazy_allocation_and_device_forwarding_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        import tempfile
        from whitematteranalysis.gpu import bspline
        from whitematteranalysis.register_two_subjects_nonrigid_bsplines import (
            RegisterTractographyNonrigid)

        rng = np.random.RandomState(404)
        fixed = rng.normal(size=(3, 19, 50)) * 20.
        moving = rng.normal(size=(3, 7, 50)) * 20.
        x = np.array([1., -2., 0.5, 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
        cpu_dev = torch.device('cpu')

        val64 = gpu.objective_value(fixed, moving, x, [1] * 4, 20., device=cpu_dev, gpu_float32=False)
        obj32 = gpu.AffineObjective(fixed, moving, device=cpu_dev, gpu_float32=True)
        self.assertEqual(obj32.fixed.dtype, torch.float32)
        val32 = obj32(x, [1] * 4, 20.)
        self.assertAlmostEqual(val32, val64, delta=max(1e-3, abs(val64) * 1e-4))
        self.assertAlmostEqual(
            gpu.inner_loop_objective(fixed, moving, 400., device=cpu_dev, gpu_float32=True),
            gpu.inner_loop_objective(fixed, moving, 400., device=cpu_dev, gpu_float32=False),
            delta=max(1e-3, abs(val64) * 1e-4))

        # Single-chunk _inner_loop_objective must not pre-allocate probability
        expected_untransformed = gpu.inner_loop_objective(
            fixed, moving, 400., device=cpu_dev, gpu_float32=False)
        F_t = gpu._prepare_fixed(fixed, cpu_dev, torch.float64)
        M_t = gpu._prepare_fixed(moving, cpu_dev, torch.float64)
        with patch.object(torch, 'empty', side_effect=AssertionError('extra probability allocation')) as empty_spy:
            single_val = gpu._inner_loop_objective(F_t, M_t, 400., cpu_dev, chunk_rows=len(M_t))
        empty_spy.assert_not_called()
        self.assertEqual(single_val, expected_untransformed)

        # RegisterTractography and RegisterTractographyNonrigid forward gpu_device
        reg = RegisterTractography()
        reg.fixed, reg.moving = fixed, moving
        reg.use_gpu, reg.maxfun = True, 2
        reg.gpu_device = cpu_dev
        with patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                patch.object(gpu, 'AffineObjective', wraps=gpu.AffineObjective) as affine_spy:
            reg.compute()
        self.assertEqual(affine_spy.call_args.kwargs.get('device'), cpu_dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            nreg = RegisterTractographyNonrigid()
            nreg.nonrigid_grid_resolution = 3
            nreg.initialize_nonrigid_grid()
            nreg.initial_transform = nreg.displacement_field_numpy.copy()
            nreg.output_directory = tmpdir
            nreg.fixed = fixed[:, :, :10]
            nreg.moving = moving[:, :3, :10]
            nreg.use_gpu, nreg.maxfun = True, 2
            nreg.gpu_device = cpu_dev
            with patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                    patch.object(bspline, 'NonrigidObjective', wraps=bspline.NonrigidObjective) as nonrigid_spy:
                nreg.compute()
            self.assertEqual(nonrigid_spy.call_args.kwargs.get('device'), cpu_dev)

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_dev = torch.device('mps')
            obj_mps = gpu.AffineObjective(fixed, moving, device=mps_dev, gpu_float32=True)
            self.assertEqual(obj_mps.fixed.device.type, 'mps')
            self.assertEqual(obj_mps.fixed.dtype, torch.float32)
            mps_val = obj_mps(x, [1] * 4, 20.)
            self.assertAlmostEqual(mps_val, val32, delta=max(1e-4, abs(val32) * 1e-4))


if __name__ == '__main__':
    unittest.main()
