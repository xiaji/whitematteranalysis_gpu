"""B-spline coordinates and optimizer results against VTK's point API."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import whitematteranalysis as wma
from whitematteranalysis.gpu import bspline, registration, registration_kernel
from whitematteranalysis.register_two_subjects_nonrigid_bsplines import (
    RegisterTractographyNonrigid, convert_transform_to_vtk)


def reference(fibers, field):
    transform = convert_transform_to_vtk(field)
    out = np.empty(fibers.shape, dtype=np.float64)
    for i in range(fibers.shape[1]):
        for j in range(fibers.shape[2]):
            out[:, i, j] = transform.TransformPoint(*fibers[:, i, j])
    return out


def samples(resolution):
    rng = np.random.RandomState(resolution)
    fibers = rng.uniform(-300, 300, (3, 53, 10))
    # Grid knots and one-ULP neighborhoods include all Zero-border cases.
    knots = -100. + np.arange(-3, resolution + 3) * (200. / (resolution - 1))
    positions = np.concatenate((knots, np.nextafter(knots, -np.inf),
                                np.nextafter(knots, np.inf)))
    fibers.reshape(3, -1)[:, :len(positions)] = positions
    field = rng.normal(size=resolution**3 * 3) * 10.
    return fibers, field


class BatchVTKTests(unittest.TestCase):
    def test_reused_coefficients(self):
        from vtk.util.numpy_support import vtk_to_numpy
        for resolution in (2, 3, 4, 6, 10):
            _, field = samples(resolution)
            cache = bspline._CoefficientGrid(resolution)
            for value in (field, -field, field * 0, field * 3):
                transform = convert_transform_to_vtk(value)
                expected = vtk_to_numpy(transform.GetCoefficientData().GetPointData().GetScalars())
                np.testing.assert_array_equal(cache(value), expected)

    def test_coordinates_bit_identical(self):
        for resolution in (2, 3, 4, 6, 10):
            fibers, field = samples(resolution)
            with self.subTest(resolution=resolution):
                np.testing.assert_array_equal(
                    bspline.transform_fiber_array_numpy(fibers, field), reference(fibers, field))

    def test_separable_prefilter_and_torch_forward_parity(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')

        prev_strict = wma.gpu.strict_enabled()
        self.addCleanup(wma.gpu.set_strict, prev_strict)
        wma.gpu.set_strict(False)

        for resolution in (2, 3, 4, 6):
            fibers, field = samples(resolution)
            fixed = fibers[:, :7].copy()
            ref_coords = reference(fibers, field)
            ref_obj = registration.inner_loop_objective(fixed, ref_coords, 400., device='cpu')
            with self.subTest(resolution=resolution):
                obj64 = bspline.NonrigidObjective(
                    fixed, fibers, resolution, device='cpu', gpu_float32=False)
                actual64 = obj64.transform(field).cpu().numpy().transpose(2, 0, 1)
                np.testing.assert_allclose(actual64, ref_coords, rtol=1e-5, atol=1e-4)
                np.testing.assert_allclose(obj64(field, 20.), ref_obj, rtol=1e-5, atol=1e-4)

                obj32 = bspline.NonrigidObjective(
                    fixed, fibers, resolution, device='cpu', gpu_float32=True)
                self.assertEqual(obj32._dtype, torch.float32)
                actual32 = obj32.transform(field)
                self.assertEqual(actual32.dtype, torch.float32)
                np.testing.assert_allclose(
                    actual32.cpu().numpy().transpose(2, 0, 1), ref_coords, rtol=1e-3, atol=1e-3)

    def test_analytic_value_and_gradient_matches_finite_differences_cpu(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch is required')

        rng = np.random.RandomState(123)
        for resolution in (3, 4):
            fixed = rng.uniform(-80., 80., (3, 11, 5))
            moving = rng.uniform(-80., 80., (3, 7, 5))
            disp = rng.normal(size=resolution ** 3 * 3) * 3.0
            objective = bspline.NonrigidObjective(
                fixed, moving, resolution, device='cpu', gpu_float32=False)
            val, grad = objective.value_and_gradient(disp, sigma=25.0)
            np.testing.assert_allclose(val, objective(disp, 25.0), rtol=1e-13, atol=1e-13)
            self.assertEqual(grad.shape, disp.shape)

            eps = 1e-6
            probe_idx = np.linspace(0, disp.size - 1, 18, dtype=int)
            for idx in probe_idx:
                d_plus = disp.copy()
                d_minus = disp.copy()
                d_plus[idx] += eps
                d_minus[idx] -= eps
                # Compare against smooth float64 forward value of the same separable operator
                v_plus, _ = objective.value_and_gradient(d_plus, 25.0)
                v_minus, _ = objective.value_and_gradient(d_minus, 25.0)
                fd = (v_plus - v_minus) / (2.0 * eps)
                self.assertAlmostEqual(grad[idx], fd, places=6)

    def test_bfgs_analytic_grad_converges_faster_and_does_not_swallow_oom(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch is required')

        rng = np.random.RandomState(77)
        fixed = rng.uniform(-60., 60., (3, 16, 4))
        moving = fixed[:, :8, :] + rng.normal(scale=0.5, size=(3, 8, 4)) + 4.0

        def run_bfgs(analytic, maxfun):
            reg = RegisterTractographyNonrigid()
            reg.nonrigid_grid_resolution = 3
            reg.initialize_nonrigid_grid()
            reg.initial_transform = reg.displacement_field_numpy.copy()
            reg.fixed = fixed.copy()
            reg.moving = moving.copy()
            reg.sigma = 25.0
            reg.maxfun = maxfun
            reg.optimizer = 'BFGS'
            reg.use_gpu = False
            reg.analytic_grad = analytic
            reg.initial_step = 2.0
            reg.final_step = 0.5
            with tempfile.TemporaryDirectory() as directory:
                reg.output_directory = directory
                reg.compute()
            return reg.objective_function_values.copy(), reg.objective_computations

        vals_fd, n_fd = run_bfgs(analytic=False, maxfun=85)
        vals_an, n_an = run_bfgs(analytic=True, maxfun=20)
        self.assertLessEqual(n_an, 20)
        self.assertLess(vals_an[-1], vals_an[0])
        self.assertLess(vals_an[-1], vals_fd[-1])

        # Unexpected RuntimeError (e.g. simulated CUDA OOM) must NOT be swallowed by BFGS
        reg_err = RegisterTractographyNonrigid()
        reg_err.nonrigid_grid_resolution = 3
        reg_err.initialize_nonrigid_grid()
        reg_err.initial_transform = reg_err.displacement_field_numpy.copy()
        reg_err.fixed = fixed
        reg_err.moving = moving
        reg_err.optimizer = 'BFGS'
        reg_err.use_gpu = False
        reg_err.analytic_grad = True
        with tempfile.TemporaryDirectory() as directory:
            reg_err.output_directory = directory
            with patch.object(bspline.NonrigidObjective, 'value_and_gradient',
                              side_effect=RuntimeError('CUDA out of memory')):
                with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                    reg_err.compute()

    def test_vectorized_write_bspline_itk_format(self):
        rng = np.random.RandomState(11)
        field = rng.normal(scale=4.0, size=3 ** 3 * 3)
        tx = convert_transform_to_vtk(field)
        with tempfile.TemporaryDirectory() as directory:
            fnames = wma.io.write_transforms_to_itk_format([tx], directory, ['nr_subj'])
            with open(fnames[0], 'r') as fh:
                text = fh.read().splitlines()

        self.assertEqual(text[0], '#Insight Transform File V1.0')
        self.assertEqual(text[2], 'Transform: BSplineTransform_double_3_3')
        self.assertTrue(text[3].startswith('Parameters: '))
        params = np.fromstring(text[3][len('Parameters: '):], sep=' ')
        self.assertEqual(params.size, 3 * 105 * 105 * 105)
        self.assertEqual(
            text[4],
            'FixedParameters: 105 105 105 -104.0 -104.0 -104.0 2 2 2 1 0 0 0 1 0 0 0 1',
        )

        # Verify corners and interior points match vtkBSplineTransform inverse directly in LPS
        spline_inv = tx.GetInverse()
        inv_lps = wma.vtk.vtkTransform() if hasattr(wma, 'vtk') else None
        import vtk
        ras_2_lps = vtk.vtkTransform()
        ras_2_lps.Scale(-1, -1, 1)
        lps_2_ras = vtk.vtkTransform()
        lps_2_ras.Scale(-1, -1, 1)
        wrapped = vtk.vtkGeneralTransform()
        wrapped.Concatenate(lps_2_ras)
        wrapped.Concatenate(spline_inv)
        wrapped.Concatenate(ras_2_lps)

        n_pts = 105 ** 3
        dx = params[:n_pts].reshape(105, 105, 105)
        dy = params[n_pts:2 * n_pts].reshape(105, 105, 105)
        dz = params[2 * n_pts:].reshape(105, 105, 105)
        for s_idx, p_idx, l_idx in ((0, 0, 0), (52, 52, 52), (40, 61, 73), (104, 104, 104)):
            lps_pt = ((l_idx - 52) * 2.0, (p_idx - 52) * 2.0, (s_idx - 52) * 2.0)
            out_pt = wrapped.TransformPoint(*lps_pt)
            expected = np.array([out_pt[0] - lps_pt[0], out_pt[1] - lps_pt[1], out_pt[2] - lps_pt[2]])
            actual = np.array([dx[s_idx, p_idx, l_idx], dy[s_idx, p_idx, l_idx], dz[s_idx, p_idx, l_idx]])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class GPUBSplineTests(unittest.TestCase):
    def test_coordinates_and_entropy(self):
        prev_strict = wma.gpu.strict_enabled()
        self.addCleanup(wma.gpu.set_strict, prev_strict)

        for resolution in (2, 3, 4, 6, 10):
            fibers, field = samples(resolution)
            fixed = fibers[:, :7].copy()
            with self.subTest(resolution=resolution):
                original = reference(fibers, field)
                wma.gpu.set_strict(True)
                objective = bspline.NonrigidObjective(fixed, fibers, resolution)
                actual = objective.transform(field).cpu().numpy().transpose(2, 0, 1)
                np.testing.assert_array_equal(actual, original)
                self.assertEqual(objective(field, 20.),
                                 registration.inner_loop_objective(fixed, original, 400.))
                with patch.object(registration_kernel, 'triton', None):
                    fallback = bspline.NonrigidObjective(fixed, fibers, resolution)
                    np.testing.assert_array_equal(
                        fallback.transform(field).cpu().numpy().transpose(2, 0, 1), original)

                # Non-strict GPU separable prefilter and float32 paths + analytic gradient
                wma.gpu.set_strict(False)
                nonstrict_obj = bspline.NonrigidObjective(fixed, fibers, resolution, gpu_float32=False)
                nonstrict_coords = nonstrict_obj.transform(field).cpu().numpy().transpose(2, 0, 1)
                np.testing.assert_allclose(nonstrict_coords, original, rtol=1e-5, atol=1e-4)

                import torch
                f32_obj = bspline.NonrigidObjective(fixed, fibers, resolution, gpu_float32=True)
                f32_coords = f32_obj.transform(field)
                self.assertEqual(f32_coords.dtype, torch.float32)
                np.testing.assert_allclose(
                    f32_coords.cpu().numpy().transpose(2, 0, 1), original, rtol=1e-3, atol=1e-3)

                if resolution in (3, 4):
                    val, grad = nonstrict_obj.value_and_gradient(field, 25.0)
                    np.testing.assert_allclose(val, nonstrict_obj(field, 25.0), rtol=1e-13, atol=1e-13)
                    eps = 1e-6
                    for idx in (0, field.size // 2, field.size - 1):
                        dp, dm = field.copy(), field.copy()
                        dp[idx] += eps
                        dm[idx] -= eps
                        vp, _ = nonstrict_obj.value_and_gradient(dp, 25.0)
                        vm, _ = nonstrict_obj.value_and_gradient(dm, 25.0)
                        self.assertAlmostEqual(grad[idx], (vp - vm) / (2.0 * eps), places=6)

    def test_complete_optimization_and_release(self):
        prev_strict = wma.gpu.strict_enabled()
        wma.gpu.set_strict(True)
        self.addCleanup(wma.gpu.set_strict, prev_strict)

        rng = np.random.RandomState(42)
        reg = RegisterTractographyNonrigid()
        reg.nonrigid_grid_resolution = 3
        reg.initialize_nonrigid_grid()
        reg.initial_transform = reg.displacement_field_numpy.copy()
        reg.fixed = rng.uniform(-120, 120, (3, 13, 10))
        reg.moving = rng.uniform(-120, 120, (3, 7, 10))
        reg.maxfun, reg.optimizer, reg.use_gpu = 100, 'Cobyla', True
        reg.initial_step, reg.final_step = 5., 1.
        with tempfile.TemporaryDirectory() as directory:
            reg.output_directory = directory
            actual = reg.compute().copy()
            values = reg.objective_function_values.copy()
            self.assertIsNone(reg._gpu_objective)
            reg.objective_function_values.clear()
            reg.objective_computations = 0
            reg.minimum_objective = np.inf
            with patch.object(registration_kernel, 'triton', None):
                expected = reg.compute().copy()
            expected_values = reg.objective_function_values.copy()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(values, expected_values)

            # Default non-strict mode (fused cross-fiber sums) stays within tight tolerance.
            wma.gpu.set_strict(False)
            reg.objective_function_values.clear()
            reg.objective_computations = 0
            reg.minimum_objective = np.inf
            nonstrict_actual = reg.compute().copy()
            nonstrict_values = reg.objective_function_values.copy()
            np.testing.assert_allclose(nonstrict_values[0], expected_values[0],
                                       rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(nonstrict_actual, expected, rtol=1e-6, atol=1e-6)

            with patch('scipy.optimize.fmin_cobyla', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    reg.compute()
            self.assertIsNone(reg._gpu_objective)

    @unittest.skipUnless(registration_kernel.triton is not None, 'Triton is required')
    def test_fused_triton_backward_kernel_and_memory(self):
        import torch

        prev_strict = wma.gpu.strict_enabled()
        self.addCleanup(wma.gpu.set_strict, prev_strict)
        rng = np.random.RandomState(314)
        device = wma.gpu.get_device()

        for dtype, rtol, atol in ((torch.float64, 1e-12, 1e-12), (torch.float32, 1e-5, 1e-5)):
            for nm, nf, p in ((17, 13, 3), (33, 600, 10), (19, 1050, 15), (9, 530, 40)):
                with self.subTest(dtype=str(dtype), nm=nm, nf=nf, p=p):
                    fixed = torch.as_tensor(rng.uniform(-60, 60, (nf, p, 3)), dtype=dtype, device=device)
                    moving = torch.as_tensor(rng.uniform(-60, 60, (nm, p, 3)), dtype=dtype, device=device)
                    sigmasq = 400.0
                    sum_k, wf_k = registration_kernel.pairwise_probability_sums_and_weighted_fixed(
                        fixed, moving, sigmasq)

                    d_fwd = registration._squared_point_distances(fixed, moving).sum(-1) / float(p)
                    d_rev = registration._squared_point_distances(fixed, moving.flip(1)).sum(-1) / float(p)
                    mask_fwd = d_fwd <= d_rev
                    d_min = torch.where(mask_fwd, d_fwd, d_rev)
                    prob = torch.exp(-d_min / sigmasq)
                    sum_ref = prob.sum(-1)
                    p_fwd = torch.where(mask_fwd, prob, torch.zeros_like(prob))
                    p_rev = prob - p_fwd
                    wf_ref = (
                        torch.matmul(p_fwd, fixed.reshape(nf, p * 3))
                        + torch.matmul(p_rev, fixed.flip(1).reshape(nf, p * 3))
                    ).reshape(nm, p, 3)

                    torch.testing.assert_close(sum_k, sum_ref, rtol=rtol, atol=atol)
                    torch.testing.assert_close(wf_k, wf_ref, rtol=rtol, atol=atol)

        # Verify NonrigidObjective.value_and_gradient fused Triton vs strict PyTorch fallback & bounded VRAM
        fixed_big = rng.uniform(-70, 70, (3, 2048, 10))
        moving_big = rng.uniform(-70, 70, (3, 2048, 10))
        disp = rng.normal(scale=2.5, size=6 ** 3 * 3)
        obj = bspline.NonrigidObjective(fixed_big, moving_big, 6, device=device, gpu_float32=False)

        wma.gpu.set_strict(True)
        val_strict, grad_strict = obj.value_and_gradient(disp, sigma=20.0)

        wma.gpu.set_strict(False)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base_alloc = torch.cuda.memory_allocated(device)
        val_fused, grad_fused = obj.value_and_gradient(disp, sigma=20.0)
        torch.cuda.synchronize(device)
        peak_delta = torch.cuda.max_memory_allocated(device) - base_alloc

        np.testing.assert_allclose(val_fused, val_strict, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(grad_fused, grad_strict, rtol=1e-10, atol=1e-10)
        # Eager (2048 x 2048 x 10) or (2048 x 2048) float64 would exceed 32 MiB/point or 32 MiB/pair
        self.assertLess(peak_delta, 128 * 1024 * 1024)
