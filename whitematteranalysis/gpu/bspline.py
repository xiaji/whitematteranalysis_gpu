"""Batched WMA B-spline transforms, matching VTK's cubic/Zero-border rules.

Coefficient filtering stays in VTK; fixed sampling geometry is prepared once.
Reference: Kitware/VTK v9.1.0 Filters/Hybrid/vtkBSplineTransform.cxx.
"""
import numpy

from .config import get_device, float32_enabled, strict_enabled
from . import registration, registration_kernel, _memory

try:
    import torch
except ImportError:
    torch = None

_PREFILTER_MATRIX_CACHE = {}


def transform_fiber_array_numpy(fibers, displacement):
    """Apply VTK's double-precision transform in C++, without Python point loops."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from ..register_two_subjects_nonrigid_bsplines import convert_transform_to_vtk

    coordinates = numpy.ascontiguousarray(fibers.transpose(1, 2, 0).reshape(-1, 3), dtype=numpy.float64)
    points, output = vtk.vtkPoints(), vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coordinates))
    output.SetDataTypeToDouble()
    convert_transform_to_vtk(displacement).TransformPoints(points, output)
    return vtk_to_numpy(output.GetData()).reshape(fibers.shape[1], fibers.shape[2], 3).transpose(2, 0, 1).copy()


def _sampling_geometry(fibers, resolution):
    coordinates = numpy.ascontiguousarray(
        fibers.transpose(1, 2, 0).reshape(-1, 3), dtype=numpy.float64)
    grid = (coordinates + 100.) / (200. / (resolution - 1))
    integer = numpy.floor(grid).astype(numpy.int64)
    fraction = grid - integer
    square = fraction * fraction
    w3 = square * fraction * (1. / 6.)
    w0 = (square - fraction) * .5 - w3 + (1. / 6.)
    w2 = fraction + w0 - w3 * 2
    w1 = 1 - w0 - w2 - w3
    weights = numpy.stack((w0, w1, w2, w3), axis=-1)
    offsets = integer[..., None] + numpy.arange(-1, 3)
    weights[offsets < 0] = 0.
    weights[offsets >= resolution] = 0.
    invalid = numpy.any((integer < -2) | (integer > resolution), axis=1)
    offsets = offsets.clip(0, resolution - 1).astype(numpy.int32)
    return coordinates, offsets, weights, invalid


if registration_kernel.triton is not None:
    triton = registration_kernel.triton
    import triton.language as tl

    @triton.jit
    def _transform(Points, Indices, Weights, Invalid, Coefficients, Output,
                   N_POINTS, RES: tl.constexpr, BLOCK: tl.constexpr):
        point = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = point < N_POINTS
        invalid = tl.load(Invalid + point, mask, other=1)
        DTYPE = Points.dtype.element_ty

        idx_base = point * 12
        ix0 = tl.load(Indices + idx_base + 0, mask, other=0).to(tl.int64)
        ix1 = tl.load(Indices + idx_base + 1, mask, other=0).to(tl.int64)
        ix2 = tl.load(Indices + idx_base + 2, mask, other=0).to(tl.int64)
        ix3 = tl.load(Indices + idx_base + 3, mask, other=0).to(tl.int64)
        iy0 = tl.load(Indices + idx_base + 4, mask, other=0).to(tl.int64)
        iy1 = tl.load(Indices + idx_base + 5, mask, other=0).to(tl.int64)
        iy2 = tl.load(Indices + idx_base + 6, mask, other=0).to(tl.int64)
        iy3 = tl.load(Indices + idx_base + 7, mask, other=0).to(tl.int64)
        iz0 = tl.load(Indices + idx_base + 8, mask, other=0).to(tl.int64)
        iz1 = tl.load(Indices + idx_base + 9, mask, other=0).to(tl.int64)
        iz2 = tl.load(Indices + idx_base + 10, mask, other=0).to(tl.int64)
        iz3 = tl.load(Indices + idx_base + 11, mask, other=0).to(tl.int64)

        wx0 = tl.load(Weights + idx_base + 0, mask, other=0.).to(DTYPE)
        wx1 = tl.load(Weights + idx_base + 1, mask, other=0.).to(DTYPE)
        wx2 = tl.load(Weights + idx_base + 2, mask, other=0.).to(DTYPE)
        wx3 = tl.load(Weights + idx_base + 3, mask, other=0.).to(DTYPE)
        wy0 = tl.load(Weights + idx_base + 4, mask, other=0.).to(DTYPE)
        wy1 = tl.load(Weights + idx_base + 5, mask, other=0.).to(DTYPE)
        wy2 = tl.load(Weights + idx_base + 6, mask, other=0.).to(DTYPE)
        wy3 = tl.load(Weights + idx_base + 7, mask, other=0.).to(DTYPE)
        wz0 = tl.load(Weights + idx_base + 8, mask, other=0.).to(DTYPE)
        wz1 = tl.load(Weights + idx_base + 9, mask, other=0.).to(DTYPE)
        wz2 = tl.load(Weights + idx_base + 10, mask, other=0.).to(DTYPE)
        wz3 = tl.load(Weights + idx_base + 11, mask, other=0.).to(DTYPE)

        dx = tl.full((BLOCK,), 0., DTYPE)
        dy = tl.full((BLOCK,), 0., DTYPE)
        dz = tl.full((BLOCK,), 0., DTYPE)
        for iz, wz in ((iz0, wz0), (iz1, wz1), (iz2, wz2), (iz3, wz3)):
            vzx = tl.full((BLOCK,), 0., DTYPE)
            vzy = tl.full((BLOCK,), 0., DTYPE)
            vzz = tl.full((BLOCK,), 0., DTYPE)
            for iy, wy in ((iy0, wy0), (iy1, wy1), (iy2, wy2), (iy3, wy3)):
                vyx = tl.full((BLOCK,), 0., DTYPE)
                vyy = tl.full((BLOCK,), 0., DTYPE)
                vyz = tl.full((BLOCK,), 0., DTYPE)
                row_base = (iz * RES + iy) * RES
                for ix, wx in ((ix0, wx0), (ix1, wx1), (ix2, wx2), (ix3, wx3)):
                    c_idx = (row_base + ix) * 3
                    cx = tl.load(Coefficients + c_idx + 0, mask, other=0.).to(DTYPE)
                    cy = tl.load(Coefficients + c_idx + 1, mask, other=0.).to(DTYPE)
                    cz = tl.load(Coefficients + c_idx + 2, mask, other=0.).to(DTYPE)
                    vyx += cx * wx
                    vyy += cy * wx
                    vyz += cz * wx
                vzx += vyx * wy
                vzy += vyy * wy
                vzz += vyz * wy
            dx += vzx * wz
            dy += vzy * wz
            dz += vzz * wz

        p_base = point * 3
        zero = tl.full((BLOCK,), 0., DTYPE)
        ox = tl.load(Points + p_base + 0, mask, other=0.)
        oy = tl.load(Points + p_base + 1, mask, other=0.)
        oz = tl.load(Points + p_base + 2, mask, other=0.)
        tl.store(Output + p_base + 0, ox + tl.where(invalid, zero, dx), mask)
        tl.store(Output + p_base + 1, oy + tl.where(invalid, zero, dy), mask)
        tl.store(Output + p_base + 2, oz + tl.where(invalid, zero, dz), mask)


def _bspline_1d_prefilter_matrix(resolution):
    """Return the exact (R, R) float64 1D VTK cubic B-spline prefilter matrix."""
    cached = _PREFILTER_MATRIX_CACHE.get(resolution)
    if cached is not None:
        return cached
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    M = numpy.zeros((resolution, resolution), dtype=numpy.float64)
    for col in range(resolution):
        impulse = numpy.zeros(resolution, dtype=numpy.float64)
        impulse[col] = 1.0
        scalars = numpy_to_vtk(numpy.ascontiguousarray(impulse), deep=True)
        image = vtk.vtkImageData()
        image.SetDimensions(resolution, 1, 1)
        image.GetPointData().SetScalars(scalars)
        filt = vtk.vtkImageBSplineCoefficients()
        filt.SetInputData(image)
        filt.Update()
        filt.UpdateWholeExtent()
        M[:, col] = vtk_to_numpy(filt.GetOutput().GetPointData().GetScalars())
    M.setflags(write=False)
    _PREFILTER_MATRIX_CACHE[resolution] = M
    return M


def _apply_separable_3d(matrix_1d, grid_4d):
    """Apply separable 1D filter matrix along (x, y, z) of an (R, R, R, 3) grid."""
    x_filtered = torch.einsum('kc,zycp->zykp', matrix_1d, grid_4d)
    y_filtered = torch.einsum('jb,zbkp->zjkp', matrix_1d, x_filtered)
    return torch.einsum('ia,ajkp->ijkp', matrix_1d, y_filtered)


class _CoefficientGrid:
    """Reuse VTK's grid and coefficient filter for one optimization."""
    def __init__(self, resolution):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        self.displacement = numpy.zeros((resolution**3, 3), dtype=numpy.float32)
        self.scalars = numpy_to_vtk(self.displacement)
        self.scalars.SetName('DisplacementField')
        self.grid = vtk.vtkImageData()
        self.grid.SetOrigin(-100., -100., -100.)
        spacing = 200. / (resolution - 1)
        self.grid.SetSpacing(spacing, spacing, spacing)
        self.grid.SetDimensions(resolution, resolution, resolution)
        self.grid.GetPointData().SetScalars(self.scalars)
        self.filter = vtk.vtkImageBSplineCoefficients()
        self.filter.SetInputData(self.grid)

    def __call__(self, displacement):
        from vtk.util.numpy_support import vtk_to_numpy

        numpy.copyto(self.displacement, numpy.asarray(displacement).reshape(-1, 3))
        self.scalars.Modified()
        self.grid.Modified()
        self.filter.Update()
        self.filter.UpdateWholeExtent()
        return vtk_to_numpy(self.filter.GetOutput().GetPointData().GetScalars())


class NonrigidObjective:
    """Fixed geometry and GPU inputs owned by one nonrigid optimization."""
    def __init__(self, fixed, moving, resolution, device=None, gpu_float32=None):
        _memory.disable_tf32()
        self.device = (device if hasattr(device, 'type')
                       else (torch.device(device) if device is not None else get_device()))
        self._dtype = torch.float32 if (
            self.device.type == 'mps' or float32_enabled(gpu_float32)
        ) else torch.float64
        self.fixed = registration._prepare_fixed(fixed, self.device, self._dtype)
        self.resolution = int(resolution)
        self.shape = moving.shape
        self.moving = moving
        self._use_triton = (registration_kernel.triton is not None
                            and self.device.type == 'cuda')

        coordinates, indices, weights, invalid = _sampling_geometry(moving, self.resolution)
        self.points = torch.as_tensor(coordinates, dtype=self._dtype, device=self.device)
        self.indices = torch.as_tensor(indices, dtype=torch.int32, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=self._dtype, device=self.device)
        self.invalid = torch.as_tensor(invalid, dtype=torch.bool, device=self.device)
        self.output = torch.empty_like(self.points)
        self._prefilter_matrix = torch.as_tensor(
            _bspline_1d_prefilter_matrix(self.resolution).copy(),
            dtype=self._dtype, device=self.device)

        if self._use_triton:
            self.coefficients = _CoefficientGrid(self.resolution)
            self._host_coefficients = torch.empty(
                (self.resolution ** 3, 3), dtype=torch.float32, pin_memory=True)
            self._coefficients_gpu = torch.empty(
                (self.resolution ** 3, 3), dtype=torch.float32, device=self.device)
        else:
            self.coefficients = None
            self._host_coefficients = None
            self._coefficients_gpu = None

        self._F_flat = None
        self._F_rev_flat = None
        self.workspace = (registration._ObjectiveWorkspace(self.fixed, moving.shape[1])
                          if self.fixed.device.type in ('cuda', 'mps') else None)

    def _compute_coefficients_tensor(self, displacement, allow_vtk_strict=True):
        if allow_vtk_strict and strict_enabled() and self._dtype == torch.float64 and self._use_triton:
            coeffs_np = self.coefficients(displacement)
            numpy.copyto(self._host_coefficients.numpy(), coeffs_np)
            self._coefficients_gpu.copy_(self._host_coefficients, non_blocking=True)
            return self._coefficients_gpu
        disp_grid = torch.as_tensor(
            numpy.ascontiguousarray(displacement, dtype=numpy.float64 if self._dtype == torch.float64 else numpy.float32),
            dtype=self._dtype, device=self.device,
        ).reshape(self.resolution, self.resolution, self.resolution, 3)
        return _apply_separable_3d(self._prefilter_matrix, disp_grid).reshape(self.resolution ** 3, 3).contiguous()

    def _transform_torch(self, coeffs_flat):
        """Pure-PyTorch vectorized B-spline interpolation using separable prefiltered coefficients."""
        n_pts = self.points.shape[0]
        if n_pts == 0:
            return self.output.reshape(self.shape[1], self.shape[2], 3)
        R = self.resolution
        ix = self.indices[:, 0, :].to(torch.int64)
        iy = self.indices[:, 1, :].to(torch.int64)
        iz = self.indices[:, 2, :].to(torch.int64)
        wx = self.weights[:, 0, :]
        wy = self.weights[:, 1, :]
        wz = self.weights[:, 2, :]

        # (N, 4, 4, 4) flat indices -> gather -> weighted tensor product reduction
        flat_idx = ((iz[:, :, None, None] * R + iy[:, None, :, None]) * R + ix[:, None, None, :]).reshape(-1)
        gathered = coeffs_flat[flat_idx].reshape(n_pts, 4, 4, 4, 3)
        vx = (gathered * wx[:, None, None, :, None]).sum(dim=3)
        vy = (vx * wy[:, None, :, None]).sum(dim=2)
        delta = (vy * wz[:, :, None]).sum(dim=1)
        delta = torch.where(self.invalid[:, None], torch.zeros_like(delta), delta)
        torch.add(self.points, delta, out=self.output)
        return self.output.reshape(self.shape[1], self.shape[2], 3)

    def transform(self, displacement):
        if strict_enabled() and not self._use_triton and self.device.type != 'mps':
            moved = transform_fiber_array_numpy(self.moving, displacement)
            return registration._prepare_fixed(moved, self.device, self._dtype)

        coeffs = self._compute_coefficients_tensor(displacement, allow_vtk_strict=True)
        if self._use_triton:
            n_points = self.points.shape[0]
            if n_points > 0:
                with torch.cuda.device(self.device):
                    _transform[(triton.cdiv(n_points, 128),)](
                        self.points, self.indices, self.weights, self.invalid, coeffs,
                        self.output, n_points, RES=self.resolution, BLOCK=128,
                        enable_fp_fusion=False)
            return self.output.reshape(self.shape[1], self.shape[2], 3)
        return self._transform_torch(coeffs)

    def value_and_gradient(self, displacement, sigma):
        """Return (objective_value, grad_displacement) analytically without finite differences."""
        R = self.resolution
        n_moving = self.shape[1]
        n_points_per_fiber = self.shape[2]
        n_fixed = self.fixed.shape[0]
        if n_moving == 0 or n_fixed == 0:
            return 0.0, numpy.zeros(R ** 3 * 3, dtype=numpy.float64)

        coeffs = self._compute_coefficients_tensor(displacement, allow_vtk_strict=False)
        if self._use_triton and self.points.shape[0] > 0:
            with torch.cuda.device(self.device):
                _transform[(triton.cdiv(self.points.shape[0], 128),)](
                    self.points, self.indices, self.weights, self.invalid, coeffs,
                    self.output, self.points.shape[0], RES=R, BLOCK=128,
                    enable_fp_fusion=False)
            moved = self.output.reshape(n_moving, n_points_per_fiber, 3)
        else:
            moved = self._transform_torch(coeffs)

        sigmasq = float(sigma) * float(sigma)
        points_f = float(n_points_per_fiber)
        grad_scale = 2.0 / (points_f * sigmasq)
        if self._F_flat is None:
            self._F_flat = self.fixed.reshape(n_fixed, n_points_per_fiber * 3).contiguous()
            self._F_rev_flat = self.fixed.flip(1).reshape(n_fixed, n_points_per_fiber * 3).contiguous()

        grad_q = torch.empty_like(moved)
        chunk_rows = (self.workspace.chunk_rows if self.workspace is not None
                      else max(1, min(n_moving, 256)))
        total_loss = 0.0
        s = 0
        while s < n_moving:
            e = min(s + chunk_rows, n_moving)
            q_c = moved[s:e]
            try:
                d_fwd = registration._squared_point_distances(self.fixed, q_c).sum(-1) / points_f
                d_rev = registration._squared_point_distances(self.fixed, q_c.flip(1)).sum(-1) / points_f
                mask_fwd = d_fwd <= d_rev
                d_min = torch.where(mask_fwd, d_fwd, d_rev)
                del d_fwd, d_rev
                prob = torch.exp(d_min.neg_().div_(sigmasq))
                sum_p = prob.sum(-1)
                denom = sum_p + 1e-20
                chunk_loss = float(torch.neg(torch.log(denom / float(n_fixed))).sum())
                p_fwd = torch.where(mask_fwd, prob, torch.zeros_like(prob))
                p_rev = prob - p_fwd
                weighted_F = (
                    torch.matmul(p_fwd, self._F_flat) + torch.matmul(p_rev, self._F_rev_flat)
                ).reshape(e - s, n_points_per_fiber, 3)
                grad_q[s:e] = (grad_scale / denom)[:, None, None] * (
                    sum_p[:, None, None] * q_c - weighted_F
                )
            except Exception as exc:
                if not _memory.is_gpu_oom(exc) or (e - s) == 1:
                    raise
                _memory.empty_cache(self.device)
                chunk_rows = max(1, (e - s) // 2)
                if self.workspace is not None:
                    self.workspace.chunk_rows = chunk_rows
                continue
            total_loss += chunk_loss
            s = e

        g_pts = grad_q.reshape(self.points.shape[0], 3)
        g_pts = torch.where(self.invalid[:, None], torch.zeros_like(g_pts), g_pts)
        dC = torch.zeros((R ** 3, 3), dtype=self._dtype, device=self.device)
        ix = self.indices[:, 0, :].to(torch.int64)
        iy = self.indices[:, 1, :].to(torch.int64)
        iz = self.indices[:, 2, :].to(torch.int64)
        wx = self.weights[:, 0, :]
        wy = self.weights[:, 1, :]
        wz = self.weights[:, 2, :]
        for a in range(4):
            w_z = wz[:, a]
            z_off = iz[:, a] * R
            for b in range(4):
                w_zy = w_z * wy[:, b]
                zy_off = (z_off + iy[:, b]) * R
                for c in range(4):
                    w = (w_zy * wx[:, c])[:, None]
                    dC.index_add_(0, zy_off + ix[:, c], w * g_pts)

        dD = _apply_separable_3d(self._prefilter_matrix.t(), dC.reshape(R, R, R, 3))
        grad_np = dD.reshape(-1).detach().cpu().numpy().astype(numpy.float64)
        return float(total_loss), grad_np

    def __call__(self, displacement, sigma):
        moved = self.transform(displacement)
        return registration._inner_loop_objective(self.fixed, moved, sigma * sigma,
                                                  self.device, workspace=self.workspace)


def bspline_objective_and_gradient(fixed, moving, displacement, resolution, sigma,
                                   device=None, gpu_float32=None):
    """Convenience wrapper returning (objective, gradient) for one nonrigid state."""
    objective = NonrigidObjective(
        fixed, moving, resolution, device=device, gpu_float32=gpu_float32)
    return objective.value_and_gradient(displacement, sigma)
