# -*- coding: utf-8 -*-

""" gpu.registration

GPU (torch) implementations of the fiber registration objective,
numerically equivalent to register_two_subjects.py:

    transform_fiber_array_numpy   (register_two_subjects.py:386)
    inner_loop_objective          (register_two_subjects.py:305)

The 15 parameter transform is still built with
register_two_subjects.convert_transform_to_vtk, but its final 4x4
matrix is applied to all fiber points as one batched computation with
the same accumulation order as vtkMatrix4x4::MultiplyPoint, so with
float64 the transformed points are identical to the CPU code.

The objective differs from the CPU only in the summation order over
fibers (~1e-15 relative difference in float64).

"""

import numpy

from .config import get_device, float32_enabled, strict_enabled
from . import registration_kernel, _memory

try:
    import torch
except ImportError:
    torch = None


def _resolve_device(device=None):
    if device is None:
        return get_device()
    return device if hasattr(device, 'type') else torch.device(device)


def transform_matrix_from_vtk(transform, mode=[1, 1, 1, 1]):
    """Build the 4x4 matrix of the 15 parameter transform exactly as
    vtk would apply it (register_two_subjects.convert_transform_to_vtk
    with scaled=True)."""

    from ..register_two_subjects import convert_transform_to_vtk

    vtktrans = convert_transform_to_vtk(transform, scaled=True, mode=mode)
    m4 = vtktrans.GetMatrix()
    M = numpy.array([m4.GetElement(i, j) for i in range(4) for j in range(4)],
                    dtype=numpy.float64)
    return M.reshape(4, 4)


def transform_fiber_array(in_array, transform, mode=[1, 1, 1, 1], device=None,
                          gpu_float32=None):
    """Batched transform_fiber_array_numpy.

    in_array: (3, N, P) float64 numpy array (R, A, S on the first axis).
    Returns a (3, N, P) float64/float32 numpy array, the same layout.
    """

    return _transform_fiber_tensor(
        in_array, transform, mode, device, gpu_float32=gpu_float32).cpu().numpy()


def _transform_fiber_tensor(in_array, transform, mode, device, gpu_float32=None):
    """Apply the affine transform and retain its result on device."""
    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    _memory.disable_tf32()
    M = transform_matrix_from_vtk(transform, mode)
    dev = _resolve_device(device)
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64

    t = torch.as_tensor(in_array, dtype=dtype, device=dev)
    x, y, z = t.unbind(0)
    Mt = torch.as_tensor(M, dtype=dtype, device=dev)

    # same accumulation order as vtkMatrix4x4::MultiplyPoint
    out_x = Mt[0, 0] * x + Mt[0, 1] * y + Mt[0, 2] * z + Mt[0, 3]
    out_y = Mt[1, 0] * x + Mt[1, 1] * y + Mt[1, 2] * z + Mt[1, 3]
    out_z = Mt[2, 0] * x + Mt[2, 1] * y + Mt[2, 2] * z + Mt[2, 3]

    return torch.stack((out_x, out_y, out_z), dim=0)


def _prepare_fixed(fixed, device, dtype):
    """(3, Nf, P) NumPy array -> contiguous (Nf, P, 3) device tensor."""

    F = numpy.ascontiguousarray(numpy.transpose(fixed, (1, 2, 0)))
    return torch.as_tensor(F, dtype=dtype, device=device)


def inner_loop_objective(fixed, moving, sigmasq, device=None, chunk_rows=None,
                         gpu_float32=None):
    """Batched inner_loop_objective.

    fixed: (3, N, P) numpy array.
    moving: (3, N, P) numpy array or device tensor.
    Returns a python float, matching register_two_subjects.py:305-330.
    """

    if torch is None:
        raise RuntimeError("torch is not available, cannot use the GPU registration backend.")

    _memory.disable_tf32()
    dev = _resolve_device(device)
    dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64

    F = _prepare_fixed(fixed, dev, dtype)

    if isinstance(moving, torch.Tensor):
        M = moving.permute(1, 2, 0).contiguous().to(device=dev, dtype=dtype)
    else:
        M = numpy.ascontiguousarray(numpy.transpose(moving, (1, 2, 0)))
        M = torch.as_tensor(M, dtype=dtype)
    return _inner_loop_objective(F, M, sigmasq, dev, chunk_rows)


def _choose_chunk_rows(F, n_moving, dev):
    itemsize = 8 if F.dtype == torch.float64 else 4
    bytes_per_row = max(1, F.shape[0] * F.shape[1] * itemsize * 6)
    free_bytes = _memory.effective_free_vram(dev)
    return max(1, min(n_moving, int(int(free_bytes * 0.5) // bytes_per_row), 8192))


class _ObjectiveWorkspace:
    """Host probability storage and adaptive tiles for one optimizer."""
    def __init__(self, fixed, moving_count):
        self.chunk_rows = _choose_chunk_rows(fixed, moving_count, fixed.device)
        self.probability = torch.empty(
            moving_count, dtype=fixed.dtype, pin_memory=(fixed.device.type == 'cuda'))


def _inner_loop_objective(F, M, sigmasq, dev, chunk_rows=None, workspace=None):
    """Entropy for prepared (N, P, 3) tensors; retain the CPU reduction."""
    dev = _resolve_device(dev)
    n_fixed = F.shape[0]
    n_moving = M.shape[0]
    if chunk_rows is not None and chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')

    if chunk_rows is None:
        chunk_rows = workspace.chunk_rows if workspace is not None else _choose_chunk_rows(F, n_moving, dev)

    probability = None
    s = 0
    while s < n_moving:
        e = min(s + chunk_rows, n_moving)
        try:
            q = M[s:e].to(dev)
            prob = _pairwise_probability(F, q, sigmasq)
        except Exception as exc:
            if not _memory.is_gpu_oom(exc):
                raise
            if e - s == 1:
                raise
            _memory.empty_cache(dev)
            chunk_rows = max(1, (e - s) // 2)
            if workspace is not None:
                workspace.chunk_rows = chunk_rows
            continue

        if s == 0 and e == n_moving:
            probability = prob
        else:
            if probability is None:
                probability = torch.empty(n_moving, dtype=F.dtype, device=dev)
            probability[s:e] = prob
        s = e

    if probability is None:
        probability = torch.empty(0, dtype=F.dtype, device=dev)

    # register_two_subjects.py:315-329: floor before dividing by n_fixed
    if workspace is not None:
        if dev.type == 'cuda':
            workspace.probability.copy_(probability, non_blocking=True)
            torch.cuda.current_stream(dev).synchronize()
        else:
            workspace.probability.copy_(probability)
        return float(workspace.probability.add_(1e-20).div_(float(n_fixed)).log_().neg_().sum())
    probability = probability.cpu()
    probability = probability + 1e-20
    probability = probability / float(n_fixed)
    entropy = torch.sum(-torch.log(probability))
    return float(entropy)


def _squared_point_distances(F, q):
    """Eager reference for environments without Triton."""
    dx = F[None, :, :, 0] - q[:, None, :, 0]
    dx.square_()
    dy = F[None, :, :, 1] - q[:, None, :, 1]
    dy.square_()
    dx.add_(dy)
    torch.sub(F[None, :, :, 2], q[:, None, :, 2], out=dy)
    dy.square_()
    return dx.add_(dy)


def _pairwise_probability(F, q, sigmasq):
    """Fuse point arithmetic and optional 2D cross-fiber sum."""
    if (registration_kernel.triton is not None and F.is_cuda and q.is_cuda
            and F.dtype == q.dtype and F.dtype in (torch.float64, torch.float32)
            and F.is_contiguous() and q.is_contiguous()
            and 1 <= F.shape[1] <= 64):
        if not strict_enabled():
            return registration_kernel.pairwise_probability_sums(F, q, sigmasq)
        return registration_kernel.pairwise_probabilities(F, q, sigmasq).sum(-1)
    points = float(F.shape[1])
    dist_forward = _squared_point_distances(F, q).sum(-1) / points
    dist_reverse = _squared_point_distances(F, q.flip(1)).sum(-1) / points
    torch.minimum(dist_forward, dist_reverse, out=dist_forward)
    return dist_forward.neg_().div_(sigmasq).exp_().sum(-1)


def objective_value(fixed, moving, transform, mode, sigma, device=None,
                    gpu_float32=None):
    """Convenience: transform the moving brain then compute the entropy,
    equivalent to RegisterTractography.objective_function."""

    moved = _transform_fiber_tensor(
        moving, transform, mode, device, gpu_float32=gpu_float32)
    return inner_loop_objective(
        fixed, moved, sigma * sigma, device=device, gpu_float32=gpu_float32)


class AffineObjective:
    """Device data shared by all evaluations of one optimizer invocation.

    Create a fresh instance when the sampled fixed/moving fibers change.
    No source NumPy arrays are retained or modified.
    """

    def __init__(self, fixed, moving, device=None, gpu_float32=None):
        _memory.disable_tf32()
        self.device = _resolve_device(device)
        self.dtype = torch.float32 if float32_enabled(gpu_float32) else torch.float64
        self.fixed = _prepare_fixed(fixed, self.device, self.dtype)
        self.moving = torch.as_tensor(moving, dtype=self.dtype, device=self.device)
        if (registration_kernel.triton is not None and self.moving.is_cuda
                and self.dtype in (torch.float64, torch.float32)):
            self.moving = self.moving.contiguous()
            self.host_matrix = torch.empty((4, 4), dtype=self.dtype, pin_memory=True)
            self.matrix = torch.empty((4, 4), dtype=self.dtype, device=self.device)
            self.transformed = torch.empty((moving.shape[1], moving.shape[2], 3),
                                           dtype=self.dtype, device=self.device)
        self.workspace = (_ObjectiveWorkspace(self.fixed, moving.shape[1])
                          if self.fixed.device.type in ('cuda', 'mps') else None)

    def __call__(self, transform, mode, sigma):
        if (registration_kernel.triton is not None and self.moving.is_cuda
                and self.dtype in (torch.float64, torch.float32)):
            matrix = transform_matrix_from_vtk(transform, mode)
            # The preceding call returned a host scalar, so the optimizer can
            # safely reuse these buffers for its next sequential evaluation.
            self.host_matrix.copy_(torch.from_numpy(matrix))
            self.matrix.copy_(self.host_matrix, non_blocking=True)
            registration_kernel.affine_transform(self.moving, self.matrix, self.transformed)
            moved = self.transformed
        else:
            moved = _transform_fiber_tensor(
                self.moving, transform, mode, self.device,
                gpu_float32=(self.dtype == torch.float32))
            moved = moved.permute(1, 2, 0).contiguous()
        return _inner_loop_objective(self.fixed, moved, sigma * sigma, self.device, workspace=self.workspace)
