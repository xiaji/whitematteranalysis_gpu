"""Normalize large similarity blocks without materializing a weight matrix."""
from .registration_kernel import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _normalize(B, A, C, Output, COUNT, N,
                   AS: tl.constexpr, CS: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offset < COUNT
        n_i64 = tl.zeros((), dtype=tl.int64) + N
        row, column = offset // n_i64, offset % n_i64
        values = tl.load(B + offset, mask, other=0.)
        a = tl.load(A + row * AS, mask, other=0.)
        c = tl.load(C + column * CS, mask, other=0.)
        # Retain the original multiplication grouping and separate rounding.
        tl.store(Output + offset, values * (a * c), mask)


def normalized_similarities(B, a, c, in_place=False, return_col_sums=False):
    """Return B * (a[:, None] * c[None, :]), preserving input values/layout.

    When in_place=True, overwrite B in place without allocating an output copy.
    When return_col_sums=True, also return column sums out.sum(0) as (out, col_sums).
    """
    import torch

    if (triton is None or not B.is_cuda or B.dtype not in (torch.float64, torch.float32)
            or B.numel() < 1 << 20
            or not (B.is_contiguous() or B.t().is_contiguous())):
        if in_place:
            out = B.mul_(a[:, None] * c[None, :])
        else:
            out = B * (a[:, None] * c[None, :])
        return (out, out.sum(0)) if return_col_sums else out
    transposed = not B.is_contiguous()
    if transposed:
        B, a, c = B.t(), c, a
    output = B if in_place else torch.empty_like(B)
    with torch.cuda.device(B.device):
        _normalize[(triton.cdiv(B.numel(), 256),)](
            B, a, c, output, B.numel(), B.shape[1],
            AS=a.stride(0), CS=c.stride(0), BLOCK=256,
            enable_fp_fusion=False)
    out = output.t() if transposed else output
    return (out, out.sum(0)) if return_col_sums else out
