"""Optional fused float64/float32 registration distances and probability sums."""

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _affine_points(Moving, Matrix, Output, COUNT, BLOCK: tl.constexpr,
                       DTYPE: tl.constexpr = tl.float64):
        point = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        count_i64 = tl.zeros((), dtype=tl.int64) + COUNT
        mask = point < count_i64
        zero = tl.full((), 0., DTYPE)
        x = tl.load(Moving + point, mask, other=zero)
        y = tl.load(Moving + count_i64 + point, mask, other=zero)
        z = tl.load(Moving + 2 * count_i64 + point, mask, other=zero)
        for axis in tl.static_range(3):
            value = (tl.load(Matrix + 4 * axis) * x + tl.load(Matrix + 4 * axis + 1) * y)
            value = value + tl.load(Matrix + 4 * axis + 2) * z
            value = value + tl.load(Matrix + 4 * axis + 3)
            tl.store(Output + 3 * point + axis, value, mask)

    @triton.jit
    def _squared_distances(F, M, Forward, Reverse, NF,
                           P: tl.constexpr, BLOCK: tl.constexpr,
                           DTYPE: tl.constexpr = tl.float64):
        tiles = tl.cdiv(NF * P, BLOCK)
        moving = (tl.program_id(0) // tiles).to(tl.int64)
        offsets = ((tl.program_id(0) % tiles) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        nf_i64 = tl.zeros((), dtype=tl.int64) + NF
        mask = offsets < nf_i64 * P
        point = offsets % P
        fixed = offsets * 3
        direct = (moving * P + point) * 3
        reverse = (moving * P + P - 1 - point) * 3
        zero = tl.full((), 0., DTYPE)
        x = tl.load(F + fixed, mask, other=zero)
        y = tl.load(F + fixed + 1, mask, other=zero)
        z = tl.load(F + fixed + 2, mask, other=zero)
        dx = x - tl.load(M + direct, mask, other=zero)
        dy = y - tl.load(M + direct + 1, mask, other=zero)
        dz = z - tl.load(M + direct + 2, mask, other=zero)
        rx = x - tl.load(M + reverse, mask, other=zero)
        ry = y - tl.load(M + reverse + 1, mask, other=zero)
        rz = z - tl.load(M + reverse + 2, mask, other=zero)
        output = moving * nf_i64 * P + offsets
        tl.store(Forward + output, (dx * dx + dy * dy) + dz * dz, mask)
        tl.store(Reverse + output, (rx * rx + ry * ry) + rz * rz, mask)

    @triton.jit
    def _point_distance(F, M, fixed, moving, point, mask,
                        P: tl.constexpr, REVERSE: tl.constexpr,
                        DTYPE: tl.constexpr = tl.float64):
        p_i64 = point.to(tl.int64)
        fi = (fixed.to(tl.int64) * P + p_i64) * 3
        order = P - 1 - p_i64 if REVERSE else p_i64
        mi = (moving.to(tl.int64) * P + order) * 3
        zero = tl.full((), 0., DTYPE)
        dx = tl.load(F + fi, mask, other=zero) - tl.load(M + mi, mask, other=zero)
        dy = tl.load(F + fi + 1, mask, other=zero) - tl.load(M + mi + 1, mask, other=zero)
        dz = tl.load(F + fi + 2, mask, other=zero) - tl.load(M + mi + 2, mask, other=zero)
        return (dx * dx + dy * dy) + dz * dz

    @triton.jit
    def _mean_squared(F, M, Forward, Reverse, NF, COUNT,
                      P: tl.constexpr, WIDTH: tl.constexpr, STEPS: tl.constexpr,
                      CHUNKS: tl.constexpr, BLOCK: tl.constexpr,
                      PROBABILITY: tl.constexpr, INV_SIGMA: tl.constexpr,
                      DTYPE: tl.constexpr = tl.float64):
        pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        lane = tl.arange(0, WIDTH).to(tl.int64)
        nf_i64 = tl.zeros((), dtype=tl.int64) + NF
        fi = pair[:, None] % nf_i64
        mi = pair[:, None] // nf_i64
        point = lane[None, :]
        mask = pair[:, None] < COUNT
        # Match Reduce.cuh's lane accumulation then ascending shuffle tree.
        forward = _point_distance(F, M, fi, mi, point, mask, P, False, DTYPE)
        reverse = _point_distance(F, M, fi, mi, point, mask, P, True, DTYPE)
        for chunk in tl.static_range(1, CHUNKS):
            p_chunk = point + chunk * WIDTH
            mask2 = mask & (p_chunk < P)
            forward += _point_distance(F, M, fi, mi, p_chunk, mask2, P, False, DTYPE)
            reverse += _point_distance(F, M, fi, mi, p_chunk, mask2, P, True, DTYPE)
        for step in tl.static_range(0, STEPS):
            af, bf = tl.split(tl.reshape(forward, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            ar, br = tl.split(tl.reshape(reverse, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            forward = af + bf
            reverse = ar + br
        inv = tl.full((), 1.0 / P, DTYPE)
        forward = tl.reshape(forward, (BLOCK,)) * inv
        reverse = tl.reshape(reverse, (BLOCK,)) * inv
        if PROBABILITY:
            distance = tl.minimum(forward, reverse, propagate_nan=tl.PropagateNan.ALL)
            values = libdevice.exp(-distance * tl.full((), INV_SIGMA, DTYPE))
            tl.store(Forward + pair, values, pair < COUNT)
        else:
            tl.store(Forward + pair, forward, pair < COUNT)
            tl.store(Reverse + pair, reverse, pair < COUNT)

    @triton.jit
    def _pairwise_probability_sums(F, M, Out, NF, NM, N_SPLITS,
                                   P: tl.constexpr, SPLIT_F: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr,
                                   SINGLE_SPLIT: tl.constexpr,
                                   INV_SIGMA: tl.constexpr, DTYPE: tl.constexpr = tl.float64):
        m_base = tl.program_id(0).to(tl.int64) * BM
        split_id = tl.program_id(1)
        f_start = split_id * SPLIT_F
        f_end = tl.where(f_start + SPLIT_F < NF, f_start + SPLIT_F, NF)

        m_idx = m_base + tl.arange(0, BM).to(tl.int64)
        valid_m = m_idx < NM
        zero = tl.full((), 0., DTYPE)
        inv = tl.full((), 1.0 / P, DTYPE)
        inv_sigma = tl.full((), INV_SIGMA, DTYPE)
        row_acc = tl.zeros((BM,), dtype=DTYPE)

        for f_base in range(f_start, f_end, BN):
            f_idx = (f_base + tl.arange(0, BN)).to(tl.int64)
            valid_f = f_idx < f_end
            fwd_acc = tl.zeros((BM, BN), dtype=DTYPE)
            rev_acc = tl.zeros((BM, BN), dtype=DTYPE)
            for p in range(P):
                f_ptrs = F + (f_idx * P + p) * 3
                m_ptrs = M + (m_idx * P + p) * 3
                mr_ptrs = M + (m_idx * P + (P - 1 - p)) * 3

                fx = tl.load(f_ptrs, valid_f, other=zero)[None, :]
                fy = tl.load(f_ptrs + 1, valid_f, other=zero)[None, :]
                fz = tl.load(f_ptrs + 2, valid_f, other=zero)[None, :]

                mx = tl.load(m_ptrs, valid_m, other=zero)[:, None]
                my = tl.load(m_ptrs + 1, valid_m, other=zero)[:, None]
                mz = tl.load(m_ptrs + 2, valid_m, other=zero)[:, None]

                dx = fx - mx
                dy = fy - my
                dz = fz - mz
                fwd_acc += (dx * dx + dy * dy) + dz * dz

                rx = tl.load(mr_ptrs, valid_m, other=zero)[:, None]
                ry = tl.load(mr_ptrs + 1, valid_m, other=zero)[:, None]
                rz = tl.load(mr_ptrs + 2, valid_m, other=zero)[:, None]

                rdx = fx - rx
                rdy = fy - ry
                rdz = fz - rz
                rev_acc += (rdx * rdx + rdy * rdy) + rdz * rdz

            d = tl.minimum(fwd_acc * inv, rev_acc * inv, propagate_nan=tl.PropagateNan.ALL)
            prob = libdevice.exp(-d * inv_sigma)
            prob = tl.where(valid_m[:, None] & valid_f[None, :], prob, zero)
            row_acc += tl.sum(prob, axis=1)

        if SINGLE_SPLIT:
            tl.store(Out + m_idx, row_acc, valid_m)
        else:
            n_splits_i64 = tl.zeros((), dtype=tl.int64) + N_SPLITS
            tl.store(Out + m_idx * n_splits_i64 + split_id.to(tl.int64), row_acc, valid_m)

    @triton.jit
    def _pairwise_probability_sums_and_weighted_fixed(
        F, M, OutSum, OutWeighted, NF, NM, N_SPLITS,
        P: tl.constexpr, SPLIT_F: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
        SINGLE_SPLIT: tl.constexpr,
        INV_SIGMA: tl.constexpr, DTYPE: tl.constexpr = tl.float64,
    ):
        m_base = tl.program_id(0).to(tl.int64) * BM
        split_id = tl.program_id(1)
        f_start = split_id * SPLIT_F
        f_end = tl.where(f_start + SPLIT_F < NF, f_start + SPLIT_F, NF)

        m_idx = m_base + tl.arange(0, BM).to(tl.int64)
        valid_m = m_idx < NM
        zero = tl.full((), 0., DTYPE)
        inv = tl.full((), 1.0 / P, DTYPE)
        inv_sigma = tl.full((), INV_SIGMA, DTYPE)
        n_splits_i64 = tl.zeros((), dtype=tl.int64) + N_SPLITS
        row_acc = tl.zeros((BM,), dtype=DTYPE)
        first_tile = True

        for f_base in range(f_start, f_end, BN):
            f_idx = (f_base + tl.arange(0, BN)).to(tl.int64)
            valid_f = f_idx < f_end
            fwd_acc = tl.zeros((BM, BN), dtype=DTYPE)
            rev_acc = tl.zeros((BM, BN), dtype=DTYPE)
            for p in range(P):
                f_ptrs = F + (f_idx * P + p) * 3
                m_ptrs = M + (m_idx * P + p) * 3
                mr_ptrs = M + (m_idx * P + (P - 1 - p)) * 3

                fx = tl.load(f_ptrs, valid_f, other=zero)[None, :]
                fy = tl.load(f_ptrs + 1, valid_f, other=zero)[None, :]
                fz = tl.load(f_ptrs + 2, valid_f, other=zero)[None, :]

                mx = tl.load(m_ptrs, valid_m, other=zero)[:, None]
                my = tl.load(m_ptrs + 1, valid_m, other=zero)[:, None]
                mz = tl.load(m_ptrs + 2, valid_m, other=zero)[:, None]

                dx = fx - mx
                dy = fy - my
                dz = fz - mz
                fwd_acc += (dx * dx + dy * dy) + dz * dz

                rx = tl.load(mr_ptrs, valid_m, other=zero)[:, None]
                ry = tl.load(mr_ptrs + 1, valid_m, other=zero)[:, None]
                rz = tl.load(mr_ptrs + 2, valid_m, other=zero)[:, None]

                rdx = fx - rx
                rdy = fy - ry
                rdz = fz - rz
                rev_acc += (rdx * rdx + rdy * rdy) + rdz * rdz

            fwd_mean = fwd_acc * inv
            rev_mean = rev_acc * inv
            mask_fwd = fwd_mean <= rev_mean
            d = tl.where(mask_fwd, fwd_mean, rev_mean)
            prob = libdevice.exp(-d * inv_sigma)
            pair_valid = valid_m[:, None] & valid_f[None, :]
            prob = tl.where(pair_valid, prob, zero)
            row_acc += tl.sum(prob, axis=1)

            p_fwd = tl.where(mask_fwd, prob, zero)
            p_rev = prob - p_fwd
            for p in range(P):
                f_ptrs = F + (f_idx * P + p) * 3
                fr_ptrs = F + (f_idx * P + (P - 1 - p)) * 3
                fx_f = tl.load(f_ptrs, valid_f, other=zero)[None, :]
                fy_f = tl.load(f_ptrs + 1, valid_f, other=zero)[None, :]
                fz_f = tl.load(f_ptrs + 2, valid_f, other=zero)[None, :]
                fx_r = tl.load(fr_ptrs, valid_f, other=zero)[None, :]
                fy_r = tl.load(fr_ptrs + 1, valid_f, other=zero)[None, :]
                fz_r = tl.load(fr_ptrs + 2, valid_f, other=zero)[None, :]

                wx = tl.sum(p_fwd * fx_f + p_rev * fx_r, axis=1)
                wy = tl.sum(p_fwd * fy_f + p_rev * fy_r, axis=1)
                wz = tl.sum(p_fwd * fz_f + p_rev * fz_r, axis=1)

                if SINGLE_SPLIT:
                    w_ptrs = OutWeighted + (m_idx * P + p) * 3
                else:
                    w_ptrs = OutWeighted + ((m_idx * n_splits_i64 + split_id.to(tl.int64)) * P + p) * 3
                if first_tile:
                    tl.store(w_ptrs + 0, wx, valid_m)
                    tl.store(w_ptrs + 1, wy, valid_m)
                    tl.store(w_ptrs + 2, wz, valid_m)
                else:
                    px = tl.load(w_ptrs + 0, valid_m, other=zero)
                    py = tl.load(w_ptrs + 1, valid_m, other=zero)
                    pz = tl.load(w_ptrs + 2, valid_m, other=zero)
                    tl.store(w_ptrs + 0, px + wx, valid_m)
                    tl.store(w_ptrs + 1, py + wy, valid_m)
                    tl.store(w_ptrs + 2, pz + wz, valid_m)
            first_tile = False

        if SINGLE_SPLIT:
            tl.store(OutSum + m_idx, row_acc, valid_m)
        else:
            tl.store(OutSum + m_idx * n_splits_i64 + split_id.to(tl.int64), row_acc, valid_m)


def _triton_dtype(tensor):
    import torch
    return tl.float32 if tensor.dtype == torch.float32 else tl.float64


def _point_reduction_layout(points):
    if points <= 32:
        steps = points.bit_length() - 1
        width = 1 << steps
    else:
        steps = 5
        width = 32
    chunks = (points + width - 1) // width
    return width, steps, chunks


def affine_transform(moving, matrix, output):
    """Write contiguous (N, P, 3) points from (3, N, P) coordinates in one launch."""
    import torch

    count = moving.shape[1] * moving.shape[2]
    if count == 0:
        return
    with torch.cuda.device(moving.device):
        _affine_points[(triton.cdiv(count, 128),)](
            moving, matrix, output, count, BLOCK=128,
            DTYPE=_triton_dtype(moving), enable_fp_fusion=False)


def squared_distances(fixed, moving):
    """Return forward/reverse (Nm, Nf, P) squared point distances.

    Inputs are contiguous float64/float32 CUDA tensors with layout (N, P, 3).
    Disable FMA to preserve the separate multiply/add rounding of PyTorch.
    """
    import torch

    shape = (moving.shape[0], fixed.shape[0], fixed.shape[1])
    forward = torch.empty(shape, dtype=fixed.dtype, device=fixed.device)
    reverse = torch.empty_like(forward)
    if forward.numel() == 0:
        return forward, reverse
    grid = (triton.cdiv(shape[1] * shape[2], 256) * shape[0],)
    with torch.cuda.device(fixed.device):
        _squared_distances[grid](fixed, moving, forward, reverse, shape[1],
                                 P=shape[2], BLOCK=256,
                                 DTYPE=_triton_dtype(fixed), enable_fp_fusion=False)
    return forward, reverse


def mean_squared_distances(fixed, moving):
    """Fuse point reduction for 1<=P<=64, matching PyTorch's CUDA sum order."""
    import torch

    points = fixed.shape[1]
    if points > 64 or points < 1:
        forward, reverse = squared_distances(fixed, moving)
        return forward.sum(-1) / float(points), reverse.sum(-1) / float(points)
    shape = (moving.shape[0], fixed.shape[0])
    forward = torch.empty(shape, dtype=fixed.dtype, device=fixed.device)
    reverse = torch.empty_like(forward)
    count = forward.numel()
    if count == 0:
        return forward, reverse
    width, steps, chunks = _point_reduction_layout(points)
    with torch.cuda.device(fixed.device):
        _mean_squared[(triton.cdiv(count, 32),)](
            fixed, moving, forward, reverse, shape[1], count,
            P=points, WIDTH=width, STEPS=steps, CHUNKS=chunks, BLOCK=32,
            PROBABILITY=False, INV_SIGMA=1., DTYPE=_triton_dtype(fixed),
            enable_fp_fusion=False)
    return forward, reverse


def pairwise_probabilities(fixed, moving, sigmasq):
    """For 1<=P<=64, fuse distance, minimum and exp; PyTorch sums over fibers."""
    import torch

    points = fixed.shape[1]
    output = torch.empty((moving.shape[0], fixed.shape[0]),
                         dtype=fixed.dtype, device=fixed.device)
    if output.numel() == 0:
        return output
    width, steps, chunks = _point_reduction_layout(points)
    with torch.cuda.device(fixed.device):
        # One warp keeps exp lanes occupied after the point reduction.
        _mean_squared[(triton.cdiv(output.numel(), 32),)](
            fixed, moving, output, output, fixed.shape[0], output.numel(),
            P=points, WIDTH=width, STEPS=steps, CHUNKS=chunks, BLOCK=32,
            PROBABILITY=True, INV_SIGMA=1. / sigmasq, DTYPE=_triton_dtype(fixed),
            num_warps=1, enable_fp_fusion=False)
    return output


def pairwise_probability_sums(fixed, moving, sigmasq):
    """2D-tiled fused cross-fiber probability row sums (Nm,) for 1<=P<=64."""
    import torch

    nf = fixed.shape[0]
    nm = moving.shape[0]
    if nm == 0:
        return torch.empty((0,), dtype=fixed.dtype, device=fixed.device)
    if nf == 0:
        return torch.zeros((nm,), dtype=fixed.dtype, device=fixed.device)
    points = fixed.shape[1]
    bm = 16
    bn = 64
    split_f = 512
    n_splits = triton.cdiv(nf, split_f)
    single_split = (n_splits == 1)
    out = (torch.empty((nm,), dtype=fixed.dtype, device=fixed.device)
           if single_split
           else torch.empty((nm, n_splits), dtype=fixed.dtype, device=fixed.device))
    with torch.cuda.device(fixed.device):
        _pairwise_probability_sums[(triton.cdiv(nm, bm), n_splits)](
            fixed, moving, out, nf, nm, n_splits,
            P=points, SPLIT_F=split_f, BM=bm, BN=bn,
            SINGLE_SPLIT=single_split,
            INV_SIGMA=1. / sigmasq, DTYPE=_triton_dtype(fixed),
            num_warps=4, enable_fp_fusion=False)
    return out if single_split else out.sum(-1)


def pairwise_probability_sums_and_weighted_fixed(fixed, moving, sigmasq):
    """2D-tiled fused forward/backward reduction returning ``(sum_p, weighted_F)``.

    Avoids materializing an ``O(Nm * Nf)`` probability matrix by streaming tiles
    in registers and reducing across ``F`` splits in ``O(Nm * N_splits * P)``.
    """
    import torch

    nf = fixed.shape[0]
    nm = moving.shape[0]
    points = fixed.shape[1]
    if nm == 0:
        return (torch.empty((0,), dtype=fixed.dtype, device=fixed.device),
                torch.empty((0, points, 3), dtype=fixed.dtype, device=fixed.device))
    if nf == 0:
        return (torch.zeros((nm,), dtype=fixed.dtype, device=fixed.device),
                torch.zeros((nm, points, 3), dtype=fixed.dtype, device=fixed.device))
    bm = 16
    bn = 64
    split_f = 512
    n_splits = triton.cdiv(nf, split_f)
    single_split = (n_splits == 1)
    if single_split:
        out_sum = torch.empty((nm,), dtype=fixed.dtype, device=fixed.device)
        out_weighted = torch.empty((nm, points, 3), dtype=fixed.dtype, device=fixed.device)
    else:
        out_sum = torch.empty((nm, n_splits), dtype=fixed.dtype, device=fixed.device)
        out_weighted = torch.empty((nm, n_splits, points, 3), dtype=fixed.dtype, device=fixed.device)
    with torch.cuda.device(fixed.device):
        _pairwise_probability_sums_and_weighted_fixed[(triton.cdiv(nm, bm), n_splits)](
            fixed, moving, out_sum, out_weighted, nf, nm, n_splits,
            P=points, SPLIT_F=split_f, BM=bm, BN=bn,
            SINGLE_SPLIT=single_split,
            INV_SIGMA=1. / sigmasq, DTYPE=_triton_dtype(fixed),
            num_warps=4, enable_fp_fusion=False)
    if single_split:
        return out_sum, out_weighted
    return out_sum.sum(-1), out_weighted.sum(1)
