"""Optional float64 distance kernels for short fibers and embedding vectors."""
from . import registration_kernel

triton = registration_kernel.triton
if triton is not None:
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _values(F, Q, fi, qi, p, mask, P: tl.constexpr,
                REVERSE: tl.constexpr, THRESHOLD2: tl.constexpr, METHOD: tl.constexpr,
                BILATERAL: tl.constexpr = False, INV_SIGMA: tl.constexpr = 0.):
        fixed = (fi * P + p) * 3
        query = (qi * P + (P - 1 - p if REVERSE else p)) * 3
        x = tl.load(F + fixed, mask, other=0.)
        qx = tl.load(Q + query, mask, other=0.)
        dx = x - qx
        dy = tl.load(F + fixed + 1, mask, other=0.) - tl.load(Q + query + 1, mask, other=0.)
        dz = tl.load(F + fixed + 2, mask, other=0.) - tl.load(Q + query + 2, mask, other=0.)
        dy2, dz2 = dy * dy, dz * dz
        d = (dx * dx + dy2) + dz2
        reflected = d
        if BILATERAL:
            rx = x + qx
            reflected = (rx * rx + dy2) + dz2
        if THRESHOLD2 != 0:
            d = d - tl.full((), THRESHOLD2, tl.float64)
            reflected = reflected - tl.full((), THRESHOLD2, tl.float64)
            d = tl.where(d < 0., 0., d)
            reflected = tl.where(reflected < 0., 0., reflected)
        if METHOD == 'Mean':
            d = libdevice.sqrt_rn(d)
            reflected = libdevice.sqrt_rn(reflected)
        elif METHOD == 'StrictSimilarity':
            inv = tl.full((), INV_SIGMA, tl.float64)
            d = libdevice.exp(-d * inv)
            reflected = libdevice.exp(-reflected * inv)
        return d, reflected

    @triton.jit
    def _reduce_orientations(a, b, P: tl.constexpr, WIDTH: tl.constexpr,
                              STEPS: tl.constexpr, METHOD: tl.constexpr, BLOCK: tl.constexpr):
        for step in tl.static_range(STEPS):
            al, ar = tl.split(tl.reshape(a, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            bl, br = tl.split(tl.reshape(b, (BLOCK, WIDTH // (1 << (step + 1)), 2)))
            if METHOD == 'Hausdorff':
                a, b = tl.maximum(al, ar, propagate_nan=tl.PropagateNan.ALL), tl.maximum(bl, br, propagate_nan=tl.PropagateNan.ALL)
            elif METHOD == 'StrictSimilarity':
                a, b = al * ar, bl * br
            else:
                a, b = al + ar, bl + br
        a, b = tl.reshape(a, (BLOCK,)), tl.reshape(b, (BLOCK,))
        if METHOD == 'Mean' or METHOD == 'MeanSquared':
            inv = tl.full((), 1. / P, tl.float64)
            a, b = a * inv, b * inv
        if METHOD == 'Mean':
            a, b = a * a, b * b
        if METHOD == 'StrictSimilarity':
            return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)
        return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _distances(F, Q, Output, NF, COUNT,
                   P: tl.constexpr, WIDTH: tl.constexpr, STEPS: tl.constexpr,
                   THRESHOLD2: tl.constexpr, METHOD: tl.constexpr, BLOCK: tl.constexpr,
                   BILATERAL: tl.constexpr = False, INV_SIGMA: tl.constexpr = 0.,
                   GAUSSIAN: tl.constexpr = False, INV_GAUSSIAN: tl.constexpr = 0.):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        p = tl.arange(0, WIDTH)[None, :]
        fi, qi = pair[:, None] % NF, pair[:, None] // NF
        mask = pair[:, None] < COUNT
        a, ra = _values(F, Q, fi, qi, p, mask, P, False, THRESHOLD2, METHOD, BILATERAL, INV_SIGMA)
        b, rb = _values(F, Q, fi, qi, p, mask, P, True, THRESHOLD2, METHOD, BILATERAL, INV_SIGMA)
        if P > WIDTH:
            mask2 = mask & (p + WIDTH < P)
            a2, ra2 = _values(F, Q, fi, qi, p + WIDTH, mask2, P, False, THRESHOLD2, METHOD, BILATERAL, INV_SIGMA)
            b2, rb2 = _values(F, Q, fi, qi, p + WIDTH, mask2, P, True, THRESHOLD2, METHOD, BILATERAL, INV_SIGMA)
            if METHOD == 'Hausdorff':
                a, b = tl.maximum(a, a2, propagate_nan=tl.PropagateNan.ALL), tl.maximum(b, b2, propagate_nan=tl.PropagateNan.ALL)
            elif METHOD == 'StrictSimilarity':
                a, b = a * a2, b * b2
            else:
                a, b = a + a2, b + b2
            if BILATERAL:
                if METHOD == 'Hausdorff':
                    ra, rb = tl.maximum(ra, ra2, propagate_nan=tl.PropagateNan.ALL), tl.maximum(rb, rb2, propagate_nan=tl.PropagateNan.ALL)
                elif METHOD == 'StrictSimilarity':
                    ra, rb = ra * ra2, rb * rb2
                else:
                    ra, rb = ra + ra2, rb + rb2
        result = _reduce_orientations(a, b, P, WIDTH, STEPS, METHOD, BLOCK)
        if BILATERAL:
            reflected = _reduce_orientations(ra, rb, P, WIDTH, STEPS, METHOD, BLOCK)
            if METHOD == 'StrictSimilarity':
                result = tl.maximum(result, reflected, propagate_nan=tl.PropagateNan.ALL)
            else:
                result = tl.minimum(result, reflected, propagate_nan=tl.PropagateNan.ALL)
        if GAUSSIAN:
            result = libdevice.exp(-result * tl.full((), INV_GAUSSIAN, tl.float64))
        tl.store(Output + pair, result, pair < COUNT)

    @triton.jit
    def _centroid_distances(Q, C, Output, NC, COUNT,
                            D: tl.constexpr, WIDTH: tl.constexpr,
                            STEPS: tl.constexpr, BLOCK: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        dimension = tl.arange(0, WIDTH)[None, :]
        mask = (pair[:, None] < COUNT) & (dimension < D)
        query = tl.load(Q + (pair[:, None] // NC) * D + dimension, mask, other=0.)
        centroid = tl.load(C + (pair[:, None] % NC) * D + dimension, mask, other=0.)
        diff = query - centroid
        values = diff * diff
        # Match cdist's descending warp-shuffle reduction, including sqrt rounding.
        for step in tl.static_range(STEPS):
            a, b = tl.split(tl.trans(tl.reshape(values,
                (BLOCK, 2, WIDTH // (1 << (step + 1)))), 0, 2, 1))
            values = a + b
        result = libdevice.sqrt_rn(tl.reshape(values, (BLOCK,)))
        tl.store(Output + pair, result, pair < COUNT)

    @triton.jit
    def _argmin_combine(val_a, idx_a, val_b, idx_b):
        a_nan = val_a != val_a
        b_nan = val_b != val_b
        a_wins = tl.where(
            a_nan | b_nan,
            tl.where(a_nan & b_nan, idx_a < idx_b, a_nan),
            (val_a < val_b) | ((val_a == val_b) & (idx_a < idx_b)),
        )
        return tl.where(a_wins, val_a, val_b), tl.where(a_wins, idx_a, idx_b)

    @triton.jit
    def _nearest_centroids(Q, C, MinDist, BestIdx, NQ, NC,
                           D: tl.constexpr, WIDTH: tl.constexpr,
                           STEPS: tl.constexpr, BLOCK_C: tl.constexpr):
        qi = tl.program_id(0).to(tl.int64)
        dim = tl.arange(0, WIDTH)
        d_mask = dim < D
        q_row = tl.load(Q + qi * D + dim, d_mask, other=0.)
        inf = tl.full((), float('inf'), tl.float64)
        nc_i64 = tl.zeros((), dtype=tl.int64) + NC
        best_dist = inf
        best_idx = nc_i64
        for base in range(0, NC, BLOCK_C):
            c_idx = base + tl.arange(0, BLOCK_C)
            c_valid = c_idx < NC
            mask = c_valid[:, None] & d_mask[None, :]
            c_vals = tl.load(C + c_idx[:, None].to(tl.int64) * D + dim[None, :], mask, other=0.)
            diff = q_row[None, :] - c_vals
            values = diff * diff
            # Match cdist's descending warp-shuffle reduction, including sqrt rounding.
            for step in tl.static_range(STEPS):
                a, b = tl.split(tl.trans(tl.reshape(values,
                    (BLOCK_C, 2, WIDTH // (1 << (step + 1)))), 0, 2, 1))
                values = a + b
            dists = libdevice.sqrt_rn(tl.reshape(values, (BLOCK_C,)))
            dists = tl.where(c_valid, dists, inf)
            idxs = tl.where(c_valid, c_idx.to(tl.int64), nc_i64)
            block_dist, block_idx = tl.reduce((dists, idxs), axis=0, combine_fn=_argmin_combine)
            best_dist, best_idx = _argmin_combine(best_dist, best_idx, block_dist, block_idx)
        tl.store(MinDist + qi, best_dist, qi < NQ)
        tl.store(BestIdx + qi, best_idx, qi < NQ)

    @triton.jit
    def _similarity_row_sums(F, Q, SimOut, DistOut, NF, NQ,
                             P: tl.constexpr, WIDTH: tl.constexpr, STEPS: tl.constexpr,
                             THRESHOLD2: tl.constexpr, BLOCK_N: tl.constexpr,
                             BILATERAL: tl.constexpr = False, INV_SIGMA: tl.constexpr = 0.,
                             COMPUTE_SQRT: tl.constexpr = False):
        qi = tl.program_id(0)
        p = tl.arange(0, WIDTH)[None, :]
        qi_col = tl.zeros((BLOCK_N, 1), dtype=tl.int32) + qi
        zero = tl.full((), 0., tl.float64)
        inv_sigma = tl.full((), INV_SIGMA, tl.float64)
        sim_acc = zero
        dist_acc = zero
        for base in range(0, NF, BLOCK_N):
            fi_vec = base + tl.arange(0, BLOCK_N)
            valid = fi_vec < NF
            fi = fi_vec[:, None]
            mask = valid[:, None]
            a, ra = _values(F, Q, fi, qi_col, p, mask, P, False, THRESHOLD2, 'Mean', BILATERAL)
            b, rb = _values(F, Q, fi, qi_col, p, mask, P, True, THRESHOLD2, 'Mean', BILATERAL)
            if P > WIDTH:
                mask2 = mask & (p + WIDTH < P)
                a2, ra2 = _values(F, Q, fi, qi_col, p + WIDTH, mask2, P, False, THRESHOLD2, 'Mean', BILATERAL)
                b2, rb2 = _values(F, Q, fi, qi_col, p + WIDTH, mask2, P, True, THRESHOLD2, 'Mean', BILATERAL)
                a, b = a + a2, b + b2
                if BILATERAL:
                    ra, rb = ra + ra2, rb + rb2
            # _values already took per-point sqrt; using 'MeanSquared' here reduces
            # and divides by P without squaring, giving unsquared mean_dist directly.
            mean_dist = _reduce_orientations(a, b, P, WIDTH, STEPS, 'MeanSquared', BLOCK_N)
            if BILATERAL:
                reflected = _reduce_orientations(ra, rb, P, WIDTH, STEPS, 'MeanSquared', BLOCK_N)
                mean_dist = tl.minimum(mean_dist, reflected, propagate_nan=tl.PropagateNan.ALL)
            d_sq = mean_dist * mean_dist
            sim = libdevice.exp(-d_sq * inv_sigma)
            sim_acc += tl.sum(tl.where(valid, sim, zero), axis=0)
            if COMPUTE_SQRT:
                dist_acc += tl.sum(tl.where(valid, mean_dist, zero), axis=0)
        tl.store(SimOut + qi, sim_acc, qi < NQ)
        if COMPUTE_SQRT:
            tl.store(DistOut + qi, dist_acc, qi < NQ)


def centroid_distances(queries, centroids):
    """Direct Euclidean distances for contiguous float64 CUDA vectors, 1<=D<=32."""
    import torch

    output = torch.empty((len(queries), len(centroids)), dtype=queries.dtype, device=queries.device)
    dimensions = queries.shape[1]
    block = 64 if dimensions <= 16 else 32
    if output.numel():
        with torch.cuda.device(queries.device):
            _centroid_distances[(triton.cdiv(output.numel(), block),)](
                queries, centroids, output, len(centroids), output.numel(),
                D=dimensions, WIDTH=triton.next_power_of_2(dimensions),
                STEPS=(dimensions - 1).bit_length(), BLOCK=block,
                num_warps=1, enable_fp_fusion=False)
    return output


def nearest_centroids(queries, centroids):
    """Fused Euclidean argmin for contiguous float64 CUDA vectors, 1<=D<=32."""
    import torch

    nq = queries.shape[0]
    nc = centroids.shape[0]
    if nc == 0:
        raise IndexError('centroids must be non-empty')
    min_distances = torch.empty((nq,), dtype=queries.dtype, device=queries.device)
    best_indices = torch.empty((nq,), dtype=torch.int64, device=queries.device)
    if nq == 0:
        return min_distances, best_indices
    dimensions = queries.shape[1]
    block_c = 64 if dimensions <= 16 else 32
    with torch.cuda.device(queries.device):
        _nearest_centroids[(nq,)](
            queries, centroids, min_distances, best_indices, nq, nc,
            D=dimensions, WIDTH=triton.next_power_of_2(dimensions),
            STEPS=(dimensions - 1).bit_length(), BLOCK_C=block_c,
            num_warps=4, enable_fp_fusion=False)
    return min_distances, best_indices


def fiber_distances(queries, fixed, threshold, method, bilateral=False, sigmasq=6400.,
                    similarity_sigmasq=None):
    import torch

    shape = (queries.shape[0], fixed.shape[0])
    output = torch.empty(shape, dtype=fixed.dtype, device=fixed.device)
    if not output.numel():
        return output
    points = fixed.shape[1]
    steps = points.bit_length() - 1
    block = 64 if points <= 16 else 32
    with torch.cuda.device(fixed.device):
        _distances[(triton.cdiv(output.numel(), block),)](
            fixed, queries, output, shape[1], output.numel(),
            P=points, WIDTH=1 << steps, STEPS=steps, THRESHOLD2=threshold * threshold,
            METHOD=method, BLOCK=block, BILATERAL=bilateral,
            INV_SIGMA=1. / sigmasq if method == 'StrictSimilarity' else 0.,
            GAUSSIAN=similarity_sigmasq is not None,
            INV_GAUSSIAN=1. / similarity_sigmasq if similarity_sigmasq is not None else 0.,
            num_warps=1, enable_fp_fusion=False)
    return output


def similarity_row_sums(queries, fixed, threshold, sigmasq, bilateral=False,
                        sqrt_distance_sums=False):
    """Fused float64 Mean-distance similarity (and optional sqrt-distance) row sums."""
    import torch

    nq = queries.shape[0]
    nf = fixed.shape[0]
    if nq == 0:
        sim_sums = torch.empty((0,), dtype=fixed.dtype, device=fixed.device)
        dist_sums = (torch.empty((0,), dtype=fixed.dtype, device=fixed.device)
                     if sqrt_distance_sums else None)
        return sim_sums, dist_sums
    if nf == 0:
        sim_sums = torch.zeros((nq,), dtype=fixed.dtype, device=fixed.device)
        dist_sums = (torch.zeros((nq,), dtype=fixed.dtype, device=fixed.device)
                     if sqrt_distance_sums else None)
        return sim_sums, dist_sums

    sim_sums = torch.empty((nq,), dtype=fixed.dtype, device=fixed.device)
    dist_sums = (torch.empty((nq,), dtype=fixed.dtype, device=fixed.device)
                 if sqrt_distance_sums else None)
    points = fixed.shape[1]
    steps = points.bit_length() - 1
    block_n = 64
    with torch.cuda.device(fixed.device):
        _similarity_row_sums[(nq,)](
            fixed, queries, sim_sums, dist_sums if sqrt_distance_sums else sim_sums,
            nf, nq,
            P=points, WIDTH=1 << steps, STEPS=steps,
            THRESHOLD2=threshold * threshold, BLOCK_N=block_n,
            BILATERAL=bilateral, INV_SIGMA=1. / sigmasq,
            COMPUTE_SQRT=sqrt_distance_sums,
            num_warps=4, enable_fp_fusion=False)
    return sim_sums, dist_sums
