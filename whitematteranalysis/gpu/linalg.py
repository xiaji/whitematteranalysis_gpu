# -*- coding: utf-8 -*-

""" gpu.linalg

GPU (torch) versions of the linear algebra steps that surround the
distance matrices in spectral clustering:

  * the Nystrom row-sum estimates and spectral embedding GEMMs
    (cluster.spectral and cluster.spectral_atlas_label), and
  * (optional, off by default) numpy.linalg.eigh on the GPU.

The GEMM results differ from CPU BLAS only in floating point
summation order (~1e-15 relative in float64), which validation shows
is far below the level that affects cluster labels.

"""

import os

import numpy

from .config import get_device, config
from . import embedding_kernel
from .._memory import SmallPages

try:
    import torch
except ImportError:
    torch = None


def eigh_enabled():
    """GPU eigendecomposition is opt-in (WMA_GPU_EIGH=1): eigh feeds the
    spectral embedding directly, so it stays on the CPU LAPACK path by
    default to keep atlas creation as close to the CPU results as
    possible."""

    return os.environ.get('WMA_GPU_EIGH', '0') in ('1', 'true', 'yes', 'on')


def eigh(M):
    """eigh for a symmetric (m, m) float64 matrix.

    Returns (e_val, e_vec) as float64 numpy arrays. Uses LAPACK unless
    the WMA_GPU_EIGH flag is set and a GPU is available.
    """

    if eigh_enabled() and config.cuda_available():
        dev = get_device()
        Mt = torch.as_tensor(M, dtype=torch.float64, device=dev)
        e_val, e_vec = torch.linalg.eigh(Mt)
        return e_val.cpu().numpy(), e_vec.cpu().numpy()
    return numpy.linalg.eigh(M)


def _as_gpu(np_array, dev):
    return torch.as_tensor(numpy.ascontiguousarray(np_array),
                           dtype=torch.float64, device=dev)


def nystrom_row_sum_2(B, row_sum_matrix):
    """GPU computation of cluster.spectral's B-part row sum estimate:

        row_sum_2 = B.sum(0) + row_sum_matrix @ B

    B: (m, n) float64. Returns float64 numpy (n,).
    """

    dev = get_device()
    Bg = _as_gpu(B, dev)
    rm = _as_gpu(row_sum_matrix, dev)
    return (Bg.sum(0) + rm @ Bg).cpu().numpy()


def nystrom_embed_B(B, e_vec, e_val):
    """GPU computation of cluster.spectral's Nystrom embedding block:

        V2 = (B.T @ e_vec) @ diag(1/e_val)

    B: (m, n) float64. Returns V2 as float64 numpy (n, m).
    """

    dev = get_device()
    Bg = _as_gpu(B, dev)
    Eg = _as_gpu(e_vec, dev)
    recip = numpy.divide(1.0, e_val)
    V2 = (Bg.t() @ Eg) * _as_gpu(recip, dev)[None, :]
    return V2.cpu().numpy()


def atlas_label_embedding(B, atlas):
    """GPU equivalent of cluster.spectral_atlas_label steps 2-5.

    B: (m, n) float64 similarity matrix (rows = atlas sample, columns =
    input fibers), as produced by _rectangular_similarity_matrix. May be
    a numpy array (uploaded here) or a float torch CUDA tensor already
    on the device (the fused path - avoids a round trip over PCIe).
    Returns (embed, cluster_idx, dist, row_sum, B_col_sums) where
    row_sum and B_col_sums feed _format_output_polydata.
    """

    embed, rows, columns, labels, distances = _atlas_embedding_block(B, atlas, _prepare_atlas(atlas))
    return embed, labels, distances, numpy.concatenate((atlas.row_sum_1, rows)), columns


def _prepare_atlas(atlas):
    dev = get_device()
    count = atlas.number_of_eigenvectors + 1
    if not numpy.isfinite(atlas.centroids).all():
        raise ValueError('array must not contain infs or NaNs')
    return tuple(_as_gpu(a, dev) for a in (
        atlas.row_sum_matrix, atlas.e_vec[:, -count:],
        numpy.divide(1., atlas.e_val[-count:]), atlas.e_vec_norm[-count:],
        numpy.sqrt(numpy.divide(1., atlas.row_sum_1)), atlas.centroids))


def _nearest_centroids(embed, centroids):
    from . import distance_kernel

    if (distance_kernel.triton is not None and embed.is_cuda
            and embed.dtype == centroids.dtype == torch.float64
            and centroids.ndim == 2 and centroids.shape[0] > 0
            and embed.shape[1] == centroids.shape[1]
            and 1 <= embed.shape[1] <= 32
            and embed.is_contiguous() and centroids.is_contiguous()):
        return distance_kernel.nearest_centroids(embed, centroids)
    distances = torch.cdist(embed, centroids, compute_mode='donot_use_mm_for_euclid_dist')
    return distances.min(1)


def _atlas_embedding_block_gpu(B, prepared, in_place=False):
    """Normalize, embed and label one block entirely on the target device."""
    dev = get_device()
    rm, Eg, recip, norm, dhat_a, centroids = prepared
    Bg = B.to(device=dev, dtype=torch.float64) if torch.is_tensor(B) else _as_gpu(B, dev)
    row_sum_2 = Bg.sum(0) + rm @ Bg
    dhat_b = torch.sqrt(torch.reciprocal(torch.where(row_sum_2 < 0, .1, row_sum_2)))
    Bg, b_col_sums = embedding_kernel.normalized_similarities(
        Bg, dhat_a, dhat_b, in_place=in_place, return_col_sums=True)
    V = ((Bg.t() @ Eg) * recip[None, :]) / norm[None, :]
    embed = V[:, :-1].flip(1) / V[:, -1:]
    del Bg
    # Direct differences avoid cancellation from the squared-norm GEMM formula.
    # min chooses the first centroid on a tie, matching scipy.cluster.vq.vq.
    distances, labels = _nearest_centroids(embed, centroids)
    return embed, row_sum_2, b_col_sums, labels, distances


def _unpack_atlas_results(embed, row_sum_2, b_col_sums, labels, distances):
    """One device-to-host transfer for compact results, with host validation."""
    host = torch.cat((embed, row_sum_2[:, None], b_col_sums[:, None],
                      labels[:, None].to(torch.float64), distances[:, None]), dim=1).cpu().numpy()
    rows = host[:, -4].copy()
    if numpy.any(rows <= 0):
        print(f"<{__name__}> Warning: negative row sum approximations:",
              numpy.count_nonzero(rows <= 0))
        rows[rows < 0] = 0.1
    if not numpy.isfinite(host[:, :-4]).all():
        raise ValueError('array must not contain infs or NaNs')
    return (host[:, :-4].copy(), rows, host[:, -3].copy(),
            host[:, -2].astype(numpy.int32), host[:, -1].copy())


def _atlas_embedding_block(B, atlas, prepared):
    """Normalize, embed and label on CUDA; copy only compact results to host."""
    return _unpack_atlas_results(*_atlas_embedding_block_gpu(B, prepared, in_place=False))


@SmallPages()
def atlas_label_embedding_blocks(blocks, atlas, number_fibers):
    """Embed each input block without retaining the full similarity or V matrices."""
    dev = get_device()
    prepared = _prepare_atlas(atlas)
    embed_g = torch.empty((number_fibers, atlas.number_of_eigenvectors), dtype=torch.float64, device=dev)
    rows_g = torch.empty(number_fibers, dtype=torch.float64, device=dev)
    columns_g = torch.empty(number_fibers, dtype=torch.float64, device=dev)
    labels_g = torch.empty(number_fibers, dtype=torch.int64, device=dev)
    distances_g = torch.empty(number_fibers, dtype=torch.float64, device=dev)
    for start, block in blocks:
        end = start + block.shape[1]
        e, r, b, idx, dist = _atlas_embedding_block_gpu(block, prepared, in_place=True)
        embed_g[start:end], rows_g[start:end], columns_g[start:end] = e, r, b
        labels_g[start:end], distances_g[start:end] = idx, dist
        del block
    embed, rows, columns, labels, distances = _unpack_atlas_results(
        embed_g, rows_g, columns_g, labels_g, distances_g)
    return embed, labels, distances, numpy.concatenate((atlas.row_sum_1, rows)), columns
