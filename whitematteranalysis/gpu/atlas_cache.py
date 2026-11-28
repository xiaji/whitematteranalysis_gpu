"""Reusable atlas coordinates and outlier probabilities; never cache subjects."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np

from . import fiber_tensors
from .config import float32_enabled, get_device


@lru_cache(maxsize=4)
def _implementation(backend, float32, device_index, device_type='cuda'):
    files = ('fibers.py', 'filter.py', 'io.py', 'similarity.py', 'gpu/_memory.py',
             'gpu/fiber_tensors.py', 'gpu/cpu_backend.py', 'gpu/distance.py',
             'gpu/distance_kernel.py', 'gpu/registration_kernel.py', 'gpu/atlas_cache.py')
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in files:
        digest.update((root / name).read_bytes())
    np_core = getattr(np, '_core', np.core)
    features = sorted(name for name, enabled in np_core._multiarray_umath.__cpu_features__.items() if enabled)
    versions = [sys.platform, platform.machine(), platform.libc_ver(), features,
                np.__version__, backend, float32, digest.hexdigest()]
    if backend == 'gpu':
        import torch
        from . import distance_kernel
        if device_type == 'cuda':
            versions += [torch.__version__, torch.version.cuda,
                         getattr(distance_kernel.triton, '__version__', None),
                         torch.cuda.get_device_name(device_index),
                         torch.cuda.get_device_capability(device_index)]
        else:
            versions += [torch.__version__, device_type]
    else:
        import scipy
        versions.append(scipy.__version__)
    return versions


def _cache_file(filename, sigmasq, method, bilateral, backend, device=None):
    directory = os.environ.get('WMA_ATLAS_CACHE')
    if directory and directory.lower() in ('off', '0', 'false'):
        return None
    if not directory:
        base = os.environ.get('LOCALAPPDATA') if os.name == 'nt' else os.environ.get('XDG_CACHE_HOME')
        directory = str(Path(base or Path.home() / '.cache') / 'whitematteranalysis' / 'atlas-outliers')
    source = Path(filename).resolve()
    stat = source.stat()
    if backend == 'gpu':
        dev = device if device is not None else get_device()
        dev_type = dev.type if hasattr(dev, 'type') else str(dev).split(':', 1)[0]
        dev_index = dev.index if (hasattr(dev, 'index') and dev_type == 'cuda') else None
    else:
        dev_type = 'cpu'
        dev_index = None
    identity = [str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                15, float(sigmasq), str(method), bool(bilateral),
                _implementation(backend, float32_enabled() if backend == 'gpu' else False,
                                dev_index, dev_type)]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(directory).expanduser() / (key + '.npz')


def load_reference(filename, sigmasq, method='Mean', bilateral=True,
                   backend='gpu', tile_bytes=None, device=None):
    """Return (N,15,3) atlas coordinates and exact leave-one-out probabilities.

    Keys include reference identity, numerical parameters, precision, backend,
    library versions and implementation. Cache writes are atomic; an unavailable
    or damaged optional cache falls back to computing the reference normally.
    """
    from .. import io, similarity

    probability_dtype = np.dtype(np.float32 if backend == 'gpu' and float32_enabled() else np.float64)
    dev = (device if device is not None else get_device()) if backend == 'gpu' else None
    path = _cache_file(filename, sigmasq, method, bilateral, backend, device=dev)
    if path is not None:
        try:
            with path.open('rb') as source, np.load(source, allow_pickle=False) as saved:
                coordinates, probabilities = saved['coordinates'], saved['probabilities']
            if (coordinates.ndim == 3 and coordinates.shape[1:] == (15, 3)
                    and coordinates.dtype == np.float64
                    and probabilities.dtype == probability_dtype
                    and probabilities.shape == (len(coordinates),)):
                return coordinates, probabilities
        except (OSError, ValueError, TypeError, AttributeError, EOFError, KeyError, zipfile.BadZipFile):
            pass
    coordinates = fiber_tensors.polydata_to_numpy(io.read_polydata(str(filename), geometry_only=True), 15)
    probabilities = np.empty(0, dtype=probability_dtype)
    if len(coordinates):
        if backend == 'gpu':
            from .distance import similarity_row_sums_from_arrays
            tile_rows = (max(1, tile_bytes // (len(coordinates) * 15 * 8 * 12))
                         if tile_bytes is not None else None)
            similarity_sums = similarity_row_sums_from_arrays(
                coordinates, coordinates, 0., distance_method=method,
                bilateral=bilateral, sigmasq=sigmasq, tile_rows=tile_rows,
                device=dev)
            probabilities = ((similarity_sums - 1.) / len(coordinates)).astype(probability_dtype, copy=False)
        else:
            from .cpu_backend import _distance_matrix as matrix
            kwargs = dict(number_of_jobs=1)
            if tile_bytes is not None:
                kwargs['tile_rows'] = max(1, tile_bytes // (len(coordinates) * 15 * 8 * 12))
            distances = matrix(coordinates, coordinates, 0., distance_method=method,
                               bilateral=bilateral, sigmasq=sigmasq, **kwargs)
            similarities = similarity.distance_to_similarity(distances, sigmasq)
            probabilities = (np.sum(similarities, axis=1) - 1.) / len(coordinates)
    if path is not None:
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.npz', delete=False) as output:
                temporary = output.name
                np.savez(output, coordinates=coordinates, probabilities=probabilities)
            os.replace(temporary, str(path))
        except OSError:
            pass
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    return coordinates, probabilities


def _spectral_cache_file(atlas_p, atlas_vtp):
    directory = os.environ.get('WMA_ATLAS_CACHE')
    if directory and directory.lower() in ('off', '0', 'false'):
        return None
    if directory:
        cache_dir = Path(directory).expanduser() / 'spectral-atlas'
    else:
        base = os.environ.get('LOCALAPPDATA') if os.name == 'nt' else os.environ.get('XDG_CACHE_HOME')
        cache_dir = Path(base or Path.home() / '.cache') / 'whitematteranalysis' / 'spectral-atlas'
    p_src = Path(atlas_p).resolve()
    v_src = Path(atlas_vtp).resolve()
    p_stat = p_src.stat()
    v_stat = v_src.stat()
    identity = [
        'spectral_atlas_v1',
        str(p_src), p_stat.st_size, p_stat.st_mtime_ns, p_stat.st_ctime_ns,
        str(v_src), v_stat.st_size, v_stat.st_mtime_ns, v_stat.st_ctime_ns,
        15,
    ]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return cache_dir / (key + '.npz')


def load_spectral_atlas(directory, atlas_name, verbose=False):
    """Load a spectral ClusterAtlas with compact binary .npz caching for inference.

    Caches only the arrays needed by ``spectral_atlas_label`` (``nystrom_coords`` in
    ``(m, 15, 3)`` float64 layout, row-sum normalization vectors, and the top
    ``number_of_eigenvectors + 1`` eigenvector/eigenvalue columns) so subsequent
    invocations avoid unpickling the full ``atlas.p`` and reading/resampling
    ``atlas.vtp``. Falls back cleanly to ``ClusterAtlas.load`` on miss or corruption.
    """
    from .. import cluster

    if not os.path.isdir(directory):
        return cluster.ClusterAtlas().load(directory, atlas_name, verbose=verbose)

    fname_base = os.path.join(directory, atlas_name)
    atlas_p = fname_base + '.p'
    atlas_vtp = fname_base + '.vtp'
    if not (os.path.exists(atlas_p) and os.path.exists(atlas_vtp)):
        return cluster.ClusterAtlas().load(directory, atlas_name, verbose=verbose)

    path = _spectral_cache_file(atlas_p, atlas_vtp)
    if path is not None:
        try:
            with path.open('rb') as source, np.load(source, allow_pickle=False) as saved:
                nystrom_coords = np.asarray(saved['nystrom_coords'], dtype=np.float64)
                row_sum_1 = np.asarray(saved['row_sum_1'], dtype=np.float64)
                row_sum_matrix = np.asarray(saved['row_sum_matrix'], dtype=np.float64)
                e_vec = np.asarray(saved['e_vec'], dtype=np.float64)
                e_val = np.asarray(saved['e_val'], dtype=np.float64)
                e_vec_norm = np.asarray(saved['e_vec_norm'], dtype=np.float64)
                centroids = np.asarray(saved['centroids'], dtype=np.float64)
                meta = json.loads(str(saved['metadata'].item()))

            k = int(meta['number_of_eigenvectors'])
            count = int(meta['top_count'])
            m = int(nystrom_coords.shape[0])
            bilateral = meta['bilateral']
            if not isinstance(bilateral, bool):
                raise ValueError('Cached bilateral field must be bool')
            if (nystrom_coords.ndim != 3 or nystrom_coords.shape[1:] != (15, 3)
                    or row_sum_1.shape != (m,) or row_sum_matrix.shape != (m,)
                    or e_vec.shape != (m, count) or e_val.shape != (count,)
                    or e_vec_norm.shape != (count,) or centroids.ndim != 2
                    or centroids.shape[1] != k):
                raise ValueError('Invalid spectral atlas cache shapes')

            compact = cluster.ClusterAtlas()
            compact.row_sum_1 = row_sum_1
            compact.row_sum_matrix = row_sum_matrix
            compact.e_vec = e_vec
            compact.e_val = e_val
            compact.e_vec_norm = e_vec_norm
            compact.centroids = centroids
            compact.number_of_eigenvectors = k
            compact.sigma = float(meta['sigma'])
            compact.threshold = float(meta['threshold'])
            compact.bilateral = bilateral
            compact.distance_method = str(meta['distance_method'])
            compact.version = meta.get('version', 1.0)
            compact.nystrom_coords = nystrom_coords
            compact.nystrom_count = m
            compact.nystrom_polydata = None
            compact.polydata_filename = str(Path(atlas_vtp).resolve())
            return compact
        except (OSError, ValueError, TypeError, AttributeError, EOFError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            pass

    full = cluster.ClusterAtlas().load(directory, atlas_name, verbose=verbose)
    nystrom_coords = fiber_tensors.polydata_to_numpy(full.nystrom_polydata, 15)
    k = int(full.number_of_eigenvectors)
    count = min(k + 1, int(full.e_vec.shape[1]))

    full.e_vec = np.ascontiguousarray(full.e_vec[:, -count:], dtype=np.float64)
    full.e_val = np.ascontiguousarray(full.e_val[-count:], dtype=np.float64)
    full.e_vec_norm = np.ascontiguousarray(full.e_vec_norm[-count:], dtype=np.float64)
    full.row_sum_1 = np.ascontiguousarray(full.row_sum_1, dtype=np.float64)
    full.row_sum_matrix = np.ascontiguousarray(full.row_sum_matrix, dtype=np.float64)
    full.centroids = np.ascontiguousarray(full.centroids, dtype=np.float64)
    full.nystrom_coords = nystrom_coords
    full.nystrom_count = int(nystrom_coords.shape[0])
    full.pinv_A = None

    if path is not None:
        meta_json = json.dumps({
            'number_of_eigenvectors': k,
            'top_count': count,
            'sigma': float(full.sigma),
            'threshold': float(full.threshold),
            'bilateral': bool(full.bilateral),
            'distance_method': str(full.distance_method),
            'version': str(full.version) if not isinstance(full.version, (int, float)) else float(full.version),
        })
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.npz', delete=False) as output:
                temporary = output.name
                np.savez(
                    output,
                    nystrom_coords=nystrom_coords,
                    row_sum_1=full.row_sum_1,
                    row_sum_matrix=full.row_sum_matrix,
                    e_vec=full.e_vec,
                    e_val=full.e_val,
                    e_vec_norm=full.e_vec_norm,
                    centroids=full.centroids,
                    metadata=np.array(meta_json),
                )
            os.replace(temporary, str(path))
        except OSError:
            pass
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    if fiber_tensors.distance_supported(full.distance_method):
        full.nystrom_polydata = None
    return full
