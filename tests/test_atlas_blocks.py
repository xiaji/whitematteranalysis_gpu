"""Streamed atlas similarities and embeddings against the previous full matrix."""
import importlib
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma
from whitematteranalysis.gpu import distance, linalg
from test_output_pipeline import baseline_module


def polydata(count):
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.random.RandomState(count).normal(size=(count * 7, 3)), deep=True))
    lines = vtk.vtkCellArray()
    for i in range(count):
        lines.InsertNextCell(7)
        for j in range(7):
            lines.InsertCellPoint(i * 7 + j)
    pd.SetPoints(points)
    pd.SetLines(lines)
    return pd


@unittest.skipUnless(wma.gpu.available(), 'CUDA is required')
class AtlasTests(unittest.TestCase):
    def test_negative_rows_and_input_preservation(self):
        import torch
        original = baseline_module('whitematteranalysis.gpu._packedbaseline',
                                   'gpu/linalg.py', 'cffed63')
        rng = np.random.RandomState(43)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=-2. * np.ones(11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        similarities = torch.full((11, 7), .5, dtype=torch.float64, device='cuda')
        before = similarities.clone()
        expected = original.atlas_label_embedding(similarities, atlas)
        actual = linalg.atlas_label_embedding(similarities, atlas)
        self.assertTrue(torch.equal(similarities, before))
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[3][11:], .1)
        atlas.centroids = np.stack((actual[0][0], actual[0][0], actual[0][0] + 1.))
        tied = linalg.atlas_label_embedding(similarities, atlas)
        self.assertEqual(tied[1][0], 0)
        self.assertEqual(tied[2][0], 0.)
        atlas.centroids[0, 0] = np.nan
        for implementation in (original, linalg):
            with self.assertRaises(ValueError):
                implementation.atlas_label_embedding(similarities, atlas)
        atlas.centroids[0, 0] = 0.
        atlas.row_sum_matrix[:] = -1.
        for implementation in (original, linalg):
            with self.assertRaises(ValueError):
                implementation.atlas_label_embedding(similarities, atlas)

    def test_streamed_results(self):
        import torch
        original = baseline_module('whitematteranalysis.gpu._oldlinalg', 'gpu/linalg.py')
        # linalg uses .config, rather than the top-level package.
        rng = np.random.RandomState(42)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        queries, fixed = polydata(13), polydata(11)
        for method in ('Mean', 'MeanSquared', 'Hausdorff', 'StrictSimilarity'):
            for bilateral in (False, True):
                for fp32 in (False, True):
                    with self.subTest(method=method, bilateral=bilateral, fp32=fp32):
                        args = (queries, fixed, .1, 20., method, bilateral)
                        full = distance.rectangular_similarity_tensor(*args, gpu_float32=fp32, tile_rows=4)
                        blocks = list(distance.iter_rectangular_similarity(*args, gpu_float32=fp32, tile_rows=4))
                        self.assertEqual([i for i, _ in blocks], [0, 4, 8, 12])
                        self.assertTrue(torch.equal(torch.cat([b for _, b in blocks], dim=1), full))
                        expected = original.atlas_label_embedding(full, atlas)
                        actual = linalg.atlas_label_embedding_blocks(iter(blocks), atlas, 13)
                        for a, b in zip(expected, actual):
                            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                        np.testing.assert_array_equal(expected[1], actual[1])
        empty = linalg.atlas_label_embedding_blocks(iter(()), atlas, 0)
        self.assertEqual(empty[0].shape, (0, 3))
        np.testing.assert_array_equal(empty[3], atlas.row_sum_1)

    def test_out_of_memory_retries(self):
        import torch
        original = distance._similarity_tensor_block
        sizes = []
        def limited(queries, *args):
            sizes.append(len(queries))
            if len(queries) > 2:
                raise torch.cuda.OutOfMemoryError('tile')
            return original(queries, *args)
        args = (polydata(7), polydata(3), 0., 20.)
        with patch.object(distance, '_similarity_tensor_block', side_effect=limited):
            blocks = list(distance.iter_rectangular_similarity(*args, tile_rows=7))
        self.assertEqual(sizes[:3], [7, 3, 1])
        self.assertEqual(sum(b.shape[1] for _, b in blocks), 7)
        with patch.object(distance, '_similarity_tensor_block', side_effect=torch.cuda.OutOfMemoryError):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                list(distance.iter_rectangular_similarity(*args, tile_rows=1))

    def test_atlas_label_embedding_blocks_single_d2h_transfer(self):
        import torch
        rng = np.random.RandomState(44)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        full = torch.rand((11, 15), dtype=torch.float64, device='cuda')
        expected = linalg.atlas_label_embedding(full.clone(), atlas)
        blocks = [(s, full[:, s:s + 4].clone()) for s in range(0, 15, 4)]
        cpu_calls = 0
        orig_cpu = torch.Tensor.cpu
        def counting_cpu(tensor, *args, **kwargs):
            nonlocal cpu_calls
            cpu_calls += 1
            return orig_cpu(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'cpu', counting_cpu):
            actual = linalg.atlas_label_embedding_blocks(iter(blocks), atlas, 15)
        self.assertEqual(cpu_calls, 1)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(expected[1], actual[1])


class AtlasBlocksCpuTests(unittest.TestCase):
    def test_atlas_label_embedding_blocks_single_d2h_transfer_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        rng = np.random.RandomState(44)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        full = torch.rand((11, 15), dtype=torch.float64)
        cpu_dev = torch.device('cpu')
        with patch.object(linalg, 'get_device', return_value=cpu_dev):
            expected = linalg.atlas_label_embedding(full.clone(), atlas)
            blocks = [(s, full[:, s:s + 4].clone()) for s in range(0, 15, 4)]
            cpu_calls = 0
            orig_cpu = torch.Tensor.cpu
            def counting_cpu(tensor, *args, **kwargs):
                nonlocal cpu_calls
                cpu_calls += 1
                return orig_cpu(tensor, *args, **kwargs)
            with patch.object(torch.Tensor, 'cpu', counting_cpu):
                actual = linalg.atlas_label_embedding_blocks(iter(blocks), atlas, 15)
            self.assertEqual(cpu_calls, 1)
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(expected[1], actual[1])
            # Verify _atlas_embedding_block_gpu exists and respects in_place=False default
            prepared = linalg._prepare_atlas(atlas)
            b_copy = full.clone()
            before = b_copy.clone()
            res = linalg._atlas_embedding_block_gpu(b_copy, prepared)
            self.assertEqual(len(res), 5)
            self.assertTrue(torch.equal(b_copy, before))
            # Verify empty blocks
            empty = linalg.atlas_label_embedding_blocks(iter(()), atlas, 0)
            self.assertEqual(empty[0].shape, (0, 3))
            np.testing.assert_array_equal(empty[3], atlas.row_sum_1)

    def test_float32_and_mps_never_construct_float64_on_device(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        rng = np.random.RandomState(44)
        atlas = types.SimpleNamespace(number_of_eigenvectors=3,
            row_sum_1=np.ones(11), row_sum_matrix=rng.uniform(0, .1, 11),
            e_vec=rng.uniform(.1, 1, (11, 11)), e_val=np.arange(1., 12.),
            e_vec_norm=np.ones(11), centroids=rng.uniform(0, 2, (5, 3)))
        full64 = torch.rand((11, 15), dtype=torch.float64)
        with patch.object(linalg, 'get_device', return_value=torch.device('cpu')):
            ref = linalg.atlas_label_embedding(full64.clone(), atlas)

        full32 = full64.to(torch.float32)
        orig_empty = torch.empty
        orig_as_tensor = torch.as_tensor
        orig_to = torch.Tensor.to

        def guard_empty(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                raise AssertionError('torch.empty(float64) called in float32 mode')
            return orig_empty(*args, **kwargs)

        def guard_as_tensor(data, *args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                raise AssertionError('torch.as_tensor(float64) called in float32 mode')
            return orig_as_tensor(data, *args, **kwargs)

        def guard_to(tensor, *args, **kwargs):
            if kwargs.get('dtype') == torch.float64 or torch.float64 in args:
                raise AssertionError('Tensor.to(float64) called in float32 mode')
            return orig_to(tensor, *args, **kwargs)

        blocks = [(s, full32[:, s:s + 4].clone()) for s in range(0, 15, 4)]
        with patch.object(linalg, 'get_device', return_value=torch.device('cpu')), \
                patch.object(torch, 'empty', side_effect=guard_empty), \
                patch.object(torch, 'as_tensor', side_effect=guard_as_tensor), \
                patch.object(torch.Tensor, 'to', guard_to):
            actual = linalg.atlas_label_embedding_blocks(
                iter(blocks), atlas, 15, gpu_float32=True)

        self.assertEqual(actual[0].dtype, np.float32)
        self.assertEqual(actual[1].dtype, np.int32)
        self.assertEqual(actual[2].dtype, np.float32)
        self.assertEqual(actual[4].dtype, np.float32)
        np.testing.assert_allclose(actual[0], ref[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(actual[1], ref[1])
        np.testing.assert_allclose(actual[2], ref[2], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual[3], ref[3], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual[4], ref[4], rtol=1e-5, atol=1e-5)

        mps_dev = torch.device('mps')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_blocks = [(s, full32[:, s:s + 4].to(mps_dev)) for s in range(0, 15, 4)]
            with patch.object(linalg, 'get_device', return_value=mps_dev), \
                    patch.object(torch, 'empty', side_effect=guard_empty), \
                    patch.object(torch, 'as_tensor', side_effect=guard_as_tensor), \
                    patch.object(torch.Tensor, 'to', guard_to):
                mps_actual = linalg.atlas_label_embedding_blocks(
                    iter(mps_blocks), atlas, 15, gpu_float32=True)
            np.testing.assert_allclose(mps_actual[0], ref[0], rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(mps_actual[1], ref[1])
        else:
            # Verify mocked mps device path also never constructs float64 tensors
            mock_blocks = [(s, full32[:, s:s + 4].clone()) for s in range(0, 15, 4)]
            with patch.object(linalg, 'get_device', return_value=mps_dev), \
                    patch.object(torch, 'empty', side_effect=lambda *a, **kw: (
                        (_ for _ in ()).throw(AssertionError('float64 on mps')) if kw.get('dtype') == torch.float64
                        else orig_empty(*a, **{**kw, 'device': torch.device('cpu')}))), \
                    patch.object(torch, 'as_tensor', side_effect=lambda d, *a, **kw: (
                        (_ for _ in ()).throw(AssertionError('float64 on mps')) if kw.get('dtype') == torch.float64
                        else orig_as_tensor(d, *a, **{**kw, 'device': torch.device('cpu')}))), \
                    patch.object(torch.Tensor, 'to', lambda t, *a, **kw: (
                        (_ for _ in ()).throw(AssertionError('float64 on mps')) if kw.get('dtype') == torch.float64
                        else orig_to(t, *a, **{k: (torch.device('cpu') if v == mps_dev else v) for k, v in kw.items()}))):
                mock_actual = linalg.atlas_label_embedding_blocks(
                    iter(mock_blocks), atlas, 15, gpu_float32=True)
            np.testing.assert_allclose(mock_actual[0], ref[0], rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(mock_actual[1], ref[1])

    def test_torch_absent_imports_all_modules_and_runs_spectral(self):
        script = """
import importlib, pkgutil, sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

sys.modules['torch'] = None
import whitematteranalysis as wma

mods = [m.name for m in pkgutil.walk_packages(wma.__path__, 'whitematteranalysis.')]
assert len(mods) >= 33, f'Unexpected module count: {len(mods)}'
for mod in sorted(mods):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        if mod == 'whitematteranalysis.tract_measurement' and exc.name == 'xlrd':
            continue
        raise

assert wma.gpu.linalg.torch is None
evals, _ = wma.gpu.linalg.eigh(np.diag([2.0, 3.0]))
np.testing.assert_allclose(evals, [2.0, 3.0])

rng = np.random.RandomState(7)
n_fibers = 110
pts = vtk.vtkPoints()
lines = vtk.vtkCellArray()
all_coords = []
for i in range(n_fibers):
    cx = -30.0 if i < 55 else 30.0
    ys = np.linspace(-20.0, 20.0, 7)
    coords = np.column_stack([np.full(7, cx), ys, np.zeros(7)]) + rng.normal(scale=0.3, size=(1, 3))
    start = len(all_coords)
    all_coords.extend(coords)
    ids = vtk.vtkIdList()
    for pid in range(start, start + 7):
        ids.InsertNextId(pid)
    lines.InsertNextCell(ids)
pts.SetData(numpy_to_vtk(np.asarray(all_coords, dtype=np.float64), deep=True))
pd = vtk.vtkPolyData()
pd.SetPoints(pts)
pd.SetLines(lines)

mask = np.ones(n_fibers, dtype=bool)
mask[102:] = False
out_pd, cidx, _, _, _, atlas, _ = wma.cluster.spectral(
    pd, number_of_clusters=2, number_of_eigenvectors=3, sigma=30.0,
    threshold=0.0, number_of_jobs=1, use_nystrom=True, nystrom_mask=mask,
    distance_method='Mean', outlier_std_threshold=np.inf, bilateral=False, use_gpu=False)
assert out_pd.GetNumberOfLines() == n_fibers
assert len(cidx) == n_fibers
assert atlas.centroids.shape == (2, 3)
"""
        proc = subprocess.run(
            [sys.executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)

        with patch.object(linalg, 'torch', None):
            evals, evecs = linalg.eigh(np.diag([4.0, 1.0]))
            np.testing.assert_allclose(evals, [1.0, 4.0])
            with self.assertRaisesRegex(RuntimeError, 'torch is not available'):
                linalg.nystrom_row_sum_2(np.ones((2, 2)), np.ones(2))
            with self.assertRaisesRegex(RuntimeError, 'torch is not available'):
                linalg.nystrom_embed_B(np.ones((2, 2)), evecs, evals)

    def test_mps_float32_nystrom_and_eigh_avoid_float64(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        gpu_config = importlib.import_module('whitematteranalysis.gpu.config')

        saved = {
            'requested': gpu_config.config.requested,
            'float32': gpu_config.config.float32,
            'allow_mps': gpu_config.config.allow_mps,
        }
        rng = np.random.RandomState(123)
        B = rng.uniform(0.1, 1.0, (8, 5))
        rm = rng.uniform(0.0, 0.2, 8)
        A = rng.normal(size=(8, 8))
        A = 0.5 * (A + A.T)
        ref_evals, ref_evecs = np.linalg.eigh(A)
        ref_row_sum = B.sum(0) + rm @ B
        ref_V2 = (B.T @ ref_evecs) * np.divide(1.0, ref_evals)[None, :]

        mps_dev = torch.device('mps')
        has_real_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        orig_as_tensor = torch.as_tensor

        def guard_as_tensor(data, *args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                raise AssertionError('torch.as_tensor(float64) requested on MPS/float32')
            if not has_real_mps and kwargs.get('device') == mps_dev:
                kwargs = {**kwargs, 'device': torch.device('cpu')}
            return orig_as_tensor(data, *args, **kwargs)

        try:
            with patch.dict(os.environ, {
                'WMA_GPU': 'auto',
                'WMA_GPU_MPS': '1',
                'WMA_GPU_FLOAT32': '1',
                'WMA_GPU_EIGH': '1',
            }):
                gpu_config._from_env()
            with patch.object(gpu_config.config, 'cuda_available', return_value=False), \
                    patch.object(gpu_config.config, 'mps_available', return_value=True), \
                    patch.object(torch, 'as_tensor', side_effect=guard_as_tensor), \
                    patch.object(torch.linalg, 'eigh',
                                 side_effect=AssertionError('torch.linalg.eigh must not run on MPS')):
                self.assertTrue(gpu_config.use_gpu())
                self.assertEqual(gpu_config.get_device(), mps_dev)
                rs = linalg.nystrom_row_sum_2(B, rm)
                V2 = linalg.nystrom_embed_B(B, ref_evecs, ref_evals)
                evals, evecs = linalg.eigh(A)
            np.testing.assert_allclose(rs, ref_row_sum, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(V2, ref_V2, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(evals, ref_evals)
            np.testing.assert_array_equal(evecs, ref_evecs)

            # Even if cuda_available is mocked True, dev.type == 'mps' must fall back to numpy.linalg.eigh
            with patch.object(gpu_config.config, 'cuda_available', return_value=True), \
                    patch.object(linalg, 'get_device', return_value=mps_dev), \
                    patch.object(torch, 'as_tensor', side_effect=guard_as_tensor), \
                    patch.object(torch.linalg, 'eigh',
                                 side_effect=AssertionError('torch.linalg.eigh must not run on MPS')):
                evals2, evecs2 = linalg.eigh(A)
            np.testing.assert_array_equal(evals2, ref_evals)
            np.testing.assert_array_equal(evecs2, ref_evecs)
        finally:
            for k, v in saved.items():
                setattr(gpu_config.config, k, v)
