"""Multi-GPU dispatch, deterministic ordering, and MPS stream-exclusion tests."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma
from whitematteranalysis import cluster, congeal_multisubject
from whitematteranalysis.gpu import atlas_cache, bspline, registration


OUTLIER_SCRIPT = Path(__file__).resolve().parents[1] / 'bin/wm_cluster_remove_outliers.py'
OUTLIER_SPEC = importlib.util.spec_from_file_location('outlier_multi_gpu_cli', OUTLIER_SCRIPT)
outlier_cli = importlib.util.module_from_spec(OUTLIER_SPEC)
OUTLIER_SPEC.loader.exec_module(outlier_cli)


def _cuda_device_count():
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _make_polydata(n_fibers, n_points=5, seed=0):
    rng = np.random.RandomState(seed)
    pd = vtk.vtkPolyData()
    pts = rng.normal(size=(n_fibers * n_points, 3)) * 20.0
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    if hasattr(cells, 'SetData'):
        offsets = numpy_to_vtk(
            np.arange(0, n_fibers * n_points + 1, n_points, dtype=np.int64),
            deep=True, array_type=vtk.VTK_ID_TYPE)
        conn = numpy_to_vtk(
            np.arange(n_fibers * n_points, dtype=np.int64),
            deep=True, array_type=vtk.VTK_ID_TYPE)
        cells.SetData(offsets, conn)
    else:
        conn = np.empty((n_fibers, n_points + 1), dtype=np.int64)
        conn[:, 0] = n_points
        conn[:, 1:] = np.arange(n_fibers * n_points, dtype=np.int64).reshape(n_fibers, n_points)
        cells.SetCells(n_fibers, numpy_to_vtk(conn.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE))
    pd.SetPoints(vtk_pts)
    pd.SetLines(cells)
    return pd


def _make_atlas(nystrom_count=11, n_eig=3, n_clusters=5, seed=42):
    rng = np.random.RandomState(seed)
    return types.SimpleNamespace(
        nystrom_polydata=_make_polydata(nystrom_count, n_points=5, seed=seed),
        threshold=0.0,
        sigma=20.0,
        distance_method='Mean',
        bilateral=False,
        number_of_eigenvectors=n_eig,
        row_sum_1=np.ones(nystrom_count, dtype=np.float64),
        row_sum_matrix=rng.uniform(0.0, 0.1, nystrom_count),
        e_vec=rng.uniform(0.1, 1.0, (nystrom_count, n_eig + 1)),
        e_val=np.arange(1.0, n_eig + 2.0),
        e_vec_norm=np.ones(n_eig + 1),
        centroids=rng.uniform(0.0, 2.0, (n_clusters, n_eig)),
    )


class MultiGpuDispatchTests(unittest.TestCase):
    def _require_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')
        return torch

    def test_outlier_cli_routes_round_robin_across_mocked_two_gpus_and_mps(self):
        torch = self._require_torch()
        saved_cfg = {name: getattr(wma.gpu.config, name)
                     for name in ('requested', 'float32', 'devices', 'device_index', 'allow_mps')}
        self.addCleanup(lambda: [setattr(wma.gpu.config, k, v) for k, v in saved_cfg.items()])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas_dir, input_dir, output_dir = (root / n for n in ('atlas', 'subject', 'output'))
            atlas_dir.mkdir()
            input_dir.mkdir()
            for idx in (1, 2, 3, 4):
                (atlas_dir / f'cluster_{idx:05d}.vtp').touch()
                (input_dir / f'cluster_{idx:05d}.vtp').touch()

            polydata = _make_polydata(3, n_points=3)
            arrays = np.zeros((3, 15, 3), dtype=np.float64)
            atlas_arrays = np.zeros((3, 15, 3), dtype=np.float64)
            seen_load_devices = []
            seen_sim_devices = []
            stream_devices = []
            lock = threading.Lock()

            def record_load(ca, *args, **kwargs):
                with lock:
                    seen_load_devices.append((Path(ca).name, kwargs.get('device')))
                return atlas_arrays, np.array([0.3, 0.5, 0.7])

            def record_sim(*args, **kwargs):
                with lock:
                    seen_sim_devices.append(kwargs.get('device'))
                return np.array([0.3, 1.5, 3.0]), np.array([3.0, 6.0, 9.0])

            def fake_stream(device=None):
                with lock:
                    stream_devices.append(device)
                return MagicMock()

            entered_cuda_devices = []

            def fake_cuda_device(dev):
                with lock:
                    entered_cuda_devices.append(dev)
                return contextlib.nullcontext()

            # 1. Mocked 2 CUDA GPUs (safe even on 1-GPU CUDA hosts)
            arguments = [
                str(OUTLIER_SCRIPT), str(input_dir), str(atlas_dir), str(output_dir),
                '-gpu', 'on', '-gpu_devices', '0,1', '-j', '2',
            ]
            with patch.object(sys, 'argv', arguments), \
                    patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'device', side_effect=fake_cuda_device), \
                    patch.object(torch.cuda, 'Stream', side_effect=fake_stream), \
                    patch.object(torch.cuda, 'stream', side_effect=lambda _s: contextlib.nullcontext()), \
                    patch.object(wma.cluster, 'load_atlas', return_value=types.SimpleNamespace(bilateral=False)), \
                    patch.object(atlas_cache, 'load_reference', side_effect=record_load), \
                    patch.object(wma.gpu.fiber_tensors, 'polydata_to_numpy', return_value=arrays), \
                    patch.object(wma.io, 'read_polydata', return_value=polydata), \
                    patch.object(wma.io, 'write_polydata'), \
                    patch.object(wma.gpu.distance, 'similarity_row_sums_from_arrays', side_effect=record_sim):
                outlier_cli.main()

            dev0, dev1 = torch.device('cuda', 0), torch.device('cuda', 1)
            by_cluster = dict(seen_load_devices)
            self.assertEqual(by_cluster['cluster_00001.vtp'], dev0)
            self.assertEqual(by_cluster['cluster_00002.vtp'], dev1)
            self.assertEqual(by_cluster['cluster_00003.vtp'], dev0)
            self.assertEqual(by_cluster['cluster_00004.vtp'], dev1)
            self.assertEqual(seen_sim_devices.count(dev0), 2)
            self.assertEqual(seen_sim_devices.count(dev1), 2)
            self.assertEqual(entered_cuda_devices.count(dev0), 2)
            self.assertEqual(entered_cuda_devices.count(dev1), 2)
            self.assertGreaterEqual(len(stream_devices), 2)
            log = (output_dir / 'subject_outlier_removed' / 'cluster_log.txt').read_text().splitlines()
            self.assertEqual([row.split()[0] for row in log[1:]], ['1', '2', '3', '4'])

            # 2. Mocked MPS with -j 2 must NOT create torch.cuda.Stream or enter torch.cuda.device
            seen_load_devices.clear()
            seen_sim_devices.clear()
            wma.gpu.config.devices = None
            wma.gpu.config.allow_mps = True
            mps_dev = torch.device('mps')
            arguments_mps = [
                str(OUTLIER_SCRIPT), str(input_dir), str(atlas_dir), str(output_dir),
                '-gpu', 'on', '-gpu_float32', '-j', '2',
            ]
            with patch.object(sys, 'argv', arguments_mps), \
                    patch.object(wma.gpu.config, 'cuda_available', return_value=False), \
                    patch.object(wma.gpu.config, 'mps_available', return_value=True), \
                    patch.object(torch.cuda, 'device',
                                 side_effect=AssertionError('must not enter CUDA device on MPS')), \
                    patch.object(torch.cuda, 'Stream',
                                 side_effect=AssertionError('must not create CUDA Stream on MPS')), \
                    patch.object(torch.cuda, 'stream',
                                 side_effect=AssertionError('must not enter CUDA stream on MPS')), \
                    patch.object(wma.cluster, 'load_atlas', return_value=types.SimpleNamespace(bilateral=False)), \
                    patch.object(atlas_cache, 'load_reference', side_effect=record_load), \
                    patch.object(wma.gpu.fiber_tensors, 'polydata_to_numpy', return_value=arrays), \
                    patch.object(wma.io, 'read_polydata', return_value=polydata), \
                    patch.object(wma.io, 'write_polydata'), \
                    patch.object(wma.gpu.distance, 'similarity_row_sums_from_arrays', side_effect=record_sim):
                outlier_cli.main()

            self.assertEqual(len(seen_load_devices), 4)
            self.assertTrue(all(dev == mps_dev for _, dev in seen_load_devices))
            self.assertTrue(all(dev == mps_dev for dev in seen_sim_devices))

    def test_spectral_atlas_label_shards_across_mocked_two_gpus(self):
        torch = self._require_torch()
        n_fibers = 16384
        atlas = _make_atlas(nystrom_count=11, n_eig=3, n_clusters=5)
        pd_1 = _make_polydata(n_fibers, n_points=5, seed=7)
        pd_2 = _make_polydata(n_fibers, n_points=5, seed=7)

        cpu_dev = torch.device('cpu')
        with patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                patch.object(wma.gpu, 'visible_device_count', return_value=1), \
                patch.object(wma.gpu, 'get_device', return_value=cpu_dev), \
                patch.object(wma.gpu.distance, 'get_device', return_value=cpu_dev), \
                patch.object(wma.gpu.linalg, 'get_device', return_value=cpu_dev):
            _, labels_1, _, embed_1 = cluster.spectral_atlas_label(pd_1, atlas, use_gpu=True)

        seen_indices = []
        seen_shard_sizes = []
        orig_blocks = wma.gpu.linalg.atlas_label_embedding_blocks

        def tracked_get_device(index=None):
            seen_indices.append(index)
            return cpu_dev

        def tracked_blocks(blocks, atlas_arg, count, **kwargs):
            seen_shard_sizes.append((count, kwargs.get('device')))
            return orig_blocks(blocks, atlas_arg, count, **kwargs)

        with patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                patch.object(wma.gpu, 'visible_device_count', return_value=2), \
                patch.object(wma.gpu, 'get_device', side_effect=tracked_get_device), \
                patch.object(wma.gpu.linalg, 'atlas_label_embedding_blocks', side_effect=tracked_blocks):
            out_2, labels_2, _, embed_2 = cluster.spectral_atlas_label(pd_2, atlas, use_gpu=True)

        self.assertIn(0, seen_indices)
        self.assertIn(1, seen_indices)
        self.assertEqual(len(seen_shard_sizes), 2)
        self.assertEqual(sum(cnt for cnt, _ in seen_shard_sizes), n_fibers)
        np.testing.assert_array_equal(labels_2, labels_1)
        np.testing.assert_allclose(embed_2, embed_1, rtol=1e-9, atol=1e-9)
        self.assertEqual(out_2.GetNumberOfLines(), n_fibers)

    def test_congeal_multisubject_dispatches_round_robin_and_skips_stream_on_mps(self):
        torch = self._require_torch()
        saved_cfg = {name: getattr(wma.gpu.config, name)
                     for name in ('requested', 'float32', 'devices', 'allow_mps')}
        self.addCleanup(lambda: [setattr(wma.gpu.config, k, v) for k, v in saved_cfg.items()])

        for mode in ('Affine', 'Rigid', 'Nonrigid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmpdir:
                reg = congeal_multisubject.MultiSubjectRegistration()
                reg.output_directory = tmpdir
                reg.random_seed = 42
                reg.render = False
                reg.mode = mode
                reg.nonrigid_grid_resolution = 3
                reg.maxfun = 6
                reg.mean_brain_size = 8
                reg.subject_brain_size = 4
                reg.points_per_fiber = 5
                reg.parallel_jobs = 2
                for i in range(3):
                    reg.add_polydata(_make_polydata(6, n_points=5, seed=10 + i), f'sub{i}')

                seen_devices = []
                entered_cuda_devices = []
                lock = threading.Lock()

                def fake_cuda_device(dev):
                    with lock:
                        entered_cuda_devices.append(dev)
                    return contextlib.nullcontext()

                class FakeObjective:
                    def __init__(self, fixed, moving, *rest, device=None, **kwargs):
                        tag = float(np.mean(moving))
                        with lock:
                            seen_devices.append((tag, device))
                        self.target = tag

                    def __call__(self, param, *args, **kwargs):
                        arr = np.asarray(param, dtype=np.float64)
                        return float(np.sum((arr[:3] - self.target) ** 2)) + 1.0

                with patch.object(congeal_multisubject, 'HAVE_PLT', 0), \
                        patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                        patch.object(torch.cuda, 'is_available', return_value=True), \
                        patch.object(torch.cuda, 'device', side_effect=fake_cuda_device), \
                        patch.object(wma.gpu, 'visible_device_count', return_value=2), \
                        patch.object(wma.gpu, 'get_device',
                                     side_effect=lambda idx=None: torch.device('cuda', 0 if idx is None else idx)), \
                        patch.object(registration, 'AffineObjective', FakeObjective), \
                        patch.object(bspline, 'NonrigidObjective', FakeObjective):
                    reg.iterate()

                self.assertEqual(len(seen_devices), 3)
                self.assertEqual(set(entered_cuda_devices),
                                 {torch.device('cuda', 0), torch.device('cuda', 1)})
                # Map each subject by its unique mean(moving) tag to verify round-robin 0, 1, 0
                subject_tags = []
                for idx in range(3):
                    pd = wma.filter.downsample(reg.polydatas[idx], reg.subject_brain_size,
                                               verbose=False, random_seed=reg.random_seed)
                    fa = wma.fibers.FiberArray()
                    fa.convert_from_polydata(pd, reg.points_per_fiber)
                    arr = np.array([fa.fiber_array_r, fa.fiber_array_a, fa.fiber_array_s])
                    subject_tags.append(float(np.mean(arr)))
                tag_to_device = dict(seen_devices)
                self.assertEqual(tag_to_device[subject_tags[0]], torch.device('cuda', 0))
                self.assertEqual(tag_to_device[subject_tags[1]], torch.device('cuda', 1))
                self.assertEqual(tag_to_device[subject_tags[2]], torch.device('cuda', 0))
                self.assertEqual(len(reg.transforms_as_array), 3)

        # Verify MPS mode runs without creating torch.cuda.Stream or entering torch.cuda.device
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = congeal_multisubject.MultiSubjectRegistration()
            reg.output_directory = tmpdir
            reg.render = False
            reg.mode = 'Affine'
            reg.maxfun = 4
            reg.mean_brain_size = 6
            reg.subject_brain_size = 3
            reg.points_per_fiber = 5
            reg.parallel_jobs = 4
            for i in range(2):
                reg.add_polydata(_make_polydata(6, n_points=5, seed=20 + i), f'sub{i}')

            mps_dev = torch.device('mps')
            seen_mps = []

            class FakeMpsObjective:
                def __init__(self, fixed, moving, *rest, device=None, **kwargs):
                    seen_mps.append(device)

                def __call__(self, param, *args, **kwargs):
                    return float(np.sum(np.asarray(param) ** 2))

            wma.gpu.config.allow_mps = True
            wma.gpu.config.float32 = True
            with patch.object(congeal_multisubject, 'HAVE_PLT', 0), \
                    patch.object(wma.gpu.config, 'cuda_available', return_value=False), \
                    patch.object(wma.gpu.config, 'mps_available', return_value=True), \
                    patch.object(torch.cuda, 'device',
                                 side_effect=AssertionError('must not enter CUDA device on MPS')), \
                    patch.object(torch.cuda, 'Stream',
                                 side_effect=AssertionError('must not create CUDA Stream on MPS')), \
                    patch.object(torch.cuda, 'stream',
                                 side_effect=AssertionError('must not enter CUDA stream on MPS')), \
                    patch.object(registration, 'AffineObjective', FakeMpsObjective):
                reg.iterate()

            self.assertEqual(seen_mps, [mps_dev, mps_dev])

    def test_congeal_multisubject_prerenders_on_main_thread_for_gpu_workers(self):
        torch = self._require_torch()
        saved_cfg = {name: getattr(wma.gpu.config, name)
                     for name in ('requested', 'float32', 'devices', 'allow_mps')}
        self.addCleanup(lambda: [setattr(wma.gpu.config, k, v) for k, v in saved_cfg.items()])

        for mode, n_gpus in (('Affine', 2), ('Nonrigid', 2), ('Affine', 1)):
            with self.subTest(mode=mode, n_gpus=n_gpus), tempfile.TemporaryDirectory() as tmpdir:
                reg = congeal_multisubject.MultiSubjectRegistration()
                reg.output_directory = tmpdir
                reg.random_seed = 42
                reg.render = True
                reg.mode = mode
                reg.nonrigid_grid_resolution = 3
                reg.maxfun = 4
                reg.mean_brain_size = 8
                reg.subject_brain_size = 4
                reg.points_per_fiber = 5
                reg.parallel_jobs = 2
                for i in range(3):
                    reg.add_polydata(_make_polydata(6, n_points=5, seed=30 + i), f'sub{i}')

                render_threads = []
                render_names = []
                worker_threads = []
                seen_devices = []
                lock = threading.Lock()

                class FakeRenderer:
                    def __init__(self):
                        self.magnification = 1

                    def save_views(self, outdir, prefix):
                        render_names.append((outdir, prefix))

                def fake_render(pd, n_fibers, verbose=False):
                    render_threads.append(threading.current_thread())
                    return FakeRenderer()

                class FakeObjective:
                    def __init__(self, fixed, moving, *rest, device=None, **kwargs):
                        with lock:
                            worker_threads.append(threading.current_thread())
                            seen_devices.append(device)

                    def __call__(self, param, *args, **kwargs):
                        return float(np.sum(np.asarray(param) ** 2))

                with patch.object(congeal_multisubject, 'HAVE_PLT', 0), \
                        patch.object(wma.render, 'render', side_effect=fake_render), \
                        patch.object(wma.gpu.config, 'cuda_available', return_value=True), \
                        patch.object(torch.cuda, 'is_available', return_value=True), \
                        patch.object(torch.cuda, 'device', side_effect=lambda _d: contextlib.nullcontext()), \
                        patch.object(torch.cuda, 'Stream', return_value=MagicMock()), \
                        patch.object(torch.cuda, 'stream', side_effect=lambda _s: contextlib.nullcontext()), \
                        patch.object(wma.gpu, 'visible_device_count', return_value=n_gpus), \
                        patch.object(wma.gpu, 'get_device',
                                     side_effect=lambda idx=None: torch.device('cuda', 0 if idx is None else idx)), \
                        patch.object(registration, 'AffineObjective', FakeObjective), \
                        patch.object(bspline, 'NonrigidObjective', FakeObjective):
                    reg.iterate()

                self.assertEqual(len(render_threads), 3)
                self.assertTrue(all(t is threading.main_thread() for t in render_threads))
                expected_suffix = '_grid_003' if mode == 'Nonrigid' else ''
                self.assertEqual(
                    [name for _, name in render_names],
                    [f'fixed_brain__subject_0000{i}_iteration_00001_sigma_020{expected_suffix}'
                     for i in range(3)],
                )
                self.assertEqual(len(seen_devices), 3)
                self.assertTrue(all(t is not threading.main_thread() for t in worker_threads))
                if n_gpus == 2:
                    self.assertEqual(set(seen_devices),
                                     {torch.device('cuda', 0), torch.device('cuda', 1)})
                else:
                    self.assertEqual(set(seen_devices), {torch.device('cuda', 0)})

    @unittest.skipUnless(wma.gpu.available() and _cuda_device_count() >= 2,
                         'Two CUDA GPUs are required')
    def test_real_two_gpu_spectral_atlas_label_parity(self):
        prev_devices = wma.gpu.config.devices
        self.addCleanup(wma.gpu.set_devices, prev_devices)

        n_fibers = 16384
        atlas = _make_atlas(nystrom_count=17, n_eig=3, n_clusters=7)
        pd_1 = _make_polydata(n_fibers, n_points=5, seed=99)
        pd_2 = _make_polydata(n_fibers, n_points=5, seed=99)

        wma.gpu.set_devices([0])
        _, labels_1, _, embed_1 = cluster.spectral_atlas_label(pd_1, atlas, use_gpu=True)
        wma.gpu.set_devices([0, 1])
        _, labels_2, _, embed_2 = cluster.spectral_atlas_label(pd_2, atlas, use_gpu=True)

        np.testing.assert_array_equal(labels_2, labels_1)
        np.testing.assert_allclose(embed_2, embed_1, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
