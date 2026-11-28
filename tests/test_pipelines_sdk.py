"""Tests for the ``whitematteranalysis.pipelines`` Python SDK."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import whitematteranalysis as wma
from whitematteranalysis.pipelines import (
    BatchResult,
    ORGConfig,
    ORGParcellationPipeline,
    SubjectResult,
    parcellate_batch,
    parcellate_subject,
)
try:
    from test_org_pipeline_cli import (
        _build_synthetic_org_atlas,
        _bundle_coords,
        _make_tract_polydata,
    )
except ImportError:
    from tests.test_org_pipeline_cli import (
        _build_synthetic_org_atlas,
        _bundle_coords,
        _make_tract_polydata,
    )

ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_subject(path, seed_offset=0):
    fibers = (
        _bundle_coords(-35.0, 0.0, 12, n_pts=21, step=3.0, seed=10 + seed_offset, jitter_scale=0.5)
        + _bundle_coords(35.0, 0.0, 12, n_pts=21, step=3.0, seed=20 + seed_offset, jitter_scale=0.5)
        + _bundle_coords(0.0, 35.0, 12, n_pts=21, step=3.0, commissural=True, seed=30 + seed_offset, jitter_scale=0.5)
    )
    pd = _make_tract_polydata(fibers)
    wma.io.write_polydata(pd, str(path))


class PipelinesSDKTests(unittest.TestCase):
    def test_config_validation_and_lazy_no_torch_import(self):
        with self.assertRaises(ValueError):
            ORGConfig(atlas_dir=None)
        with self.assertRaises(ValueError):
            ORGConfig(atlas_dir='/tmp', atlas_family='invalid')
        with self.assertRaises(ValueError):
            ORGConfig(atlas_dir='/tmp', reg_mode='bad')
        with self.assertRaises(ValueError):
            ORGConfig(atlas_dir='/tmp', keep_intermediates=3)

        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT)
        code = (
            "import sys; sys.modules['torch'] = None; "
            "import whitematteranalysis as wma; "
            "assert hasattr(wma.pipelines, 'ORGParcellationPipeline'); "
            "print('SDK_NO_TORCH_OK')"
        )
        proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn('SDK_NO_TORCH_OK', proc.stdout)

    def test_parcellate_subject_and_load_tracts(self):
        with tempfile.TemporaryDirectory(prefix='wma_sdk_subj_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)
            subj_path = root / 'sub-01.vtp'
            _write_synthetic_subject(subj_path, seed_offset=1)
            out_dir = root / 'out'

            res = parcellate_subject(
                subj_path,
                out_dir,
                atlas_dir,
                reg_mode='rig',
                jobs=1,
                gpu=False,
            )
            self.assertIsInstance(res, SubjectResult)
            self.assertEqual(res.case_id, 'sub-01')
            self.assertEqual(res.output_dir, out_dir.resolve() / 'sub-01')
            self.assertEqual(res.qc_summary['hardening_backend'], 'pure_python_affine')
            self.assertGreater(res.qc_summary['retained_clustered_fibers'], 0)
            self.assertIn('T_AF_left.vtp', res.tract_paths)

            pd_no_ext = res.load_tract('T_AF_left')
            pd_with_ext = res.load_tract('T_AF_left.vtp')
            self.assertEqual(pd_no_ext.GetNumberOfLines(), pd_with_ext.GetNumberOfLines())
            with self.assertRaises(KeyError):
                res.load_tract('T_NONEXISTENT')

            all_nonempty = res.load_all_tracts(nonempty_only=True)
            self.assertGreaterEqual(len(all_nonempty), 1)
            for pd_tract in all_nonempty.values():
                self.assertGreater(pd_tract.GetNumberOfLines(), 0)

    def test_pipeline_reuses_resident_atlas_and_enforces_closed_state(self):
        with tempfile.TemporaryDirectory(prefix='wma_sdk_reuse_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)
            s1 = root / 'sub-A.vtp'
            s2 = root / 'sub-B.vtp'
            _write_synthetic_subject(s1, seed_offset=1)
            _write_synthetic_subject(s2, seed_offset=2)
            out_dir = root / 'out'

            reg_atlas_reads = []
            orig_read = wma.io.read_polydata

            def _counting_read(path, *args, **kwargs):
                if Path(path).name == 'registration_atlas.vtk':
                    reg_atlas_reads.append(path)
                return orig_read(path, *args, **kwargs)

            with patch.object(wma.io, 'read_polydata', side_effect=_counting_read), \
                    patch.object(wma.cluster, 'load_atlas', wraps=wma.cluster.load_atlas) as atlas_loader:
                with ORGParcellationPipeline(atlas_dir, gpu=False, jobs=1, keep_resident=True) as pipe:
                    self.assertEqual(pipe.atlas_family, 'org')
                    r1 = pipe.run(s1, out_dir)
                    r2 = pipe.run(s2, out_dir, case_id='custom-B')

            self.assertEqual(r1.case_id, 'sub-A')
            self.assertEqual(r2.case_id, 'custom-B')
            self.assertTrue((out_dir / 'custom-B' / 'qc_summary.json').is_file())
            self.assertEqual(len(reg_atlas_reads), 1)
            self.assertEqual(atlas_loader.call_count, 1)

            with self.assertRaises(RuntimeError):
                pipe.run(s1, out_dir)

    def test_parcellate_batch_sequences_and_continue_on_error(self):
        with tempfile.TemporaryDirectory(prefix='wma_sdk_batch_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)
            good_path = root / 'sub-good.vtp'
            _write_synthetic_subject(good_path, seed_offset=3)
            empty_path = root / 'sub-empty.vtp'
            wma.io.write_polydata(_make_tract_polydata([]), str(empty_path))

            out_continue = root / 'batch_continue'
            batch = parcellate_batch(
                [('case_empty', empty_path), ('case_good', good_path)],
                out_continue,
                atlas_dir,
                gpu=False,
                jobs=1,
                continue_on_error=True,
            )
            self.assertIsInstance(batch, BatchResult)
            self.assertEqual(batch.total, 2)
            self.assertEqual(batch.succeeded, 1)
            self.assertEqual(batch.failed, 1)
            self.assertIn('case_empty', batch.errors)
            self.assertEqual([r.case_id for r in batch.results], ['case_good'])
            self.assertTrue(batch.summary_path.is_file())
            disk_summary = json.loads(batch.summary_path.read_text(encoding='utf-8'))
            self.assertEqual(disk_summary['succeeded_subjects'], 1)
            self.assertEqual(disk_summary['failed_subjects'], 1)

            # Fail-fast mode raises immediately and still writes batch_qc_summary.json
            out_failfast = root / 'batch_failfast'
            with self.assertRaises(RuntimeError):
                parcellate_batch(
                    [empty_path, good_path],
                    out_failfast,
                    atlas_dir,
                    gpu=False,
                    jobs=1,
                    continue_on_error=False,
                )
            self.assertTrue((out_failfast / 'batch_qc_summary.json').is_file())

    def test_gpu_config_is_scoped_per_pipeline_run_and_restored(self):
        with tempfile.TemporaryDirectory(prefix='wma_sdk_gpu_scope_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)
            subj_path = root / 'sub-scope.vtp'
            _write_synthetic_subject(subj_path, seed_offset=4)

            gpu_cfg = wma.gpu.config
            orig_state = (
                gpu_cfg.requested,
                gpu_cfg.float32,
                gpu_cfg.strict,
                list(gpu_cfg.devices) if gpu_cfg.devices is not None else None,
                gpu_cfg.device_index,
            )
            try:
                gpu_cfg.requested = None
                gpu_cfg.float32 = False
                gpu_cfg.strict = False
                gpu_cfg.devices = None

                cpu_pipe = ORGParcellationPipeline(
                    atlas_dir, gpu=False, gpu_float32=False, gpu_strict=False, jobs=1
                )
                gpu_pipe = ORGParcellationPipeline(
                    atlas_dir, gpu=True, gpu_float32=True, gpu_strict=True, gpu_devices=[1, 0], jobs=1
                )
                # Constructing gpu_pipe must not mutate process-global GPU config.
                self.assertIsNone(gpu_cfg.requested)
                self.assertFalse(gpu_cfg.float32)
                self.assertFalse(gpu_cfg.strict)
                self.assertIsNone(gpu_cfg.devices)

                observed = {}

                def _fake_run_subject(case_id, input_path, args, session, workers, slicer_path):
                    observed['requested'] = wma.gpu.config.requested
                    observed['float32'] = wma.gpu.config.float32
                    observed['strict'] = wma.gpu.config.strict
                    observed['devices'] = list(wma.gpu.config.devices) if wma.gpu.config.devices is not None else None
                    out_d = Path(args.output) / case_id
                    (out_d / 'AnatomicalTracts').mkdir(parents=True, exist_ok=True)
                    return out_d, {'tracts': {}, 'retained_clustered_fibers': 0, 'stage_timings_s': {}}

                with patch('whitematteranalysis.pipelines.org._run_subject', side_effect=_fake_run_subject):
                    gpu_pipe.run(subj_path, root / 'out_gpu')
                    self.assertEqual(observed, {
                        'requested': True,
                        'float32': True,
                        'strict': True,
                        'devices': [1, 0],
                    })
                    # Global state restored immediately after gpu_pipe.run().
                    self.assertIsNone(gpu_cfg.requested)
                    self.assertFalse(gpu_cfg.float32)
                    self.assertFalse(gpu_cfg.strict)
                    self.assertIsNone(gpu_cfg.devices)

                    gpu_pipe.close()
                    cpu_pipe.run(subj_path, root / 'out_cpu')
                    self.assertEqual(observed, {
                        'requested': False,
                        'float32': False,
                        'strict': False,
                        'devices': None,
                    })
                    cpu_pipe.close()

                # Global state remains untouched after both runs.
                self.assertIsNone(gpu_cfg.requested)
                self.assertFalse(gpu_cfg.float32)
                self.assertFalse(gpu_cfg.strict)
                self.assertIsNone(gpu_cfg.devices)
            finally:
                (
                    gpu_cfg.requested,
                    gpu_cfg.float32,
                    gpu_cfg.strict,
                    gpu_cfg.devices,
                    gpu_cfg.device_index,
                ) = orig_state

    def test_case_id_path_traversal_is_rejected_without_touching_outside_files(self):
        with tempfile.TemporaryDirectory(prefix='wma_sdk_case_id_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)
            subj_path = root / 'sub-01.vtp'
            _write_synthetic_subject(subj_path, seed_offset=5)

            victim_tracts = root / 'victim' / 'AnatomicalTracts'
            victim_tracts.mkdir(parents=True)
            sentinel = victim_tracts / 'keep.txt'
            sentinel.write_text('do not delete', encoding='utf-8')
            out_dir = root / 'out'

            with ORGParcellationPipeline(atlas_dir, gpu=False, atomic_publish=False, jobs=1) as pipe:
                for bad_id in ('../victim', '..', '.', '', 'sub/nested', '/tmp/escape', 'sub\\win'):
                    with self.assertRaises(ValueError, msg=f'Expected ValueError for {bad_id!r}'):
                        pipe.run(subj_path, out_dir, case_id=bad_id)
                with self.assertRaises(ValueError):
                    pipe.run_batch([('../victim', subj_path)], out_dir)

            self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'do not delete')

    def test_resident_gpu_and_ref_caches_invalidate_on_precision_change(self):
        from whitematteranalysis.pipelines import org as org_mod

        with tempfile.TemporaryDirectory(prefix='wma_sdk_prec_cache_') as tmp:
            root = Path(tmp)
            atlas_dir = root / 'atlas'
            _build_synthetic_org_atlas(atlas_dir)

            gpu_cfg = wma.gpu.config
            orig_f32 = gpu_cfg.float32
            fake_dev = type('FakeDev', (), {'type': 'cuda', 'index': 0, '__str__': lambda self: 'cuda:0'})()
            prepared_calls = []

            def _fake_prepare(atlas, device=None):
                tag = (bool(wma.gpu.float32_enabled()), str(device))
                prepared_calls.append(tag)
                return tag

            try:
                with ORGParcellationPipeline(atlas_dir, keep_resident=True, jobs=1) as pipe:
                    with patch.object(wma.gpu, 'use_gpu', return_value=True), \
                            patch.object(wma.gpu, 'get_device', return_value=fake_dev), \
                            patch.object(wma.gpu, 'visible_device_count', return_value=1), \
                            patch.object(wma.gpu.linalg, 'prepare_atlas_gpu', side_effect=_fake_prepare):
                        wma.gpu.set_float32(False)
                        p1 = pipe._session.get_prepared_gpu()
                        p2 = pipe._session.get_prepared_gpu()
                        self.assertIs(p1, p2)
                        self.assertEqual(prepared_calls, [(False, 'cuda:0')])

                        wma.gpu.set_float32(True)
                        p3 = pipe._session.get_prepared_gpu()
                        self.assertIsNot(p3, p1)
                        self.assertEqual(prepared_calls, [(False, 'cuda:0'), (True, 'cuda:0')])

                    ref_cache = {}
                    ref_loads = []

                    def _fake_load_ref(*args, **kwargs):
                        val = ('ref', bool(wma.gpu.float32_enabled()))
                        ref_loads.append(val)
                        return val

                    with patch.object(org_mod, 'load_reference', side_effect=_fake_load_ref):
                        wma.gpu.set_float32(False)
                        r1 = org_mod._load_cluster_reference(
                            'c1.vtp', 400.0, 'Mean', True, 'gpu', ref_cache=ref_cache
                        )
                        wma.gpu.set_float32(True)
                        r2 = org_mod._load_cluster_reference(
                            'c1.vtp', 400.0, 'Mean', True, 'gpu', ref_cache=ref_cache
                        )
                        self.assertEqual(r1, ('ref', False))
                        self.assertEqual(r2, ('ref', True))
                        self.assertEqual(len(ref_loads), 2)
            finally:
                gpu_cfg.float32 = orig_f32


if __name__ == '__main__':
    unittest.main()
