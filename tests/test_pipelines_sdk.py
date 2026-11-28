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


if __name__ == '__main__':
    unittest.main()
