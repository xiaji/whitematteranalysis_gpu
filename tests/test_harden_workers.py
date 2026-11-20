"""Exercise the Slicer launcher and completion bookkeeping without Slicer."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import whitematteranalysis as wma

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'bin' / 'wm_harden_transform.py'
SPEC = importlib.util.spec_from_file_location('wm_harden_transform', SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class HardenWorkerTests(unittest.TestCase):
    def test_run_slicer_builds_structured_command_and_isolates_environment(self):
        thread_limits = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')
        with tempfile.TemporaryDirectory(prefix='wma slicer launcher ') as temporary:
            root = Path(temporary)
            log = root / 'worker log.txt'
            profile = root / 'worker profile.json'
            command_paths = [root / name for name in ('input file.vtp', 'transform file.tfm',
                           'Slicer app', 'helper script.py', 'output files', 'split output')]
            slicer, helper, outdir, split = command_paths[2], command_paths[3], command_paths[4], command_paths[5]
            base_expected = [str(slicer), '--no-main-window',
                '--disable-modules',
                '--python-script', str(helper), str(command_paths[0]), str(command_paths[1]),
                '1', str(outdir), '2', '4', '--split-output-directory', str(split),
                '--profile-output', str(profile), '--python-code', 'slicer.app.quit()']

            # 1) Unwrapped when DISPLAY is set (even on headless Linux with xvfb-run)
            with patch.dict(os.environ, {'PYTHONPATH': 'parent path',
                                         'WMA_VTP_COMPRESSION': 'zlib',
                                         'DISPLAY': ':0'}, clear=True), \
                    patch.object(sys, 'platform', 'linux'), \
                    patch.object(launcher.shutil, 'which', return_value='/usr/bin/xvfb-run'), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], True, slicer,
                                     helper, outdir, 2, 4, log, split, profile)
                command = run.call_args.args[0]
                env = run.call_args.kwargs['env']
                self.assertEqual(list(map(str, command)), base_expected)
                self.assertEqual(run.call_args.kwargs['stdout'].name, str(log))
                self.assertIs(run.call_args.kwargs['stderr'], launcher.subprocess.STDOUT)
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertNotIn('PYTHONPATH', env)
                self.assertEqual(env['WMA_VTP_COMPRESSION'], 'zlib')
                self.assertEqual({name: env[name] for name in thread_limits},
                                 {name: '1' for name in thread_limits})
                self.assertIn('WMA_HARDEN_LAUNCH_TIME', env)
                self.assertEqual(os.environ.get('PYTHONPATH'), 'parent path')
                self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), 'zlib')

            # 2) Unwrapped when WMA_NO_XVFB=1 or xvfb-run missing
            with patch.dict(os.environ, {'WMA_NO_XVFB': '1'}, clear=True), \
                    patch.object(sys, 'platform', 'linux'), \
                    patch.object(launcher.shutil, 'which', return_value='/usr/bin/xvfb-run'), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], False, slicer,
                                     helper, outdir, 0, 1, log, split)
                self.assertEqual(run.call_args.args[0][0], str(slicer))
                self.assertEqual(run.call_args.kwargs['env']['WMA_VTP_COMPRESSION'], 'lz4')

            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(sys, 'platform', 'linux'), \
                    patch.object(launcher.shutil, 'which', return_value=None), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], False, slicer,
                                     helper, outdir, 0, 1, log, split)
                self.assertEqual(run.call_args.args[0][0], str(slicer))

            # 3) Wrapped with xvfb-run -a on headless Linux when DISPLAY unset and xvfb-run available
            with patch.dict(os.environ, {'PYTHONPATH': 'parent path',
                                         'WMA_VTP_COMPRESSION': 'zlib'}, clear=True), \
                    patch.object(sys, 'platform', 'linux'), \
                    patch.object(launcher.shutil, 'which', return_value='/usr/bin/xvfb-run'), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], True, slicer,
                                     helper, outdir, 2, 4, log, split, profile)
                command = run.call_args.args[0]
                env = run.call_args.kwargs['env']
                self.assertEqual(list(map(str, command)), ['/usr/bin/xvfb-run', '-a'] + base_expected)
                self.assertNotIn('PYTHONPATH', env)
                self.assertEqual({name: env[name] for name in thread_limits},
                                 {name: '1' for name in thread_limits})
                self.assertIn('WMA_HARDEN_LAUNCH_TIME', env)

            # 4) Bypassed when no_xvfb=True
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(sys, 'platform', 'linux'), \
                    patch.object(launcher.shutil, 'which', return_value='/usr/bin/xvfb-run'), \
                    patch.object(launcher.subprocess, 'run') as run:
                launcher._run_slicer(command_paths[0], command_paths[1], True, slicer,
                                     helper, outdir, 2, 4, log, split, profile, no_xvfb=True)
                self.assertEqual(list(map(str, run.call_args.args[0])), base_expected)

    def test_main_keeps_profile_artifacts_only_when_requested(self):
        with tempfile.TemporaryDirectory(prefix='wma harden main ') as temporary:
            root = Path(temporary)
            source, output = root / 'input files', root / 'output files'
            source.mkdir()
            output.mkdir()
            (source / 'cluster_00001.vtp').write_bytes(b'input')
            (source / 'cluster_00002.vtp').write_bytes(b'input')
            transform = root / 'transform file.tfm'
            transform.write_text('unused by fake workers')
            slicer = root / 'Slicer app'
            slicer.touch()
            split = root / 'split output'

            def fake_worker(polydata, transform, inverse, slicer_path, helper, outdir,
                            partition, partitions, log, split_output_dir=None,
                            profile_output=None, input_status=None):
                names = sorted(path.name for path in source.glob('*.vtp'))[partition::partitions]
                for name in names:
                    (outdir / name).write_bytes(b'hardened')
                    if split_output_dir is not None:
                        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                       'tracts_commissural'):
                            (split_output_dir / folder / name).write_bytes(b'split')
                log.write_text('fake worker log')
                if profile_output is not None:
                    Path(profile_output).write_text(json.dumps({'files': 1, 'success': True}))

            args = [str(source), str(output), str(slicer), '-t', str(transform), '-j', '2',
                    '--split-output-directory', str(split)]
            with patch.object(launcher, '_run_slicer', side_effect=fake_worker) as worker:
                with patch('sys.argv', ['wm_harden_transform.py'] + args + ['--profile']):
                    launcher.main()
                self.assertEqual(worker.call_count, 2)
                self.assertTrue((output / 'transform_complete.log').is_file())
                profiles = list(output.glob('wma-harden-*/*.json'))
                self.assertEqual(len(profiles), 2)
                for profile in profiles:
                    self.assertEqual(json.loads(profile.read_text()), {'files': 1, 'success': True})
                self.assertTrue(all((split / folder / name).is_file()
                    for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                   'tracts_commissural')
                    for name in ('cluster_00001.vtp', 'cluster_00002.vtp')))

            for path in output.glob('wma-harden-*'):
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
            (output / 'transform_complete.log').unlink()
            with patch.object(launcher, '_run_slicer', side_effect=fake_worker) as worker:
                with patch('sys.argv', ['wm_harden_transform.py'] + args):
                    launcher.main()
                self.assertEqual(worker.call_count, 2)
                self.assertTrue((output / 'transform_complete.log').is_file())
                self.assertFalse(list(output.glob('wma-harden-*')))

    def _make_cluster_polydata(self, lines_coords, hemi_labels, tensors=False):
        pd = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pt_hemi = []
        for lidx, coords in enumerate(lines_coords):
            ids = vtk.vtkIdList()
            for xyz in coords:
                ids.InsertNextId(pts.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2])))
                pt_hemi.append(hemi_labels[lidx])
            lines.InsertNextCell(ids)
        pd.SetPoints(pts)
        pd.SetLines(lines)
        harr = numpy_to_vtk(np.asarray(pt_hemi, dtype=np.float64), deep=True)
        harr.SetName('HemisphereLocation')
        pd.GetPointData().AddArray(harr)
        if tensors:
            tarr = numpy_to_vtk(np.zeros((pd.GetNumberOfPoints(), 9), dtype=np.float32), deep=True)
            tarr.SetName('tensors')
            pd.GetPointData().AddArray(tarr)
            pd.GetPointData().SetActiveTensors('tensors')
        return pd

    def test_pure_python_and_slicer_none_linear_hardening_with_split(self):
        with tempfile.TemporaryDirectory(prefix='wma pure python harden ') as temporary:
            root = Path(temporary)
            source = root / 'input'
            source.mkdir()
            c1 = self._make_cluster_polydata(
                [[[-10.0, 0.0, 0.0], [-10.0, 5.0, 0.0]],
                 [[10.0, 0.0, 0.0], [10.0, 5.0, 0.0]]],
                hemi_labels=[1.0, 2.0])
            c2 = self._make_cluster_polydata(
                [[[-5.0, 1.0, 0.0], [5.0, 1.0, 0.0]]],
                hemi_labels=[3.0])
            wma.io.write_polydata(c1, str(source / 'cluster_00001.vtp'))
            wma.io.write_polydata(c2, str(source / 'cluster_00002.vtp'))

            tx = vtk.vtkTransform()
            tx.Translate(2.0, -3.0, 4.0)
            tx.RotateZ(15.0)
            tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['case'])[0])

            for mode_name, extra_args in (
                ('pure_python_flag', [str(source), str(root / 'out_flag'), '-t', str(tfm),
                                      '-i', '-j', '2', '--pure-python',
                                      '--split-output-directory', str(root / 'split_flag')]),
                ('slicer_none', [str(source), str(root / 'out_none'), 'none', '-t', str(tfm),
                                 '-i', '-j', '2',
                                 '--split-output-directory', str(root / 'split_none')]),
            ):
                with self.subTest(mode=mode_name):
                    outdir = root / ('out_flag' if mode_name == 'pure_python_flag' else 'out_none')
                    splitdir = root / ('split_flag' if mode_name == 'pure_python_flag' else 'split_none')
                    with patch.object(launcher, '_run_slicer',
                                      side_effect=AssertionError('Slicer must not run in pure-python mode')), \
                            patch('sys.argv', ['wm_harden_transform.py'] + extra_args):
                        launcher.main()
                    self.assertTrue((outdir / 'transform_complete.log').is_file())
                    out_c1 = wma.io.read_polydata(str(outdir / 'cluster_00001.vtp'))
                    tf = vtk.vtkTransformPolyDataFilter()
                    tf.SetInputData(c1)
                    tf.SetTransform(tx.GetLinearInverse())
                    tf.Update()
                    np.testing.assert_allclose(
                        vtk_to_numpy(out_c1.GetPoints().GetData()),
                        vtk_to_numpy(tf.GetOutput().GetPoints().GetData()),
                        atol=1e-5,
                    )
                    self.assertEqual(
                        wma.io.read_polydata(str(splitdir / 'tracts_left_hemisphere' / 'cluster_00001.vtp')).GetNumberOfLines(), 1)
                    self.assertEqual(
                        wma.io.read_polydata(str(splitdir / 'tracts_right_hemisphere' / 'cluster_00001.vtp')).GetNumberOfLines(), 1)
                    self.assertEqual(
                        wma.io.read_polydata(str(splitdir / 'tracts_commissural' / 'cluster_00002.vtp')).GetNumberOfLines(), 1)

            # Reject tensor-bearing inputs in pure-python mode
            tensor_src = root / 'tensor_input'
            tensor_src.mkdir()
            c_tens = self._make_cluster_polydata(
                [[[-10.0, 0.0, 0.0], [-10.0, 5.0, 0.0]]],
                hemi_labels=[1.0], tensors=True)
            wma.io.write_polydata(c_tens, str(tensor_src / 'cluster_00001.vtp'))
            with patch('sys.argv', ['wm_harden_transform.py', str(tensor_src),
                                    str(root / 'out_tensor'), 'none', '-t', str(tfm), '-i']):
                with self.assertRaisesRegex(RuntimeError, 'tensor|Slicer'):
                    launcher.main()

    def test_slicer_auto_prepare_command_runs_once_and_avoids_redundant_reads(self):
        with tempfile.TemporaryDirectory(prefix='wma auto prepare ') as temporary:
            root = Path(temporary)
            catalogue = root / 'catalogue'
            catalogue.mkdir()
            for name in ('cluster_00001.vtp', 'cluster_00002.vtp'):
                (catalogue / name).write_bytes(b'catalogue')

            fake_slicer = root / 'Slicer'
            fake_slicer.write_text('#!/bin/sh\nexit 0\n')

            tx = vtk.vtkTransform()
            tx.Translate(2.0, -3.0, 4.0)
            linear_tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['lin'])[0])
            bspline_tfm = root / 'bspline.tfm'
            bspline_tfm.write_text(
                '#Insight Transform File V1.0\n# Transform 0\n'
                'Transform: BSplineTransform_double_3_3\nParameters: 0 0 0\n'
                'FixedParameters: 4 4 4 -100 -100 -100 50 50 50 1 0 0 0 1 0 0 0 1\n'
            )

            pd_free = self._make_cluster_polydata(
                [[[-10.0, 0.0, 0.0], [-10.0, 5.0, 0.0]]], hemi_labels=[1.0])
            pd_tensor = self._make_cluster_polydata(
                [[[-10.0, 0.0, 0.0], [-10.0, 5.0, 0.0]]], hemi_labels=[1.0], tensors=True)

            # 1) BSpline transform with Slicer='auto' + --prepare-command: goes directly to _run_prepared (1 prepare call)
            in_bspline = root / 'in_bspline'
            out_bspline = root / 'out_bspline'
            in_bspline.mkdir()
            prep_calls = []

            def fake_prepare_bspline(command, check=False):
                prep_calls.append(list(command))
                for name in ('cluster_00001.vtp', 'cluster_00002.vtp'):
                    wma.io.write_polydata(pd_free, str(in_bspline / name))

            def fake_slicer_worker(polydata, transform, inverse, slicer_path, helper, outdir,
                                   partition, partitions, log, split_output_dir=None,
                                   profile_output=None, input_status=None):
                import time
                while not Path(input_status).is_file():
                    time.sleep(0.01)
                self.assertEqual(Path(input_status).read_text(), 'ready')
                for name in sorted(p.name for p in in_bspline.glob('*.vtp'))[partition::partitions]:
                    (outdir / name).write_bytes(b'hardened')
                log.write_text('ok')

            stdout_bspline = io.StringIO()
            with patch.object(launcher.shutil, 'which', return_value=str(fake_slicer)), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_prepare_bspline), \
                    patch.object(launcher, '_run_slicer', side_effect=fake_slicer_worker) as slicer_mock, \
                    patch('sys.stdout', stdout_bspline), \
                    patch('sys.argv', [
                        'wm_harden_transform.py', str(in_bspline), str(out_bspline), 'auto',
                        '-t', str(bspline_tfm), '-i', '-j', '2',
                        '--input-catalogue', str(catalogue), '--prepare-command', 'echo', 'prep']):
                launcher.main()
            self.assertEqual(len(prep_calls), 1)
            self.assertEqual(slicer_mock.call_count, 2)
            self.assertIsNotNone(slicer_mock.call_args_list[0].args[11])
            self.assertNotIn('pure-Python affine', stdout_bspline.getvalue())

            # 2) Linear .tfm + tensor-bearing VTP with Slicer='auto' + --prepare-command: prepares ONCE, falls back cleanly
            in_tensor = root / 'in_tensor'
            out_tensor = root / 'out_tensor'
            in_tensor.mkdir()
            prep_calls_tensor = []

            def fake_prepare_tensor(command, check=False):
                prep_calls_tensor.append(list(command))
                for name in ('cluster_00001.vtp', 'cluster_00002.vtp'):
                    wma.io.write_polydata(pd_tensor, str(in_tensor / name))

            def fake_slicer_tensor(polydata, transform, inverse, slicer_path, helper, outdir,
                                   partition, partitions, log, split_output_dir=None,
                                   profile_output=None, input_status=None):
                for name in sorted(p.name for p in in_tensor.glob('*.vtp'))[partition::partitions]:
                    (outdir / name).write_bytes(b'hardened')
                log.write_text('ok')

            stdout_tensor = io.StringIO()
            with patch.object(launcher.shutil, 'which', return_value=str(fake_slicer)), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_prepare_tensor), \
                    patch.object(launcher, '_run_slicer', side_effect=fake_slicer_tensor) as slicer_mock, \
                    patch('sys.stdout', stdout_tensor), \
                    patch('sys.argv', [
                        'wm_harden_transform.py', str(in_tensor), str(out_tensor), 'auto',
                        '-t', str(linear_tfm), '-i', '-j', '2',
                        '--input-catalogue', str(catalogue), '--prepare-command', 'echo', 'prep']):
                launcher.main()
            self.assertEqual(len(prep_calls_tensor), 1)
            self.assertEqual(slicer_mock.call_count, 2)
            self.assertIsNone(slicer_mock.call_args_list[0].args[11])
            self.assertNotIn('pure-Python affine', stdout_tensor.getvalue())

            # 3) Linear .tfm + tensor-free VTPs with Slicer='auto' + --prepare-command: prepares ONCE, reads each VTP ONCE
            in_free = root / 'in_free'
            out_free = root / 'out_free'
            in_free.mkdir()
            prep_calls_free = []

            def fake_prepare_free(command, check=False):
                prep_calls_free.append(list(command))
                for name in ('cluster_00001.vtp', 'cluster_00002.vtp'):
                    wma.io.write_polydata(pd_free, str(in_free / name))

            stdout_free = io.StringIO()
            with patch.object(launcher.shutil, 'which', return_value=str(fake_slicer)), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_prepare_free), \
                    patch.object(launcher, '_run_slicer', side_effect=AssertionError('Slicer must not run')), \
                    patch.object(wma.io, 'read_polydata', wraps=wma.io.read_polydata) as read_spy, \
                    patch('sys.stdout', stdout_free), \
                    patch('sys.argv', [
                        'wm_harden_transform.py', str(in_free), str(out_free), 'auto',
                        '-t', str(linear_tfm), '-i', '-j', '2',
                        '--input-catalogue', str(catalogue), '--prepare-command', 'echo', 'prep']):
                launcher.main()
            self.assertEqual(len(prep_calls_free), 1)
            self.assertEqual(read_spy.call_count, 2)
            self.assertIn('pure-Python affine', stdout_free.getvalue())
            self.assertTrue((out_free / 'transform_complete.log').is_file())


if __name__ == '__main__':
    unittest.main()
