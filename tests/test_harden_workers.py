"""Exercise the Slicer launcher and completion bookkeeping without Slicer."""
import importlib.util
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


if __name__ == '__main__':
    unittest.main()
