"""Exercise the input-preparation gate without launching Slicer."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import whitematteranalysis as wma


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'wm_harden_transform_preparation', ROOT / 'bin/wm_harden_transform.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class HardenPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='wma preparation gate ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputdir = self.root / 'input files'
        self.inputdir.mkdir()
        self.status = self.root / 'logs/input.status'
        self.status.parent.mkdir()
        self.expected = {'cluster_00001.vtp', 'cluster_00002.vtk'}
        self.started = [threading.Event(), threading.Event()]
        self.finished = []
        self.outputs = []

    def workers(self):
        def run(index):
            self.started[index].set()
            try:
                while not self.status.is_file():
                    time.sleep(.01)
                if self.status.read_text() != 'ready':
                    raise RuntimeError('Input preparation failed.')
                self.outputs.append(index)
            except RuntimeError:
                pass
            finally:
                self.finished.append(index)
        return [lambda index=index: run(index) for index in range(2)]

    def wait_for_workers(self):
        self.assertTrue(all(event.wait(5) for event in self.started))

    def write_expected_inputs(self):
        for name in self.expected:
            (self.inputdir / name).write_bytes(b'fixture')

    def test_workers_wait_until_producer_completes_and_catalogue_matches(self):
        def produce(command, check):
            self.assertEqual(command, ['produce fixtures'])
            self.assertTrue(check)
            self.wait_for_workers()
            self.assertFalse(self.status.exists())
            self.assertEqual(self.finished, [])
            self.write_expected_inputs()

        with patch.object(launcher.subprocess, 'run', side_effect=produce):
            elapsed = launcher._run_prepared(self.workers(), 2, ['produce fixtures'],
                self.status, self.inputdir, self.expected)
        self.assertGreaterEqual(elapsed, 0.)
        self.assertEqual(self.status.read_text(), 'ready')
        self.assertEqual(sorted(self.outputs), [0, 1])
        self.assertEqual(sorted(self.finished), [0, 1])

    def test_producer_failure_is_preserved_and_releases_all_workers(self):
        error = subprocess.CalledProcessError(17, ['prepare'], 'producer failed')

        def fail(command, check):
            self.wait_for_workers()
            raise error

        with patch.object(launcher.subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertEqual(self.outputs, [])
        self.assertEqual(list(self.inputdir.iterdir()), [])

    def test_successful_producer_with_wrong_catalogue_fails_gate(self):
        def produce(command, check):
            self.wait_for_workers()
            (self.inputdir / 'cluster_00001.vtp').write_bytes(b'fixture')

        with patch.object(launcher.subprocess, 'run', side_effect=produce):
            with self.assertRaisesRegex(RuntimeError, 'filenames do not match'):
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertEqual(self.outputs, [])

    def test_sigterm_releases_workers_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(command, check):
            self.wait_for_workers()
            signal.raise_signal(signal.SIGTERM)

        with patch.object(launcher.subprocess, 'run', side_effect=terminate):
            with self.assertRaises(SystemExit) as raised:
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_auto_fallback_preparation_json_includes_pre_fallback_prepare_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = root / 'catalogue'
            inputdir = root / 'inputs'
            outdir = root / 'outputs'
            catalogue.mkdir()
            inputdir.mkdir()
            slicer = root / 'Slicer'
            slicer.write_text('#!/bin/sh\nexit 0\n')
            slicer.chmod(0o755)

            def make_tensor_pd(path):
                pd = vtk.vtkPolyData()
                pts = vtk.vtkPoints()
                pts.InsertNextPoint(0.0, 0.0, 0.0)
                pts.InsertNextPoint(1.0, 0.0, 0.0)
                lines = vtk.vtkCellArray()
                lines.InsertNextCell(2)
                lines.InsertCellPoint(0)
                lines.InsertCellPoint(1)
                pd.SetPoints(pts)
                pd.SetLines(lines)
                tens = numpy_to_vtk(np.tile(np.eye(3).ravel(), (2, 1)), deep=True)
                tens.SetName('Tensors')
                pd.GetPointData().SetTensors(tens)
                wma.io.write_polydata(pd, str(path))

            make_tensor_pd(catalogue / 'cluster_00001.vtp')
            tx = vtk.vtkTransform()
            tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['id'])[0])

            def fake_prepare(cmd, check=False):
                time.sleep(0.05)
                make_tensor_pd(inputdir / 'cluster_00001.vtp')

            def fake_slicer(polydata, transform, inverse, slicer_path, helper, out_d,
                            partition, partitions, log, *args, **kwargs):
                time.sleep(0.05)
                make_tensor_pd(Path(out_d) / 'cluster_00001.vtp')

            argv_auto = [
                'wm_harden_transform.py', '--profile', '-t', str(tfm), '-j', '1',
                '--input-catalogue', str(catalogue),
                str(inputdir), str(outdir), 'auto',
                '--prepare-command', 'produce-tensor',
            ]
            with patch.dict(os.environ, {'PATH': f"{root}{os.pathsep}{os.environ.get('PATH', '')}"}), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_prepare), \
                    patch.object(launcher, '_run_slicer', side_effect=fake_slicer), \
                    patch.object(sys, 'argv', argv_auto):
                launcher.main()

            prep_files = list(outdir.glob('wma-harden-*/preparation.json'))
            self.assertEqual(len(prep_files), 1)
            prep_info = json.loads(prep_files[0].read_text())
            self.assertGreaterEqual(prep_info['prepare_s'], 0.04)
            self.assertGreaterEqual(prep_info['combined_s'], prep_info['prepare_s'] + 0.04)

    def test_keyboard_interrupt_releases_workers_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(command, check):
            self.wait_for_workers()
            raise KeyboardInterrupt()

        with patch.object(launcher.subprocess, 'run', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                launcher._run_prepared(self.workers(), 2, ['prepare'], self.status,
                                       self.inputdir, self.expected)
        self.assertEqual(self.status.read_text(), 'failed')
        self.assertEqual(sorted(self.finished), [0, 1])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_auto_fallback_preparation_json_includes_pre_fallback_prepare_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = root / 'catalogue'
            inputdir = root / 'inputs'
            outdir = root / 'outputs'
            catalogue.mkdir()
            inputdir.mkdir()
            slicer = root / 'Slicer'
            slicer.write_text('#!/bin/sh\nexit 0\n')
            slicer.chmod(0o755)

            def make_tensor_pd(path):
                pd = vtk.vtkPolyData()
                pts = vtk.vtkPoints()
                pts.InsertNextPoint(0.0, 0.0, 0.0)
                pts.InsertNextPoint(1.0, 0.0, 0.0)
                lines = vtk.vtkCellArray()
                lines.InsertNextCell(2)
                lines.InsertCellPoint(0)
                lines.InsertCellPoint(1)
                pd.SetPoints(pts)
                pd.SetLines(lines)
                tens = numpy_to_vtk(np.tile(np.eye(3).ravel(), (2, 1)), deep=True)
                tens.SetName('Tensors')
                pd.GetPointData().SetTensors(tens)
                wma.io.write_polydata(pd, str(path))

            make_tensor_pd(catalogue / 'cluster_00001.vtp')
            tx = vtk.vtkTransform()
            tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['id'])[0])

            def fake_prepare(cmd, check=False):
                time.sleep(0.05)
                make_tensor_pd(inputdir / 'cluster_00001.vtp')

            def fake_slicer(polydata, transform, inverse, slicer_path, helper, out_d,
                            partition, partitions, log, *args, **kwargs):
                time.sleep(0.05)
                make_tensor_pd(Path(out_d) / 'cluster_00001.vtp')

            argv_auto = [
                'wm_harden_transform.py', '--profile', '-t', str(tfm), '-j', '1',
                '--input-catalogue', str(catalogue),
                str(inputdir), str(outdir), 'auto',
                '--prepare-command', 'produce-tensor',
            ]
            with patch.dict(os.environ, {'PATH': f"{root}{os.pathsep}{os.environ.get('PATH', '')}"}), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_prepare), \
                    patch.object(launcher, '_run_slicer', side_effect=fake_slicer), \
                    patch.object(sys, 'argv', argv_auto):
                launcher.main()

            prep_files = list(outdir.glob('wma-harden-*/preparation.json'))
            self.assertEqual(len(prep_files), 1)
            prep_info = json.loads(prep_files[0].read_text())
            self.assertGreaterEqual(prep_info['prepare_s'], 0.04)
            self.assertGreaterEqual(prep_info['combined_s'], prep_info['prepare_s'] + 0.04)


if __name__ == '__main__':
    unittest.main()
