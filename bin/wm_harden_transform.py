#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from joblib import Parallel, delayed, effective_n_jobs
import numpy
import vtk

_repo_root = Path(__file__).resolve().parents[1]
if (_repo_root / 'whitematteranalysis').is_dir():
    _repo_str = str(_repo_root)
    if _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)

import whitematteranalysis as wma
from whitematteranalysis._hemisphere import read_mask_location_from_vtk, split_polydata
from whitematteranalysis._memory import SmallPages


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Harden transforms with independent Slicer processes.",
                                     epilog="Original script by Fan Zhang, fzhang@bwh.harvard.edu.")
    parser.add_argument('-v', '--version', action='version', version='1.0')
    parser.add_argument('inputDirectory', help='Directory of input VTK/VTP files.')
    parser.add_argument('outputDirectory', help='Directory of transformed results.')
    parser.add_argument('Slicer', nargs='?', default=None,
                        help="Path of 3D Slicer, or 'none'/'auto' for pure-Python linear hardening.")
    transforms = parser.add_mutually_exclusive_group(required=True)
    transforms.add_argument('-t', dest='transform_file', help='One transform for all input files.')
    transforms.add_argument('-ts', dest='transform_folder', help='Sorted transforms matching sorted input files.')
    parser.add_argument('-i', action='store_true', dest='inverse_transform', help='Invert the transform.')
    parser.add_argument('-j', dest='numberOfJobs', type=int, default=1,
                        help='Number of Slicer workers. A shared transform uses one batch per worker.')
    parser.add_argument('--split-output-directory', default=None,
                        help='Also write hemisphere subsets from each hardened file.')
    parser.add_argument('--pure-python', action='store_true', dest='pure_python',
                        help='Apply linear ITK .tfm transforms directly in Python (tensor-free polydatas only).')
    parser.add_argument('-no_xvfb', '--no-xvfb', action='store_true', dest='no_xvfb',
                        help='Disable automatic xvfb-run wrapper on headless Linux.')
    parser.add_argument('--profile', action='store_true',
                        help='Keep worker logs and JSON timings after successful processing.')
    parser.add_argument('--input-catalogue',
                        help='Directory listing expected input filenames before preparation.')
    parser.add_argument('--prepare-command', nargs=argparse.REMAINDER,
                        help='Prepare inputs while Slicer starts; place this command last.')
    return parser


def _run_slicer(polydata, transform, inverse, slicer_path, helper, outdir, partition,
                partitions, log, split_output_dir=None, profile_output=None, input_status=None,
                no_xvfb=False):
    command = [str(slicer_path), '--no-main-window', '--disable-modules',
               '--python-script', helper,
               str(polydata), str(transform), str(int(inverse)), str(outdir),
               str(partition), str(partitions)]
    if split_output_dir is not None:
        command.extend(['--split-output-directory', str(split_output_dir)])
    if profile_output is not None:
        command.extend(['--profile-output', str(profile_output)])
    if input_status is not None:
        command.extend(['--input-status', str(input_status)])
    command.extend(['--python-code', 'slicer.app.quit()'])
    use_xvfb = (
        not no_xvfb
        and sys.platform.startswith('linux')
        and not os.environ.get('DISPLAY')
        and os.environ.get('WMA_NO_XVFB', '').strip().lower() not in ('1', 'true', 'yes', 'on')
    )
    if use_xvfb:
        xvfb = shutil.which('xvfb-run')
        if xvfb:
            command = [xvfb, '-a'] + command
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        environment[name] = '1'
    if split_output_dir is not None:
        environment.setdefault('WMA_VTP_COMPRESSION', 'lz4')
    if profile_output is not None:
        environment['WMA_HARDEN_LAUNCH_TIME'] = str(time.monotonic())
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env=environment)


def _publish_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def _cancel_preparation(signum, frame):
    raise SystemExit(128 + signum)


def _run_prepared(runs, workers, command, status, inputdir, expected):
    previous = signal.signal(signal.SIGTERM, _cancel_preparation)
    released = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                futures = [pool.submit(run) for run in runs]
                started = time.perf_counter()
                subprocess.run(command, check=True)
                prepare_seconds = time.perf_counter() - started
                actual = {Path(path).name for path in wma.io.list_vtk_files(str(inputdir))}
                if actual != expected:
                    raise RuntimeError('Prepared input filenames do not match the catalogue.')
                _publish_status(status, 'ready')
                released = True
            finally:
                if not released:
                    _publish_status(status, 'failed')
            for future in futures:
                future.result()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return prepare_seconds


def _split_hardened_polydata(polydata, out_name, split_output_dir):
    found, labels = read_mask_location_from_vtk(polydata)
    if not found:
        raise RuntimeError('HemisphereLocation is missing from transformed file: ' + str(out_name))
    if (labels == 0).sum() > 1:
        raise RuntimeError('Hemisphere labels are incomplete: ' + str(out_name))
    split_polydata(polydata, labels, os.path.basename(str(out_name)), str(split_output_dir))


def _load_linear_matrices(transform_paths, inverse, allow_fallback):
    matrix_cache = {}
    for tf_path in transform_paths:
        key = str(tf_path)
        if key not in matrix_cache:
            mat = wma.transform.read_itk_affine_matrix_ras(key, inverse=inverse)
            if mat is None:
                if allow_fallback:
                    return None
                raise RuntimeError(
                    'Pure-Python hardening requires linear ITK affine .tfm transforms; '
                    'unsupported transform: ' + key)
            matrix_cache[key] = mat
    return matrix_cache


def _is_lps_polydata(polydata):
    """Return True if polydata FieldData SPACE explicitly declares LPS coordinates."""
    if polydata is None:
        return False
    field_data = polydata.GetFieldData()
    if field_data is None:
        return False
    arr = field_data.GetAbstractArray('SPACE')
    if arr is None or not hasattr(arr, 'GetValue') or arr.GetNumberOfValues() == 0:
        return False
    return str(arr.GetValue(0)).strip().upper() == 'LPS'


def _set_ras_space_field(polydata):
    space = vtk.vtkStringArray()
    space.SetName('SPACE')
    space.InsertNextValue('RAS')
    polydata.GetFieldData().AddArray(space)


def _run_pure_python(inputs, transform_paths, matrix_cache, outdir, workers,
                     split_output_dir=None, allow_fallback=False):
    cached_polydatas = {}
    if allow_fallback:
        for pd_path in inputs:
            out_name = outdir / Path(pd_path).name
            if out_name.exists():
                continue
            pd = wma.io.read_polydata(str(pd_path))
            if wma.transform.has_tensor_arrays(pd):
                cached_polydatas.clear()
                return False
            cached_polydatas[str(pd_path)] = pd

    print('Harden %d files with pure-Python affine (%d worker(s)).' % (len(inputs), workers), flush=True)

    def _harden_one(item):
        pd_path, tf_path = item
        out_name = outdir / Path(pd_path).name
        if out_name.exists():
            cached_polydatas.pop(str(pd_path), None)
            if split_output_dir is None:
                return
            split_files = [split_output_dir / folder / Path(pd_path).name
                           for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere',
                                          'tracts_commissural')]
            if all(path.is_file() for path in split_files):
                return
            with SmallPages():
                _split_hardened_polydata(wma.io.read_polydata(str(out_name)), out_name, split_output_dir)
            return

        with SmallPages():
            pd = cached_polydatas.pop(str(pd_path), None)
            if pd is None:
                pd = wma.io.read_polydata(str(pd_path))
            if wma.transform.has_tensor_arrays(pd):
                raise RuntimeError(
                    'Pure-Python hardening cannot rotate tensor/vector/normal arrays; '
                    'use 3D Slicer with SlicerDMRI: ' + str(pd_path))
            is_lps = _is_lps_polydata(pd)
            if pd.GetNumberOfCells() == 0 and not is_lps:
                if os.path.splitext(str(out_name))[1] == os.path.splitext(str(pd_path))[1]:
                    shutil.copyfile(str(pd_path), str(out_name))
                else:
                    wma.io.write_polydata(pd, str(out_name))
                hardened = pd
            else:
                mat = matrix_cache[str(tf_path)]
                if is_lps:
                    mat = mat @ numpy.diag([-1.0, -1.0, 1.0, 1.0])
                hardened = wma.transform.apply_affine_to_polydata(pd, mat)
                if out_name.suffix.lower() == '.vtp' or is_lps:
                    _set_ras_space_field(hardened)
                wma.io.write_polydata(hardened, str(out_name))
            if split_output_dir is not None:
                _split_hardened_polydata(hardened, out_name, split_output_dir)

    items = list(zip(inputs, transform_paths))
    if workers <= 1:
        for item in items:
            _harden_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_harden_one, items))
    return True


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    inputdir, outdir = map(lambda p: Path(p).resolve(), (args.inputDirectory, args.outputDirectory))
    split_output_dir = (Path(args.split_output_directory).resolve()
                        if args.split_output_directory else None)
    if bool(args.input_catalogue) != bool(args.prepare_command):
        parser.error('--input-catalogue and --prepare-command must be used together.')
    catalogue = Path(args.input_catalogue).resolve() if args.input_catalogue else inputdir
    if not catalogue.is_dir():
        parser.error('Input catalogue does not exist: ' + str(catalogue))
    if not args.prepare_command and not inputdir.is_dir():
        parser.error('Input directory does not exist: ' + str(inputdir))

    slicer_arg = args.Slicer
    slicer_mode_keyword = slicer_arg.lower() if isinstance(slicer_arg, str) else None
    if not args.pure_python and slicer_arg is None:
        parser.error('Slicer executable is required unless --pure-python or Slicer=none/auto is given.')

    pure_python_requested = args.pure_python or slicer_mode_keyword == 'none'
    auto_requested = (not pure_python_requested) and slicer_mode_keyword == 'auto'

    slicer_path = None
    if not pure_python_requested:
        if auto_requested:
            found_slicer = shutil.which('Slicer')
            slicer_path = Path(found_slicer).resolve() if found_slicer else None
        else:
            slicer_path = Path(slicer_arg).resolve()
            if not slicer_path.is_file():
                parser.error('Slicer executable does not exist: ' + str(slicer_path))

    outdir.mkdir(parents=True, exist_ok=True)
    if split_output_dir is not None:
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            (split_output_dir / folder).mkdir(parents=True, exist_ok=True)
    inputs = [str(inputdir / Path(path).name) for path in wma.io.list_vtk_files(str(catalogue))]
    if args.prepare_command and not inputs:
        parser.error('The preparation catalogue must contain input VTK/VTP files.')
    workers = min(effective_n_jobs(args.numberOfJobs), max(1, len(inputs)))
    helper = str(Path(__file__).resolve().with_name('harden_transform_with_slicer.py'))
    if args.transform_folder:
        transform_path = Path(args.transform_folder).resolve()
        if not transform_path.is_dir():
            parser.error('Transform directory does not exist: ' + str(transform_path))
        transforms = wma.io.list_transform_files(str(transform_path))
        if len(inputs) != len(transforms):
            parser.error('The input and transform file counts must match.')
        transform_list = transforms
        tasks = [(polydata, transform, 0, 1) for polydata, transform in zip(inputs, transforms)]
    else:
        transform_path = Path(args.transform_file).resolve()
        if not transform_path.is_file():
            parser.error('Transform file does not exist: ' + str(transform_path))
        transform_list = [transform_path] * len(inputs)
        tasks = [(inputdir, transform_path, index, workers) for index in range(workers)] if inputs else []

    if not tasks:
        return

    marker = outdir / 'transform_complete.log'
    if marker.exists():
        marker.unlink()

    prepare_already_ran = False
    prepare_seconds = None
    expected = {Path(path).name for path in inputs}

    if pure_python_requested or auto_requested:
        matrix_cache = _load_linear_matrices(
            transform_list, args.inverse_transform, allow_fallback=auto_requested)
        if matrix_cache is not None:
            if args.prepare_command:
                prep_started = time.perf_counter()
                subprocess.run(args.prepare_command, check=True)
                prepare_seconds = time.perf_counter() - prep_started
                prepare_already_ran = True
                actual = {Path(path).name for path in wma.io.list_vtk_files(str(inputdir))}
                if actual != expected:
                    raise RuntimeError('Prepared input filenames do not match the catalogue.')
            if _run_pure_python(inputs, transform_list, matrix_cache, outdir,
                                workers, split_output_dir, allow_fallback=auto_requested):
                actual = {Path(path).name for path in wma.io.list_vtk_files(str(outdir))}
                if not expected.issubset(actual):
                    raise RuntimeError('Transformed files are missing in ' + str(outdir))
                if split_output_dir is not None:
                    for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                        split_files = {Path(path).name for path in wma.io.list_vtk_files(str(split_output_dir / folder))}
                        if not expected.issubset(split_files):
                            raise RuntimeError('Hemisphere outputs are missing from %s' % folder)
                marker.write_text('Transforms completed for %d files.\n' % len(inputs))
                print('Transforms completed for %d files.' % len(inputs), flush=True)
                return
        if slicer_path is None or not slicer_path.is_file():
            raise RuntimeError('Slicer executable is required for non-linear transforms or tensor-bearing inputs.')

    print('Harden %d files with %d Slicer worker(s).' % (len(inputs), workers), flush=True)
    logs = Path(tempfile.mkdtemp(prefix='wma-harden-', dir=str(outdir)))
    print('Slicer worker logs: ' + str(logs), flush=True)
    run_prepare_via_gate = bool(args.prepare_command and not prepare_already_ran)
    status = logs / 'input.status' if run_prepare_via_gate else None
    xvfb_kw = {'no_xvfb': True} if args.no_xvfb else {}
    runs = [partial(_run_slicer, polydata, transform, args.inverse_transform, slicer_path, helper,
            outdir, index, count, logs/('worker_%04d.log' % task), split_output_dir,
            logs/('worker_%04d.json' % task) if args.profile else None, status, **xvfb_kw)
        for task, (polydata, transform, index, count) in enumerate(tasks)]
    started = time.perf_counter()
    if run_prepare_via_gate:
        prepare_seconds = _run_prepared(runs, workers, args.prepare_command, status, inputdir, expected)
        if args.profile:
            (logs / 'preparation.json').write_text(json.dumps({
                'prepare_s': prepare_seconds, 'combined_s': time.perf_counter() - started,
                'command': args.prepare_command}, indent=2) + '\n')
    else:
        Parallel(n_jobs=workers, prefer='threads')(delayed(run)() for run in runs)
        if args.profile and prepare_already_ran:
            (logs / 'preparation.json').write_text(json.dumps({
                'prepare_s': prepare_seconds, 'combined_s': time.perf_counter() - started,
                'command': args.prepare_command}, indent=2) + '\n')
    actual = {Path(path).name for path in wma.io.list_vtk_files(str(outdir))}
    if not expected.issubset(actual):
        raise RuntimeError('Transformed files are missing; inspect Slicer logs: ' + str(logs))
    if split_output_dir is not None:
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            split_files = {Path(path).name for path in wma.io.list_vtk_files(str(split_output_dir / folder))}
            if not expected.issubset(split_files):
                raise RuntimeError('Hemisphere outputs are missing from %s; inspect Slicer logs: %s'
                                   % (folder, logs))
    if not args.profile:
        shutil.rmtree(logs)
    marker.write_text('Transforms completed for %d files.\n' % len(inputs))
    print('Transforms completed for %d files.' % len(inputs), flush=True)


if __name__ == '__main__':
    main()
