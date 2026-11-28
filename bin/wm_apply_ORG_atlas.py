#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-process in-memory ORG atlas parcellation pipeline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import gc
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy
import vtk
from vtk.util.numpy_support import numpy_to_vtk

_repo_root = Path(__file__).resolve().parents[1]
if (_repo_root / 'whitematteranalysis').is_dir():
    _repo_str = str(_repo_root)
    if _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)

import whitematteranalysis as wma
from whitematteranalysis.gpu.atlas_cache import load_reference

from whitematteranalysis._hemisphere import (
    assess_polydata,
    load_location_policy,
    read_mask_location_from_vtk,
    split_polydata_subsets,
)
from whitematteranalysis._memory import SmallPages


HEMISPHERIC_TRACTS = (
    "T_AF", "T_CB", "T_CPC", "T_MdLF", "T_PLIC", "T_SLF-I", "T_SLF-II", "T_SLF-III",
    "T_EC", "T_EmC", "T_ICP", "T_ILF", "T_IOFF", "T_UF",
    "T_Intra-CBLM-I&P", "T_Intra-CBLM-PaT", "T_CR-F", "T_CR-P", "T_CST",
    "T_SF", "T_SO", "T_SP", "T_TF", "T_TO", "T_TP",
    "T_Sup-F", "T_Sup-FP", "T_Sup-O", "T_Sup-OT", "T_Sup-P", "T_Sup-PO", "T_Sup-PT", "T_Sup-T",
)

COMMISSURAL_TRACTS = (
    "T_CC1", "T_CC2", "T_CC3", "T_CC4", "T_CC5", "T_CC6", "T_CC7", "T_MCP",
)

_PRESETS = {
    'rigid_affine_fast': {
        'sigma': [30, 30, 10, 7.5, 5],
        'iterations': [2, 2, 2, 2, 2],
        'maxfun': [45, 45, 60, 75, 90],
        'mean_brain_size': [1000, 1000, 3000, 4000, 5000],
        'subject_brain_size': [250, 250, 1500, 1750, 2000],
        'initial_step': [10, 10, 5, 5, 5],
        'final_step': [5, 5, 2, 2, 2],
        'points_per_fiber': 10,
        'rigid_scale': [1, 1, 1, 0, 0],
    },
    'affine': {
        'sigma': [20, 10, 10, 5, 5],
        'iterations': [3, 3, 3, 3, 3],
        'maxfun': [50, 80, 80, 80, 100],
        'mean_brain_size': [2000, 3000, 3000, 3500, 4000],
        'subject_brain_size': [1000, 1500, 1500, 1500, 1500],
        'initial_step': [10, 8, 5, 3, 2],
        'final_step': [2, 2, 1, 1, 1],
        'points_per_fiber': 10,
    },
    'nonrigid': {
        'grid_resolution': [4, 5, 6, 7, 8],
        'initial_step': [5, 4, 3, 2, 1.5],
        'final_step': [3, 3, 2, 1, 1],
        'sigma': [5, 3, 2, 2, 2],
        'iterations': [10, 10, 8, 5, 2],
        'mean_brain_size': [1000, 1000, 1000, 1000, 1000],
        'subject_brain_size': [500, 500, 500, 500, 500],
        'maxfun': [205, 390, 670, 1049, 1556],
        'points_per_fiber': 3,
    },
}


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Apply the ORG white matter atlas in a single in-memory Python process.")
    parser.add_argument('-i', '--input', required=True,
                        help='Input tractography (.vtk or .vtp) in RAS coordinates.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output root directory; subject outputs are published under <output>/<caseID>.')
    parser.add_argument('-a', '--atlas', required=True,
                        help='Atlas root directory (ORG, East-West, NABA, or parent atlas folder).')
    parser.add_argument('--atlas-family', choices=('auto', 'org', 'east-west', 'naba'), default='auto',
                        help='Atlas family layout: auto (default), org, east-west, or naba.')
    parser.add_argument('-s', '--slicer', default=None,
                        help='Path to 3D Slicer executable (optional for tensor-free rigid mode).')
    parser.add_argument('-r', '--reg-mode', choices=('rig', 'nonrig'), default='rig',
                        help='Registration mode: rig (rigid+affine, default) or nonrig (affine+nonrigid).')
    parser.add_argument('-l', '--fiber-length', type=float, default=40.0,
                        help='Minimum fiber length in mm (default: 40).')
    parser.add_argument('--fiber-length-max', type=float, default=260.0,
                        help='Maximum fiber length in mm for registration sampling (default: 260).')
    parser.add_argument('-t', '--transform', default=None,
                        help='Optional initial ITK .tfm transform into adult atlas size.')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of CPU/IO worker threads (default: 1).')
    parser.add_argument('--keep-intermediates', type=int, choices=(0, 1, 2), default=0,
                        help='0: AnatomicalTracts + qc_summary only (default); '
                             '1: also keep SeparatedClusters and transforms; 2: keep all stages.')
    parser.add_argument('--no-atomic-publish', action='store_true', default=False,
                        help='Write directly to <output>/<caseID> instead of staging and atomic rename.')
    parser.add_argument('-no_xvfb', '--no-xvfb', action='store_true', dest='no_xvfb',
                        help='Disable automatic xvfb-run wrapper when calling Slicer.')
    parser.add_argument('--outlier-std', type=float, default=2.0,
                        help='Outlier standard deviation threshold (default: 2.0).')
    parser.add_argument('--outlier-sigma', type=float, default=20.0,
                        help='Local sigma in mm for cluster outlier removal (default: 20.0).')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detailed per-cluster diagnostics.')
    wma.gpu.add_gpu_arguments(parser)
    return parser


def _effective_workers(requested):
    if requested is None or requested == 0:
        return 1
    if requested < 0:
        return max(1, (os.cpu_count() or 1) + 1 + requested)
    return max(1, int(requested))


def _matrix_from_vtk_transform(vtk_tx):
    m = vtk_tx.GetMatrix()
    return numpy.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=numpy.float64)


def _add_ras_space_field(polydata):
    space = vtk.vtkStringArray()
    space.SetName('SPACE')
    space.InsertNextValue('RAS')
    polydata.GetFieldData().AddArray(space)


def _run_registration_stage(subject_pd, atlas_pd, subject_id, atlas_id, preset_name,
                            fiber_length, fiber_length_max, reg_outdir, save_iter_tfms):
    preset = _PRESETS[preset_name]
    register = wma.congeal_to_atlas.SubjectToAtlasRegistration()
    register.output_directory = str(reg_outdir)
    register.input_polydata_filename = subject_id
    register.fiber_length = fiber_length
    register.fiber_length_max = fiber_length_max
    register.save_iteration_transforms = save_iter_tfms
    if preset_name == 'nonrigid':
        register.mode = "Nonrigid"
    register.set_subject(subject_pd, subject_id)
    register.set_atlas(atlas_pd, atlas_id)
    register.points_per_fiber = preset['points_per_fiber']

    rigid_scale = preset.get('rigid_scale')
    for scale_idx in range(len(preset['sigma'])):
        register.sigma = preset['sigma'][scale_idx]
        register.initial_step = preset['initial_step'][scale_idx]
        register.final_step = preset['final_step'][scale_idx]
        register.maxfun = preset['maxfun'][scale_idx]
        register.mean_brain_size = preset['mean_brain_size'][scale_idx]
        register.subject_brain_size = preset['subject_brain_size'][scale_idx]
        if preset_name == 'nonrigid':
            register.nonrigid_grid_resolution = preset['grid_resolution'][scale_idx]
            register.update_nonrigid_grid()
        if rigid_scale is not None:
            register.mode = "Rigid" if rigid_scale[scale_idx] else "Affine"
        for _ in range(preset['iterations'][scale_idx]):
            register.iterate()

    return register.transform


def _parse_mrml_clusters(mrml_path, valid_cluster_names):
    members = []
    text = Path(mrml_path).read_text(encoding='utf-8', errors='replace')
    for line in text.splitlines():
        for match in re.findall(r'cluster_\d+\.vtp', line):
            if match not in valid_cluster_names:
                raise RuntimeError(f"MRML {mrml_path} references missing cluster {match}")
            members.append(match)
    return members


def _collect_tract_tasks(fc_atlas_dir, atlas_cluster_names, location_data,
                         hemispheric_tracts=None, commissural_tracts=None):
    valid_names = set(atlas_cluster_names)
    hemi_defs = hemispheric_tracts if hemispheric_tracts is not None else HEMISPHERIC_TRACTS
    comm_defs = commissural_tracts if commissural_tracts is not None else COMMISSURAL_TRACTS
    standard_hemi = [(t, fc_atlas_dir / f"{t}.mrml") for t in hemi_defs]
    standard_comm = [(t, fc_atlas_dir / f"{t}.mrml") for t in comm_defs]
    all_standard_present = all(p.is_file() for _, p in standard_hemi + standard_comm)
    if len(atlas_cluster_names) >= 800 and not all_standard_present:
        missing = [str(p) for _, p in standard_hemi + standard_comm if not p.is_file()]
        raise FileNotFoundError(f"Missing atlas MRML files: {missing[:5]}")

    tasks = []
    seen_mrml = set()
    for tract, mrml_path in standard_hemi:
        if not mrml_path.is_file():
            continue
        seen_mrml.add(mrml_path.name)
        members = _parse_mrml_clusters(mrml_path, valid_names)
        tasks.append((f"{tract}_left.vtp", 'tracts_left_hemisphere', members))
        tasks.append((f"{tract}_right.vtp", 'tracts_right_hemisphere', members))

    for tract, mrml_path in standard_comm:
        if not mrml_path.is_file():
            continue
        seen_mrml.add(mrml_path.name)
        members = _parse_mrml_clusters(mrml_path, valid_names)
        tasks.append((f"{tract}.vtp", 'tracts_commissural', members))

    # Support custom T_*.mrml definitions in synthetic or custom atlases
    for mrml_path in sorted(fc_atlas_dir.glob('T_*.mrml')):
        if mrml_path.name in seen_mrml:
            continue
        tract = mrml_path.stem
        members = _parse_mrml_clusters(mrml_path, valid_names)
        is_comm = False
        if members:
            idx_map = {name: i for i, name in enumerate(atlas_cluster_names)}
            labels = [location_data[idx_map[m]][1] for m in members if m in idx_map]
            is_comm = bool(labels) and all(lbl in (b'c', b'ng') for lbl in labels)
        if is_comm:
            tasks.append((f"{tract}.vtp", 'tracts_commissural', members))
        else:
            tasks.append((f"{tract}_left.vtp", 'tracts_left_hemisphere', members))
            tasks.append((f"{tract}_right.vtp", 'tracts_right_hemisphere', members))

    if not tasks:
        raise FileNotFoundError(f"No T_*.mrml tract definitions found in {fc_atlas_dir}")
    return tasks


def _assemble_single_tract(member_pds, output_path, indexed_source=None):
    if indexed_source is not None:
        (hardened_geometry, all_lines, point_specs, cell_specs,
         tensor_names, extra_per_line_point_arrays) = indexed_source
        pd_appended = wma.filter._build_polydata_from_indexed_lines(
            hardened_geometry,
            all_lines,
            point_specs=point_specs,
            cell_specs=cell_specs,
            tensor_names=tensor_names,
            extra_per_line_point_arrays=extra_per_line_point_arrays,
        )
        _add_ras_space_field(pd_appended)
        wma.io.write_polydata(pd_appended, str(output_path))
        return int(pd_appended.GetNumberOfLines()), int(pd_appended.GetNumberOfPoints())

    appender = vtk.vtkAppendPolyData()
    for c_idx, pd_cluster in enumerate(member_pds):
        pd_copy = vtk.vtkPolyData()
        pd_copy.ShallowCopy(pd_cluster)
        vtk_array = numpy_to_vtk(
            numpy.full(pd_copy.GetNumberOfPoints(), c_idx, dtype=numpy.int32), deep=True)
        vtk_array.SetName('cluster_idx')
        pd_copy.GetPointData().AddArray(vtk_array)
        if vtk.vtkVersion().GetVTKMajorVersion() >= 6.0:
            appender.AddInputData(pd_copy)
        else:
            appender.AddInput(pd_copy)
    appender.Update()
    pd_appended = appender.GetOutput()
    wma.io.write_polydata(pd_appended, str(output_path))
    return int(pd_appended.GetNumberOfLines()), int(pd_appended.GetNumberOfPoints())


@contextlib.contextmanager
def _default_lz4_compression():
    prev_comp = os.environ.get('WMA_VTP_COMPRESSION')
    if not prev_comp:
        os.environ['WMA_VTP_COMPRESSION'] = 'lz4'
    try:
        yield
    finally:
        if prev_comp is None:
            os.environ.pop('WMA_VTP_COMPRESSION', None)
        else:
            os.environ['WMA_VTP_COMPRESSION'] = prev_comp


def _invoke_slicer_hardening(in_dir, out_dir, tfm_path, slicer_path, jobs,
                             inverse=True, split_output_dir=None, no_xvfb=False):
    harden_cli = str(Path(__file__).resolve().with_name('wm_harden_transform.py'))
    cmd = [
        sys.executable, harden_cli,
        '-j', str(jobs),
        '-t', str(tfm_path),
    ]
    if inverse:
        cmd.append('-i')
    if split_output_dir is not None:
        cmd.extend(['--split-output-directory', str(split_output_dir)])
    if no_xvfb:
        cmd.append('-no_xvfb')
    cmd.extend(['--profile', str(in_dir), str(out_dir), str(slicer_path)])
    env = os.environ.copy()
    if not env.get('WMA_VTP_COMPRESSION'):
        env['WMA_VTP_COMPRESSION'] = 'lz4'
    if (_repo_root / 'whitematteranalysis').is_dir():
        existing_pp = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{_repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(_repo_root)
    subprocess.run(cmd, env=env, check=True)
    # Inspect profile json to detect FiberBundleNode vs ModelNode backend
    backend = 'slicer_fiber_bundle'
    for pjson in sorted(Path(out_dir).glob('wma-harden-*/worker_*.json')):
        try:
            info = json.loads(pjson.read_text())
            if 'backend' in info:
                backend = info['backend']
                break
        except Exception:
            pass
    for log_dir in Path(out_dir).glob('wma-harden-*'):
        if log_dir.is_dir():
            shutil.rmtree(log_dir, ignore_errors=True)
    return backend


def _dispatch_cluster_workers(atlas_cluster_paths, backend, workers, worker_fn):
    tasks = list(enumerate(atlas_cluster_paths))
    if backend == 'gpu' and workers > 1:
        import torch
        base_device = wma.gpu.get_device()
        n_gpus = wma.gpu.visible_device_count() if base_device.type != 'mps' else 1
        thread_local = threading.local()

        def _run_gpu(item):
            c_idx, ca_path = item
            dev = wma.gpu.get_device(c_idx % n_gpus) if n_gpus > 1 else base_device
            dev_ctx = (torch.cuda.device(dev)
                       if dev.type == 'cuda' and torch.cuda.is_available()
                       else contextlib.nullcontext())
            with dev_ctx:
                if dev.type == 'cuda':
                    streams = getattr(thread_local, 'streams', None)
                    if streams is None:
                        streams = {}
                        thread_local.streams = streams
                    stream = streams.get(dev.index)
                    if stream is None:
                        stream = torch.cuda.Stream(device=dev)
                        streams[dev.index] = stream
                    stream_ctx = torch.cuda.stream(stream)
                else:
                    stream_ctx = contextlib.nullcontext()
                with stream_ctx:
                    worker_fn(c_idx, ca_path, device=dev)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run_gpu, tasks))
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: worker_fn(item[0], item[1]), tasks))
    else:
        for c_idx, ca_path in tasks:
            worker_fn(c_idx, ca_path)


def _load_cluster_reference(ca_path, sigmasq, distance_method, atlas_bilateral,
                            backend, device=None, ref_cache=None, ref_lock=None):
    if ref_cache is not None:
        key = (ca_path, float(sigmasq), str(distance_method), bool(atlas_bilateral), str(backend))
        if ref_lock is not None:
            with ref_lock:
                cached = ref_cache.get(key)
        else:
            cached = ref_cache.get(key)
        if cached is not None:
            return cached
        loaded = load_reference(
            ca_path, sigmasq, distance_method, atlas_bilateral, backend, device=device)
        if ref_lock is not None:
            with ref_lock:
                ref_cache.setdefault(key, loaded)
                return ref_cache[key]
        ref_cache[key] = loaded
        return loaded
    return load_reference(
        ca_path, sigmasq, distance_method, atlas_bilateral, backend, device=device)


def _compute_cluster_outlier_scores(subject_array, atlas_array, total_similarity_atlas,
                                    sigmasq, outlier_std, atlas_bilateral, backend,
                                    device=None, verbose=False, c_num=0):
    distance_method = 'Mean'
    n_atlas = len(atlas_array)
    if backend == 'gpu':
        similarity_sums, _ = wma.gpu.distance.similarity_row_sums_from_arrays(
            subject_array, atlas_array, 0.0, distance_method=distance_method,
            bilateral=atlas_bilateral, sigmasq=sigmasq,
            sqrt_distance_sums=False, device=device)
        total_similarity_subject = similarity_sums / n_atlas
        (total_similarity_subject, total_similarity_atlas,
         refined) = wma.gpu.distance.refine_outlier_probabilities_from_arrays(
            subject_array, atlas_array, total_similarity_subject,
            total_similarity_atlas, outlier_std,
            sigmasq=sigmasq, bilateral=atlas_bilateral,
            distance_method=distance_method)
        if verbose and refined:
            print(f"cluster {c_num} refined near-cutoff probabilities with CPU/NumPy")
    else:
        matrix = wma.gpu.cpu_backend._distance_matrix
        cluster_distances = matrix(
            subject_array, atlas_array, 0.0,
            distance_method=distance_method, bilateral=atlas_bilateral,
            number_of_jobs=1).T
        cluster_similarity_subject = wma.similarity.distance_to_similarity(
            cluster_distances, sigmasq)
        total_similarity_subject = numpy.sum(cluster_similarity_subject, axis=0) / n_atlas

    mean_sim_atlas = numpy.mean(total_similarity_atlas)
    cluster_std_atlas = numpy.std(total_similarity_atlas)
    cutoff = mean_sim_atlas - outlier_std * cluster_std_atlas
    return total_similarity_subject, cutoff


_PT_RANGE_40 = numpy.arange(40, dtype=numpy.float64)


def _assess_cluster_hemisphere_indexed(kept_lines, c_idx, location_data, geometry,
                                       has_existing_hemi, existing_hemi_labels):
    n_kept = len(kept_lines)
    if n_kept == 0:
        return numpy.zeros(0, dtype=numpy.float64)
    if has_existing_hemi:
        return numpy.asarray(existing_hemi_labels[kept_lines], dtype=numpy.float64)

    label_group = None if location_data is None else location_data[c_idx][1]
    labels = numpy.zeros(n_kept, dtype=numpy.float64)
    if label_group in (b'c', b'ng'):
        labels[:] = 3.0
        return labels

    offsets, conn, points = geometry
    starts = offsets[:-1][kept_lines]
    lengths = offsets[1:][kept_lines] - starts
    steps = (lengths.astype(numpy.float64) - 1.0) / 39.0
    local_ids = numpy.rint(_PT_RANGE_40 * steps[:, numpy.newaxis]).astype(numpy.int64)
    global_ids = conn[starts[:, numpy.newaxis] + local_ids]
    r_vals = points[global_ids, 0].astype(numpy.float64, copy=False)
    frac_left = numpy.count_nonzero(r_vals < 0.0, axis=1) / 40.0

    thresh = 0.5001 if label_group == b'h' else 0.6
    is_left = frac_left > thresh
    is_right = frac_left < (1.0 - thresh)
    is_comm = ~(is_left | is_right)
    labels[is_left] = 1.0
    labels[is_right] = 2.0
    if numpy.any(is_comm):
        if label_group == b'h':
            labels[is_comm] = 1.0 if int(numpy.count_nonzero(is_left)) <= int(numpy.count_nonzero(is_right)) else 2.0
        else:
            labels[is_comm] = 3.0
    return labels


def _outlier_and_assess_indexed(cluster_line_indices, queries_15, geometry,
                                has_existing_hemi, existing_hemi_labels,
                                atlas_cluster_paths, atlas_cluster_names,
                                atlas_bilateral, location_data, outlier_sigma,
                                outlier_std, backend, workers, verbose=False,
                                ref_cache=None, ref_lock=None):
    num_clusters = len(atlas_cluster_paths)
    kept_by_hemi = {
        'tracts_left_hemisphere': {},
        'tracts_right_hemisphere': {},
        'tracts_commissural': {},
    }
    stats = [{'removed': 0, 'retained': 0} for _ in range(num_clusters)]
    assessed_hemi = None if has_existing_hemi else numpy.zeros(len(queries_15), dtype=numpy.float32)
    sigmasq = outlier_sigma * outlier_sigma
    distance_method = 'Mean'
    empty_idx = numpy.empty(0, dtype=numpy.int64)

    def _process_one(c_idx, ca_path, device=None):
        c_num = c_idx + 1
        name = atlas_cluster_names[c_idx]
        c_lines = cluster_line_indices[c_idx]
        n_subj = int(len(c_lines))
        if n_subj == 0:
            kept_by_hemi['tracts_left_hemisphere'][name] = empty_idx
            kept_by_hemi['tracts_right_hemisphere'][name] = empty_idx
            kept_by_hemi['tracts_commissural'][name] = empty_idx
            return

        atlas_array, total_similarity_atlas = _load_cluster_reference(
            ca_path, sigmasq, distance_method, atlas_bilateral, backend,
            device=device, ref_cache=ref_cache, ref_lock=ref_lock)
        n_atlas = len(atlas_array)

        if n_atlas == 0:
            keep_lines = c_lines
            n_rejected = 0
            n_retained = n_subj
        else:
            subject_array = queries_15[c_lines]
            total_similarity_subject, cutoff = _compute_cluster_outlier_scores(
                subject_array, atlas_array, total_similarity_atlas,
                sigmasq, outlier_std, atlas_bilateral, backend,
                device=device, verbose=verbose, c_num=c_num)
            keep_mask = total_similarity_subject >= cutoff
            keep_lines = c_lines[keep_mask]
            n_retained = int(len(keep_lines))
            n_rejected = n_subj - n_retained

        labels = _assess_cluster_hemisphere_indexed(
            keep_lines, c_idx, location_data, geometry,
            has_existing_hemi, existing_hemi_labels)
        if len(numpy.where(labels == 0)[0]) > 1:
            raise RuntimeError(f"Hemisphere labels incomplete for cluster {c_num:05d}")
        if assessed_hemi is not None and n_retained > 0:
            assessed_hemi[keep_lines] = labels.astype(numpy.float32)

        kept_by_hemi['tracts_left_hemisphere'][name] = keep_lines[labels == 1]
        kept_by_hemi['tracts_right_hemisphere'][name] = keep_lines[labels == 2]
        kept_by_hemi['tracts_commissural'][name] = keep_lines[labels == 3]
        stats[c_idx]['removed'] = n_rejected
        stats[c_idx]['retained'] = n_retained

    _dispatch_cluster_workers(atlas_cluster_paths, backend, workers, _process_one)
    total_removed = sum(s['removed'] for s in stats)
    total_retained = sum(s['retained'] for s in stats)
    nonempty = sum(1 for s in stats if s['retained'] > 0)
    return kept_by_hemi, assessed_hemi, total_removed, total_retained, nonempty


def _outlier_and_assess_clusters(cluster_pds, atlas_cluster_paths, atlas_bilateral,
                                 location_data, outlier_sigma, outlier_std,
                                 backend, workers, verbose=False,
                                 ref_cache=None, ref_lock=None):
    num_clusters = len(atlas_cluster_paths)
    cleaned_pds = [None] * num_clusters
    log_rows = [None] * num_clusters
    stats = [{'removed': 0, 'retained': 0} for _ in range(num_clusters)]
    sigmasq = outlier_sigma * outlier_sigma
    distance_method = 'Mean'

    def _process_one(c_idx, ca_path, device=None):
        c_num = c_idx + 1
        pd_subject = cluster_pds[c_idx]
        cluster_pds[c_idx] = None
        n_subj = pd_subject.GetNumberOfLines()
        if n_subj == 0:
            with SmallPages():
                pd_out, labels, _ = assess_polydata(pd_subject, c_idx, location_data)
            if len(numpy.where(labels == 0)[0]) > 1:
                raise RuntimeError(f"Hemisphere labels incomplete for cluster {c_num:05d}")
            cleaned_pds[c_idx] = pd_out
            log_rows[c_idx] = f"{c_num}\t{n_subj}\t 0 \t 0 \t 0 \t 0 \t 0 \t 0 \t 0"
            return

        if backend:
            atlas_array, total_similarity_atlas = _load_cluster_reference(
                ca_path, sigmasq, distance_method, atlas_bilateral, backend,
                device=device, ref_cache=ref_cache, ref_lock=ref_lock)
            n_atlas = len(atlas_array)
        else:
            pd_atlas = wma.io.read_polydata(ca_path)
            n_atlas = pd_atlas.GetNumberOfLines()

        if n_atlas == 0:
            with SmallPages():
                pd_out, labels, _ = assess_polydata(pd_subject, c_idx, location_data)
            if len(numpy.where(labels == 0)[0]) > 1:
                raise RuntimeError(f"Hemisphere labels incomplete for cluster {c_num:05d}")
            cleaned_pds[c_idx] = pd_out
            stats[c_idx]['retained'] = n_subj
            log_rows[c_idx] = f"{c_num}\t{n_subj}\t{n_subj}\t0\t0.0\t0\t0\t0\t0"
            return

        if backend:
            subject_array = wma.gpu.fiber_tensors.polydata_to_numpy(pd_subject, 15)
            if backend == 'gpu':
                similarity_sums, distance_sums = wma.gpu.distance.similarity_row_sums_from_arrays(
                    subject_array, atlas_array, 0.0, distance_method=distance_method,
                    bilateral=atlas_bilateral, sigmasq=sigmasq,
                    sqrt_distance_sums=True, device=device)
                total_similarity_subject = similarity_sums / n_atlas
                cluster_mean_distances = distance_sums / n_atlas
                (total_similarity_subject, total_similarity_atlas,
                 refined) = wma.gpu.distance.refine_outlier_probabilities_from_arrays(
                    subject_array, atlas_array, total_similarity_subject,
                    total_similarity_atlas, outlier_std,
                    sigmasq=sigmasq, bilateral=atlas_bilateral,
                    distance_method=distance_method)
                if verbose and refined:
                    print(f"cluster {c_num} refined near-cutoff probabilities with CPU/NumPy")
            else:
                matrix = wma.gpu.cpu_backend._distance_matrix
                cluster_distances = matrix(
                    subject_array, atlas_array, 0.0,
                    distance_method=distance_method, bilateral=atlas_bilateral,
                    number_of_jobs=1).T
                cluster_similarity_subject = wma.similarity.distance_to_similarity(
                    cluster_distances, sigmasq)
                total_similarity_subject = numpy.sum(cluster_similarity_subject, axis=0) / n_atlas
        else:
            cluster_distances = wma.cluster._pairwise_distance_matrix(
                pd_atlas, 0.0, number_of_jobs=1, bilateral=atlas_bilateral,
                distance_method=distance_method)
            cluster_similarity_atlas = wma.similarity.distance_to_similarity(
                cluster_distances, sigmasq)
            cluster_distances = wma.cluster._rectangular_distance_matrix(
                pd_subject, pd_atlas, 0.0, number_of_jobs=1, bilateral=atlas_bilateral,
                distance_method=distance_method)
            cluster_similarity_subject = wma.similarity.distance_to_similarity(
                cluster_distances, sigmasq)
            total_similarity_atlas = (numpy.sum(cluster_similarity_atlas, axis=1) - 1.0) / n_atlas
            total_similarity_subject = numpy.sum(cluster_similarity_subject, axis=0) / n_atlas

        mean_sim_atlas = numpy.mean(total_similarity_atlas)
        cluster_std_atlas = numpy.std(total_similarity_atlas)
        cutoff = mean_sim_atlas - outlier_std * cluster_std_atlas
        reject_idx = [fidx for fidx, sim in enumerate(total_similarity_subject) if sim < cutoff]
        n_rejected = len(reject_idx)
        n_retained = n_subj - n_rejected

        mask = numpy.ones(n_subj)
        mask[reject_idx] = 0
        pd_c = wma.filter.mask(pd_subject, mask, verbose=False,
                               preserve_point_data=True, preserve_cell_data=True)
        with SmallPages():
            pd_c, labels, _ = assess_polydata(pd_c, c_idx, location_data)
        if len(numpy.where(labels == 0)[0]) > 1:
            raise RuntimeError(f"Hemisphere labels incomplete for cluster {c_num:05d}")

        if backend != 'gpu':
            cluster_distances = numpy.sqrt(cluster_distances)
            cluster_mean_distances = numpy.mean(cluster_distances, axis=0)
        keep_mask = mask == 1
        mean_dist_before = float(numpy.mean(cluster_mean_distances))
        mean_dist_after = float(numpy.mean(cluster_mean_distances[keep_mask])) if n_retained else 0.0
        mean_prob_before = float(numpy.mean(total_similarity_subject))
        mean_prob_after = float(numpy.mean(total_similarity_subject[keep_mask])) if n_retained else 0.0

        cleaned_pds[c_idx] = pd_c
        stats[c_idx]['removed'] = n_rejected
        stats[c_idx]['retained'] = n_retained
        log_rows[c_idx] = (
            f"{c_num}\t{n_subj}\t{n_retained}\t{n_rejected}\t"
            f"{float(n_rejected) / n_subj}\t{mean_dist_before}\t{mean_dist_after}\t"
            f"{mean_prob_before}\t{mean_prob_after}"
        )

    _dispatch_cluster_workers(atlas_cluster_paths, backend, workers, _process_one)

    total_removed = sum(s['removed'] for s in stats)
    total_retained = sum(s['retained'] for s in stats)
    nonempty = sum(1 for s in stats if s['retained'] > 0)
    return cleaned_pds, log_rows, total_removed, total_retained, nonempty


def _empty_like_schema(template_pd):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    if template_pd.GetPoints() is not None:
        pts.SetDataType(template_pd.GetPoints().GetDataType())
    pd.SetPoints(pts)
    pd.SetLines(vtk.vtkCellArray())
    for src, dst in (
        (template_pd.GetPointData(), pd.GetPointData()),
        (template_pd.GetCellData(), pd.GetCellData()),
    ):
        for idx in range(src.GetNumberOfArrays()):
            arr = src.GetAbstractArray(idx)
            if arr is None:
                continue
            clone = arr.NewInstance()
            clone.SetName(arr.GetName())
            clone.SetNumberOfComponents(arr.GetNumberOfComponents())
            dst.AddArray(clone)
    return pd


def _clean_stale_outputs(work_dir):
    """Remove downstream outputs before an in-place --no-atomic-publish rerun."""
    for rel in (
        'AnatomicalTracts',
        'anatomical_tracts_complete.log',
        'qc_summary.json',
        'FiberClustering/InitialClusters',
        'FiberClustering/OutlierRemovedClusters',
        'FiberClustering/TransformedClusters',
        'FiberClustering/SeparatedClusters',
        'TransformedTracts',
        'SeparatedTracts',
        'InvTransformedTracts',
        'TractRegistration',
    ):
        target = work_dir / rel
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    fc_dir = work_dir / 'FiberClustering'
    if fc_dir.is_dir() and not any(fc_dir.iterdir()):
        fc_dir.rmdir()


def run_pipeline(args):
    t_start = time.perf_counter()
    timings = {}

    raw_input = Path(args.input)
    case_id = os.path.splitext(raw_input.name)[0]
    input_path = raw_input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input tractography file does not exist: {input_path}")

    resolved_atlas = wma.atlases.resolve_atlas(
        args.atlas, family=getattr(args, 'atlas_family', 'auto'), validate_mrml=True)
    atlas_base = resolved_atlas.base_dir
    reg_atlas_path = resolved_atlas.reg_atlas_path
    fc_atlas_dir = resolved_atlas.fc_dir
    atlas_cluster_paths = list(resolved_atlas.cluster_paths)
    atlas_cluster_names = list(resolved_atlas.cluster_names)
    location_data = resolved_atlas.location_data
    tract_tasks = _collect_tract_tasks(
        fc_atlas_dir,
        atlas_cluster_names,
        location_data,
        hemispheric_tracts=resolved_atlas.hemispheric_tracts,
        commissural_tracts=resolved_atlas.commissural_tracts,
    )

    wma.gpu.apply_gpu_arguments(args)
    workers = _effective_workers(args.jobs)

    t0 = time.perf_counter()
    input_pd = wma.io.read_polydata(str(input_path))
    if input_pd is None or input_pd.GetNumberOfLines() == 0:
        raise RuntimeError(f"Input polydata contains no fibers: {input_path}")
    timings['load_input'] = time.perf_counter() - t0

    input_fibers = int(input_pd.GetNumberOfLines())
    has_tensors = wma.transform.has_tensor_arrays(input_pd)

    pre_fwd_matrix = None
    pre_inv_matrix = None
    if args.transform:
        pre_tfm_path = Path(args.transform).resolve()
        if not pre_tfm_path.is_file():
            raise FileNotFoundError(f"Initial transform file not found: {pre_tfm_path}")
        pre_fwd_matrix = wma.transform.read_itk_affine_matrix_ras(str(pre_tfm_path), inverse=False)
        pre_inv_matrix = wma.transform.read_itk_affine_matrix_ras(str(pre_tfm_path), inverse=True)

    tier1_eligible = (
        args.reg_mode == 'rig'
        and not has_tensors
        and (args.transform is None or pre_fwd_matrix is not None)
    )
    slicer_path = None
    if args.slicer:
        slicer_candidate = Path(args.slicer)
        if slicer_candidate.is_file():
            slicer_path = slicer_candidate.resolve()
        elif os.sep not in args.slicer and (not os.altsep or os.altsep not in args.slicer):
            found_slicer = shutil.which(args.slicer)
            if found_slicer:
                slicer_path = Path(found_slicer).resolve()
        else:
            slicer_path = slicer_candidate.resolve()
    if not tier1_eligible:
        if slicer_path is None or not slicer_path.is_file():
            raise RuntimeError(
                "3D Slicer (-s/--slicer) is required for nonrigid registration, "
                "non-linear pre-transforms, or tensor-bearing tractography.")

    output_root = Path(args.output).resolve()
    final_dir = output_root / case_id
    if not args.no_atomic_publish:
        if final_dir.exists() and any(final_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {final_dir}")
        output_root.mkdir(parents=True, exist_ok=True)
        work_dir = output_root / f'.staging_{case_id}_{os.getpid()}'
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    else:
        final_dir.mkdir(parents=True, exist_ok=True)
        work_dir = final_dir
        _clean_stale_outputs(work_dir)

    try:
        with tempfile.TemporaryDirectory(prefix='wma_org_work_') as tmp_scratch:
            scratch_dir = Path(tmp_scratch)
            keep = int(args.keep_intermediates)

            # Optional initial pre-transform
            if args.transform:
                if pre_fwd_matrix is not None and not has_tensors:
                    input_pd = wma.transform.apply_affine_to_polydata(input_pd, pre_fwd_matrix)
                else:
                    pre_in = scratch_dir / 'pre_in'
                    pre_out = scratch_dir / 'pre_out'
                    pre_in.mkdir()
                    with _default_lz4_compression():
                        wma.io.write_polydata(input_pd, str(pre_in / input_path.name))
                    del input_pd
                    _invoke_slicer_hardening(
                        pre_in, pre_out, pre_tfm_path, slicer_path, workers,
                        inverse=False, no_xvfb=args.no_xvfb)
                    input_pd = wma.io.read_polydata(str(pre_out / input_path.name))

            # Registration stage
            t0 = time.perf_counter()
            atlas_reg_pd = wma.io.read_polydata(str(reg_atlas_path), geometry_only=True)
            atlas_id = os.path.splitext(reg_atlas_path.name)[0]
            reg_root = (work_dir / 'TractRegistration') if keep >= 1 else (scratch_dir / 'TractRegistration')

            if args.reg_mode == 'rig':
                reg_case_dir = reg_root / case_id
                reg_out_dir = reg_case_dir / 'output_tractography'
                reg_out_dir.mkdir(parents=True, exist_ok=True)
                fwd_transform = _run_registration_stage(
                    input_pd, atlas_reg_pd, case_id, atlas_id, 'rigid_affine_fast',
                    args.fiber_length, args.fiber_length_max,
                    reg_case_dir if keep >= 2 else (scratch_dir / 'reg_iters'),
                    save_iter_tfms=(keep >= 2),
                )
                del atlas_reg_pd
                timings['registration'] = time.perf_counter() - t0

                t0 = time.perf_counter()
                fwd_matrix = _matrix_from_vtk_transform(fwd_transform)
                inv_matrix = numpy.linalg.inv(fwd_matrix)
                reg_pd = wma.transform.apply_affine_to_polydata(input_pd, fwd_matrix)
                del input_pd
                gc.collect()
                if keep >= 1 or not tier1_eligible:
                    wma.io.write_transforms_to_itk_format([fwd_transform], str(reg_out_dir), [case_id])
                if keep >= 2:
                    wma.io.write_polydata(reg_pd, str(reg_out_dir / f'{case_id}_reg.vtk'))
                rig_tfm_path = reg_out_dir / f'itk_txform_{case_id}.tfm'
                nonrig_tfm_path = None
                timings['forward_transform'] = time.perf_counter() - t0
            else:
                aff_case_dir = reg_root / case_id
                aff_out_dir = aff_case_dir / 'output_tractography'
                aff_out_dir.mkdir(parents=True, exist_ok=True)
                aff_tx = _run_registration_stage(
                    input_pd, atlas_reg_pd, case_id, atlas_id, 'affine',
                    args.fiber_length, args.fiber_length_max,
                    aff_case_dir if keep >= 2 else (scratch_dir / 'reg_aff_iters'),
                    save_iter_tfms=(keep >= 2),
                )
                affine_pd = wma.transform.apply_affine_to_polydata(input_pd, aff_tx)
                del input_pd
                wma.io.write_transforms_to_itk_format([aff_tx], str(aff_out_dir), [case_id])
                rig_tfm_path = aff_out_dir / f'itk_txform_{case_id}.tfm'

                nonrig_id = f'{case_id}_reg'
                nonrig_case_dir = reg_root / nonrig_id
                nonrig_out_dir = nonrig_case_dir / 'output_tractography'
                nonrig_out_dir.mkdir(parents=True, exist_ok=True)
                nonrig_tx = _run_registration_stage(
                    affine_pd, atlas_reg_pd, nonrig_id, atlas_id, 'nonrigid',
                    args.fiber_length, args.fiber_length_max,
                    nonrig_case_dir if keep >= 2 else (scratch_dir / 'reg_nonrig_iters'),
                    save_iter_tfms=(keep >= 2),
                )
                del atlas_reg_pd
                timings['registration'] = time.perf_counter() - t0

                t0 = time.perf_counter()
                reg_pd = wma.transform.apply_affine_to_polydata(affine_pd, nonrig_tx)
                del affine_pd
                gc.collect()
                wma.io.write_transforms_to_itk_format([nonrig_tx], str(nonrig_out_dir), [nonrig_id])
                if keep >= 2:
                    wma.io.write_polydata(reg_pd, str(nonrig_out_dir / f'{nonrig_id}_reg.vtk'))
                nonrig_tfm_path = nonrig_out_dir / f'itk_txform_{nonrig_id}.tfm'
                timings['forward_transform'] = time.perf_counter() - t0

            # Classification-stage minimum-length preprocessing (default 40 mm, matching wm_cluster_from_atlas.py)
            t0 = time.perf_counter()
            registered_fibers = int(reg_pd.GetNumberOfLines())
            filtered_reg_pd = wma.filter.preprocess(
                reg_pd, args.fiber_length, return_indices=False,
                preserve_point_data=True, preserve_cell_data=True, verbose=False)
            del reg_pd
            gc.collect()
            filtered_fibers = int(filtered_reg_pd.GetNumberOfLines())
            excluded_short_fibers = registered_fibers - filtered_fibers
            timings['length_filter'] = time.perf_counter() - t0

            # Spectral atlas labeling + in-memory per-cluster split or indexed partition
            t0 = time.perf_counter()
            atlas = wma.cluster.load_atlas(str(fc_atlas_dir), 'atlas', for_labeling=True)
            number_of_clusters = int(atlas.centroids.shape[0])
            atlas_bilateral = bool(atlas.bilateral)
            if number_of_clusters != len(atlas_cluster_names):
                raise RuntimeError(
                    f"Atlas centroid count ({number_of_clusters}) does not match "
                    f"cluster VTP count ({len(atlas_cluster_names)})")

            backend = 'gpu' if wma.gpu.use_gpu() else ('cpu' if wma.gpu.config.cpu_fast_enabled() else None)
            use_indexed = (
                tier1_eligible
                and keep == 0
                and backend is not None
                and filtered_fibers > 0
                and wma.filter._lines_offsets_connectivity(filtered_reg_pd) is not None
            )
            queries_15 = None
            cluster_line_indices = None
            labeled_pd = None
            if filtered_fibers > 0:
                if backend is not None:
                    queries_15 = wma.gpu.fiber_tensors.polydata_to_numpy(filtered_reg_pd, 15)
                labeled_pd, cluster_numbers, color, embed = wma.cluster.spectral_atlas_label(
                    filtered_reg_pd, atlas, number_of_jobs=workers, queries_15=queries_15)
                if use_indexed:
                    labels_arr = numpy.asarray(cluster_numbers, dtype=numpy.int64)
                    valid_order = numpy.flatnonzero(labels_arr >= 0)
                    valid_order = valid_order[numpy.argsort(labels_arr[valid_order], kind='stable')]
                    boundaries = numpy.searchsorted(
                        labels_arr[valid_order], numpy.arange(number_of_clusters + 1))
                    cluster_line_indices = [
                        valid_order[boundaries[c]:boundaries[c + 1]]
                        for c in range(number_of_clusters)
                    ]
                    del labels_arr, valid_order, boundaries
                else:
                    cluster_pds = wma.cluster.mask_all_clusters(
                        labeled_pd, cluster_numbers, number_of_clusters,
                        preserve_point_data=True, preserve_cell_data=True, verbose=args.verbose,
                        number_of_jobs=workers)
                    if keep >= 2:
                        init_dir = work_dir / 'FiberClustering' / 'InitialClusters' / case_id
                        init_dir.mkdir(parents=True, exist_ok=True)
                        init_fnames = [str(init_dir / name) for name in atlas_cluster_names]
                        with _default_lz4_compression():
                            wma.io.write_polydatas_parallel(
                                cluster_pds, init_fnames, jobs=workers, use_threads=True)
                    del labeled_pd
                    labeled_pd = None
                del cluster_numbers, color, embed
            else:
                use_indexed = False
                empty_pd = _empty_like_schema(filtered_reg_pd)
                cluster_pds = [_empty_like_schema(empty_pd) for _ in range(number_of_clusters)]
            del filtered_reg_pd, atlas
            gc.collect()
            timings['clustering'] = time.perf_counter() - t0

            # Threaded / multi-GPU outlier removal & hemisphere assessment
            t0 = time.perf_counter()
            indexed_assembly = None
            if use_indexed:
                labeled_geometry = wma.filter._lines_offsets_connectivity(labeled_pd)
                has_existing_hemi, existing_hemi_labels = read_mask_location_from_vtk(labeled_pd)
                (kept_by_hemi, assessed_hemi, outlier_removed,
                 retained_fibers, nonempty_clusters) = _outlier_and_assess_indexed(
                    cluster_line_indices, queries_15, labeled_geometry,
                    has_existing_hemi, existing_hemi_labels,
                    atlas_cluster_paths, atlas_cluster_names, atlas_bilateral,
                    location_data, args.outlier_sigma, args.outlier_std,
                    backend, workers, verbose=args.verbose)
                del cluster_line_indices, queries_15, labeled_geometry, existing_hemi_labels
                gc.collect()
                timings['outlier_and_hemisphere'] = time.perf_counter() - t0

                # Single-pass vectorized affine hardening (points only) + direct tract assembly
                t0 = time.perf_counter()
                hardening_backend = 'pure_python_affine'
                total_inv_matrix = (pre_inv_matrix @ inv_matrix) if pre_inv_matrix is not None else inv_matrix
                hardened_pd = wma.transform.apply_affine_to_polydata(labeled_pd, total_inv_matrix)
                del labeled_pd
                gc.collect()
                hardened_geometry = wma.filter._lines_offsets_connectivity(hardened_pd)
                point_specs, cell_specs, tensor_names = wma.filter._extract_polydata_schema_and_arrays(
                    hardened_pd, preserve_point_data=True, preserve_cell_data=True)
                indexed_assembly = (
                    kept_by_hemi, assessed_hemi, hardened_geometry,
                    point_specs, cell_specs, tensor_names,
                )
                timings['hardening_and_separation'] = time.perf_counter() - t0
            else:
                del queries_15
                cleaned_pds, log_rows, outlier_removed, retained_fibers, nonempty_clusters = _outlier_and_assess_clusters(
                    cluster_pds, atlas_cluster_paths, atlas_bilateral,
                    location_data, args.outlier_sigma, args.outlier_std,
                    backend, workers, verbose=args.verbose)
                del cluster_pds
                gc.collect()

                if keep >= 2:
                    outrem_dir = work_dir / 'FiberClustering' / 'OutlierRemovedClusters' / f'{case_id}_outlier_removed'
                    outrem_dir.mkdir(parents=True, exist_ok=True)
                    with _default_lz4_compression():
                        wma.io.write_polydatas_parallel(
                            cleaned_pds, [str(outrem_dir / name) for name in atlas_cluster_names],
                            jobs=workers, use_threads=True)
                    with (outrem_dir / 'cluster_log.txt').open('w') as lf:
                        lf.write('cluster\t input_fibers\t output_fibers\t number_removed\t percentage_removed\t '
                                 'mean_distance_before\t mean_distance_after\t mean_probability_before\t mean_probability_after\n')
                        for row in log_rows:
                            lf.write(row + '\n')
                    (outrem_dir / 'cluster_location_by_hemisphere.log').write_text(
                        '<wm_assess_cluster_location_by_hemisphere.py> Done!!!')
                timings['outlier_and_hemisphere'] = time.perf_counter() - t0

                # Tiered hardening + hemisphere separation
                t0 = time.perf_counter()
                separated_map = {
                    'tracts_left_hemisphere': {},
                    'tracts_right_hemisphere': {},
                    'tracts_commissural': {},
                }

                if tier1_eligible:
                    hardening_backend = 'pure_python_affine'
                    total_inv_matrix = (pre_inv_matrix @ inv_matrix) if pre_inv_matrix is not None else inv_matrix
                    hardened_for_save = [None] * number_of_clusters if keep >= 2 else None

                    def _harden_and_split_one(c_idx):
                        name = atlas_cluster_names[c_idx]
                        pd_clean = cleaned_pds[c_idx]
                        cleaned_pds[c_idx] = None
                        with SmallPages():
                            pd_h = wma.transform.apply_affine_to_polydata(pd_clean, total_inv_matrix)
                            _add_ras_space_field(pd_h)
                            found, labels = read_mask_location_from_vtk(pd_h)
                            if not found or (labels == 0).sum() > 1:
                                raise RuntimeError(f"Hemisphere labels missing or incomplete for {name}")
                            left_pd, right_pd, comm_pd = split_polydata_subsets(pd_h, labels)
                        separated_map['tracts_left_hemisphere'][name] = left_pd
                        separated_map['tracts_right_hemisphere'][name] = right_pd
                        separated_map['tracts_commissural'][name] = comm_pd
                        if hardened_for_save is not None:
                            hardened_for_save[c_idx] = pd_h

                    if workers > 1:
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            list(pool.map(_harden_and_split_one, range(number_of_clusters)))
                    else:
                        for idx in range(number_of_clusters):
                            _harden_and_split_one(idx)

                    del cleaned_pds
                    gc.collect()

                    if keep >= 2 and hardened_for_save is not None:
                        tfm_dir = work_dir / 'FiberClustering' / 'TransformedClusters' / case_id
                        tfm_dir.mkdir(parents=True, exist_ok=True)
                        with _default_lz4_compression():
                            wma.io.write_polydatas_parallel(
                                hardened_for_save, [str(tfm_dir / name) for name in atlas_cluster_names],
                                jobs=workers, use_threads=True)
                        (tfm_dir / 'transform_complete.log').write_text(
                            f'Transforms completed for {number_of_clusters} files.\n')
                        del hardened_for_save

                    if keep >= 1:
                        sep_dir = work_dir / 'FiberClustering' / 'SeparatedClusters'
                        with _default_lz4_compression():
                            for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                                (sep_dir / folder).mkdir(parents=True, exist_ok=True)
                                wma.io.write_polydatas_parallel(
                                    [separated_map[folder][name] for name in atlas_cluster_names],
                                    [str(sep_dir / folder / name) for name in atlas_cluster_names],
                                    jobs=workers, use_threads=True)
                        (sep_dir / 'hemisphere_separation_complete.log').write_text(
                            f'Hemisphere separation completed for {number_of_clusters} clusters.\n')
                else:
                    outrem_work = (work_dir / 'FiberClustering' / 'OutlierRemovedClusters' / f'{case_id}_outlier_removed') \
                        if keep >= 2 else (scratch_dir / 'OutlierRemovedWork')
                    if keep < 2:
                        outrem_work.mkdir(parents=True, exist_ok=True)
                        with _default_lz4_compression():
                            wma.io.write_polydatas_parallel(
                                cleaned_pds, [str(outrem_work / name) for name in atlas_cluster_names],
                                jobs=workers, use_threads=True)
                    del cleaned_pds
                    gc.collect()

                    tfm_dir = (work_dir / 'FiberClustering' / 'TransformedClusters' / case_id) \
                        if keep >= 2 else (scratch_dir / 'TransformedClusters')
                    sep_final_dir = (work_dir / 'FiberClustering' / 'SeparatedClusters') \
                        if keep >= 1 else (scratch_dir / 'SeparatedClusters')
                    rig_target_dir = (scratch_dir / 'TransformedClustersPre') if args.transform else tfm_dir
                    rig_split_dir = None if args.transform else sep_final_dir
                    tfm_dir.mkdir(parents=True, exist_ok=True)
                    rig_target_dir.mkdir(parents=True, exist_ok=True)
                    sep_final_dir.mkdir(parents=True, exist_ok=True)

                    if args.reg_mode == 'nonrig':
                        tmp_nonrig = scratch_dir / 'tmp_nonrig'
                        tmp_nonrig.mkdir(parents=True, exist_ok=True)
                        _invoke_slicer_hardening(
                            outrem_work, tmp_nonrig, nonrig_tfm_path, slicer_path, workers,
                            inverse=True, no_xvfb=args.no_xvfb)
                        hardening_backend = _invoke_slicer_hardening(
                            tmp_nonrig, rig_target_dir, rig_tfm_path, slicer_path, workers,
                            inverse=True, split_output_dir=rig_split_dir, no_xvfb=args.no_xvfb)
                    else:
                        hardening_backend = _invoke_slicer_hardening(
                            outrem_work, rig_target_dir, rig_tfm_path, slicer_path, workers,
                            inverse=True, split_output_dir=rig_split_dir, no_xvfb=args.no_xvfb)

                    if args.transform:
                        hardening_backend = _invoke_slicer_hardening(
                            rig_target_dir, tfm_dir, pre_tfm_path, slicer_path, workers,
                            inverse=True, split_output_dir=sep_final_dir, no_xvfb=args.no_xvfb)

                    if keep >= 1:
                        (sep_final_dir / 'hemisphere_separation_complete.log').write_text(
                            f'Hemisphere separation completed for {number_of_clusters} clusters.\n')

                    for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                        for name in atlas_cluster_names:
                            separated_map[folder][name] = wma.io.read_polydata(str(sep_final_dir / folder / name))

                timings['hardening_and_separation'] = time.perf_counter() - t0

            # Parallel anatomical tract assembly
            t0 = time.perf_counter()
            tracts_dir = work_dir / 'AnatomicalTracts'
            tracts_dir.mkdir(parents=True, exist_ok=True)

            if indexed_assembly is not None:
                (kept_by_hemi, assessed_hemi, hardened_geometry,
                 point_specs, cell_specs, tensor_names) = indexed_assembly

                def _run_tract(task):
                    out_name, folder, members = task
                    member_lines = [kept_by_hemi[folder][m] for m in members]
                    nonempty_lines = [arr for arr in member_lines if len(arr) > 0]
                    all_lines = numpy.concatenate(nonempty_lines) if nonempty_lines else numpy.empty(0, dtype=numpy.int64)
                    cluster_idx_per_line = (
                        numpy.concatenate([
                            numpy.full(len(arr), m_idx, dtype=numpy.int32)
                            for m_idx, arr in enumerate(member_lines) if len(arr) > 0
                        ])
                        if all_lines.size else numpy.empty(0, dtype=numpy.int32)
                    )
                    extra_arrays = []
                    if assessed_hemi is not None:
                        hemi_vals = (assessed_hemi[all_lines]
                                     if all_lines.size else numpy.empty(0, dtype=numpy.float32))
                        extra_arrays.append(('HemisphereLocation', hemi_vals, numpy.float32))
                    extra_arrays.append(('cluster_idx', cluster_idx_per_line, numpy.int32))
                    n_lines, n_pts = _assemble_single_tract(
                        None, tracts_dir / out_name,
                        indexed_source=(
                            hardened_geometry, all_lines, point_specs,
                            cell_specs, tensor_names, extra_arrays,
                        ),
                    )
                    return out_name, n_lines, n_pts
            else:
                def _run_tract(task):
                    out_name, folder, members = task
                    member_pds = [separated_map[folder][m] for m in members]
                    n_lines, n_pts = _assemble_single_tract(member_pds, tracts_dir / out_name)
                    return out_name, n_lines, n_pts

            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    tract_results = list(pool.map(_run_tract, tract_tasks))
            else:
                tract_results = [_run_tract(t) for t in tract_tasks]

            if indexed_assembly is not None:
                del indexed_assembly, hardened_pd, hardened_geometry, kept_by_hemi, assessed_hemi
            else:
                del separated_map
            gc.collect()

            if any(not (tracts_dir / out_name).is_file() for out_name, _, _ in tract_results):
                raise RuntimeError('Anatomical tract outputs are incomplete.')
            (tracts_dir / 'anatomical_tracts_complete.log').write_text(
                f'Anatomical tract appending completed for {len(tract_results)} files.\n')
            timings['tract_assembly'] = time.perf_counter() - t0
            timings['total'] = time.perf_counter() - t_start

            qc_summary = {
                'case_id': case_id,
                'input_file': str(input_path),
                'atlas_dir': str(atlas_base),
                'atlas_family': resolved_atlas.family,
                'reg_mode': args.reg_mode,
                'fiber_length_mm': float(args.fiber_length),
                'fiber_length_max_mm': float(args.fiber_length_max),
                'hardening_backend': hardening_backend,
                'has_tensor_arrays': bool(has_tensors),
                'gpu_used': bool(wma.gpu.use_gpu()),
                'gpu_float32': bool(wma.gpu.float32_enabled()),
                'input_fibers': input_fibers,
                'registered_fibers': registered_fibers,
                'filtered_fibers': filtered_fibers,
                'excluded_short_fibers': excluded_short_fibers,
                'total_clusters': number_of_clusters,
                'nonempty_clusters': nonempty_clusters,
                'outlier_removed_fibers': outlier_removed,
                'retained_clustered_fibers': retained_fibers,
                'anatomical_tracts_expected': len(tract_tasks),
                'anatomical_tracts_written': len(tract_results),
                'tracts': {
                    name: {'fibers': n_lines, 'points': n_pts}
                    for name, n_lines, n_pts in tract_results
                },
                'stage_timings_s': {k: round(v, 6) for k, v in timings.items()},
            }
            (work_dir / 'qc_summary.json').write_text(
                json.dumps(qc_summary, indent=2) + '\n', encoding='utf-8')

        if not args.no_atomic_publish:
            if final_dir.exists() and not any(final_dir.iterdir()):
                final_dir.rmdir()
            os.replace(str(work_dir), str(final_dir))
    except BaseException:
        if not args.no_atomic_publish and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return final_dir


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
