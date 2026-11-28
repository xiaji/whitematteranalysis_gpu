"""End-to-end tests for bin/wm_apply_ORG_atlas.py and Slicer-Core ModelNode fallback."""
import builtins
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import whitematteranalysis as wma

ROOT = Path(__file__).resolve().parents[1]
ORG_CLI_PATH = ROOT / 'bin' / 'wm_apply_ORG_atlas.py'
SLICER_HELPER_PATH = ROOT / 'bin' / 'harden_transform_with_slicer.py'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tract_polydata(lines_coords, tags=None, tensors=False, hemi_labels=None):
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    pt_tags = []
    pt_hemi = []
    for line_idx, coords in enumerate(lines_coords):
        ids = vtk.vtkIdList()
        for xyz in coords:
            pid = points.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            ids.InsertNextId(pid)
            if tags is not None:
                pt_tags.append(tags[line_idx])
            if hemi_labels is not None:
                pt_hemi.append(hemi_labels[line_idx])
        lines.InsertNextCell(ids)
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    if tags is not None:
        arr = numpy_to_vtk(np.asarray(pt_tags, dtype=np.float32), deep=True)
        arr.SetName('fiber_tag')
        polydata.GetPointData().AddArray(arr)
    if hemi_labels is not None:
        harr = numpy_to_vtk(np.asarray(pt_hemi, dtype=np.float64), deep=True)
        harr.SetName('HemisphereLocation')
        polydata.GetPointData().AddArray(harr)
    if tensors:
        n_pts = polydata.GetNumberOfPoints()
        tarr = numpy_to_vtk(np.zeros((n_pts, 9), dtype=np.float32), deep=True)
        tarr.SetName('tensors')
        polydata.GetPointData().AddArray(tarr)
        polydata.GetPointData().SetActiveTensors('tensors')
    return polydata


def _bundle_coords(center_x, center_z, n_fibers, n_pts=21, step=3.0, commissural=False, seed=0, jitter_scale=0.35):
    rng = np.random.RandomState(seed)
    fibers = []
    for _ in range(n_fibers):
        jitter = rng.normal(scale=jitter_scale, size=(1, 3))
        if commissural:
            xs = np.linspace(-step * (n_pts - 1) / 2.0, step * (n_pts - 1) / 2.0, n_pts)
            coords = np.column_stack([
                xs,
                np.full(n_pts, 0.0),
                np.full(n_pts, center_z),
            ]) + jitter
        else:
            ys = np.linspace(-step * (n_pts - 1) / 2.0, step * (n_pts - 1) / 2.0, n_pts)
            coords = np.column_stack([
                np.full(n_pts, center_x),
                ys,
                np.full(n_pts, center_z),
            ]) + jitter
        fibers.append(coords)
    return fibers


def _build_synthetic_org_atlas(atlas_root):
    reg_dir = atlas_root / 'ORG-RegAtlas-100HCP'
    fc_dir = atlas_root / 'ORG-800FC-100HCP'
    reg_dir.mkdir(parents=True)
    fc_dir.mkdir(parents=True)

    # Three well-separated anatomical groups: left, right, commissural.
    # Spread fibers inside each bundle so the 115x115 Nystrom A matrix stays
    # well-conditioned across NumPy/LAPACK versions (avoids PSD complex noise).
    left_fibers = _bundle_coords(-35.0, 0.0, 40, n_pts=21, step=3.0, seed=1, jitter_scale=1.0)
    right_fibers = _bundle_coords(35.0, 0.0, 40, n_pts=21, step=3.0, seed=2, jitter_scale=1.0)
    comm_fibers = _bundle_coords(0.0, 35.0, 40, n_pts=21, step=3.0, commissural=True, seed=3, jitter_scale=1.0)
    all_atlas_fibers = left_fibers + right_fibers + comm_fibers
    atlas_pd = _make_tract_polydata(all_atlas_fibers)
    wma.io.write_polydata(atlas_pd, str(reg_dir / 'registration_atlas.vtk'))

    # Build a valid Nystrom spectral atlas over these 3 bundles
    nystrom_mask = np.ones(len(all_atlas_fibers), dtype=bool)
    nystrom_mask[115:] = False  # 115 in A (>= 100), 5 in B
    np.random.seed(42)
    out_pd, cluster_idx, _, _, _, atlas, _ = wma.cluster.spectral(
        atlas_pd,
        number_of_clusters=3,
        number_of_eigenvectors=3,
        sigma=25.0,
        threshold=0.0,
        number_of_jobs=1,
        use_nystrom=True,
        nystrom_mask=nystrom_mask,
        distance_method='MeanSquared',
        outlier_std_threshold=np.inf,
        bilateral=False,
        use_gpu=False,
    )
    atlas.save(str(fc_dir), 'atlas')

    cluster_pds = wma.cluster.mask_all_clusters(
        out_pd, cluster_idx, 3, preserve_point_data=True, preserve_cell_data=True, verbose=False)
    hemi_rows = ['Cluster Index\tLocation Label\n']
    hemispheric_clusters = []
    commissural_clusters = []
    for c_idx, c_pd in enumerate(cluster_pds):
        fname = f'cluster_{c_idx + 1:05d}.vtp'
        wma.io.write_polydata(c_pd, str(fc_dir / fname))
        fa = wma.fibers.FiberArray()
        fa.hemispheres = True
        fa.hemisphere_percent_threshold = 0.6
        fa.convert_from_polydata(c_pd, points_per_fiber=15)
        if fa.number_commissure > fa.number_left_hem + fa.number_right_hem:
            hemi_rows.append(f'cluster_{c_idx + 1:05d}\tc\n')
            commissural_clusters.append(fname)
        else:
            hemi_rows.append(f'cluster_{c_idx + 1:05d}\th\n')
            hemispheric_clusters.append(fname)

    (fc_dir / 'cluster_hemisphere_location.txt').write_text(''.join(hemi_rows))
    (fc_dir / 'T_AF.mrml').write_text(
        ''.join(f'<FiberBundleStorage fileName="{name}"/>\n' for name in hemispheric_clusters))
    (fc_dir / 'T_CC1.mrml').write_text(
        ''.join(f'<FiberBundleStorage fileName="{name}"/>\n' for name in commissural_clusters))


class FakeMRMLScene:
    BatchProcessState = 1

    def __init__(self):
        self.nodes = []
        self._counter = 0

    def StartState(self, _state):
        pass

    def EndState(self, _state):
        pass

    def AddNewNodeByClass(self, cls_name, name=''):
        self._counter += 1
        node_id = f'{cls_name}{self._counter}'
        if cls_name == 'vtkMRMLTransformNode':
            node = FakeTransformNode(node_id, self)
        elif cls_name == 'vtkMRMLModelNode':
            node = FakeModelNode(node_id, self)
        else:
            raise AssertionError(f'Unexpected MRML node class: {cls_name}')
        self.nodes.append(node)
        return node

    def AddNode(self, node):
        if node not in self.nodes:
            node.scene = self
            self.nodes.append(node)
        return node

    def RemoveNode(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            node.scene = None

    def GetNumberOfNodes(self):
        return len(self.nodes)


class FakeNode:
    def __init__(self, node_id, scene):
        self.node_id = node_id
        self.scene = scene

    def GetID(self):
        return self.node_id

    def GetScene(self):
        return self.scene


class FakeTransformStorageNode(FakeNode):
    def __init__(self, node_id, scene):
        super().__init__(node_id, scene)
        self.filename = None

    def SetFileName(self, filename):
        self.filename = filename

    def ReadData(self, node):
        mat = wma.transform.read_itk_affine_matrix_ras(self.filename, inverse=False)
        if mat is None:
            return 0
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, float(mat[i, j]))
        tx = vtk.vtkTransform()
        tx.SetMatrix(vtk_mat)
        node.transform = tx
        return 1


class FakeTransformNode(FakeNode):
    def __init__(self, node_id, scene):
        super().__init__(node_id, scene)
        self.storage = None
        self.transform = vtk.vtkTransform()

    def SetAndObserveStorageNodeID(self, storage_id):
        for n in self.scene.nodes:
            if n.GetID() == storage_id:
                self.storage = n

    def GetStorageNode(self):
        return self.storage

    def Inverse(self):
        self.transform.Inverse()


class FakeModelStorageNode(FakeNode):
    CoordinateSystemRAS = 1

    def __init__(self, node_id, scene):
        super().__init__(node_id, scene)
        self.filename = None
        self.coord_sys = None

    def SetCoordinateSystem(self, cs):
        self.coord_sys = cs

    def SetFileName(self, filename):
        self.filename = filename

    def ReadData(self, model_node):
        pd = wma.io.read_polydata(self.filename)
        model_node.SetAndObservePolyData(pd)
        return 1

    def WriteData(self, model_node):
        wma.io.write_polydata(model_node.GetPolyData(), self.filename)
        return 1


class FakeModelDisplayNode(FakeNode):
    pass


class FakeModelNode(FakeNode):
    def __init__(self, node_id, scene):
        super().__init__(node_id, scene)
        self.polydata = None
        self.storage = None
        self.transform_id = None
        self.display_nodes = []

    def CreateDefaultStorageNode(self):
        self.scene._counter += 1
        return FakeModelStorageNode(f'vtkMRMLModelStorageNode{self.scene._counter}', self.scene)

    def SetAndObserveStorageNodeID(self, storage_id):
        for n in self.scene.nodes:
            if n.GetID() == storage_id:
                self.storage = n

    def GetStorageNode(self):
        return self.storage

    def SetAndObservePolyData(self, pd):
        self.polydata = pd

    def GetPolyData(self):
        return self.polydata

    def SetAndObserveTransformNodeID(self, tid):
        self.transform_id = tid

    def GetNumberOfDisplayNodes(self):
        return len(self.display_nodes)

    def GetNthDisplayNode(self, idx):
        return self.display_nodes[idx]


class FakeTransformLogic:
    def __init__(self, scene):
        self.scene = scene

    def hardenTransform(self, node):
        tx_node = next(n for n in self.scene.nodes if n.GetID() == node.transform_id)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(node.GetPolyData())
        tf.SetTransform(tx_node.transform)
        tf.Update()
        node.SetAndObservePolyData(tf.GetOutput())
        return 1


def _assert_native_roundtrip_points(test_case, vtp_path, native_fibers, atol=5e-4):
    pd = wma.io.read_polydata(str(vtp_path))
    if pd.GetNumberOfLines() == 0:
        return 0
    tags = vtk_to_numpy(pd.GetPointData().GetArray('fiber_tag'))
    pts = vtk_to_numpy(pd.GetPoints().GetData())
    lines = pd.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    checked = 0
    while lines.GetNextCell(id_list):
        pids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        tag = int(round(float(tags[pids[0]])))
        test_case.assertGreater(tag, 0, f'Unexpected tag {tag} in {vtp_path}')
        expected = np.asarray(native_fibers[tag - 1], dtype=np.float64)
        np.testing.assert_allclose(
            pts[pids], expected, atol=atol,
            err_msg=f'Point round-trip mismatch in {vtp_path} for fiber {tag}',
        )
        checked += 1
    return checked


def _mock_slicer_hardening(in_dir, out_dir, tfm_path, slicer_path, jobs,
                           inverse=True, split_output_dir=None, no_xvfb=False):
    from whitematteranalysis._hemisphere import read_mask_location_from_vtk, split_polydata

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_output_dir is not None:
        split_output_dir = Path(split_output_dir)
        for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
            (split_output_dir / folder).mkdir(parents=True, exist_ok=True)
    mat = wma.transform.read_itk_affine_matrix_ras(str(tfm_path), inverse=inverse)
    for pd_file in wma.io.list_vtk_files(str(in_dir)):
        name = Path(pd_file).name
        out_file = out_dir / name
        if out_file.exists():
            hardened = wma.io.read_polydata(str(out_file))
        else:
            pd = wma.io.read_polydata(pd_file)
            hardened = (wma.transform.apply_affine_to_polydata(pd, mat)
                        if pd.GetNumberOfCells() > 0 else pd)
            wma.io.write_polydata(hardened, str(out_file))
        if split_output_dir is not None:
            split_files = [
                split_output_dir / folder / name
                for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural')
            ]
            if not all(p.is_file() for p in split_files):
                found, labels = read_mask_location_from_vtk(hardened)
                assert found and (labels == 0).sum() <= 1
                split_polydata(hardened, labels, name, str(split_output_dir))
    (out_dir / 'transform_complete.log').write_text('mocked\n')
    return 'mock_slicer_model_node'


class ORGPipelineCLITests(unittest.TestCase):
    def setUp(self):
        cfg_fields = ('requested', 'float32', 'strict', 'devices', 'allow_mps')
        saved_cfg = {name: getattr(wma.gpu.config, name) for name in cfg_fields}
        self.addCleanup(lambda: [setattr(wma.gpu.config, k, v) for k, v in saved_cfg.items()])

    def test_single_process_org_cli_end_to_end_and_short_fiber_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            long_left = _bundle_coords(-35.0, 0.0, 6, n_pts=21, step=3.0, seed=10)
            long_right = _bundle_coords(35.0, 0.0, 6, n_pts=21, step=3.0, seed=11)
            long_comm = _bundle_coords(0.0, 35.0, 6, n_pts=21, step=3.0, commissural=True, seed=12)
            # Short fibers (< 40 mm: 6 points * 3.0 mm = 18 mm)
            short_fibers = _bundle_coords(-35.0, 0.0, 5, n_pts=6, step=3.0, seed=13)
            long_fibers = long_left + long_right + long_comm
            all_fibers = long_fibers + short_fibers
            tags = [float(i + 1) for i in range(len(long_fibers))] + [-999.0] * len(short_fibers)
            subj_pd = _make_tract_polydata(all_fibers, tags=tags)
            subj_path = root / 'sub-synth.vtp'
            wma.io.write_polydata(subj_pd, str(subj_path))

            out_root = root / 'out_root'
            # Invoke CLI via subprocess from a non-repo working directory without --slicer
            env = dict(os.environ, WMA_GPU='off', CUDA_VISIBLE_DEVICES='',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
            proc = subprocess.run(
                [sys.executable, str(ORG_CLI_PATH),
                 '-i', str(subj_path),
                 '-o', str(out_root),
                 '-a', str(atlas_root),
                 '-r', 'rig',
                 '-l', '40',
                 '-j', '2',
                 '-gpu', 'off'],
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)

            case_dir = out_root / 'sub-synth'
            tracts_dir = case_dir / 'AnatomicalTracts'
            self.assertTrue((tracts_dir / 'anatomical_tracts_complete.log').is_file())
            for expected_tract in ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CC1.vtp'):
                tract_file = tracts_dir / expected_tract
                self.assertTrue(tract_file.is_file(), expected_tract)
                t_pd = wma.io.read_polydata(str(tract_file))
                self.assertGreater(t_pd.GetNumberOfLines(), 0, expected_tract)
                tag_arr = vtk_to_numpy(t_pd.GetPointData().GetArray('fiber_tag'))
                self.assertFalse(np.any(tag_arr < 0), f'Short fiber leaked into {expected_tract}')

            qc = json.loads((case_dir / 'qc_summary.json').read_text())
            self.assertEqual(qc['hardening_backend'], 'pure_python_affine')
            self.assertEqual(qc['input_fibers'], len(all_fibers))
            self.assertEqual(qc['registered_fibers'], len(all_fibers))
            self.assertEqual(qc['filtered_fibers'], len(long_fibers))
            self.assertEqual(qc['excluded_short_fibers'], len(short_fibers))
            self.assertEqual(qc['anatomical_tracts_written'], 3)
            self.assertFalse(list(out_root.glob('.staging_*')))
            # Default keep-intermediates=0 leaves no intermediate cluster folders
            self.assertFalse((case_dir / 'FiberClustering').exists())
            self.assertFalse((case_dir / 'TractRegistration').exists())

            # Verify --keep-intermediates 2 + --no-atomic-publish + linear -t pre-transform round-trip
            tx = vtk.vtkTransform()
            tx.Translate(25.0, -30.0, 40.0)
            tx.RotateZ(20.0)
            pre_tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['pre'])[0])
            out_keep2 = root / 'out_keep2'
            proc2 = subprocess.run(
                [sys.executable, str(ORG_CLI_PATH),
                 '-i', str(subj_path),
                 '-o', str(out_keep2),
                 '-a', str(atlas_root),
                 '-t', str(pre_tfm),
                 '-r', 'rig',
                 '-l', '40',
                 '-j', '1',
                 '-gpu', 'off',
                 '--keep-intermediates', '2',
                 '--no-atomic-publish'],
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(proc2.returncode, 0, proc2.stdout)
            keep2_case = out_keep2 / 'sub-synth'
            self.assertTrue((keep2_case / 'TractRegistration' / 'sub-synth' / 'output_tractography' / 'itk_txform_sub-synth.tfm').is_file())
            self.assertTrue((keep2_case / 'FiberClustering' / 'InitialClusters' / 'sub-synth' / 'cluster_00001.vtp').is_file())
            self.assertTrue((keep2_case / 'FiberClustering' / 'OutlierRemovedClusters' / 'sub-synth_outlier_removed' / 'cluster_log.txt').is_file())
            self.assertTrue((keep2_case / 'FiberClustering' / 'TransformedClusters' / 'sub-synth' / 'transform_complete.log').is_file())
            self.assertTrue((keep2_case / 'FiberClustering' / 'SeparatedClusters' / 'hemisphere_separation_complete.log').is_file())
            self.assertTrue((keep2_case / 'AnatomicalTracts' / 'anatomical_tracts_complete.log').is_file())
            self.assertTrue((keep2_case / 'qc_summary.json').is_file())
            checked_tracts = sum(
                _assert_native_roundtrip_points(self, keep2_case / 'AnatomicalTracts' / tract, long_fibers)
                for tract in ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CC1.vtp')
            )
            self.assertGreater(checked_tracts, 0)
            sep_root = keep2_case / 'FiberClustering' / 'SeparatedClusters'
            checked_sep = sum(
                _assert_native_roundtrip_points(self, sep_vtp, long_fibers)
                for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural')
                for sep_vtp in sorted((sep_root / folder).glob('cluster_*.vtp'))
            )
            self.assertGreater(checked_sep, 0)

    def test_atomic_publish_cleans_up_staging_on_assembly_failure(self):
        org_cli = _load_module('wm_apply_ORG_atlas_test', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            fibers = (_bundle_coords(-35.0, 0.0, 4, n_pts=21, step=3.0, seed=20)
                      + _bundle_coords(35.0, 0.0, 4, n_pts=21, step=3.0, seed=21)
                      + _bundle_coords(0.0, 35.0, 4, n_pts=21, step=3.0, commissural=True, seed=22))
            subj_path = root / 'sub-fail.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers), str(subj_path))
            out_root = root / 'out_fail'

            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_root),
                '-a', str(atlas_root), '-r', 'rig', '-gpu', 'off']), \
                    patch.object(org_cli, '_assemble_single_tract',
                                 side_effect=RuntimeError('induced tract assembly failure')):
                with self.assertRaisesRegex(RuntimeError, 'induced tract assembly failure'):
                    org_cli.main()

            self.assertFalse((out_root / 'sub-fail').exists())
            self.assertFalse(list(out_root.glob('.staging_*')))

    def test_tensor_input_requires_slicer_in_rig_mode(self):
        org_cli = _load_module('wm_apply_ORG_atlas_tensor_req', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)
            fibers = _bundle_coords(-35.0, 0.0, 4, n_pts=21, step=3.0, seed=30)
            subj_path = root / 'sub-tensor.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers, tensors=True), str(subj_path))
            out_root = root / 'out_tensor'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_root),
                '-a', str(atlas_root), '-r', 'rig', '-gpu', 'off']):
                with self.assertRaisesRegex(RuntimeError, 'Slicer'):
                    org_cli.main()

    def test_harden_transform_with_slicer_model_node_fallback_and_tensor_rejection(self):
        scene = FakeMRMLScene()

        class FakeFactoryManager:
            searchPaths = []

        class FakeModuleManager:
            def factoryManager(self):
                return FakeFactoryManager()

        class FakeApp:
            def moduleManager(self):
                return FakeModuleManager()

        fake_slicer = types.ModuleType('slicer')
        fake_slicer.mrmlScene = scene
        fake_slicer.app = FakeApp()

        # Patch AddNewNodeByClass to handle FakeTransformStorageNode too
        orig_add = scene.AddNewNodeByClass

        def add_node_by_class(cls_name, name=''):
            if cls_name == 'vtkMRMLTransformStorageNode':
                scene._counter += 1
                node = FakeTransformStorageNode(f'{cls_name}{scene._counter}', scene)
                scene.nodes.append(node)
                return node
            return orig_add(cls_name, name)

        scene.AddNewNodeByClass = add_node_by_class

        fake_transforms_mod = types.ModuleType('vtkSlicerTransformsModuleLogicPython')
        fake_transforms_mod.vtkSlicerTransformLogic = lambda: FakeTransformLogic(scene)

        real_import = builtins.__import__

        def mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'vtkSlicerTractographyDisplayModuleLogicPython':
                raise ImportError('SlicerDMRI is not installed')
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tfm_path = root / 'tx.tfm'
            tx = vtk.vtkTransform()
            tx.Translate(2.0, -3.0, 4.0)
            wma.io.write_transforms_to_itk_format([tx], str(root), ['test'])
            tfm_path = root / 'itk_txform_test.tfm'

            # 1) Tensor-free VTP succeeds with ModelNode fallback + split_output_dir
            in_free = root / 'in_free'
            out_free = root / 'out_free'
            split_free = root / 'split_free'
            in_free.mkdir()
            out_free.mkdir()
            for sub in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                (split_free / sub).mkdir(parents=True)

            coords = _bundle_coords(-20.0, 0.0, 2, n_pts=5, step=2.0, seed=1) + \
                     _bundle_coords(20.0, 0.0, 2, n_pts=5, step=2.0, seed=2)
            pd_free = _make_tract_polydata(coords, hemi_labels=[1.0, 1.0, 2.0, 2.0])
            wma.io.write_polydata(pd_free, str(in_free / 'cluster_00001.vtp'))

            with patch.dict(sys.modules, {
                'slicer': fake_slicer,
                'vtkSlicerTransformsModuleLogicPython': fake_transforms_mod,
            }):
                helper = _load_module('slicer_helper_fallback_test', SLICER_HELPER_PATH)
                argv = [
                    str(SLICER_HELPER_PATH), str(in_free), str(tfm_path), '1', str(out_free),
                    '--split-output-directory', str(split_free),
                ]
                with patch.object(builtins, '__import__', side_effect=mocked_import), \
                        patch.object(sys, 'argv', argv):
                    helper.main()

            self.assertEqual(scene.GetNumberOfNodes(), 0)
            hardened = wma.io.read_polydata(str(out_free / 'cluster_00001.vtp'))
            np.testing.assert_allclose(
                vtk_to_numpy(hardened.GetPoints().GetData())[0],
                np.asarray(coords[0][0]) - np.array([2.0, -3.0, 4.0]),
                atol=1e-5,
            )
            self.assertEqual(
                wma.io.read_polydata(str(split_free / 'tracts_left_hemisphere' / 'cluster_00001.vtp')).GetNumberOfLines(),
                2,
            )
            self.assertEqual(
                wma.io.read_polydata(str(split_free / 'tracts_right_hemisphere' / 'cluster_00001.vtp')).GetNumberOfLines(),
                2,
            )

            # 2) VTP with 9-component tensors must fail clearly without SlicerDMRI
            in_tensor = root / 'in_tensor'
            out_tensor = root / 'out_tensor'
            in_tensor.mkdir()
            out_tensor.mkdir()
            pd_tensor = _make_tract_polydata(coords, tensors=True, hemi_labels=[1.0, 1.0, 2.0, 2.0])
            wma.io.write_polydata(pd_tensor, str(in_tensor / 'cluster_00001.vtp'))

            with patch.dict(sys.modules, {
                'slicer': fake_slicer,
                'vtkSlicerTransformsModuleLogicPython': fake_transforms_mod,
            }):
                helper = _load_module('slicer_helper_tensor_reject_test', SLICER_HELPER_PATH)
                argv = [str(SLICER_HELPER_PATH), str(in_tensor), str(tfm_path), '1', str(out_tensor)]
                with patch.object(builtins, '__import__', side_effect=mocked_import), \
                        patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'tensor|SlicerDMRI|9-component'):
                        helper.main()
            self.assertEqual(scene.GetNumberOfNodes(), 0)

    def test_org_pipeline_never_forks_after_threads_and_writes_serially_at_j1(self):
        org_cli = _load_module('wm_apply_ORG_atlas_nofork', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)
            fibers = (_bundle_coords(-35.0, 0.0, 4, n_pts=21, step=3.0, seed=40)
                      + _bundle_coords(35.0, 0.0, 4, n_pts=21, step=3.0, seed=41)
                      + _bundle_coords(0.0, 35.0, 4, n_pts=21, step=3.0, commissural=True, seed=42))
            subj_path = root / 'sub-nofork.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers), str(subj_path))

            # Direct check on write_polydatas_parallel(use_threads=True) on mocked Linux fork host
            direct_pd = _make_tract_polydata(fibers[:2])
            direct_paths = [str(root / 'direct_0.vtp'), str(root / 'direct_1.vtp')]
            with patch.object(sys, 'platform', 'linux'), \
                    patch.object(multiprocessing, 'get_all_start_methods', return_value=['fork', 'spawn']), \
                    patch.object(multiprocessing, 'get_context',
                                 side_effect=AssertionError('fork context must not be used when use_threads=True')):
                wma.io.write_polydatas_parallel([direct_pd, direct_pd], direct_paths, jobs=2, use_threads=True)
            self.assertTrue(all(Path(p).is_file() for p in direct_paths))

            for jobs in (1, 2):
                out_root = root / f'out_j{jobs}'
                recorded_write_calls = []
                orig_parallel_write = wma.io.write_polydatas_parallel

                def spy_write(polydatas, filenames, jobs=None, use_threads=False):
                    recorded_write_calls.append((jobs, use_threads))
                    if jobs == 1:
                        with patch.object(concurrent.futures, 'ThreadPoolExecutor',
                                          side_effect=AssertionError('-j 1 must write serially without ThreadPoolExecutor')):
                            return orig_parallel_write(polydatas, filenames, jobs=jobs, use_threads=use_threads)
                    return orig_parallel_write(polydatas, filenames, jobs=jobs, use_threads=use_threads)

                with self.subTest(jobs=jobs), \
                        patch.object(sys, 'platform', 'linux'), \
                        patch.object(multiprocessing, 'get_all_start_methods', return_value=['fork', 'spawn']), \
                        patch.object(multiprocessing, 'get_context',
                                     side_effect=AssertionError('ORG pipeline must never fork()')), \
                        patch.object(wma.io, 'write_polydatas_parallel', side_effect=spy_write), \
                        patch.object(sys, 'argv', [
                            str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_root),
                            '-a', str(atlas_root), '-r', 'rig', '-j', str(jobs),
                            '-gpu', 'off', '--keep-intermediates', '2']):
                    org_cli.main()

                # keep_intermediates=2 calls write_polydatas_parallel 6 times (init, outrem, tfm, 3x sep)
                self.assertEqual(len(recorded_write_calls), 6)
                self.assertTrue(all(call == (jobs, True) for call in recorded_write_calls), recorded_write_calls)

    def test_non_tier1_pretransform_keep1_roundtrip_and_no_atomic_rerun_cleanup(self):
        org_cli = _load_module('wm_apply_ORG_atlas_nontier1', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            fake_slicer = root / 'FakeSlicer'
            fake_slicer.write_text('#!/bin/sh\nexit 0\n')

            fibers1 = (
                _bundle_coords(-35.0, 0.0, 5, n_pts=21, step=3.0, seed=50)
                + _bundle_coords(35.0, 0.0, 5, n_pts=21, step=3.0, seed=51)
                + _bundle_coords(0.0, 35.0, 5, n_pts=21, step=3.0, commissural=True, seed=52)
            )
            tags1 = [float(i + 1) for i in range(len(fibers1))]
            subj_path = root / 'sub-nontier1.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers1, tags=tags1, tensors=True), str(subj_path))

            tx = vtk.vtkTransform()
            tx.Translate(30.0, -25.0, 15.0)
            tx.RotateZ(15.0)
            pre_tfm = Path(wma.io.write_transforms_to_itk_format([tx], str(root), ['pre'])[0])

            out_keep1 = root / 'out_keep1'
            with patch.object(org_cli, '_invoke_slicer_hardening', side_effect=_mock_slicer_hardening), \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_keep1),
                        '-a', str(atlas_root), '-s', str(fake_slicer), '-t', str(pre_tfm),
                        '-r', 'rig', '-j', '1', '-gpu', 'off', '--keep-intermediates', '1']):
                org_cli.main()

            keep1_case = out_keep1 / 'sub-nontier1'
            sep_dir = keep1_case / 'FiberClustering' / 'SeparatedClusters'
            self.assertTrue((sep_dir / 'hemisphere_separation_complete.log').is_file())
            qc1 = json.loads((keep1_case / 'qc_summary.json').read_text())
            self.assertEqual(qc1['hardening_backend'], 'mock_slicer_model_node')
            self.assertTrue(qc1['has_tensor_arrays'])

            # Verify empty-string WMA_VTP_COMPRESSION='' restoration and unset preservation in _default_lz4_compression
            with patch.dict(os.environ, {'WMA_VTP_COMPRESSION': ''}):
                with org_cli._default_lz4_compression():
                    self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), 'lz4')
                self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), '')
            with patch.dict(os.environ, {'WMA_VTP_COMPRESSION': 'zlib'}):
                with org_cli._default_lz4_compression():
                    self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), 'zlib')
                self.assertEqual(os.environ.get('WMA_VTP_COMPRESSION'), 'zlib')

            # Verify non-Tier-1 -t --keep-intermediates 2 runs pre_tfm once and stores native coords in TransformedClusters/
            out_keep2 = root / 'out_keep2_t'
            with patch.object(org_cli, '_invoke_slicer_hardening', side_effect=_mock_slicer_hardening) as mock_inv, \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_keep2),
                        '-a', str(atlas_root), '-s', str(fake_slicer), '-t', str(pre_tfm),
                        '-r', 'rig', '-j', '1', '-gpu', 'off', '--keep-intermediates', '2']):
                org_cli.main()
            # 3 hardening invocations total: forward pre_tfm, rig inverse -> scratch, pre_tfm inverse -> TransformedClusters + SeparatedClusters
            self.assertEqual(mock_inv.call_count, 3)
            tfm_vtp_count = sum(
                _assert_native_roundtrip_points(self, p, fibers1)
                for p in sorted((out_keep2 / 'sub-nontier1' / 'FiberClustering' / 'TransformedClusters' / 'sub-nontier1').glob('cluster_*.vtp'))
            )
            self.assertEqual(tfm_vtp_count, len(fibers1))

            checked_tracts = sum(
                _assert_native_roundtrip_points(self, keep1_case / 'AnatomicalTracts' / tract, fibers1)
                for tract in ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CC1.vtp')
            )
            self.assertEqual(checked_tracts, qc1['retained_clustered_fibers'])
            checked_sep = sum(
                _assert_native_roundtrip_points(self, sep_vtp, fibers1)
                for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural')
                for sep_vtp in sorted((sep_dir / folder).glob('cluster_*.vtp'))
            )
            self.assertEqual(checked_sep, qc1['retained_clustered_fibers'])

            # Rerun twice into the same --no-atomic-publish --keep-intermediates 2 directory
            # with changed input fibers while hardening mock skips existing files.
            out_rerun = root / 'out_rerun'
            with patch.object(org_cli, '_invoke_slicer_hardening', side_effect=_mock_slicer_hardening), \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_rerun),
                        '-a', str(atlas_root), '-s', str(fake_slicer),
                        '-r', 'rig', '-j', '1', '-gpu', 'off',
                        '--keep-intermediates', '2', '--no-atomic-publish']):
                org_cli.main()

            fibers2 = (
                _bundle_coords(-35.0, 0.0, 4, n_pts=21, step=3.0, seed=60)
                + _bundle_coords(35.0, 0.0, 4, n_pts=21, step=3.0, seed=61)
                + _bundle_coords(0.0, 35.0, 4, n_pts=21, step=3.0, commissural=True, seed=62)
            )
            tags2 = [float(1001 + i) for i in range(len(fibers2))]
            wma.io.write_polydata(_make_tract_polydata(fibers2, tags=tags2, tensors=True), str(subj_path))

            with patch.object(org_cli, '_invoke_slicer_hardening', side_effect=_mock_slicer_hardening), \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_rerun),
                        '-a', str(atlas_root), '-s', str(fake_slicer),
                        '-r', 'rig', '-j', '1', '-gpu', 'off',
                        '--keep-intermediates', '2', '--no-atomic-publish']):
                org_cli.main()

            rerun_case = out_rerun / 'sub-nontier1'
            qc_rerun = json.loads((rerun_case / 'qc_summary.json').read_text())
            self.assertEqual(qc_rerun['input_fibers'], len(fibers2))
            for tract in ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CC1.vtp'):
                t_pd = wma.io.read_polydata(str(rerun_case / 'AnatomicalTracts' / tract))
                self.assertGreater(t_pd.GetNumberOfLines(), 0)
                tag_arr = vtk_to_numpy(t_pd.GetPointData().GetArray('fiber_tag'))
                self.assertTrue(np.all(tag_arr >= 1001.0), f'Stale run-1 fibers leaked into {tract}: {tag_arr}')

    def test_symlink_case_id_zero_atlas_cluster_empty_schema_and_slicer_which(self):
        org_cli = _load_module('wm_apply_ORG_atlas_edge_cases', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            fibers = (
                _bundle_coords(-35.0, 0.0, 5, n_pts=21, step=3.0, seed=70)
                + _bundle_coords(35.0, 0.0, 5, n_pts=21, step=3.0, seed=71)
                + _bundle_coords(0.0, 35.0, 5, n_pts=21, step=3.0, commissural=True, seed=72)
            )
            real_subj = root / 'tractography.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers), str(real_subj))
            sym_subj = root / 'sub-alias.vtp'
            os.symlink(str(real_subj), str(sym_subj))

            # Clear cluster_00001.vtp to 0 fibers to verify n_atlas == 0 retains subject fibers in QC & logs
            fc_dir = atlas_root / 'ORG-800FC-100HCP'
            empty_cluster = vtk.vtkPolyData()
            empty_cluster.SetPoints(vtk.vtkPoints())
            empty_cluster.SetLines(vtk.vtkCellArray())
            wma.io.write_polydata(empty_cluster, str(fc_dir / 'cluster_00001.vtp'))

            out_sym = root / 'out_sym'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(sym_subj), '-o', str(out_sym),
                '-a', str(atlas_root), '-r', 'rig', '-j', '1', '-gpu', 'off',
                '--keep-intermediates', '2']):
                org_cli.main()

            sym_case = out_sym / 'sub-alias'
            self.assertTrue(sym_case.is_dir())
            self.assertFalse((out_sym / 'tractography').exists())
            qc_sym = json.loads((sym_case / 'qc_summary.json').read_text())
            self.assertEqual(qc_sym['case_id'], 'sub-alias')
            self.assertEqual(qc_sym['retained_clustered_fibers'], len(fibers))
            self.assertEqual(qc_sym['nonempty_clusters'], 3)
            log_lines = (
                sym_case / 'FiberClustering' / 'OutlierRemovedClusters' /
                'sub-alias_outlier_removed' / 'cluster_log.txt'
            ).read_text().splitlines()
            row1 = log_lines[1].split('\t')
            self.assertGreater(int(row1[1]), 0)
            self.assertEqual(row1[2], row1[1])

            # _empty_like_schema handles vtkStringArray in PointData/CellData when filtered_fibers == 0
            str_pd = _make_tract_polydata(fibers[:2])
            s_pt = vtk.vtkStringArray()
            s_pt.SetName('PointLabel')
            for _ in range(str_pd.GetNumberOfPoints()):
                s_pt.InsertNextValue('pt')
            str_pd.GetPointData().AddArray(s_pt)
            s_cell = vtk.vtkStringArray()
            s_cell.SetName('CellLabel')
            for _ in range(str_pd.GetNumberOfLines()):
                s_cell.InsertNextValue('cell')
            str_pd.GetCellData().AddArray(s_cell)
            empty_schema = org_cli._empty_like_schema(str_pd)
            self.assertIsInstance(empty_schema.GetPointData().GetAbstractArray('PointLabel'), vtk.vtkStringArray)
            self.assertIsInstance(empty_schema.GetCellData().GetAbstractArray('CellLabel'), vtk.vtkStringArray)

            empty_subj = root / 'sub-emptyfilter.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers[:2], tags=[1.0, 2.0]), str(empty_subj))
            out_empty = root / 'out_empty'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(empty_subj), '-o', str(out_empty),
                '-a', str(atlas_root), '-r', 'rig', '-l', '500', '-j', '1', '-gpu', 'off']):
                org_cli.main()
            qc_empty = json.loads((out_empty / 'sub-emptyfilter' / 'qc_summary.json').read_text())
            self.assertEqual(qc_empty['filtered_fibers'], 0)
            self.assertEqual(qc_empty['retained_clustered_fibers'], 0)

            # Bare executable name supplied to -s resolves via shutil.which
            bin_dir = root / 'path_bin'
            bin_dir.mkdir()
            slicer_bin = bin_dir / 'FakeSlicerOnPath'
            slicer_bin.write_text('#!/bin/sh\nexit 0\n')
            slicer_bin.chmod(0o755)
            tensor_subj = root / 'sub-path-slicer.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers, tensors=True), str(tensor_subj))
            out_which = root / 'out_which'
            with patch.dict(os.environ, {'PATH': f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}), \
                    patch.object(org_cli, '_invoke_slicer_hardening', side_effect=_mock_slicer_hardening) as mock_harden, \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(tensor_subj), '-o', str(out_which),
                        '-a', str(atlas_root), '-s', 'FakeSlicerOnPath',
                        '-r', 'rig', '-j', '1', '-gpu', 'off']):
                org_cli.main()
            self.assertEqual(Path(mock_harden.call_args.args[3]).resolve(), slicer_bin.resolve())

    def test_zero_intermediate_indexed_pipeline_matches_keep2_exactly(self):
        org_cli = _load_module('wm_apply_ORG_atlas_indexed_parity', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            fibers = (
                _bundle_coords(-35.0, 0.0, 6, n_pts=21, step=3.0, seed=101)
                + _bundle_coords(35.0, 0.0, 6, n_pts=21, step=3.0, seed=102)
                + _bundle_coords(0.0, 35.0, 6, n_pts=21, step=3.0, commissural=True, seed=103)
            )
            tags = [float(i + 1) for i in range(len(fibers))]
            subj_path = root / 'sub-parity.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers, tags=tags), str(subj_path))

            out_ref = root / 'out_keep2'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_ref),
                '-a', str(atlas_root), '-r', 'rig', '-j', '2', '-gpu', 'off',
                '--keep-intermediates', '2']):
                org_cli.main()

            out_idx = root / 'out_keep0'
            with patch.object(wma.cluster, 'mask_all_clusters',
                              side_effect=AssertionError('keep=0 must skip mask_all_clusters')), \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_idx),
                        '-a', str(atlas_root), '-r', 'rig', '-j', '2', '-gpu', 'off',
                        '--keep-intermediates', '0']):
                org_cli.main()

            qc_ref = json.loads((out_ref / 'sub-parity' / 'qc_summary.json').read_text())
            qc_idx = json.loads((out_idx / 'sub-parity' / 'qc_summary.json').read_text())
            for key in (
                'input_fibers', 'registered_fibers', 'filtered_fibers',
                'excluded_short_fibers', 'total_clusters', 'nonempty_clusters',
                'outlier_removed_fibers', 'retained_clustered_fibers',
                'anatomical_tracts_expected', 'anatomical_tracts_written', 'tracts',
            ):
                self.assertEqual(qc_idx[key], qc_ref[key], f'QC mismatch for {key}')

            for tract_name in ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CC1.vtp'):
                pd_ref = wma.io.read_polydata(str(out_ref / 'sub-parity' / 'AnatomicalTracts' / tract_name))
                pd_idx = wma.io.read_polydata(str(out_idx / 'sub-parity' / 'AnatomicalTracts' / tract_name))
                self.assertEqual(pd_idx.GetNumberOfLines(), pd_ref.GetNumberOfLines())
                self.assertEqual(pd_idx.GetNumberOfPoints(), pd_ref.GetNumberOfPoints())
                np.testing.assert_allclose(
                    vtk_to_numpy(pd_idx.GetPoints().GetData()),
                    vtk_to_numpy(pd_ref.GetPoints().GetData()),
                    rtol=1e-6, atol=1e-6,
                )
                for pt_arr in ('fiber_tag', 'HemisphereLocation', 'cluster_idx'):
                    np.testing.assert_array_equal(
                        vtk_to_numpy(pd_idx.GetPointData().GetArray(pt_arr)),
                        vtk_to_numpy(pd_ref.GetPointData().GetArray(pt_arr)),
                    )
                for cell_arr in ('ClusterNumber', 'EmbeddingCoordinate'):
                    np.testing.assert_allclose(
                        vtk_to_numpy(pd_idx.GetCellData().GetArray(cell_arr)),
                        vtk_to_numpy(pd_ref.GetCellData().GetArray(cell_arr)),
                        rtol=1e-6, atol=1e-6,
                    )
                space_idx = pd_idx.GetFieldData().GetAbstractArray('SPACE')
                self.assertIsNotNone(space_idx)
                self.assertEqual(space_idx.GetValue(0), 'RAS')

    def test_batch_mode_reuses_resident_atlas_and_handles_manifests_and_errors(self):
        org_cli = _load_module('wm_apply_ORG_atlas_batch', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas_root = root / 'ORG-Atlas'
            _build_synthetic_org_atlas(atlas_root)

            fibers_a = (
                _bundle_coords(-35.0, 0.0, 5, n_pts=21, step=3.0, seed=111)
                + _bundle_coords(35.0, 0.0, 5, n_pts=21, step=3.0, seed=112)
                + _bundle_coords(0.0, 35.0, 5, n_pts=21, step=3.0, commissural=True, seed=113)
            )
            fibers_b = (
                _bundle_coords(-35.0, 0.0, 4, n_pts=21, step=3.0, seed=114)
                + _bundle_coords(35.0, 0.0, 4, n_pts=21, step=3.0, seed=115)
                + _bundle_coords(0.0, 35.0, 4, n_pts=21, step=3.0, commissural=True, seed=116)
            )
            in_dir = root / 'subjects'
            in_dir.mkdir()
            sub_a = in_dir / 'sub-01.vtp'
            sub_b = in_dir / 'sub-02.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers_a, tags=[float(i + 1) for i in range(len(fibers_a))]), str(sub_a))
            wma.io.write_polydata(_make_tract_polydata(fibers_b, tags=[float(i + 100) for i in range(len(fibers_b))]), str(sub_b))

            # Single-subject reference for sub-02
            out_single = root / 'out_single'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(sub_b), '-o', str(out_single),
                '-a', str(atlas_root), '-r', 'rig', '-j', '2', '-gpu', 'off']):
                org_cli.main()

            # Batch mode via --input-dir loads registration atlas & spectral atlas only ONCE
            out_batch = root / 'out_batch'
            orig_read = wma.io.read_polydata
            reg_reads = []

            def _counting_read(path, *a, **kw):
                if Path(path).name == 'registration_atlas.vtk':
                    reg_reads.append(path)
                return orig_read(path, *a, **kw)

            with patch.object(wma.io, 'read_polydata', side_effect=_counting_read), \
                    patch.object(wma.cluster, 'load_atlas', wraps=wma.cluster.load_atlas) as atlas_loader, \
                    patch.object(sys, 'argv', [
                        str(ORG_CLI_PATH), '--input-dir', str(in_dir), '-o', str(out_batch),
                        '-a', str(atlas_root), '-r', 'rig', '-j', '2', '-gpu', 'off']):
                org_cli.main()

            self.assertEqual(len(reg_reads), 1)
            self.assertEqual(atlas_loader.call_count, 1)
            self.assertEqual(atlas_loader.call_args.kwargs.get('for_labeling'), True)

            batch_qc = json.loads((out_batch / 'batch_qc_summary.json').read_text())
            self.assertEqual(batch_qc['total_subjects'], 2)
            self.assertEqual(batch_qc['succeeded_subjects'], 2)
            self.assertEqual(batch_qc['failed_subjects'], 0)

            qc_single_b = json.loads((out_single / 'sub-02' / 'qc_summary.json').read_text())
            qc_batch_b = json.loads((out_batch / 'sub-02' / 'qc_summary.json').read_text())
            self.assertEqual(qc_batch_b['tracts'], qc_single_b['tracts'])
            self.assertEqual(qc_batch_b['retained_clustered_fibers'], qc_single_b['retained_clustered_fibers'])

            # -i <dir> + per-subject transform directory support
            tfm_dir = root / 'tfms'
            tfm_dir.mkdir()
            tx = vtk.vtkTransform()
            tx.Translate(5.0, -3.0, 2.0)
            wma.io.write_transforms_to_itk_format([tx], str(tfm_dir), ['sub-01'])
            wma.io.write_transforms_to_itk_format([tx], str(tfm_dir), ['sub-02'])
            out_dir_i = root / 'out_batch_i_tfm'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '-i', str(in_dir), '-o', str(out_dir_i),
                '-a', str(atlas_root), '-t', str(tfm_dir), '-r', 'rig', '-j', '1', '-gpu', 'off']):
                org_cli.main()
            self.assertTrue((out_dir_i / 'sub-01' / 'qc_summary.json').is_file())
            self.assertTrue((out_dir_i / 'sub-02' / 'qc_summary.json').is_file())

            # --input-list with blank/comment lines + --continue-on-error + empty polydata failure
            bad_sub = root / 'sub-bad.vtp'
            empty_pd = vtk.vtkPolyData()
            empty_pd.SetPoints(vtk.vtkPoints())
            empty_pd.SetLines(vtk.vtkCellArray())
            wma.io.write_polydata(empty_pd, str(bad_sub))
            manifest = root / 'subjects.txt'
            manifest.write_text(f"# batch list\n\n{sub_a.name}\n{bad_sub}\n", encoding='utf-8')
            # Write relative sub_a from root? Put symlink/copy or use full path so relative resolves cleanly
            manifest.write_text(f"# batch list\n\nsubjects/{sub_a.name}\n{bad_sub}\n", encoding='utf-8')
            out_err = root / 'out_continue_err'
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '--input-list', str(manifest), '-o', str(out_err),
                '-a', str(atlas_root), '-r', 'rig', '-j', '1', '-gpu', 'off', '--continue-on-error']):
                with self.assertRaises(SystemExit) as cm:
                    org_cli.main()
                self.assertNotEqual(cm.exception.code, 0)
            err_qc = json.loads((out_err / 'batch_qc_summary.json').read_text())
            self.assertEqual(err_qc['succeeded_subjects'], 1)
            self.assertEqual(err_qc['failed_subjects'], 1)
            self.assertEqual(err_qc['subjects'][0]['status'], 'ok')
            self.assertEqual(err_qc['subjects'][1]['status'], 'failed')

            # Duplicate case_id across .vtp and .vtk raises ValueError before processing
            dup_vtk = root / 'sub-01.vtk'
            wma.io.write_polydata(_make_tract_polydata(fibers_a), str(dup_vtk))
            dup_manifest = root / 'dup.txt'
            dup_manifest.write_text(f"{sub_a}\n{dup_vtk}\n", encoding='utf-8')
            with patch.object(sys, 'argv', [
                str(ORG_CLI_PATH), '--input-list', str(dup_manifest), '-o', str(root / 'out_dup'),
                '-a', str(atlas_root), '-r', 'rig', '-gpu', 'off']):
                with self.assertRaisesRegex(ValueError, 'Duplicate case_id'):
                    org_cli.main()


if __name__ == '__main__':
    unittest.main()
