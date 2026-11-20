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


def _bundle_coords(center_x, center_z, n_fibers, n_pts=21, step=3.0, commissural=False, seed=0):
    rng = np.random.RandomState(seed)
    fibers = []
    for _ in range(n_fibers):
        jitter = rng.normal(scale=0.35, size=(1, 3))
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

    # Three well-separated anatomical groups: left, right, commissural
    left_fibers = _bundle_coords(-35.0, 0.0, 40, n_pts=21, step=3.0, seed=1)
    right_fibers = _bundle_coords(35.0, 0.0, 40, n_pts=21, step=3.0, seed=2)
    comm_fibers = _bundle_coords(0.0, 35.0, 40, n_pts=21, step=3.0, commissural=True, seed=3)
    all_atlas_fibers = left_fibers + right_fibers + comm_fibers
    atlas_pd = _make_tract_polydata(all_atlas_fibers)
    wma.io.write_polydata(atlas_pd, str(reg_dir / 'registration_atlas.vtk'))

    import contextlib
    # Build a valid Nystrom spectral atlas over these 3 bundles
    nystrom_mask = np.ones(len(all_atlas_fibers), dtype=bool)
    nystrom_mask[115:] = False  # 115 in A (>= 100), 5 in B
    np.random.seed(42)
    try:
        import torch
        has_torch = torch is not None
    except ImportError:
        has_torch = False
    spectral_ctx = (
        contextlib.nullcontext() if has_torch
        else patch.object(wma.cluster.gpu, 'linalg',
                          types.SimpleNamespace(eigh=np.linalg.eigh), create=True)
    )
    with spectral_ctx:
        out_pd, cluster_idx, _, _, _, atlas, _ = wma.cluster.spectral(
            atlas_pd,
            number_of_clusters=3,
            number_of_eigenvectors=3,
            sigma=30.0,
            threshold=0.0,
            number_of_jobs=1,
            use_nystrom=True,
            nystrom_mask=nystrom_mask,
            distance_method='Mean',
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


class ORGPipelineCLITests(unittest.TestCase):
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
            all_fibers = long_left + long_right + long_comm + short_fibers
            tags = [1.0] * (len(long_left) + len(long_right) + len(long_comm)) + [-999.0] * len(short_fibers)
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
            self.assertEqual(qc['filtered_fibers'], len(long_left) + len(long_right) + len(long_comm))
            self.assertEqual(qc['excluded_short_fibers'], len(short_fibers))
            self.assertEqual(qc['anatomical_tracts_written'], 3)
            self.assertFalse(list(out_root.glob('.staging_*')))
            # Default keep-intermediates=0 leaves no intermediate cluster folders
            self.assertFalse((case_dir / 'FiberClustering').exists())
            self.assertFalse((case_dir / 'TractRegistration').exists())

            # Verify --keep-intermediates 2 + --no-atomic-publish + linear -t pre-transform
            tx = vtk.vtkTransform()
            tx.Translate(2.5, -1.5, 3.0)
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


if __name__ == '__main__':
    unittest.main()
