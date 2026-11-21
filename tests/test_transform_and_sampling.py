"""Tests for whitematteranalysis.transform and vectorized registration sampling."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import whitematteranalysis as wma
from test_output_pipeline import fixture as tensor_fixture


def _build_polydata(coords_list, point_dtype=np.float64):
    """Build a vtkPolyData from a list of (N_i, 3) fiber coordinate arrays."""
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    all_pts = []
    for coords in coords_list:
        start = len(all_pts)
        all_pts.extend(coords)
        ids = vtk.vtkIdList()
        for pid in range(start, len(all_pts)):
            ids.InsertNextId(pid)
        lines.InsertNextCell(ids)
    pts_arr = np.asarray(all_pts, dtype=point_dtype).reshape(-1, 3)
    points.SetData(numpy_to_vtk(pts_arr, deep=True))
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def _vtk_transform_polydata(polydata, transform):
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()


class TransformModuleTests(unittest.TestCase):
    def test_lazy_attribute_import_on_wma_package(self):
        from whitematteranalysis import transform as direct_mod
        self.assertIs(wma.transform, direct_mod)

    def test_has_tensor_arrays_detection(self):
        empty = vtk.vtkPolyData()
        self.assertFalse(wma.transform.has_tensor_arrays(empty))

        coords = [np.column_stack([np.arange(5, dtype=np.float64),
                                   np.zeros(5), np.zeros(5)])]
        pd = _build_polydata(coords)
        self.assertFalse(wma.transform.has_tensor_arrays(pd))

        # Scalars, RGB uint8 colors, 10-component embeddings must return False
        scalar = numpy_to_vtk(np.arange(5, dtype=np.float64), deep=True)
        scalar.SetName('scalar')
        pd.GetPointData().AddArray(scalar)
        pd.GetPointData().SetActiveScalars('scalar')

        rgb = numpy_to_vtk(np.zeros((5, 3), dtype=np.uint8), deep=True)
        rgb.SetName('EmbeddingColor')
        pd.GetPointData().AddArray(rgb)

        embed = numpy_to_vtk(np.zeros((1, 10), dtype=np.float32), deep=True)
        embed.SetName('EmbeddingCoordinate')
        pd.GetCellData().AddArray(embed)

        cluster_num = numpy_to_vtk(np.zeros(1, dtype=np.int32), deep=True)
        cluster_num.SetName('ClusterNumber')
        pd.GetCellData().AddArray(cluster_num)
        self.assertFalse(wma.transform.has_tensor_arrays(pd))

        # 9-component array in PointData
        pd_pt9 = _build_polydata(coords)
        arr9 = numpy_to_vtk(np.zeros((5, 9), dtype=np.float32), deep=True)
        arr9.SetName('tensor1')
        pd_pt9.GetPointData().AddArray(arr9)
        self.assertTrue(wma.transform.has_tensor_arrays(pd_pt9))

        # 9-component array in CellData
        pd_cell9 = _build_polydata(coords)
        carr9 = numpy_to_vtk(np.zeros((1, 9), dtype=np.float32), deep=True)
        carr9.SetName('cell_tensors')
        pd_cell9.GetCellData().AddArray(carr9)
        self.assertTrue(wma.transform.has_tensor_arrays(pd_cell9))

        # SetTensors / SetVectors / SetNormals on PointData or CellData
        for setter in ('SetTensors', 'SetVectors', 'SetNormals'):
            comp = 9 if setter == 'SetTensors' else 3
            for loc in ('GetPointData', 'GetCellData'):
                count = 5 if loc == 'GetPointData' else 1
                pd_attr = _build_polydata(coords)
                arr = numpy_to_vtk(np.zeros((count, comp), dtype=np.float32), deep=True)
                arr.SetName(setter + '_arr')
                getattr(pd_attr, loc)().AddArray(arr)
                getattr(getattr(pd_attr, loc)(), setter)(arr)
                self.assertTrue(wma.transform.has_tensor_arrays(pd_attr),
                                f'{setter} on {loc}')

    def test_apply_affine_to_polydata_matches_vtk_and_chunks_cleanly(self):
        rng = np.random.RandomState(123)
        fibers = [rng.normal(scale=40.0, size=(8, 3)) for _ in range(15)]

        tx = vtk.vtkTransform()
        tx.Translate(4.5, -3.25, 7.125)
        tx.RotateX(15.0)
        tx.RotateY(-22.5)
        tx.RotateZ(9.0)
        tx.Scale(1.04, 0.96, 1.02)
        m_vtk = tx.GetMatrix()
        m_np = np.array([[m_vtk.GetElement(i, j) for j in range(4)] for i in range(4)],
                        dtype=np.float64)

        for dtype in (np.float32, np.float64):
            pd = _build_polydata(fibers, point_dtype=dtype)
            # Attach non-tensor attributes to verify preservation
            pt_scalar = numpy_to_vtk(np.arange(pd.GetNumberOfPoints(), dtype=np.float32), deep=True)
            pt_scalar.SetName('HemisphereLocation')
            pd.GetPointData().AddArray(pt_scalar)
            cell_scalar = numpy_to_vtk(np.arange(pd.GetNumberOfLines(), dtype=np.int32), deep=True)
            cell_scalar.SetName('ClusterNumber')
            pd.GetCellData().AddArray(cell_scalar)
            pd.GetCellData().SetActiveScalars('ClusterNumber')
            text = vtk.vtkStringArray()
            text.SetName('SPACE')
            text.InsertNextValue('RAS')
            pd.GetFieldData().AddArray(text)

            before_pts = vtk_to_numpy(pd.GetPoints().GetData()).copy()
            expected_pd = _vtk_transform_polydata(pd, tx)
            expected_pts = vtk_to_numpy(expected_pd.GetPoints().GetData())

            for transform_arg in (tx, m_vtk, m_np):
                with self.subTest(dtype=dtype, arg=type(transform_arg).__name__):
                    with patch.object(vtk, 'vtkTransformPolyDataFilter',
                                      side_effect=AssertionError('should not delegate without tensors')):
                        actual_pd = wma.transform.apply_affine_to_polydata(
                            pd, transform_arg, chunk_points=17)
                    actual_pts = vtk_to_numpy(actual_pd.GetPoints().GetData())
                    self.assertEqual(actual_pts.dtype, dtype)
                    if dtype == np.float32:
                        np.testing.assert_allclose(actual_pts, expected_pts, atol=1e-5, rtol=1e-6)
                    else:
                        np.testing.assert_allclose(actual_pts, expected_pts, atol=1e-12, rtol=1e-13)
                    # Input untouched
                    np.testing.assert_array_equal(vtk_to_numpy(pd.GetPoints().GetData()), before_pts)
                    # Metadata & connectivity preserved
                    np.testing.assert_array_equal(
                        vtk_to_numpy(actual_pd.GetPointData().GetArray('HemisphereLocation')),
                        vtk_to_numpy(pd.GetPointData().GetArray('HemisphereLocation')))
                    np.testing.assert_array_equal(
                        vtk_to_numpy(actual_pd.GetCellData().GetArray('ClusterNumber')),
                        vtk_to_numpy(pd.GetCellData().GetArray('ClusterNumber')))
                    self.assertEqual(actual_pd.GetCellData().GetScalars().GetName(), 'ClusterNumber')
                    self.assertEqual(actual_pd.GetFieldData().GetAbstractArray('SPACE').GetValue(0), 'RAS')

    def test_apply_affine_empty_and_tensor_delegation(self):
        tx = vtk.vtkTransform()
        tx.Translate(2.0, -3.0, 4.0)

        # Empty polydata with 0 points
        empty_with_pts = vtk.vtkPolyData()
        empty_with_pts.SetPoints(vtk.vtkPoints())
        empty_with_pts.SetLines(vtk.vtkCellArray())
        out_empty = wma.transform.apply_affine_to_polydata(empty_with_pts, tx)
        self.assertEqual(out_empty.GetNumberOfPoints(), 0)
        self.assertEqual(out_empty.GetNumberOfLines(), 0)

        # Empty polydata with no vtkPoints object
        empty_no_pts = vtk.vtkPolyData()
        out_empty2 = wma.transform.apply_affine_to_polydata(empty_no_pts, tx)
        self.assertEqual(out_empty2.GetNumberOfPoints(), 0)

        # Delegation when 9-component tensors exist
        tensor_pd = tensor_fixture()
        expected = _vtk_transform_polydata(tensor_pd, tx)
        with patch.object(vtk, 'vtkTransformPolyDataFilter',
                          wraps=vtk.vtkTransformPolyDataFilter) as filter_cls:
            actual = wma.transform.apply_affine_to_polydata(tensor_pd, tx)
        self.assertGreaterEqual(filter_cls.call_count, 1)
        np.testing.assert_array_equal(
            vtk_to_numpy(actual.GetPoints().GetData()),
            vtk_to_numpy(expected.GetPoints().GetData()))

    def test_read_itk_affine_matrix_ras_against_io_and_centered_fixture(self):
        tx = vtk.vtkTransform()
        tx.Translate(12.5, -7.25, 3.75)
        tx.RotateX(11.0)
        tx.RotateY(-19.0)
        tx.RotateZ(27.0)
        tx.Scale(1.03, 0.98, 1.01)

        coords = [np.array([[10.0, -5.0, 2.0], [20.0, 15.0, -8.0]], dtype=np.float64)]
        pd = _build_polydata(coords, point_dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fnames = wma.io.write_transforms_to_itk_format([tx], str(root), ['subj'])
            tfm_path = Path(fnames[0])

            mat_fwd = wma.transform.read_itk_affine_matrix_ras(str(tfm_path), inverse=False)
            mat_inv = wma.transform.read_itk_affine_matrix_ras(str(tfm_path), inverse=True)
            self.assertIsNotNone(mat_fwd)
            self.assertIsNotNone(mat_inv)

            expected_fwd = np.array([[tx.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)])
            tx_inv = tx.GetLinearInverse()
            expected_inv = np.array([[tx_inv.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)])
            np.testing.assert_allclose(mat_fwd, expected_fwd, atol=1e-12, rtol=1e-13)
            np.testing.assert_allclose(mat_inv, expected_inv, atol=1e-12, rtol=1e-13)

            fwd_pd = wma.transform.apply_affine_to_polydata(pd, mat_fwd)
            vtk_fwd_pd = _vtk_transform_polydata(pd, tx)
            np.testing.assert_allclose(
                vtk_to_numpy(fwd_pd.GetPoints().GetData()),
                vtk_to_numpy(vtk_fwd_pd.GetPoints().GetData()),
                atol=1e-12, rtol=1e-13)

            inv_pd = wma.transform.apply_affine_to_polydata(fwd_pd, mat_inv)
            np.testing.assert_allclose(
                vtk_to_numpy(inv_pd.GetPoints().GetData()),
                vtk_to_numpy(pd.GetPoints().GetData()),
                atol=1e-12, rtol=1e-13)

            # Independently constructed ITK .tfm fixture with non-zero FixedParameters center
            A = np.array([[0.9, -0.2, 0.1],
                          [0.25, 1.05, -0.15],
                          [-0.05, 0.12, 0.98]], dtype=np.float64)
            t = np.array([-4.0, 7.5, 2.25], dtype=np.float64)
            c = np.array([10.0, -20.0, 5.0], dtype=np.float64)
            centered_tfm = root / 'centered.tfm'
            centered_tfm.write_text(
                '#Insight Transform File V1.0\n'
                '# Transform 0\n'
                'Transform: MatrixOffsetTransformBase_double_3_3\n'
                'Parameters: ' + ' '.join(repr(float(x)) for x in list(A.ravel()) + list(t)) + '\n'
                'FixedParameters: ' + ' '.join(repr(float(x)) for x in c) + '\n'
            )

            m_inv = wma.transform.read_itk_affine_matrix_ras(str(centered_tfm), inverse=True)
            m_fwd = wma.transform.read_itk_affine_matrix_ras(str(centered_tfm), inverse=False)
            self.assertIsNotNone(m_inv)
            self.assertIsNotNone(m_fwd)

            # Verify exact L(p_lps) = A(p_lps - c) + c + t in RAS
            p_ras = np.array([14.2, -9.5, 22.1], dtype=np.float64)
            C3 = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
            p_lps = C3 * p_ras
            q_lps = A @ (p_lps - c) + c + t
            expected_q_ras = C3 * q_lps

            actual_q_ras = (m_inv @ np.append(p_ras, 1.0))[:3]
            np.testing.assert_allclose(actual_q_ras, expected_q_ras, atol=1e-12, rtol=1e-13)
            recovered_p_ras = (m_fwd @ np.append(actual_q_ras, 1.0))[:3]
            np.testing.assert_allclose(recovered_p_ras, p_ras, atol=1e-12, rtol=1e-13)

            # BSpline .tfm and HDF5 binary files return None
            bspline_tfm = root / 'bspline.tfm'
            bspline_tfm.write_text(
                '#Insight Transform File V1.0\n'
                '# Transform 0\n'
                'Transform: BSplineTransform_double_3_3\n'
                'Parameters: 0 0 0\n'
                'FixedParameters: 4 4 4 -100 -100 -100 50 50 50 1 0 0 0 1 0 0 0 1\n'
            )
            self.assertIsNone(wma.transform.read_itk_affine_matrix_ras(str(bspline_tfm)))

            h5_file = root / 'transform.h5'
            h5_file.write_bytes(b'\x89HDF\r\n\x1a\n\x00\x00\xff')
            self.assertIsNone(wma.transform.read_itk_affine_matrix_ras(str(h5_file)))

            # Singular linear matrix raises explicit ValueError context message
            singular_tfm = root / 'singular.tfm'
            singular_tfm.write_text(
                '#Insight Transform File V1.0\n'
                '# Transform 0\n'
                'Transform: AffineTransform_double_3_3\n'
                'Parameters: 0 0 0 0 0 0 0 0 0 0 0 0\n'
                'FixedParameters: 0 0 0\n'
            )
            with self.assertRaisesRegex(ValueError, 'Singular linear transform matrix'):
                wma.transform.read_itk_affine_matrix_ras(str(singular_tfm), inverse=False)


class DirectSamplingTests(unittest.TestCase):
    def _non_integer_float64_polydata(self):
        # Include short (< 3mm), medium eligible (3..7mm), and long (> 7mm) fibers
        # with non-integer float64 coordinates (10.123456789) to test float32 rounding.
        fibers = []
        for idx, count in enumerate((8, 7, 6, 5, 4, 3, 2, 9)):
            pts = np.zeros((count, 3), dtype=np.float64)
            for p in range(count):
                pts[p] = (10.123456789 + idx * 5.987654321,
                          float(p) + 0.111111111,
                          -3.333333333 + idx * 0.777777777)
            fibers.append(pts)
        return _build_polydata(fibers, point_dtype=np.float64)

    def test_sample_preprocessed_fiber_array_matches_polydata_bit_for_bit(self):
        for dtype in (np.float64, np.float32):
            raw = self._non_integer_float64_polydata()
            if dtype == np.float32:
                pts = numpy_to_vtk(vtk_to_numpy(raw.GetPoints().GetData()).astype(np.float32), deep=True)
                raw.GetPoints().SetData(pts)
            for sample_count in (2, 4, 10):
                with self.subTest(dtype=dtype.__name__, sample_count=sample_count):
                    reg = wma.congeal_to_atlas.SubjectToAtlasRegistration()
                    reg.fiber_length = 3.0
                    reg.fiber_length_max = 7.0
                    reg.points_per_fiber = 15

                    np.random.seed(99)
                    expected_pd = reg._sample_preprocessed_polydata(raw, 'subject', sample_count, 2026)
                    expected_state = np.random.get_state()
                    fa = wma.fibers.FiberArray()
                    fa.convert_from_polydata(expected_pd, reg.points_per_fiber)
                    expected = np.array([fa.fiber_array_r, fa.fiber_array_a, fa.fiber_array_s])

                    np.random.seed(99)
                    actual = reg._sample_preprocessed_fiber_array(raw, 'subject', sample_count, 2026)
                    actual_state = np.random.get_state()

                    self.assertEqual(actual.dtype, np.float64)
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_state[1], expected_state[1])
                    self.assertEqual(actual_state[2:], expected_state[2:])

    def test_sample_preprocessed_fiber_array_fallback_and_empty(self):
        raw = self._non_integer_float64_polydata()
        reg = wma.congeal_to_atlas.SubjectToAtlasRegistration()
        reg.fiber_length = 3.0
        reg.fiber_length_max = 7.0
        reg.points_per_fiber = 15

        expected_pd = reg._sample_preprocessed_polydata(raw, 'atlas', 4, 555)
        fa = wma.fibers.FiberArray()
        fa.convert_from_polydata(expected_pd, reg.points_per_fiber)
        expected = np.array([fa.fiber_array_r, fa.fiber_array_a, fa.fiber_array_s])

        # Force geometry=None in cache to exercise fallback
        cached = reg._preprocessed['atlas']
        reg._preprocessed['atlas'] = (cached[0], cached[1], None)
        actual = reg._sample_preprocessed_fiber_array(raw, 'atlas', 4, 555)
        np.testing.assert_array_equal(actual, expected)

        # Empty polydata
        empty = vtk.vtkPolyData()
        out_empty = reg._sample_preprocessed_fiber_array(empty, 'empty', 4, 555)
        self.assertEqual(out_empty.shape, (3, 0, reg.points_per_fiber))
        self.assertEqual(out_empty.dtype, np.float64)


if __name__ == '__main__':
    unittest.main()
