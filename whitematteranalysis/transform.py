# -*- coding: utf-8 -*-
"""Vectorized affine transforms and ITK transform parsing for tractography."""

from pathlib import Path
import numpy
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def has_tensor_arrays(polydata):
    """Return True if polydata contains tensor, vector, normal, or 9-component arrays."""
    if polydata is None:
        return False
    for data in (polydata.GetPointData(), polydata.GetCellData()):
        if data is None:
            continue
        if (data.GetTensors() is not None
                or data.GetVectors() is not None
                or data.GetNormals() is not None):
            return True
        for idx in range(data.GetNumberOfArrays()):
            arr = data.GetAbstractArray(idx)
            if (arr is not None
                    and isinstance(arr, vtk.vtkDataArray)
                    and arr.GetNumberOfComponents() == 9):
                return True
    return False


def read_itk_affine_matrix_ras(tfm_path, inverse=False):
    """Parse a linear ITK .tfm file into a 4x4 RAS homogeneous matrix.

    ITK stores linear transforms in LPS as the inverse (resampling) transform:
    ``L(p) = A @ (p - c) + c + t = A @ p + (t + c - A @ c)`` where
    ``L = C @ T_fwd^{-1} @ C`` and ``C = diag(-1, -1, 1, 1)``.

    Returns:
        numpy.ndarray | None:
            - ``inverse=True`` returns ``C @ L @ C`` (atlas -> subject, matching
              ``wm_harden_transform.py -i``).
            - ``inverse=False`` returns ``(C @ L @ C)^{-1}`` (subject -> atlas).
            - ``None`` for non-affine, BSpline, multi-transform, or binary/HDF5 files.
    """
    path = Path(tfm_path)
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    if '\x00' in text:
        return None

    transform_types = []
    params = None
    fixed_params = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == 'Transform':
            transform_types.append(value)
        elif key == 'Parameters':
            try:
                params = [float(tok) for tok in value.split()]
            except ValueError:
                return None
        elif key == 'FixedParameters':
            try:
                fixed_params = [float(tok) for tok in value.split()] if value else []
            except ValueError:
                return None

    if len(transform_types) != 1 or params is None or len(params) != 12:
        return None
    tx_type = transform_types[0]
    if not (tx_type.startswith('AffineTransform_')
            or tx_type.startswith('MatrixOffsetTransformBase_')):
        return None
    if fixed_params is None or len(fixed_params) == 0:
        center = numpy.zeros(3, dtype=numpy.float64)
    elif len(fixed_params) == 3:
        center = numpy.asarray(fixed_params, dtype=numpy.float64)
    else:
        return None

    A = numpy.asarray(params[:9], dtype=numpy.float64).reshape(3, 3)
    t = numpy.asarray(params[9:12], dtype=numpy.float64)
    offset = t + center - A @ center

    L = numpy.eye(4, dtype=numpy.float64)
    L[:3, :3] = A
    L[:3, 3] = offset

    C = numpy.diag([-1.0, -1.0, 1.0, 1.0])
    ras_inv = C @ L @ C
    if inverse:
        return ras_inv
    try:
        return numpy.linalg.inv(ras_inv)
    except numpy.linalg.LinAlgError as exc:
        raise ValueError(f"Singular linear transform matrix in {tfm_path}") from exc


def _extract_linear_matrix(transform_or_matrix):
    """Return a (4, 4) float64 matrix for linear transforms, or None for non-linear VTK transforms."""
    if isinstance(transform_or_matrix, vtk.vtkMatrix4x4):
        m = transform_or_matrix
        return numpy.array(
            [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=numpy.float64)
    if isinstance(transform_or_matrix, vtk.vtkLinearTransform):
        m = transform_or_matrix.GetMatrix()
        return numpy.array(
            [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=numpy.float64)
    if isinstance(transform_or_matrix, vtk.vtkAbstractTransform):
        return None
    mat = numpy.asarray(transform_or_matrix, dtype=numpy.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got shape {mat.shape}")
    return mat


def _to_vtk_transform(transform_or_matrix):
    if isinstance(transform_or_matrix, vtk.vtkAbstractTransform):
        return transform_or_matrix
    vtk_mat = vtk.vtkMatrix4x4()
    if isinstance(transform_or_matrix, vtk.vtkMatrix4x4):
        vtk_mat.DeepCopy(transform_or_matrix)
    else:
        mat = numpy.asarray(transform_or_matrix, dtype=numpy.float64)
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, float(mat[i, j]))
    tx = vtk.vtkTransform()
    tx.SetMatrix(vtk_mat)
    return tx


def apply_affine_to_polydata(polydata, transform_or_matrix, chunk_points=262144):
    """Apply an affine transform to polydata points in bounded float64 chunks.

    Delegates to ``vtkTransformPolyDataFilter`` whenever ``has_tensor_arrays(polydata)``
    is ``True`` or the transform is non-linear.
    """
    if polydata is None:
        raise ValueError("polydata must not be None")

    matrix = _extract_linear_matrix(transform_or_matrix)
    if matrix is None or has_tensor_arrays(polydata):
        transformer = vtk.vtkTransformPolyDataFilter()
        if vtk.vtkVersion().GetVTKMajorVersion() >= 6.0:
            transformer.SetInputData(polydata)
        else:
            transformer.SetInput(polydata)
        transformer.SetTransform(_to_vtk_transform(transform_or_matrix))
        transformer.Update()
        return transformer.GetOutput()

    out = vtk.vtkPolyData()
    out.ShallowCopy(polydata)

    in_pts = polydata.GetPoints()
    n_pts = in_pts.GetNumberOfPoints() if in_pts is not None else 0
    if in_pts is None or n_pts == 0:
        if in_pts is not None:
            empty_pts = vtk.vtkPoints()
            empty_pts.SetDataType(in_pts.GetDataType())
            empty_pts.SetNumberOfPoints(0)
            out.SetPoints(empty_pts)
        return out

    in_data = in_pts.GetData()
    in_arr = vtk_to_numpy(in_data)
    out_data = vtk.vtkDataArray.CreateDataArray(in_data.GetDataType())
    out_data.SetNumberOfComponents(3)
    out_data.SetNumberOfTuples(n_pts)
    if in_data.GetName():
        out_data.SetName(in_data.GetName())
    out_arr = vtk_to_numpy(out_data)
    out_dtype = out_arr.dtype

    m00, m01, m02, m03 = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3])
    m10, m11, m12, m13 = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3])
    m20, m21, m22, m23 = float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3])
    projective = bool(
        matrix[3, 0] != 0.0 or matrix[3, 1] != 0.0
        or matrix[3, 2] != 0.0 or matrix[3, 3] != 1.0
    )
    m30, m31, m32, m33 = float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3])

    step = max(1, int(chunk_points))
    for first in range(0, n_pts, step):
        last = min(first + step, n_pts)
        pts = numpy.asarray(in_arr[first:last], dtype=numpy.float64)
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        ox = px * m00 + py * m01 + pz * m02 + m03
        oy = px * m10 + py * m11 + pz * m12 + m13
        oz = px * m20 + py * m21 + pz * m22 + m23
        if projective:
            ow = px * m30 + py * m31 + pz * m32 + m33
            ox = ox / ow
            oy = oy / ow
            oz = oz / ow
        out_arr[first:last] = numpy.column_stack((ox, oy, oz)).astype(out_dtype, copy=False)

    out_data.Modified()
    new_pts = vtk.vtkPoints()
    new_pts.SetData(out_data)
    out.SetPoints(new_pts)
    return out
