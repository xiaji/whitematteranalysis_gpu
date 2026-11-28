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


def _parse_itk_text_blocks(text):
    """Parse ITK text transform blocks into [(tx_type, params_list, fixed_params_list), ...]."""
    blocks = []
    cur_type = None
    cur_params = None
    cur_fixed = None

    def _flush():
        nonlocal cur_type, cur_params, cur_fixed
        if cur_type is not None:
            blocks.append((cur_type, cur_params or [], cur_fixed or []))
        cur_type = None
        cur_params = None
        cur_fixed = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == 'Transform':
            _flush()
            cur_type = value
        elif key == 'Parameters':
            try:
                cur_params = [float(tok) for tok in value.split()] if value else []
            except ValueError:
                return None
        elif key == 'FixedParameters':
            try:
                cur_fixed = [float(tok) for tok in value.split()] if value else []
            except ValueError:
                return None
    _flush()
    return blocks or None


def _parse_itk_h5_blocks(tfm_path):
    """Parse ITK HDF5 (.h5) transform groups if h5py is available, else return None."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(str(tfm_path), 'r') as h5f:
            if 'TransformGroup' not in h5f:
                return None
            grp = h5f['TransformGroup']
            blocks = []
            for key in sorted(grp.keys(), key=lambda k: int(k)):
                sub = grp[key]
                raw_type = sub['TransformType'][()]
                if isinstance(raw_type, numpy.ndarray):
                    raw_type = raw_type.flat[0]
                if isinstance(raw_type, (bytes, numpy.bytes_)):
                    tx_type = raw_type.decode('utf-8', errors='replace')
                else:
                    tx_type = str(raw_type)
                tx_type = tx_type.rstrip('\x00').strip()
                p_key = 'TransformParameters' if 'TransformParameters' in sub else (
                    'TranformParameters' if 'TranformParameters' in sub else None)
                f_key = 'TransformFixedParameters' if 'TransformFixedParameters' in sub else (
                    'TranformFixedParameters' if 'TranformFixedParameters' in sub else None)
                params = numpy.asarray(sub[p_key][()], dtype=numpy.float64).ravel().tolist() if p_key else []
                fixed = numpy.asarray(sub[f_key][()], dtype=numpy.float64).ravel().tolist() if f_key else []
                blocks.append((tx_type, params, fixed))
            return blocks or None
    except Exception:
        return None


def _lps_matrix_from_linear_block(tx_type, params, fixed_params):
    """Return 4x4 LPS matrix for an ITK AffineTransform / MatrixOffsetTransformBase block."""
    if not (tx_type.startswith('AffineTransform_')
            or tx_type.startswith('MatrixOffsetTransformBase_')):
        return None
    if len(params) != 12:
        return None
    if len(fixed_params) == 0:
        center = numpy.zeros(3, dtype=numpy.float64)
    elif len(fixed_params) == 3:
        center = numpy.asarray(fixed_params, dtype=numpy.float64)
    else:
        return None
    A = numpy.asarray(params[:9], dtype=numpy.float64).reshape(3, 3)
    t = numpy.asarray(params[9:12], dtype=numpy.float64)
    L = numpy.eye(4, dtype=numpy.float64)
    L[:3, :3] = A
    L[:3, 3] = t + center - A @ center
    return L


def _vtk_bspline_from_lps_block(tx_type, params, fixed_params):
    """Return vtkBSplineTransform in LPS from an ITK BSplineTransform_*_3_3 block, or None."""
    if not (tx_type.startswith('BSplineTransform_') and tx_type.endswith('_3_3')):
        return None
    if len(fixed_params) != 18:
        return None
    dims = []
    for v in fixed_params[:3]:
        iv = int(round(v))
        if abs(v - iv) > 1e-6 or iv < 4:
            return None
        dims.append(iv)
    nx, ny, nz = dims
    origin = tuple(float(v) for v in fixed_params[3:6])
    spacing = tuple(float(v) for v in fixed_params[6:9])
    if any(s <= 0.0 for s in spacing):
        return None
    direction = numpy.asarray(fixed_params[9:18], dtype=numpy.float64).reshape(3, 3)
    if not numpy.allclose(direction, numpy.eye(3, dtype=numpy.float64), atol=1e-6, rtol=0.0):
        return None
    n_grid = nx * ny * nz
    if len(params) != 3 * n_grid:
        return None

    from vtk.util.numpy_support import numpy_to_vtk
    coeffs = numpy.ascontiguousarray(
        numpy.asarray(params, dtype=numpy.float64).reshape(3, n_grid).T)
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    scalars = numpy_to_vtk(coeffs, deep=True)
    scalars.SetName('DisplacementField')
    image.GetPointData().SetScalars(scalars)

    bspline = vtk.vtkBSplineTransform()
    bspline.SetBorderModeToZero()
    if hasattr(bspline, 'SetCoefficientData'):
        bspline.SetCoefficientData(image)
    else:
        bspline.SetCoefficients(image)
    bspline._coefficients_image = image
    return bspline


def _matrix_to_vtk_transform(matrix):
    vtk_mat = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_mat.SetElement(i, j, float(matrix[i, j]))
    tx = vtk.vtkTransform()
    tx.SetMatrix(vtk_mat)
    return tx


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

    blocks = _parse_itk_text_blocks(text)
    if blocks is None or len(blocks) != 1:
        return None
    L = _lps_matrix_from_linear_block(*blocks[0])
    if L is None:
        return None

    C = numpy.diag([-1.0, -1.0, 1.0, 1.0])
    ras_inv = C @ L @ C
    if inverse:
        return ras_inv
    try:
        return numpy.linalg.inv(ras_inv)
    except numpy.linalg.LinAlgError as exc:
        raise ValueError(f"Singular linear transform matrix in {tfm_path}") from exc


def read_itk_transform_ras(tfm_path, inverse=False):
    """Parse an ITK linear, BSpline, or Composite transform (.tfm/.h5) into RAS space.

    Returns:
        numpy.ndarray | vtk.vtkGeneralTransform | None:
            - ``(4, 4)`` float64 matrix when the transform is purely linear.
            - ``vtk.vtkGeneralTransform`` when a valid 3D cubic BSpline component
              is present (already conjugated from LPS to RAS).
            - ``None`` when the file is unreadable or contains unsupported types.
    """
    path = Path(tfm_path)
    blocks = None
    if path.suffix.lower() in ('.h5', '.hdf5'):
        blocks = _parse_itk_h5_blocks(path)
    if blocks is None:
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return _parse_itk_h5_blocks(path) if path.suffix.lower() not in ('.h5', '.hdf5') else None
        if '\x00' in text:
            return None
        blocks = _parse_itk_text_blocks(text)
    if not blocks:
        return None

    if blocks[0][0].startswith('CompositeTransform_'):
        child_blocks = blocks[1:]
        if not child_blocks:
            return None
    else:
        child_blocks = blocks

    parsed_lps = []
    all_linear = True
    for tx_type, params, fixed_params in child_blocks:
        mat_lps = _lps_matrix_from_linear_block(tx_type, params, fixed_params)
        if mat_lps is not None:
            parsed_lps.append(('linear', mat_lps))
            continue
        bspline_lps = _vtk_bspline_from_lps_block(tx_type, params, fixed_params)
        if bspline_lps is not None:
            all_linear = False
            parsed_lps.append(('bspline', bspline_lps))
            continue
        return None

    C = numpy.diag([-1.0, -1.0, 1.0, 1.0])
    if all_linear:
        # ITK composite applies child_blocks[-1] first: L_comp = L_0 @ L_1 @ ...
        L_comp = numpy.eye(4, dtype=numpy.float64)
        for _, mat_lps in reversed(parsed_lps):
            L_comp = mat_lps @ L_comp
        ras_inv = C @ L_comp @ C
        if inverse:
            return ras_inv
        try:
            return numpy.linalg.inv(ras_inv)
        except numpy.linalg.LinAlgError as exc:
            raise ValueError(f"Singular linear transform matrix in {tfm_path}") from exc

    ras_to_lps = vtk.vtkTransform()
    ras_to_lps.Scale(-1.0, -1.0, 1.0)
    lps_to_ras = vtk.vtkTransform()
    lps_to_ras.Scale(-1.0, -1.0, 1.0)

    gt = vtk.vtkGeneralTransform()
    gt.Concatenate(lps_to_ras)
    if inverse:
        # PreMultiply evaluates last-concatenated first: p -> T_0(T_1(...T_k(p)))
        for kind, obj in parsed_lps:
            gt.Concatenate(_matrix_to_vtk_transform(obj) if kind == 'linear' else obj)
    else:
        # Inverse of T_0 o ... o T_k is T_k^{-1} o ... o T_0^{-1}
        for kind, obj in reversed(parsed_lps):
            if kind == 'linear':
                try:
                    inv_mat = numpy.linalg.inv(obj)
                except numpy.linalg.LinAlgError as exc:
                    raise ValueError(f"Singular linear transform matrix in {tfm_path}") from exc
                gt.Concatenate(_matrix_to_vtk_transform(inv_mat))
            else:
                inv_bs = vtk.vtkBSplineTransform()
                inv_bs.DeepCopy(obj)
                inv_bs.Inverse()
                gt.Concatenate(inv_bs)
    gt.Concatenate(ras_to_lps)
    return gt


def compose_ras_transforms(transforms):
    """Compose RAS transforms given in application order [first, second, ...]."""
    active = [t for t in transforms if t is not None]
    if not active:
        return numpy.eye(4, dtype=numpy.float64)
    matrices = [_extract_linear_matrix(t) for t in active]
    if all(m is not None for m in matrices):
        combined = numpy.eye(4, dtype=numpy.float64)
        for mat in matrices:
            combined = mat @ combined
        return combined
    gt = vtk.vtkGeneralTransform()
    for tx in reversed(active):
        gt.Concatenate(_to_vtk_transform(tx))
    return gt


def apply_transform_to_polydata(polydata, transform_or_matrix, chunk_points=262144):
    """Apply a linear or VTK non-linear transform to polydata without Slicer.

    Linear transforms delegate to :func:`apply_affine_to_polydata`. Non-linear
    transforms on tensor-free polydatas shallow-copy cell/point attributes and
    call ``TransformPoints`` directly; tensor-bearing polydatas fall back to
    ``vtkTransformPolyDataFilter``.
    """
    if polydata is None:
        raise ValueError("polydata must not be None")
    matrix = _extract_linear_matrix(transform_or_matrix)
    if matrix is not None:
        return apply_affine_to_polydata(polydata, matrix, chunk_points=chunk_points)
    if has_tensor_arrays(polydata) or not isinstance(transform_or_matrix, vtk.vtkAbstractTransform):
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
    out_pts = vtk.vtkPoints()
    if in_pts is not None:
        out_pts.SetDataType(in_pts.GetDataType())
    if in_pts is None or n_pts == 0:
        if in_pts is not None:
            out_pts.SetNumberOfPoints(0)
            out.SetPoints(out_pts)
        return out

    transform_or_matrix.TransformPoints(in_pts, out_pts)
    in_name = in_pts.GetData().GetName()
    if in_name:
        out_pts.GetData().SetName(in_name)
    out.SetPoints(out_pts)
    return out


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
