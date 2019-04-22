"""
Testing pymcr.constraints

"""

import numpy as np

from numpy.testing import assert_allclose

from pymcr.constraints import (ConstraintNonneg, ConstraintNorm,
                               ConstraintCumsumNonneg, ConstraintZeroEndPoints,
                               ConstraintZeroCumSumEndPoints, ConstraintCompressAbove,
                               ConstraintCompressBelow, ConstraintCutAbove, ConstraintCutBelow,
                               ConstraintReplaceZeros, ConstraintPlanarize)

import pytest

def test_nonneg():
    A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]])
    A_nn = np.array([[1, 2, 3], [0, 0, 0], [1, 2, 3]])

    constr_nn = ConstraintNonneg(copy=True)
    out = constr_nn.transform(A)
    assert_allclose(A_nn, out)

    constr_nn = ConstraintNonneg(copy=False)
    out = constr_nn.transform(A)
    assert_allclose(A_nn, A)

def test_cumsumnonneg():
    """ Cum-Sum Nonnegativity Constraint """
    A = np.array([[2, -2, 3, -2], [-1, -2, -3, 7], [1, -2, -3, 7]])
    A_nn_ax1 = np.array([[2, 0, 3, -2], [0, 0, 0, 7], [1, 0, 0, 7]])
    A_nn_ax0 = np.array([[2, 0, 3, 0], [-1, 0, 0, 7], [1, 0, 0, 7]])

    # Axis -1
    constr_nn = ConstraintCumsumNonneg(copy=True, axis=-1)
    out = constr_nn.transform(A)
    assert_allclose(A_nn_ax1, out)

    # Axis 0
    constr_nn = ConstraintCumsumNonneg(copy=False, axis=0)
    out = constr_nn.transform(A)
    assert_allclose(A_nn_ax0, A)

def test_zeroendpoints():
    """ 0-Endpoints Constraint """
    A = np.array([[1, 2, 3, 4], [3, 6, 9, 12], [4, 8, 12, 16]]).astype(np.float)
    A_ax1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float)
    A_ax0 = np.array([[0, 0, 0, 0], [0.5, 1, 1.5, 2], [0, 0, 0, 0]]).astype(np.float)

    # Axis 0
    constr_ax0 = ConstraintZeroEndPoints(copy=True, axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(A_ax0, out)

    # Axis -1
    constr_ax1 = ConstraintZeroEndPoints(copy=True, axis=-1)
    out = constr_ax1.transform(A)
    assert_allclose(A_ax1, out)

    with pytest.raises(TypeError):
        constr_ax1 = ConstraintZeroEndPoints(copy=True, axis=3)

    # Axis 0 -- NOT copies
    constr_ax0 = ConstraintZeroEndPoints(copy=False, axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(A_ax0, A)

def test_zeroendpoints_span():
    """ 0-Endpoints Constraint """
    A = np.array([[1, 2, 3, 4], [3, 6, 9, 12], [4, 8, 12, 16]]).astype(np.float)

    # Axis 1
    constr_ax1 = ConstraintZeroEndPoints(copy=True, axis=1, span=2)
    out = constr_ax1.transform(A)
    assert_allclose(out[:, [0,1]].mean(axis=1), 0)
    assert_allclose(out[:, [1,2]].mean(axis=1), 0)

    # Axis 0
    constr_ax0 = ConstraintZeroEndPoints(copy=True, axis=0, span=2)
    out = constr_ax0.transform(A)
    assert_allclose(out[[0,1], :].mean(axis=0), 0)
    assert_allclose(out[[1,2], :].mean(axis=0), 0)

    # effective an assert_not_equal
    assert_allclose([q != 0 for q in out[:,0]], True)
    assert_allclose([q != 0 for q in out[:,-1]], True)

    # Axis 1 -- no copy
    constr_ax1 = ConstraintZeroEndPoints(copy=False, axis=1, span=2)
    out = constr_ax1.transform(A)
    assert_allclose(A[:, [0,1]].mean(axis=1), 0)
    assert_allclose(A[:, [1,2]].mean(axis=1), 0)

def test_zerocumsumendpoints():
    """ Cum-Sum 0-Endpoints Constraint """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_diff1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype(np.float)
    A_diff0 = np.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]).astype(np.float)

    # A_ax1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # A_ax0 = np.array([[0, 0, 0], [0.5, 1, 1.5], [0, 0, 0]])

    # Axis 0
    constr_ax0 = ConstraintZeroCumSumEndPoints(copy=True, axis=0)
    out = constr_ax0.transform(A_diff0)
    assert_allclose(out, 0)
    assert_allclose(np.cumsum(out, axis=0), 0)

    # Axis -1
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=True, axis=-1)
    out = constr_ax1.transform(A_diff1)
    assert_allclose(out, 0)
    assert_allclose(np.cumsum(out, axis=1), 0)

    # Axis = -1 -- NOT copy
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=False, axis=-1)
    out = constr_ax1.transform(A_diff1)
    assert_allclose(A_diff1, 0)
    assert_allclose(np.cumsum(A_diff1, axis=1), 0)

def test_zerocumsumendpoints_nodes():
    """ Cum-Sum 0-Endpoints Constraint with defined nodes"""
    
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    A_transform_ax0 = np.array([[0, 0, -1, 5], [0, 0, 0, -4], [0, 0, 1, -1]])
    A_transform_ax1 = np.array([[-3, -2, -3, 8], [-1, 0, 0, 1], [-2, -1, 0, 3]])

    # COPY Axis 0, node=0 (same as endpoints)
    constr_ax0 = ConstraintZeroCumSumEndPoints(copy=True, nodes=[0], axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(out, A_transform_ax0)
    # assert_allclose(np.cumsum(out, axis=0), 0)

    # COPY Axis -1, node=0 (same as endpoints)
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=True, nodes=[0], axis=-1)
    out = constr_ax1.transform(A)
    assert_allclose(out, A_transform_ax1)
    # assert_allclose(np.cumsum(out, axis=1), 0)

    # OVERWRITE, Axis = 0, node=0 (same as endpoints)
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax0 = ConstraintZeroCumSumEndPoints(copy=False, nodes=[0], axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(A, A_transform_ax0)

     # OVERWRITE, Axis = 1, node=0 (same as endpoints)
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=False, nodes=[0], axis=-1)
    out = constr_ax1.transform(A)
    assert_allclose(A, A_transform_ax1)

    # COPY, Axis = 0, Nodes = all
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax0 = ConstraintZeroCumSumEndPoints(copy=True, nodes=[0,1,2], axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(out, 0)

    # COPY, Axis = 1, Nodes = all
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=True, nodes=[0,1,2,3], axis=1)
    out = constr_ax1.transform(A)
    assert_allclose(out, 0)

    # OVERWRITE, Axis = 0, Nodes = all
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax0 = ConstraintZeroCumSumEndPoints(copy=False, nodes=[0,1,2], axis=0)
    out = constr_ax0.transform(A)
    assert_allclose(A, 0)

    # OVERWRITE, Axis = 1, Nodes = all
    A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(np.float)
    constr_ax1 = ConstraintZeroCumSumEndPoints(copy=False, nodes=[0,1,2,3], axis=1)
    out = constr_ax1.transform(A)
    assert_allclose(A, 0)

def test_norm():
    """ Test normalization """
    # A must be dtype.float for in-place math (copy=False)
    constr_norm = ConstraintNorm(axis=0, copy=False)
    A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]]) # dtype: int32
    with pytest.raises(TypeError):
        out = constr_norm.transform(A)

    # Axis must be 0,1, or -1
    with pytest.raises(ValueError):
        constr_norm = ConstraintNorm(axis=2, copy=False)

    A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]], dtype=np.float)
    A_norm0 = A / A.sum(axis=0)[None,:]
    A_norm1 = A / A.sum(axis=1)[:,None]

    constr_norm = ConstraintNorm(axis=0, copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_norm0, out)

    constr_norm = ConstraintNorm(axis=1, copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_norm1, out)

    constr_norm = ConstraintNorm(axis=-1, copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_norm1, out)

    constr_norm = ConstraintNorm(axis=0, copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_norm0, A)

    A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]], dtype=np.float)
    constr_norm = ConstraintNorm(axis=1, copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_norm1, A)

def test_norm_fixed_axes():
    # AXIS = 1
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float)
    A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.3, 0.6, 0.0]], dtype=np.float)

    # Fixed axes must be integers
    with pytest.raises(TypeError):
        constr_norm = ConstraintNorm(axis=1, fix=2.2, copy=True)

    # Dtype must be integer related
    with pytest.raises(TypeError):
        constr_norm = ConstraintNorm(axis=1, fix=np.array([2.2]), copy=True)

    # COPY: True
    # Fix of type int
    constr_norm = ConstraintNorm(axis=1, fix=2, copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type list
    constr_norm = ConstraintNorm(axis=1, fix=[2, 3], copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type tuple
    constr_norm = ConstraintNorm(axis=1, fix=(2), copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type ndarray
    constr_norm = ConstraintNorm(axis=1, fix=np.array([2]), copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # COPY: False
    A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.3, 0.6, 0.0]], dtype=np.float)

    # Fix of type int
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float)
    constr_norm = ConstraintNorm(axis=1, fix=2, copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type list
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float)
    constr_norm = ConstraintNorm(axis=1, fix=[2,3], copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type tuple
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float)
    constr_norm = ConstraintNorm(axis=1, fix=(2), copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type ndarray
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float)
    constr_norm = ConstraintNorm(axis=1, fix=np.array([2]), copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # AXIS = 0
    # Lazy, so just transposed
    A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.3, 0.9, 0.6, 0.0]], dtype=np.float).T
    A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.3, 0.6, 0.0]], dtype=np.float).T
    # COPY: True
    # Fix of type int
    constr_norm = ConstraintNorm(axis=0, fix=2, copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type list
    constr_norm = ConstraintNorm(axis=0, fix=[2,3], copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type tuple
    constr_norm = ConstraintNorm(axis=0, fix=(2), copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # Fix of type ndarray
    constr_norm = ConstraintNorm(axis=0, fix=np.array([2]), copy=True)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, out)

    # COPY: False
    A_fix2_ax1 = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.1, 0.3, 0.6]], dtype=np.float).T

    # Fix of type int
    A = np.array([[0.0, 0.2, 1.0], [0.25, 0.25, 0.0], [0.3, 0.9, 0.6]], dtype=np.float).T
    constr_norm = ConstraintNorm(axis=0, fix=2, copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type list
    A = np.array([[0.0, 0.2, 1.0], [0.25, 0.25, 0.0], [0.3, 0.9, 0.6]], dtype=np.float).T
    constr_norm = ConstraintNorm(axis=0, fix=[2], copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type tuple
    A = np.array([[0.0, 0.2, 1.0], [0.25, 0.25, 0.0], [0.3, 0.9, 0.6]], dtype=np.float).T
    constr_norm = ConstraintNorm(axis=0, fix=(2), copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

    # Fix of type ndarray
    A = np.array([[0.0, 0.2, 1.0], [0.25, 0.25, 0.0], [0.3, 0.9, 0.6]], dtype=np.float).T
    constr_norm = ConstraintNorm(axis=0, fix=np.array([2]), copy=False)
    out = constr_norm.transform(A)
    assert_allclose(A_fix2_ax1, A)

def test_cut_below():
    """ Test cutting below (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform = np.array([[0, 0, 0, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)

    constr = ConstraintCutBelow(copy=True, value=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform)

    # No Copy
    constr = ConstraintCutBelow(copy=False, value=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform)

def test_cut_below_exclude():
    """ Test cutting below (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform_excl_0_ax1 = np.array([[1, 0, 0, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform_excl_0_ax0 = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)

    # COPY
    constr = ConstraintCutBelow(copy=True, value=4, exclude=0, exclude_axis=1)
    out = constr.transform(A)
    assert_allclose(out, A_transform_excl_0_ax1)

    constr = ConstraintCutBelow(copy=True, value=4, exclude=0, exclude_axis=0)
    out = constr.transform(A)
    assert_allclose(out, A_transform_excl_0_ax0)

    # OVERWRITE
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    constr = ConstraintCutBelow(copy=False, value=4, exclude=0, exclude_axis=1)
    out = constr.transform(A)
    assert_allclose(A, A_transform_excl_0_ax1)

    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    constr = ConstraintCutBelow(copy=False, value=4, exclude=0, exclude_axis=0)
    out = constr.transform(A)
    assert_allclose(A, A_transform_excl_0_ax0)

def test_cut_below_nonzerosum():
    """
    Test cutting below (and not equal to) a value with the constrain that
        no columns (axis=-1) will all become 0
    """
    A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
    cutoff = 0.7
    A_correct = np.array([[0.0, 0.7], [0.4, 0.6], [0.5, 0.5]])

    constr = ConstraintCutBelow(copy=True, value=cutoff, axis_sumnz=-1)
    out = constr.transform(A)
    assert_allclose(out, A_correct)

    constr = ConstraintCutBelow(copy=False, value=cutoff, axis_sumnz=-1)
    constr.transform(A)
    assert_allclose(A, A_correct)

def test_cut_below_nonzerosum_exclude():
    """
    Test cutting below (and not equal to) a value with the constrain that
        no columns (axis=-1) will all become 0
    """
    A = np.array([[0.3, 0.7], [0.3, 0.7], [0.7, 0.3]])
    cutoff = 0.7
    A_correct = np.array([[0.3, 0.7], [0.3, 0.7], [0.7, 0.0]])

    # COPY
    constr = ConstraintCutBelow(copy=True, value=cutoff, axis_sumnz=-1,
                                exclude=0, exclude_axis=-1)
    out = constr.transform(A)
    assert_allclose(out, A_correct)

    # OVERWRITE
    constr = ConstraintCutBelow(copy=False, value=cutoff, axis_sumnz=-1,
                                exclude=0, exclude_axis=-1)
    _ = constr.transform(A)
    assert_allclose(A, A_correct)

    # constr = ConstraintCutBelow(copy=False, value=cutoff, axis_sumnz=-1)
    # constr.transform(A)
    # assert_allclose(A, A_correct)

def test_cut_above_nonzerosum():
    """
    Test cutting above (and not equal to) a value with the constrain that
        no columns (axis=-1) will all become 0
    """
    A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
    cutoff = 0.4
    A_correct = np.array([[0.3, 0.0], [0.4, 0.0], [0.5, 0.5]])

    constr = ConstraintCutAbove(copy=True, value=cutoff, axis_sumnz=-1)
    out = constr.transform(A)
    assert_allclose(out, A_correct)

    # NO Copy
    A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
    cutoff = 0.4
    A_correct = np.array([[0.3, 0.0], [0.4, 0.0], [0.5, 0.5]])

    constr = ConstraintCutAbove(copy=False, value=cutoff, axis_sumnz=-1)
    constr.transform(A)
    assert_allclose(A, A_correct)

def test_compress_below():
    """ Test compressing below (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform = np.array([[4, 4, 4, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)

    constr = ConstraintCompressBelow(copy=True, value=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform)

    # No Copy
    constr = ConstraintCompressBelow(copy=False, value=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform)

def test_cut_above():
    """ Test cutting above (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform = np.array([[1, 2, 3, 4], [4, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float)

    constr = ConstraintCutAbove(copy=True, value=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform)

    # No Copy
    constr = ConstraintCutAbove(copy=False, value=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform)

def test_cut_above_exclude():
    """ Test cutting above (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform_excl_0_ax1 = np.array([[1, 2, 3, 4], [4, 0, 0, 0], [7, 0, 0, 0]]).astype(np.float)
    A_transform_excl_2_ax0 = np.array([[1, 2, 3, 4], [4, 0, 0, 0], [7, 8, 9, 10]]).astype(np.float)

    constr = ConstraintCutAbove(copy=True, value=4, exclude=0, exclude_axis=-1)
    out = constr.transform(A)
    assert_allclose(out, A_transform_excl_0_ax1)

    constr = ConstraintCutAbove(copy=True, value=4, exclude=2, exclude_axis=0)
    out = constr.transform(A)
    assert_allclose(out, A_transform_excl_2_ax0)

    # No Copy
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    constr = ConstraintCutAbove(copy=False, value=4, exclude=0, exclude_axis=-1)
    _ = constr.transform(A)
    assert_allclose(A, A_transform_excl_0_ax1)

    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    constr = ConstraintCutAbove(copy=False, value=4, exclude=2, exclude_axis=0)
    _ = constr.transform(A)
    assert_allclose(A, A_transform_excl_2_ax0)
def test_compress_above():
    """ Test compressing above (and not equal to) a value """
    A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(np.float)
    A_transform = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [4, 4, 4, 4]]).astype(np.float)

    constr = ConstraintCompressAbove(copy=True, value=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform)

    # No Copy
    constr = ConstraintCompressAbove(copy=False, value=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform)

def test_replace_zeros():
    """ Test replace zeros using a single feature """
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=0, feature = 0
    A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 1.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=1, feature = 0
    A_transform_ax1 = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [1.0, 0.0, 0.0, 0.0]], dtype=np.float)

    # Axis 0, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=0, feature=0)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax0)

    # Axis 1, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=0)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax1)

    # Axis 0, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=0, feature=0)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax0)

    # Axis 1, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=1, feature=0)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax1)

def test_replace_zeros_multifeature():
    """ Replace zeros using multiple features """

    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=0, feature = [0,1]
    A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 0.5],
                                [0.25, 0.25, 0.0, 0.5],
                                [0.3, 0.9, 0.6, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=1, feature = [0,1]
    A_transform_ax1 = np.array([[0.5, 0.5, 0.0, 0.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [0.5, 0.5, 0.0, 0.0]], dtype=np.float)

    # Axis 0, copy=True, feature=[0,1]
    constr = ConstraintReplaceZeros(copy=True, axis=0, feature=[0,1])
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax0)

    # Axis 1, copy=True, feature=[0,1]
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=[0,1])
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax1)

    # Axis 1, copy=True, feature=np.array([0,1])
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=np.array([0,1]))
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax1)

    # Axis 1, copy=True, feature=None
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=None)
    out = constr.transform(A)
    assert_allclose(out, A)

    # Axis 0, copy=FALSE, feature=[0,1]
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=0, feature=[0,1])
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax0)

    # Axis 1, copy=FALSE, feature=[0,1]
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=1, feature=[0,1])
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax1)

def test_replace_zeros_non1fval():
    """ Test replace zeros using a single feature with fval != 1 """
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=0, feature = 0
    A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 4.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=1, feature = 0
    A_transform_ax1 = np.array([[4.0, 0.0, 0.0, 0.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [4.0, 0.0, 0.0, 0.0]], dtype=np.float)

    # Axis 0, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=0, feature=0, fval=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax0)

    # Axis 1, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=0, fval=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax1)

    # Axis 0, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=0, feature=0, fval=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax0)

    # Axis 1, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=1, feature=0, fval=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax1)

def test_replace_zeros_non1fval_multifeature():
    """ Test replace zeros using a 2 features with fval != 1 """
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=0, feature = 0
    A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 2.0],
                                [0.25, 0.25, 0.0, 2.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=np.float)
    # Axis=1, feature = 0
    A_transform_ax1 = np.array([[2.0, 2.0, 0.0, 0.0],
                                [0.25, 0.25, 0.0, 0.0],
                                [0.3, 0.9, 0.6, 0.0],
                                [2.0, 2.0, 0.0, 0.0]], dtype=np.float)

    # Axis 0, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=0, feature=[0,1], fval=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax0)

    # Axis 1, copy=True
    constr = ConstraintReplaceZeros(copy=True, axis=1, feature=[0,1], fval=4)
    out = constr.transform(A)
    assert_allclose(out, A_transform_ax1)

    # Axis 0, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=0, feature=[0,1], fval=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax0)

    # Axis 1, copy=FALSE
    A = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0, 0.0],
                  [0.3, 0.9, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float)

    constr = ConstraintReplaceZeros(copy=False, axis=1, feature=[0,1], fval=4)
    out = constr.transform(A)
    assert_allclose(A, A_transform_ax1)

def test_planarize_no_noise():
    """ Test ConstraintPlanarize with no noise """
    C_img = np.zeros((10, 20, 2))  # Y, X, Target
    x = np.arange(C_img.shape[1])
    y = np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = np.meshgrid(x,y)

    C_0_ideal = 0.1*X + 0.3*Y + 3
    C_img[:,:,0] = C_0_ideal
    C_img[:,:,1] = 0.2*X + 0.4*Y + 3

    C_ravel = C_img.reshape((-1, n_targets))

    # COPY
    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True)
    out = constr.transform(C_ravel)
    out_img = out.reshape(C_img.shape)

    assert_allclose(out_img[..., 1], C_img[..., 1])
    # Data is a plane. Should return the same plane to within numerical error
    assert_allclose(out_img[...,0], C_0_ideal)

    # OVERWRITE
    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=False)
    _ = constr.transform(C_ravel)
    assert_allclose(C_ravel.reshape(C_img.shape)[..., 0], C_img[..., 0])
    assert_allclose(C_ravel.reshape(C_img.shape)[..., 1], C_img[..., 1])


def test_planarize_noisy():
    """ Test ConstraintPlanarize """
    C_img = np.zeros((10, 20, 2))  # Y, X, Target
    x = np.arange(C_img.shape[1])
    y = np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = np.meshgrid(x,y)

    C_img[:,:,0] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10,20)
    C_img[:,:,1] = 0.1*X + 0.3*Y + 3

    C_ravel = C_img.reshape((-1, n_targets))

    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True)
    out = constr.transform(C_ravel)
    out_img = out.reshape(C_img.shape)

    assert_allclose(out_img[..., 1], C_img[..., 1])

    assert np.sum((out_img[..., 0] - C_img[..., 1])**2) < np.sum((C_img[...,0] - C_img[..., 1])**2)

def test_planarize_noisy_list_target():
    """ Test ConstraintPlanarize """
    C_img = np.zeros((10, 20, 2))  # Y, X, Target
    x = np.arange(C_img.shape[1])
    y = np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = np.meshgrid(x,y)

    C_img[:,:,0] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10,20)
    C_img[:,:,1] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10,20)
    C_ideal_01 = 0.1*X + 0.3*Y + 3

    C_ravel = C_img.reshape((-1, n_targets))

    constr = ConstraintPlanarize([0, 1], (10, 20), scaler=None, copy=True)
    out = constr.transform(C_ravel)
    out_img = out.reshape(C_img.shape)

    # Test that RSS is better after planarize than before
    assert np.sum((C_img[...,0] - C_ideal_01)**2) > np.sum((out_img[...,0] - C_ideal_01)**2)
    assert np.sum((C_img[...,1] - C_ideal_01)**2) > np.sum((out_img[...,1] - C_ideal_01)**2)

    # Two output targets are not identical
    assert np.sum((out_img[..., 0] - out_img[..., 1])**2) > 0

def test_planarize_err_type_input():
    """ Inputting a target that is not a list, tuple, or ndarray results in Type Error """
    with pytest.raises(TypeError):
        constr = ConstraintPlanarize({'a' : 0}, (10, 20), scaler=None, copy=True)

def test_planarize_set_scaler():
    """ Test setting or not setting scaler """

    C_img = np.zeros((10, 20, 2))  # Y, X, Target
    x = np.arange(C_img.shape[1])
    y = np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = np.meshgrid(x,y)

    C_img[:,:,0] = 0.1*X + 0.3*Y - 2.5
    C_img[:,:,1] = 0.1*X + 0.3*Y - 2.5

    C_ravel = C_img.reshape((-1, n_targets))

    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True)
    out = constr.transform(C_ravel)

    assert constr.scaler is not None
    assert constr.scaler > 100

    constr = ConstraintPlanarize(0, (10, 20), scaler=1.0, copy=True)
    out = constr.transform(C_ravel)

    assert constr.scaler is not None
    assert constr.scaler == 1.0




def test_planarize_use_above_and_below_on_plane():
    """ Test ConstraintPlanarize """
    C_img = np.zeros((10, 20, 2))  # Y, X, Target
    x = np.arange(C_img.shape[1])
    y = np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = np.meshgrid(x,y)

    C_img[:,:,0] = 0.1*X + 0.3*Y - 2.5
    C_img[:,:,1] = 0.1*X + 0.3*Y - 2.5

    C_ravel = C_img.reshape((-1, n_targets))

    # COPY -- DO NOT Apply limits to plane
    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True, use_vals_above=0,
                                 use_vals_below=1, lims_to_plane=False)
    out = constr.transform(C_ravel)

    assert C_ravel.min() < 0
    assert C_ravel.max() > 1
    assert out[:,0].min() < 0
    assert out[:,0].max() > 1
    assert out[:,1].min() < 0
    assert out[:,1].max() > 1

    # COPY -- Apply limits to plane
    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True, use_vals_above=0,
                                 use_vals_below=1, lims_to_plane=True)
    out = constr.transform(C_ravel)

    assert C_ravel.min() < 0
    assert C_ravel.max() > 1
    assert out[:,0].min() >= 0
    assert out[:,0].max() <= 1
    assert out[:,1].min() < 0
    assert out[:,1].max() > 1

    # OVERWRITE
    assert C_ravel[:,0].min() < 0
    assert C_ravel[:,0].max() > 1
    assert C_ravel[:,1].min() < 0
    assert C_ravel[:,1].max() > 1


    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=False, use_vals_above=0,
                                 use_vals_below=1, lims_to_plane=True)
    out = constr.transform(C_ravel)

    assert C_ravel[:,0].min() >= 0
    assert C_ravel[:,0].max() <= 1
    assert C_ravel[:,1].min() < 0
    assert C_ravel[:,1].max() > 1
