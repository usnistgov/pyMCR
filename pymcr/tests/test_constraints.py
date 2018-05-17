"""
Testing pymcr.constraints

"""

import numpy as np

from numpy.testing import assert_allclose

from pymcr.constraints import (ConstraintNonneg, ConstraintNorm, 
                               ConstraintCumsumNonneg, ConstraintZeroEndPoints,
                               ConstraintZeroCumSumEndPoints)

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

def test_norm():

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