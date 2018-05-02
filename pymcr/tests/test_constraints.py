"""
Testing pymcr.constraints

"""

import numpy as np

from numpy.testing import assert_allclose

from pymcr.constraints import ConstraintNonneg, ConstraintNorm

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