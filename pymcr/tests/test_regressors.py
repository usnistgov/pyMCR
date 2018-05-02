"""
Testing pymcr.regressors

"""

import numpy as np

from numpy.testing import (assert_equal, assert_array_equal, 
                           assert_allclose)

from pymcr.regressors import OLS, NNLS

import pytest   

@pytest.fixture(params=[OLS, NNLS])
def Methods(request):
    return request.param

def test_instantiation(Methods):
    """ Test correct instantiation of classes """

    regr = Methods()

    assert regr.coef_ is None
    assert regr.X_ is None
    assert regr.residual_ is None
    assert hasattr(regr, 'fit')

def test_dimensionality(Methods):
    """ Test correct dimension of regressor methods """

    A = np.array([[1,1,0],[1,0,1],[0,0,1]])
    x = np.array([1,2,3])
    X = x[:,None]
    B = np.dot(A,X)
    b = np.dot(A,x)

    regr = Methods()
    
    # b = 1D
    regr.fit(A,b)
    assert_equal(regr.X_.ndim, 1)
    assert_equal(regr.X_.shape[0], regr.X_.size)
    assert_equal(regr.X_.shape[0], A.shape[-1])
    assert_array_equal(x.shape, regr.X_.shape)
    assert_array_equal(x.T.shape, regr.coef_.shape)

    # B = 2D
    regr.fit(A,B)
    assert_equal(regr.X_.ndim, 2)
    assert_array_equal(regr.X_.shape, [A.shape[-1], B.shape[-1]])

    assert_array_equal(X.shape, regr.X_.shape)
    assert_array_equal(X.T.shape, regr.coef_.shape)

def test_basic_positive_least_squares(Methods):
    """

    """
    A = np.array([[1,1,0],[1,0,1],[0,0,1]])
    x = np.array([1,2,3])
    X = x[:,None]
    B = np.dot(A,X)
    b = np.dot(A,x)

    regr = Methods()
    
    # b is 1D
    regr.fit(A,b)
    assert_allclose(x, regr.X_)
    
    # b is 1D->2D
    regr.fit(A,b[:,None])
    assert_allclose(X, regr.X_)

    # b = 2D
    regr.fit(A,B)
    assert_allclose(X, regr.X_)

def test_nnls_negative_x():
    """ Test nnls """
    A = np.array([[1,1,0],[1,0,1],[0,0,1]])
    x = np.array([1,-2,3])
    b = np.dot(A,x)

    regr = NNLS()

    # b is 1D
    regr.fit(A,b)
    assert_allclose(True, regr.X_ >= 0)  # Must be non-negative
    assert_allclose(False, regr.X_ == x)  # Must be non-negative
    # assert_allclose(x, regr.X_)

def test_ols_negative_x():
    """ Test nnls """
    A = np.array([[1,1,0],[1,0,1],[0,0,1]])
    x = np.array([1,-2,3])
    b = np.dot(A,x)

    regr = OLS()

    # b is 1D
    regr.fit(A,b)
    assert_allclose(x, regr.X_)
    
    
    

