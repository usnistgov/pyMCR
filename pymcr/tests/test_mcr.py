import numpy as np


from numpy.testing import assert_allclose, assert_equal, assert_array_less

import pytest

from pymcr.mcr import McrAls
from pymcr.metrics import mse
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

@pytest.fixture(scope="function")
def dataset():
    """ Setups dataset """

    M = 21
    N = 21
    P = 101
    n_components = 2

    C_img = np.zeros((M,N,n_components))
    C_img[...,0] = np.dot(np.ones((M,1)),np.linspace(0,1,N)[None,:])
    C_img[...,1] = 1 - C_img[...,0]

    St_known = np.zeros((n_components, P))
    St_known[0,40:60] = 1
    St_known[1,60:80] = 2

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    yield C_known, D_known, St_known
    
def test_mcr_ideal_default(dataset):
    """ Provides C/St_known so optimal should be 1 iteration """

    C_known, D_known, St_known = dataset

    mcrals = McrAls()
    mcrals.fit(D_known, ST=St_known)
    assert_equal(1, mcrals.n_iter_opt)
    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

    mcrals.fit(D_known, C=C_known)
    assert_equal(2, mcrals.n_iter_opt)
    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

def test_mcr_ideal_str_regressors(dataset):
    """ Test MCR """

    C_known, D_known, St_known = dataset

    mcrals = McrAls(c_regr='OLS', st_regr='OLS')
    mcrals.fit(D_known, ST=St_known, verbose=True)
    assert_equal(1, mcrals.n_iter_opt)

    mcrals = McrAls(c_regr='NNLS', st_regr='NNLS')
    mcrals.fit(D_known, ST=St_known)
    assert_equal(1, mcrals.n_iter_opt)

    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

    mcrals = McrAls(c_regr='OLS', st_regr='OLS')
    mcrals.fit(D_known, C=C_known)

    # Turns out some systems get it in 1 iteration, some in 2
    # assert_equal(1, mcrals.n_iter_opt)
    assert_equal(True, mcrals.n_iter_opt<=2)

    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

    # Seeding with a constant of 0.1 for C, actually leads to a bad local
    # minimum; thus, the err_change gets really small with a relatively bad 
    # error. This is not really a test, but it does test out breaking
    # from tol_err_change
    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='OLS', st_regr='OLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10, tol_n_above_min=100)
    mcrals.fit(D_known, C=C_known*0 + 0.1)

    # Seeding with a constant of 0.1 for C, actually leads to a bad local
    # minimum; thus, the err_change gets really small with a relatively bad 
    # error. The tol_err_change is set to None, so it makes it to max_iter.
    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='OLS', st_regr='OLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=None, tol_n_above_min=100)
    mcrals.fit(D_known, C=C_known*0 + 0.1)
    assert_equal(mcrals.n_iter, 50)

def test_mcr_semilearned(dataset):
    """ """

    C_known, D_known, St_known = dataset
    
    M = 21
    N = 21
    P = 101
    n_components = 3

    C_img = np.zeros((M,N,n_components))
    C_img[...,0] = np.dot(np.ones((M,1)),np.linspace(0,1,N)[None,:])
    C_img[...,1] = np.dot(np.linspace(0,1,M)[:, None], np.ones((1,N)))
    C_img[...,2] = 1 - C_img[...,0] - C_img[...,1]
    C_img = C_img / C_img.sum(axis=-1)[:,:,None]

    St_known = np.zeros((n_components, P))
    St_known[0,30:50] = 1
    St_known[1,50:70] = 2
    St_known[1,70:90] = 3
    St_known += 1

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    ST_guess = 1 * St_known
    ST_guess[2, :] = np.random.randn(P)

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10, 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)

    mcrals.fit(D_known, ST=ST_guess, st_fix=[0,1])
    assert_equal(mcrals.ST_[0,:], St_known[0,:])
    assert_equal(mcrals.ST_[1,:], St_known[1,:])
    

def test_mcr_errors():
    
    # Providing both C and S^T estimates
    with pytest.raises(TypeError):
        mcrals = McrAls()
        mcrals.fit(np.random.randn(10,5), C=np.random.randn(10,3),
                   ST=np.random.randn(3,5))

    # Providing no estimates
    with pytest.raises(TypeError):
        mcrals = McrAls()
        mcrals.fit(np.random.randn(10,5))

    # Unknown regression method
    with pytest.raises(ValueError):
        mcrals = McrAls(c_regr='NOTREAL')

    # regression object with no fit method
    with pytest.raises(ValueError):
        mcrals = McrAls(c_regr=print)