import numpy as np

from numpy.testing import assert_allclose, assert_equal, assert_array_less, assert_raises

from pymcr.mcr import McrAls
from pymcr.metrics import mse
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

def test_mcr():
    M = 21
    N = 21
    P = 101
    n_components = 2

    C_img = np.zeros((M,N,n_components))
    C_img[...,0] = np.dot(np.ones((M,1)),np.linspace(0,1,N)[None,:])
    C_img[...,1] = 1 - C_img[...,0]

    ST_known = np.zeros((n_components, P))
    ST_known[0,40:60] = 1
    ST_known[1,60:80] = 2

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, ST_known)

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10, 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)
    mcrals._saveall_st = False
    mcrals._saveall_c = False
    mcrals.fit(D_known, ST=ST_known)

    assert_equal(1, mcrals.n_iter_opt)

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='OLS', st_regr='OLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)
    mcrals.fit(D_known, ST=ST_known)
    assert_equal(1, mcrals.n_iter_opt)
    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='NNLS', st_regr='NNLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)
    mcrals.fit(D_known, ST=ST_known)
    assert_equal(1, mcrals.n_iter_opt)

    assert ((mcrals.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrals.D_opt_ - D_known)**2).mean() < 1e-10

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='OLS', st_regr='OLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)
    mcrals.fit(D_known, C=C_known)
    assert_equal(1, mcrals.n_iter_opt)

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
                    tol_err_change=1e-10)
    mcrals.fit(D_known, C=C_known*0 + 0.1)

    # Seeding with a constant of 0.1 for C, actually leads to a bad local
    # minimum; thus, the err_change gets really small with a relatively bad 
    # error. This is not really a test, but it does test out breaking
    # from tol_err_change
    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='OLS', st_regr='OLS', 
                    st_constraints=[ConstraintNonneg()], 
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=None)
    mcrals.fit(D_known, C=C_known*0 + 0.1)
    assert_equal(mcrals.n_iter, 50)

def test_mcr_errors():
    
    # Providing both C and S^T estimates
    with assert_raises(TypeError):
        mcrals = McrAls()
        mcrals.fit(np.random.randn(10,5), C=np.random.randn(10,3),
                   ST=np.random.randn(3,5))

    # Providing no estimates
    with assert_raises(TypeError):
        mcrals = McrAls()
        mcrals.fit(np.random.randn(10,5))

    # Unknown regression method
    with assert_raises(ValueError):
        mcrals = McrAls(c_regr='NOTREAL')

    # regression object with no fit method
    with assert_raises(ValueError):
        mcrals = McrAls(c_regr=print)