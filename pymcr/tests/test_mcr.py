import numpy as np


from numpy.testing import assert_allclose, assert_equal, assert_array_less

import pytest

import pymcr
from pymcr.mcr import McrAR
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

    mcrar = McrAR()
    mcrar.fit(D_known, ST=St_known)
    assert_equal(1, mcrar.n_iter_opt)
    assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

    mcrar.fit(D_known, C=C_known)
    assert_equal(2, mcrar.n_iter_opt)
    assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

def test_mcr_ideal_str_regressors(dataset):
    """ Test MCR with string-provded regressors"""

    C_known, D_known, St_known = dataset

    mcrar = McrAR(c_regr='OLS', st_regr='OLS')
    mcrar.fit(D_known, ST=St_known, verbose=True)
    assert_equal(1, mcrar.n_iter_opt)
    assert isinstance(mcrar.c_regressor, pymcr.regressors.OLS)
    assert isinstance(mcrar.st_regressor, pymcr.regressors.OLS)

    mcrar = McrAR(c_regr='NNLS', st_regr='NNLS')
    mcrar.fit(D_known, ST=St_known)
    assert_equal(1, mcrar.n_iter_opt)
    assert isinstance(mcrar.c_regressor, pymcr.regressors.NNLS)
    assert isinstance(mcrar.st_regressor, pymcr.regressors.NNLS)
    assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

    # Provided C_known this time
    mcrar = McrAR(c_regr='OLS', st_regr='OLS')
    mcrar.fit(D_known, C=C_known)

    # Turns out some systems get it in 1 iteration, some in 2
    # assert_equal(1, mcrar.n_iter_opt)
    assert_equal(True, mcrar.n_iter_opt<=2)

    assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

def test_mcr_max_iterations(dataset):
    """ Test MCR exits at max_iter"""

    C_known, D_known, St_known = dataset

    # Seeding with a constant of 0.1 for C, actually leads to a bad local
    # minimum; thus, the err_change gets really small with a relatively bad
    # error. The tol_err_change is set to None, so it makes it to max_iter.
    mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_increase=None, tol_n_increase=None,
                    tol_err_change=None, tol_n_above_min=None)
    mcrar.fit(D_known, C=C_known*0 + 0.1)
    assert mcrar.exit_max_iter_reached

def test_mcr_tol_increase(dataset):
    """ Test MCR exits due error increasing above a tolerance fraction"""

    C_known, D_known, St_known = dataset

    # Seeding with a constant of 0.1 for C, actually leads to a bad local
    # minimum; thus, the err_change gets really small with a relatively bad
    # error.
    mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_increase=0, tol_n_increase=None,
                    tol_err_change=None, tol_n_above_min=None)
    mcrar.fit(D_known, C=C_known*0 + 0.1)
    assert mcrar.exit_tol_increase

def test_mcr_tol_n_increase(dataset):
    """
    Test MCR exits due iterating n times with an increase in error

    Note: On some CI systems, the minimum err bottoms out; thus, tol_n_above_min
    needed to be set to 0 to trigger a break.
    """

    C_known, D_known, St_known = dataset

    mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_increase=None, tol_n_increase=0,
                    tol_err_change=None, tol_n_above_min=None)
    mcrar.fit(D_known, C=C_known*0 + 0.01)
    assert mcrar.exit_tol_n_increase

def test_mcr_tol_err_change(dataset):
    """ Test MCR exits due error increasing by a value """

    C_known, D_known, St_known = dataset

    mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_increase=None, tol_n_increase=None,
                    tol_err_change=1e-20, tol_n_above_min=None)
    mcrar.fit(D_known, C=C_known)
    assert mcrar.exit_tol_err_change

def test_mcr_tol_n_above_min(dataset):
    """
    Test MCR exits due to half-terating n times with error above the minimum error.

    Note: On some CI systems, the minimum err bottoms out; thus, tol_n_above_min
    needed to be set to 0 to trigger a break.
    """

    C_known, D_known, St_known = dataset

    mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_increase=None, tol_n_increase=None,
                    tol_err_change=None, tol_n_above_min=0)
    mcrar.fit(D_known, C=C_known*0 + 0.1)
    assert mcrar.exit_tol_n_above_min


def test_mcr_st_semilearned():
    """ Test when St items are fixed, i.e., enforced to be the same as the input, always """

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
    St_known[2,70:90] = 3
    St_known += 1

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    ST_guess = 1 * St_known
    ST_guess[2, :] = np.random.randn(P)

    mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)

    mcrar.fit(D_known, ST=ST_guess, st_fix=[0,1])
    assert_equal(mcrar.ST_[0,:], St_known[0,:])
    assert_equal(mcrar.ST_[1,:], St_known[1,:])

def test_mcr_c_semilearned():
    """ Test when C items are fixed, i.e., enforced to be the same as the input, always """

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
    St_known[2,70:90] = 3
    St_known += 1

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    C_guess = 1 * C_known
    C_guess[:, 2] = np.abs(np.random.randn(int(M*N))+0.1)

    mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)

    mcrar.fit(D_known, C=C_guess, c_fix=[0,1])
    assert_equal(mcrar.C_[:, 0], C_known[:, 0])
    assert_equal(mcrar.C_[:, 1], C_known[:, 1])

def test_mcr_semilearned_both_c_st():
    """
    Test the special case when C & ST are provided, requiring C-fix ST-fix to
    be provided
    """

    M = 21
    N = 21
    P = 101
    n_components = 3

    C_img = np.zeros((M,N,n_components))
    C_img[...,0] = np.dot(np.ones((M,1)),np.linspace(0.1,1,N)[None,:])
    C_img[...,1] = np.dot(np.linspace(0.1,1,M)[:, None], np.ones((1,N)))
    C_img[...,2] = 1 - C_img[...,0] - C_img[...,1]
    C_img = C_img / C_img.sum(axis=-1)[:,:,None]

    St_known = np.zeros((n_components, P))
    St_known[0,30:50] = 1
    St_known[1,50:70] = 2
    St_known[2,70:90] = 3
    St_known += 1

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    C_guess = 1 * C_known
    C_guess[:, 2] = np.abs(np.random.randn(int(M*N)))

    mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-10)

    mcrar.fit(D_known, C=C_guess, ST=St_known, c_fix=[0,1], st_fix=[0], c_first=True)
    assert_equal(mcrar.C_[:, 0], C_known[:, 0])
    assert_equal(mcrar.C_[:, 1], C_known[:, 1])
    assert_equal(mcrar.ST_[0, :], St_known[0, :])

    # ST-solve first
    mcrar.fit(D_known, C=C_guess, ST=St_known, c_fix=[0,1], st_fix=[0], c_first=False)
    assert_equal(mcrar.C_[:, 0], C_known[:, 0])
    assert_equal(mcrar.C_[:, 1], C_known[:, 1])
    assert_equal(mcrar.ST_[0, :], St_known[0, :])

def test_mcr_errors():

    # Providing both C and S^T estimates without C_fix and St_fix
    with pytest.raises(TypeError):
        mcrar = McrAR()
        mcrar.fit(np.random.randn(10,5), C=np.random.randn(10,3),
                   ST=np.random.randn(3,5))

    # Providing both C and S^T estimates without both C_fix and St_fix
    with pytest.raises(TypeError):
        mcrar = McrAR()
        # Only c_fix
        mcrar.fit(np.random.randn(10,5), C=np.random.randn(10,3),
                   ST=np.random.randn(3,5), c_fix=[0])

    with pytest.raises(TypeError):
        mcrar = McrAR()
        # Only st_fix
        mcrar.fit(np.random.randn(10,5), C=np.random.randn(10,3),
                   ST=np.random.randn(3,5), st_fix=[0])

    # Providing no estimates
    with pytest.raises(TypeError):
        mcrar = McrAR()
        mcrar.fit(np.random.randn(10,5))

    # Unknown regression method
    with pytest.raises(ValueError):
        mcrar = McrAR(c_regr='NOTREAL')

    # regression object with no fit method
    with pytest.raises(ValueError):
        mcrar = McrAR(c_regr=print)

def test_props_features_samples_targets(dataset):
    """ Test mcrar properties for features, targets, samples """
    C_known, D_known, St_known = dataset

    mcrar = McrAR()
    mcrar.fit(D_known, ST=St_known)

    assert mcrar.n_targets == C_known.shape[-1]  # n_components
    assert mcrar.n_samples == D_known.shape[0]
    assert mcrar.n_features == D_known.shape[-1]