""" Test rank metrics """
import numpy as np

from pymcr.rank import pca, rsd, ind, rod

def test_pca_full():
    """ Test PCA (full rank)"""
    D = np.vstack((np.eye(3), -np.eye(3)))
    # Axis 0 is most variability
    # Axis 1 is next
    # Axis 2 is smallest
    D[[1, 4], 1] *= 1e-1
    D[[2, 5], 2] *= 1e-2

    T, W, s2 = pca(D)

    assert np.allclose(np.eye(3), np.abs(W))
    assert np.allclose(np.abs(D), np.abs(T))


def test_pca_n_components():
    """ Test PCA (2 components)"""
    D = np.vstack((np.eye(3), -np.eye(3)))
    # Axis 0 is most variability
    # Axis 1 is next
    # Axis 2 is smallest
    D[[1, 4], 1] *= 1e-1
    D[[2, 5], 2] *= 1e-2

    T, W, s2 = pca(D, n_components=2)
    D_m = D - D.mean(axis=0, keepdims=True)
    X = np.dot(D_m.T, D)

    assert np.allclose(np.eye(3, 2), np.abs(W))
    assert np.allclose(np.abs(D)[:,:2], np.abs(T))

    # Sorting now built into pca
    # assert np.allclose(np.eye(3, 2), np.abs(W[:,np.flipud(np.argsort(s2))]))
    # assert np.allclose(np.abs(D)[:,:2], np.abs(T[:, np.flipud(np.argsort(s2))]))


def test_rsd_length():
    """ Test if RSD's length is correct """
    X = np.random.randn(10, 100)
    n_rank = np.min(X.shape)
    assert len(rsd(X)) == n_rank - 1


def test_ind_length():
    """ Test if IND's length is correct """
    X = np.random.randn(10, 100)
    n_rank = np.min(X.shape)
    assert len(ind(X)) == n_rank - 1


def test_rod_length():
    """ Test if ROD's length is correct """
    X = np.random.randn(10, 100)
    n_rank = np.min(X.shape)
    assert len(rod(X)) == n_rank - 1 - 2
