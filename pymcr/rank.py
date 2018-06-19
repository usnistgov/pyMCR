""" Functions to estimate number of factors / rank """
import numpy as _np
from numpy.linalg import svd as _svd
from scipy.sparse.linalg import svds as _svds

from pymcr.condition import standardize as _standardize

__all__ = ['ind', 'rod', 'pca']


def pca(D, n_components=None):
    """
    Principle component analysis

    Parameters
    ----------
    D : ndarray [n_sample, n_features]
        Data
    n_components : int
        Number of components to calculate (using scipy.sparse.linalg.svds). If
        None use numpy.linalg.svd

    Returns
    -------
    Tuple with 3 items: Scores (T), Loadings (W), eigenvalues (singular values-squared)

    """
    if n_components is None:
        W, s2, Wt = _svd(_np.dot((D - D.mean(axis=0, keepdims=True)).T,
                                D - D.mean(axis=0, keepdims=True)),
                         full_matrices=False)
    else:
        W, s2, Wt = _svds(_np.dot((D - D.mean(axis=0, keepdims=True)).T,
                                D - D.mean(axis=0, keepdims=True)), k=n_components)

        # svds does not sort by variance; thus, manually sorting from biggest to
        # smallest variance
        sort_vec = _np.flipud(_np.argsort(s2))
        W = W[:, sort_vec]
        Wt = Wt[sort_vec, :]
        s2 = s2[sort_vec]

    assert _np.allclose(W, Wt.T)
    T = _np.dot(D, W)

    return (T, W, s2)

def ind(D_actual, ul_rank=100):
    """ Malinowski's indicator function """
    n_samples = D_actual.shape[0]
    n_max_rank = _np.min([ul_rank, _np.min(D_actual.shape)-1])
    error_squared = _np.zeros(n_max_rank)

    T, W, s2 = pca(D_actual)

    # ! PCA only centers the mean, it does not divide by the standard deviation
    # PCA forces data matrices to be normalized.
    # Therefore, D_actual also needs to be normalized.
    # D_scale = _standardize(D_actual)
    D_scale = D_actual - D_actual.mean(axis=0, keepdims=True)
    # U, S, _ = _svd(D_actual)
    # T = U * S

    error_squared = _np.sum(D_scale**2) - _np.sum(_np.cumsum(T[:,:-1]**2, axis=-1), axis=0)

    indicator = _np.sqrt(error_squared) / (n_samples - _np.arange(1, n_max_rank+1))**2

    # n_samples = D_actual.shape[0]
    # n_features = D_actual.shape[1]

    # assert s2.size == n_features, 'Number of samples must be larger than number of features'
    # k_vec = _np.arange(n_features) + 1

    # eigen_values = _np.sqrt(s2)

    # indicator = _np.sqrt(_np.cumsum(eigen_values) / (n_samples * (n_features - k_vec))) / (n_samples - k_vec)**2

    return indicator


def rod(D_actual, ul_rank=100):
    """ Ratio of derivatives """
    IND = ind(D_actual, ul_rank)
    ROD = ( IND[0:(len(IND)-2)] - IND[1:(len(IND)-1)] ) \
          / ( IND[1:(len(IND)-1)] - IND[2:len(IND)] )
    return ROD

if __name__ == '__main__':
    D = _np.vstack((_np.eye(3), -_np.eye(3)))
    ind(D)