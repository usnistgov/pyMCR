
import numpy as np
from numpy.linalg import svd

__all__ = ['ind', 'rod']


def ind(D_actual, ul_rank=100):
    """ Malinowski's indicator function """
    n_samples = D_actual.shape[0]
    n_max_rank = np.min([ul_rank, np.min(D_actual.shape)-1])
    error_squared = np.zeros(n_max_rank)

    # PCA forces data matrices to be normalized.
    # Therefore, D_actual also needs to be normalized.
    D_scale = _scale(D_actual)
    U, S, _ = svd(D_actual)
    T = U * S
    for n_rank in range(1, n_max_rank+1):
        error_squared[n_rank - 1] = np.sum(np.square(D_scale)) - np.sum(np.square(T[:, :n_rank]))
    indicator = np.sqrt(error_squared) /\
                np.square([n_samples - L for L in np.arange(1, n_max_rank+1)])
    return indicator


def rod(D_actual, ul_rank=100):
    """ Ratio of derivatives """
    IND = ind(D_actual, ul_rank)
    ROD = ( IND[0:(len(IND)-2)] - IND[1:(len(IND)-1)] ) \
          / ( IND[1:(len(IND)-1)] - IND[2:len(IND)] )
    return ROD


def _scale(X, with_mean=True, with_std=True):
    Xsc = X
    if with_mean:
        Xsc -= X.mean(axis=0)
    if with_std:
        Xsc /= X.std(axis=0)
    return Xsc
