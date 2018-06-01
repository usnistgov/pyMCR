
import numpy as _np
from numpy.linalg import svd as _svd

__all__ = ['ind', 'rod']


def ind(D_actual, ul_rank=100):
    """ Malinowski's indicator function """
    n_samples = D_actual.shape[0]
    n_max_rank = _np.min([ul_rank, _np.min(D_actual.shape)-1])
    error_squared = _np.zeros(n_max_rank)

    # PCA forces data matrices to be normalized.
    # Therefore, D_actual also needs to be normalized.
    D_scale = _scale(D_actual)
    U, S, _ = _svd(D_actual)
    T = U * S
    for n_rank in range(1, n_max_rank+1):
        error_squared[n_rank - 1] = _np.sum(_np.square(D_scale)) - _np.sum(_np.square(T[:, :n_rank]))
    indicator = _np.sqrt(error_squared) /\
                _np.square([n_samples - L for L in _np.arange(1, n_max_rank+1)])
    return indicator


def rod(D_actual, ul_rank=100):
    """ Ratio of derivatives """
    IND = ind(D_actual, ul_rank)
    ROD = ( IND[0:(len(IND)-2)] - IND[1:(len(IND)-1)] ) \
          / ( IND[1:(len(IND)-1)] - IND[2:len(IND)] )
    return ROD


def _scale(X, mean_ctr=True, with_std=True, axis=-1, copy=True):
    """
    Standardization of data
    
    Parameters
    ----------

    X : ndarray
        Data array

    mean_ctr : bool
        Mean-center data

    with_std : bool
        Normalize by the standard deviation of the data
    
    axis : int
        Axis from which to calculate mean and standard deviation

    copy : bool
        Copy data (X) if True, overwite if False

    """

    if copy:
        Xsc = 1*X
    else:
        Xsc = X

    if mean_ctr:
        Xsc -= X.mean(axis=axis, keepdims=True)
    if with_std:
        Xsc /= X.std(axis=axis, keepdims=True)
    return Xsc
