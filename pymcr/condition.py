""" Functions to condition / preprocess data """
import numpy as _np

__all__ = ['standardize']

def standardize(X, mean_ctr=True, with_std=True, axis=-1, copy=True):
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