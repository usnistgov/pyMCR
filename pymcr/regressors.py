"""
Built-in least squares / regression methods.

All models will follow the formalism, AX = B, solve for X.

NOTE: coef_ will be X.T, which is the formalism that scikit-learn follows

"""

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)

import numpy as _np

from scipy.linalg import lstsq as _lstsq
from scipy.optimize import nnls as _nnls

class LinearRegression(_ABC):
    """ Abstract class for linear regression methods """
    def __init__(self):
        self.X_ = None
        self.residual_ = None

    @property
    def coef_(self):
        """ The transposed form of X. This is the formalism of scikit-learn """
        if self.X_ is None:
            return None
        else:
            return self.X_.T

    @_abstractmethod
    def fit(self, A, B):
        """ AX = B, solve for X """

class OLS(LinearRegression):
    """
    Ordinary least squares regression

    AX = B, solve for X (coefficients.T)

    Attributes
    ----------
    coef_ : ndarray
        Regression coefficients (X.T)

    residual_ : ndarray
        Residual (sum-of-squares)

    rank_ : int
        Effective rank of matrix A

    svs_ : ndarray
        Singular values of matrix A

    Notes
    -----
    This is simply a wrapped version of Ordinary Least Squares
    (scipy.linalg.lstsq).

    coef_ is X.T, which is the formalism of scikit-learn

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rank_ = None
        self.svs_ = None

    def fit(self, A, B):
        """ Solve for X: AX = B"""
        self.X_, self.residual_, self.rank_, self.svs_ = _lstsq(A, B)

class NNLS(LinearRegression):
    """
    Non-negative constrained least squares regression

    AX = B, solve for X (coeffients.T)

    Attributes
    ----------
    coef_ : ndarray
        Regression coefficients

    residual_ : ndarray
        Residual (sum-of-squares)

    Notes
    -----
    This is simply a wrapped version of NNLS
    (scipy.optimize.nnls).

    coef_ is X.T, which is the formalism of scikit-learn
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, A, B):
        """ Solve for X: AX = B"""

        if B.ndim == 2:
            N = B.shape[-1]
        else:
            N = 0

        self.X_ = _np.zeros((A.shape[-1], N))
        self.residual_ = _np.zeros((N))

        # nnls is Ax = b; thus, need to iterate along
        # columns of B
        if N == 0:
            self.X_, self.residual_ = _nnls(A, B)
        else:
            for num in range(N):
                self.X_[:, num], self.residual_[num] = _nnls(A, B[:, num])
