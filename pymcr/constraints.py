"""
Built-in constraints

All classes need a transform class. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)

import numpy as _np

__all__ = ['ConstraintNonneg', 'ConstraintCumsumNonneg', 
           'ConstraintZeroEndPoints', 'ConstraintZeroCumSumEndPoints',
           'ConstraintNorm']

class Constraint(_ABC):
    """ Abstract class for constraints """

    @_abstractmethod
    def transform(self, A):
        """ Transform A input based on constraint """

class ConstraintNonneg(Constraint):
    """
    Non-negativity constraint. All negative entries made 0.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, copy=False):
        """ A must be non-negative"""
        self.copy = copy

    def transform(self, A):
        """ Apply nonnegative constraint"""
        if self.copy:
            return A*(A > 0)
        else:
            A *= (A > 0)
            return A

class ConstraintCumsumNonneg(Constraint):
    """
    Cumulative-Summation non-negativity constraint. All negative entries made 0.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, axis=-1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        if self.copy:
            return A*(_np.cumsum(A, self.axis) > 0)
        else:
            A *= (_np.cumsum(A, self.axis) > 0)
            return A

class ConstraintZeroEndPoints(Constraint):
    """
    Enforce the endpoints (or the mean over a range) is zero

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    axis : int
        Axis to operate on

    span : int
        Number of pixels along the ends to average.
    """
    def __init__(self, axis=-1, span=1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        if [0,1,-1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis
        self.span = span

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        pix_vec = _np.arange(A.shape[self.axis])
        if (self.axis == 0):
            if self.span == 1:
                slope = (A[-1,:] - A[0,:]) / (pix_vec[-1] - pix_vec[0])
                intercept = A[0,:]
            else:
                slope = ((A[-self.span:,:].mean(axis=0) - A[:self.span,:].mean(axis=0)) / 
                          (pix_vec[-self.span:].mean() - 
                           pix_vec[:self.span].mean()))
                intercept = (A[:self.span, :] - _np.dot(pix_vec[:self.span, None],
                                                        slope[None, :])).mean(axis=0)
            if self.copy:
                return A - _np.dot(pix_vec[:,None], slope[None,:]) - intercept[None,:]
            else:
                 A -= (_np.dot(pix_vec[:,None], slope[None,:]) + intercept[None,:])
                 return A
        else:
            if self.span == 1:
                slope = (A[:, -1] - A[:, 0]) / (pix_vec[-1] - pix_vec[0])
                intercept = A[:,0]
            else:
                slope = ((A[:, -self.span:].mean(axis=1) - A[:, :self.span].mean(axis=1)) / 
                          (pix_vec[-self.span:].mean() - 
                           pix_vec[:self.span].mean()))
                intercept = (A[:, :self.span] - _np.dot(slope[:,None], 
                                                        pix_vec[None,:self.span])).mean(axis=1)

            if self.copy:
                return A - _np.dot(slope[:,None], pix_vec[None,:]) - intercept[:, None]
            else:
                A -= (_np.dot(slope[:,None], pix_vec[None,:]) + intercept[:, None])
                return A

class ConstraintZeroCumSumEndPoints(Constraint):
    """
    Enforce the endpoints of the cumsum (or the mean over a range) is near-zero.
    Note: this is an approximation.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    axis : int
        Axis to operate on

    span : int
        Number of pixels along the ends to average.
    """
    def __init__(self, axis=-1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        if [0,1,-1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        meaner = A.mean(self.axis)

        if (self.axis == 0):
            if self.copy:
                return A - meaner[None,:]
            else:
                A -= meaner[None,:]
                return A
        else:
            if self.copy:
                return A - meaner[:, None]
            else:
                A -= meaner[:, None]
                return A

class ConstraintNorm(Constraint):
    """
    Normalization constraint.

    Parameters
    ----------
    axis : int
        Which axis of input matrix A to apply normalization acorss.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, axis=-1, copy=False):
        """Normalize along axis"""
        self.copy = copy
        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply normalization constraint """
        if self.copy:
            if self.axis == 0:
                return A / A.sum(axis=self.axis)[None, :]
            else:
                return A / A.sum(axis=self.axis)[:, None]
        else:
            if A.dtype != _np.float:
                raise TypeError('A.dtype must be float for in-place math (copy=False)')

            if self.axis == 0:
                A /= A.sum(axis=self.axis)[None, :]
            else:
                A /= A.sum(axis=self.axis)[:, None]
            return A
