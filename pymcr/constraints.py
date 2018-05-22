"""
Built-in constraints

All classes need a transform class. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)

import numpy as _np

__all__ = ['ConstraintNonneg', 'ConstraintCumsumNonneg',
           'ConstraintZeroEndPoints', 'ConstraintZeroCumSumEndPoints',
           'ConstraintNorm', 'ConstraintCutBelow', 'ConstraintCutBelow',
           'ConstraintCompressBelow', 'ConstraintCutAbove',
           'ConstraintCompressAbove']

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
    fix : list
        Keep fix-axes as-is and normalize the remaining axes based on the
        residual of the fixed axes.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, axis=-1, fix=None, copy=False):
        """Normalize along axis"""
        self.copy = copy
        if fix is None:
            self.fix = fix
        elif isinstance(fix, int):
            self.fix = [fix]
        elif isinstance(fix, (list, tuple)):
            self.fix = fix
        elif isinstance(fix, _np.ndarray):
            if _np.issubdtype(fix.dtype, _np.integer):
                self.fix = fix.tolist()
            else:
                raise TypeError('ndarrays must be of dtype int')
        else:
            raise TypeError('Parameter fix must be of type None, int, list, tuple, ndarray')

        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply normalization constraint """
        if self.copy:
            if self.axis == 0:
                if not self.fix:  # No fixed axes
                    return A / A.sum(axis=self.axis)[None, :]
                else:  # Fixed axes
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[0]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[not_fix_locs, :].sum(axis=0)[None, :]
                    div[div == 0] = 1
                    scaler[not_fix_locs, :] = ((1 - A[fix_locs, :].sum(axis=0)[None,:]) / div)

                    return A * scaler
            else:  # Axis = 1 / -1
                if not self.fix:  # No fixed axes
                    return A / A.sum(axis=self.axis)[:, None]
                else:  # Fixed axis
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[-1]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[:, not_fix_locs].sum(axis=-1)[:,None]
                    div[div == 0] = 1
                    scaler[:, not_fix_locs] = ((1 - A[:, fix_locs].sum(axis=-1)[:,None]) / div)

                    return A * scaler
        else:  # Overwrite original data
            if A.dtype != _np.float:
                raise TypeError('A.dtype must be float for in-place math (copy=False)')

            if self.axis == 0:
                if not self.fix:  # No fixed axes
                    A /= A.sum(axis=self.axis)[None, :]
                    return A
                else:  # Fixed axes
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[0]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[not_fix_locs, :].sum(axis=0)[None,:]
                    div[div == 0] = 1
                    scaler[not_fix_locs, :] = ((1 - A[fix_locs, :].sum(axis=0)[None,:]) / div)
                    A *= scaler
                    return A
            else:  # Axis = 1 / -1
                if not self.fix:  # No fixed axes
                    A /= A.sum(axis=self.axis)[:, None]
                    return A
                else:  # Fixed axis
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[-1]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[:, not_fix_locs].sum(axis=-1)[:,None]
                    div[div == 0] = 1
                    scaler[:, not_fix_locs] = ((1 - A[:, fix_locs].sum(axis=-1)[:,None]) / div)
                    A *= scaler
                    return A



class ConstraintCutBelow(Constraint):
    """
    Cut values below (and not-equal to) a certain threshold.

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : int
        If not None, cut below value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, value=0, axis_sumnz=None, copy=False):
        """ """
        self.copy = copy
        self.value = value
        self.axis = axis_sumnz

    def transform(self, A):
        """ Apply cut-below value constraint"""
        if self.axis is None:
            if self.copy:
                return A*(A >= self.value)
            else:
                A *= (A >= self.value)
                return A
        else:
            if self.copy:
                return A*(_np.alltrue(A < self.value, axis=self.axis, keepdims=True) +
                          (A >= self.value))
            else:
                A *= (_np.alltrue(A < self.value, axis=self.axis, keepdims=True) +
                      (A >= self.value))
                return A

class ConstraintCompressBelow(Constraint):
    """
    Compress values below (and not-equal to) a certain threshold (set to value)

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        self.copy = copy
        self.value = value

    def transform(self, A):
        """ Apply compress-below value constraint"""
        if self.copy:
            return A*(A >= self.value) + self.value*(A < self.value)
        else:
            temp = self.value*(A < self.value)
            A *= (A >= self.value)
            A += temp
            return A

class ConstraintCutAbove(Constraint):
    """
    Cut values above (and not-equal to) a certain threshold

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : int
        If not None, cut above value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, value=0, axis_sumnz=None, copy=False):
        """ """
        self.copy = copy
        self.value = value
        self.axis = axis_sumnz

    def transform(self, A):
        """ Apply cut-above value constraint"""
        if self.axis is None:
            if self.copy:
                return A*(A <= self.value)
            else:
                A *= (A <= self.value)
                return A
        else:
            if self.copy:
                return A*(_np.alltrue(A > self.value, axis=self.axis, keepdims=True) +
                          (A <= self.value))
            else:
                A *= (_np.alltrue(A > self.value, axis=self.axis, keepdims=True) +
                      (A <= self.value))
                return A

class ConstraintCompressAbove(Constraint):
    """
    Compress values above (and not-equal to) a certain threshold (set to value)

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        self.copy = copy
        self.value = value

    def transform(self, A):
        """ Apply compress-above value constraint"""
        if self.copy:
            return A*(A <= self.value) + self.value*(A > self.value)
        else:
            temp = self.value*(A > self.value)
            A *= (A <= self.value)
            A += temp
            return A