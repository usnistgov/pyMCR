"""
Built-in constraints

All classes need a transform class. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)

import numpy as _np

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
