"""
Numpy-based methods for fitting/regression/inversion
"""

from numpy import dot
from numpy.linalg import pinv

def ols_c(H, ST):
    """
    CS^T = H

    Solve for C
    """

    return dot(H, pinv(ST))

def ols_s(H, C):
    """
    CS^T = H

    Solve for S^T
    """
    
    return dot(pinv(C), H)
