"""
Numpy-based methods for fitting/regression/inversion
"""

from numpy import dot, zeros
from numpy.linalg import pinv
from scipy.optimize import nnls

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

def nnls_c(H, ST):
    """
    Solution using scipy.optimize.nnls (non-negative least squares)
    """
    temp = zeros((H.shape[0], ST.shape[0]))
    for num, h in enumerate(H):
        # temp[num, :] = nnls(ST.T, h)[0]

        # Maybe faster?
        temp[num, :] = nnls(dot(ST, ST.T), dot(ST, h))[0]
    
    return temp

def nnls_s(H, C):
    """
    Solution using scipy.optimize.nnls (non-negative least squares)
    """
    temp = zeros((C.shape[-1], H.shape[-1]))
    for num, h in enumerate(H.T):
        # temp[:, num] = nnls(C, h)[0]

        # Maybe faster?
        temp[:, num] = nnls(dot(C.T,C), dot(C.T,h))[0]
    
    return temp
