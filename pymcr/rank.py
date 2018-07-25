""" Functions to estimate number of factors / rank """
import numpy as _np
from numpy.linalg import svd as _svd
from scipy.sparse.linalg import svds as _svds

from pymcr.condition import standardize as _standardize

__all__ = ['rsd', 'ind', 'rod', 'pca']


def pca(D, n_components=None):
    """
    Principal component analysis

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
    Dcenter = D - D.mean(axis=0, keepdims=True)
    if n_components is None:
        W, s2, Wt = _svd(_np.dot(Dcenter.T, Dcenter),
                         full_matrices=False)
        # Note: s2 contains trivial values.
        # Ex) Let D n x d matrix (n >= d),
        # s2 is n-length vector,
        # though the mathematical rank of the metrics is at most d
    else:
        if n_components == Dcenter.shape[0]:
            n_components -= 1
        W, s2, Wt = _svds(_np.dot(Dcenter.T, Dcenter), k=n_components)

        # svds does not sort by variance; thus, manually sorting from biggest to
        # smallest variance
        sort_vec = _np.flipud(_np.argsort(s2))
        W = W[:, sort_vec]
        Wt = Wt[sort_vec, :]
        s2 = s2[sort_vec]
        # FIXME: T.var(axis=0) is not equal to s2 values.

    # SVD decomposes A into U * S * V^T
    # It is thought that U == Wt is false.
    T = _np.dot(D, W)
    # Note: T.mean(axis=0) is almost zeros
    return T, W, s2


def rsd(D_actual):
    """
    The residual standard deviation (RSD)

    Parameters
    ----------
    D_actual: ndarray [n_sample, n_features]
        Spectral data matrix

    Returns
    -------
    RSD, a measure of the lack of fit of a PCA model to a data set.
    The number of PCA components is q - 1, when the rank of input data is q.
    Centering preceding PCA reduces the rank by one.

    RSD is computed over l from 1 to q - 2 by definition,
    where l is the number of principal components,
    q is the value of the rank of X

    """
    n_rank = _np.min(D_actual.shape)
    n_samples = D_actual.shape[0]
    pca_scores, _, _ = pca(D_actual, n_rank-1)
    q = pca_scores.shape[1]
    variances = pca_scores.var(axis=0)
    csum = _np.cumsum(variances[::-1])[::-1]
    RSD = _np.sqrt( csum / ( n_samples * (q-1) ) )
    return RSD[1:]


def ind(D_actual):
    """
    Malinowski's indicator function

    Parameters
    ----------
    D_actual : ndarray [n_sample, n_features]
        Data

    Returns
    -------
    IND, ul_rank-length vector

    """
    n_rank = _np.min(D_actual.shape)# q
    RSD = rsd(D_actual)# the length is q-2
    denominator = _np.square(_np.arange(n_rank-2, 0, -1))# (q-1)^2, (q-2)^2, ..., 2^2, 1^2
    IND = _np.divide(RSD, denominator)
    return IND


def rod(D_actual):
    """

    Ratio of Derivatives (ROD)

    argmax(ROD) is thought to correspond to the chemical rank of the data.
    For example, a mixture spectrum consists of three pure components,
    the chemical rank is three. The chemical rank of the mixture spectra
    is expected to be three.

    Parameters
    ----------
    D_actual : ndarray [n_sample, n_features]
        Data

    Returns
    -------
    ROD, ul_rank-length vector

    """

    IND = ind(D_actual)
    n_ind = len(IND)
    ROD = ( IND[0:(n_ind-2)] - IND[1:(n_ind-1)] ) \
          / ( IND[1:(n_ind-1)] - IND[2:n_ind] )
    return _np.array([0, 0]+list(ROD))


if __name__ == '__main__':
    D = _np.vstack((_np.eye(3), -_np.eye(3)))
    print(ind(D))