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


def rsd(X):
    n_rank = _np.min(X.shape)
    n_samples = X.shape[0]
    pca_scores, _, _ = pca(X, n_rank)
    variances = pca_scores.var(axis=0)
    csum = _np.cumsum(variances[::-1])[::-1]
    rsd_values = _np.sqrt( csum / ( n_samples * (n_rank-1) ) )
    return rsd_values


def ind(D_actual, ul_rank=100):
    """
    Malinowski's indicator function

    Parameters
    ----------
    D_actual : ndarray [n_sample, n_features]
        Data
    ul_rank : int
        The upper limit of the rank. Too large chemical rank
        doesn't have reasonable meaning.

    Returns
    -------
    IND, ul_rank-length vector

    """
    n_samples = D_actual.shape[0]
    n_max_rank = _np.min([ul_rank, _np.min(D_actual.shape)-1])

    T, W, s2 = pca(D_actual)
    D_centered = _standardize(D_actual, with_std=False)

    # FIXME:
    # Correct the definition of IND function, based on the following work:
    # "An automated procedure to predict the number of components in spectroscopic data"

    # error_squared is equal to projection errors of PCA.
    square_errors = _np.sum(_np.square(D_centered)) - \
                    _np.cumsum(s2[0:n_max_rank], axis=-1)

    # indicator is a standardized statistic
    l_vector = _np.arange(n_max_rank) + 1
    print(l_vector)
    indicator = _np.sqrt(square_errors) / \
                _np.square(n_samples - l_vector)
    return indicator, square_errors


def rod(D_actual, ul_rank=100):
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
    ul_rank : int
        The upper limit of the rank. Too large chemical rank
        doesn't have reasonable meaning.

    Returns
    -------
    ROD, ul_rank-length vector

    """

    IND = ind(D_actual, ul_rank)
    ROD = ( IND[0:(len(IND)-2)] - IND[1:(len(IND)-1)] ) \
          / ( IND[1:(len(IND)-1)] - IND[2:len(IND)] )
    return ROD


if __name__ == '__main__':
    D = _np.vstack((_np.eye(3), -_np.eye(3)))
    print(ind(D))