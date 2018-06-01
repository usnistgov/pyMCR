
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# FIXME: avoid sklearn implementations.
#        Just use numpy libraries: e.g. PCA => numpy.linalg.svd
def ind(D_actual, scale=True, ul_rank=100):
    """ Malinowski's indicator function """
    n_samples = D_actual.shape[0]
    n_max_rank = np.min([ul_rank, np.min(D_actual.shape)-1])
    error_squared = np.zeros(n_max_rank)

    # PCA forces data matrices to be centered.
    # Therefore, D_actual also needs to be centered.
    D_scale = StandardScaler(with_std=scale).fit_transform(D_actual)

    if scale:
        model = Pipeline([('scale', StandardScaler()),
                          ('pca', PCA(n_components=n_max_rank))])
    else:
        model = Pipeline([('pca', PCA(n_components=n_max_rank))])
    pca_scores = model.fit_transform(D_actual)
    for n_rank in range(1, n_max_rank+1):
        error_squared[n_rank - 1] = np.sum(np.square(D_scale)) - np.sum(np.square(pca_scores[:, :n_rank]))
    indicator = np.sqrt(error_squared) /\
                np.square([n_samples - L for L in np.arange(1, n_max_rank+1)])
    return indicator


def rod(D_actual, scale=True, ul_rank=100):
    """ Ratio of derivatives """
    IND = ind(D_actual, scale, ul_rank)
    ROD = ( IND[0:(len(IND)-2)] - IND[1:(len(IND)-1)] ) \
          / ( IND[1:(len(IND)-1)] - IND[2:len(IND)] )
    return ROD
