"""
Metric used in pyMCR
"""

import numpy as _np

def mse(ideal, under_test):
    """
    Mean square error
    """
    return _np.sum((ideal - under_test)**2)/ideal.size

def mean_rel_dif(new, old, only_non_zero=True):
    """ Relative difference """
    
    
    mrd = (new-old)
    
    loc_n0 = _np.where(old != 0.0)
    mrd[loc_n0] = mrd[loc_n0]/old[loc_n0]
    
    if loc_n0[0].size == 0:
        return None
    else:
        if only_non_zero:
            mrd = _np.sum(mrd[loc_n0])
            mrd /= _np.size(new[loc_n0])
        else:
            mrd = _np.sum(mrd)
            mrd /= _np.size(new)
        return mrd
    