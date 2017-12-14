"""
Metric used in pyMCR
"""

import numpy as _np

def mse(ideal, under_test):
    """
    Mean square error
    """
    return _np.sum((ideal - under_test)**2)/ideal.size