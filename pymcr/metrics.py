"""
Metrics used in pyMCR

All functions must take C, ST, D_actual, D_calculated
"""

import numpy as np

def mse(C, ST, D_actual, D_calculated):
    """ Mean square error"""
    return ((D_actual - D_calculated)**2).sum()/D_actual.size
	
def lof(C, ST, D_actual, D_calculated):
    return 100*np.sqrt(np.sum((D_actual - D_calculated)**2)/np.sum((D_actual**2)))
