""" 
pyMCR: Pythonic Multivariate Curve Resolution - Alternating Regression

pyMCR is a small package for performing multivariate curve resolution. Currently, 
it implements a simple alternating regression scheme (MCR-AR). The most common 
implementation is with ordinary least-squares regression, MCR-ALS.

See Also
--------
https://pages.nist.gov/pyMCR

"""
from . import mcr
from . import constraints
from . import metrics
from . import regressors
from . import condition
