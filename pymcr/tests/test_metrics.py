"""
Testing pymcr.regressors

"""

import numpy as np

from numpy.testing import (assert_equal, assert_array_equal, 
                           assert_allclose)

from pymcr.metrics import mse

def test_mse():
    A = np.ones((3,3))
    B = np.eye(3)
    mse(None, None, A, B)

# import pytest  