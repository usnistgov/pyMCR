"""
Testing external regressors if library if installed

"""

import numpy as np

from numpy.testing import (assert_equal, assert_array_equal, 
                           assert_allclose)

try:
    from sklearn.linear_model import LinearRegression, Ridge
except:
    flag_sklearn = False
else:
    flag_sklearn = True

import pytest   

@pytest.mark.skipif(flag_sklearn == False, reason='sklearn not installed, skipping.')
class TestSklearn:
    """ Test some scikit-learn methods. Skips if sklearn is not installed """
    
    def test_sklearn_linear_regression(self):
        """ Test sklearn linear regression """

        A = np.array([[1,1,0],[1,0,1],[0,0,1]])
        x = np.array([1,2,3])
        X = x[:,None]
        B = np.dot(A,X)
        b = np.dot(A,x)

        # HAVE to NOT fit intercept
        regr = LinearRegression(fit_intercept=False)

        assert hasattr(regr, 'fit')

        regr.fit(A,B)
        assert_allclose(X.T, regr.coef_)

        regr.fit(A,b)
        assert_allclose(x.T, regr.coef_)
