"""
Testing metrics

"""

import numpy as np
import numpy.testing

from numpy.testing import assert_array_almost_equal

import pymcr.metrics

def test_mse():
    ideal = np.ones(100)
    under_test = np.zeros(100)

    np.testing.assert_equal(pymcr.metrics.mse(ideal, under_test), 1.0)
    np.testing.assert_equal(pymcr.metrics.mse(ideal, ideal), 0.0)

def test_mrd():
    new = np.ones(100)
    old = np.zeros(100)
    old2 = 2*np.ones(100)
    old_combo = np.append(old, old2)

    np.testing.assert_equal(pymcr.metrics.mean_rel_dif(new, old), None)
    np.testing.assert_equal(pymcr.metrics.mean_rel_dif(new, old2), -0.5)
    np.testing.assert_equal(pymcr.metrics.mean_rel_dif(new, new), 0.0)
    np.testing.assert_equal(pymcr.metrics.mean_rel_dif(np.append(new, new),
                                                       old_combo, 
                                                       only_non_zero=True), -0.5)
    np.testing.assert_equal(pymcr.metrics.mean_rel_dif(np.append(new, new),
                                                       old_combo, 
                                                       only_non_zero=False), 0.25)