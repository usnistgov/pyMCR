""" Test data conditioning """
import numpy as np

from numpy.testing import (assert_equal, assert_array_equal,
                           assert_allclose)

from pymcr.condition import (standardize)

def test_scale():
    """ Test data standardization """
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    A_sub_mean_ax0 = np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
    A_sub_mean_ax1 = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

    A_just_std_ax0 = np.array([[0.66666667, 1.33333333, 2.0], [2.66666667, 3.33333333, 4.0]])
    A_just_std_ax1 = np.array([[1.22474487, 2.44948974, 3.67423461],
                               [4.89897949, 6.12372436, 7.34846923]])

    A_standardized_ax0 = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    A_standardized_ax1 = np.array([[-1.22474487, 0.0, 1.22474487], [-1.22474487, 0.0, 1.22474487]])

    # MEAN-CENTER, COPY, AXIS=0
    out = standardize(A, mean_ctr=True, with_std=False, axis=0, copy=True)
    assert_allclose(out, A_sub_mean_ax0)

    # MEAN-CENTER, COPY, AXIS=1
    out = standardize(A, mean_ctr=True, with_std=False, axis=1, copy=True)
    assert_allclose(out, A_sub_mean_ax1)

    # STD-NORM, COPY, AXIS=0
    out = standardize(A, mean_ctr=False, with_std=True, axis=0, copy=True)
    assert_allclose(out, A_just_std_ax0)

    # STD-NORM, COPY, AXIS=1
    out = standardize(A, mean_ctr=False, with_std=True, axis=1, copy=True)
    assert_allclose(out, A_just_std_ax1)

    # FULL STANDARDIZE, COPY, AXIS=0
    out = standardize(A, mean_ctr=True, with_std=True, axis=0, copy=True)
    assert_allclose(out, A_standardized_ax0)

    # FULL STANDARDIZE, COPY, AXIS=1
    out = standardize(A, mean_ctr=True, with_std=True, axis=1, copy=True)
    assert_allclose(out, A_standardized_ax1)

    # OVERWRITE VARIANTS
    # MEAN-CENTER, OVERWRITE, AXIS=0
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=True, with_std=False, axis=0, copy=False)
    assert_allclose(A, A_sub_mean_ax0)

    # MEAN-CENTER, OVERWRITE, AXIS=1
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=True, with_std=False, axis=1, copy=False)
    assert_allclose(A, A_sub_mean_ax1)

    # STD-NORM, OVERWRITE, AXIS=0
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=False, with_std=True, axis=0, copy=False)
    assert_allclose(A, A_just_std_ax0)

    # STD-NORM, OVERWRITE, AXIS=1
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=False, with_std=True, axis=1, copy=False)
    assert_allclose(A, A_just_std_ax1)

    # FULL STANDARDIZE, OVERWRITE, AXIS=0
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=True, with_std=True, axis=0, copy=False)
    assert_allclose(A, A_standardized_ax0)

    # FULL STANDARDIZE, OVERWRITE, AXIS=1
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _ = standardize(A, mean_ctr=True, with_std=True, axis=1, copy=False)
    assert_allclose(A, A_standardized_ax1)
