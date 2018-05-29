"""
Built-in constraints

All classes need a transform class. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)

import numpy as _np

__all__ = ['ConstraintNonneg', 'ConstraintCumsumNonneg',
           'ConstraintZeroEndPoints', 'ConstraintZeroCumSumEndPoints',
           'ConstraintNorm', 'ConstraintCutBelow', 'ConstraintCutBelow',
           'ConstraintCompressBelow', 'ConstraintCutAbove',
           'ConstraintCompressAbove']

class Constraint(_ABC):
    """ Abstract class for constraints """

    @_abstractmethod
    def transform(self, A):
        """ Transform A input based on constraint """

class ConstraintNonneg(Constraint):
    """
    Non-negativity constraint. All negative entries made 0.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, copy=False):
        """ A must be non-negative"""
        self.copy = copy

    def transform(self, A):
        """ Apply nonnegative constraint"""
        if self.copy:
            return A*(A > 0)
        else:
            A *= (A > 0)
            return A

class ConstraintCumsumNonneg(Constraint):
    """
    Cumulative-Summation non-negativity constraint. All negative entries made 0.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, axis=-1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        if self.copy:
            return A*(_np.cumsum(A, self.axis) > 0)
        else:
            A *= (_np.cumsum(A, self.axis) > 0)
            return A

class ConstraintZeroEndPoints(Constraint):
    """
    Enforce the endpoints (or the mean over a range) is zero

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    axis : int
        Axis to operate on

    span : int
        Number of pixels along the ends to average.
    """
    def __init__(self, axis=-1, span=1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        if [0,1,-1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis
        self.span = span

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        pix_vec = _np.arange(A.shape[self.axis])
        if (self.axis == 0):
            if self.span == 1:
                slope = (A[-1,:] - A[0,:]) / (pix_vec[-1] - pix_vec[0])
                intercept = A[0,:]
            else:
                slope = ((A[-self.span:,:].mean(axis=0) - A[:self.span,:].mean(axis=0)) /
                          (pix_vec[-self.span:].mean() -
                           pix_vec[:self.span].mean()))
                intercept = (A[:self.span, :] - _np.dot(pix_vec[:self.span, None],
                                                        slope[None, :])).mean(axis=0)
            if self.copy:
                return A - _np.dot(pix_vec[:,None], slope[None,:]) - intercept[None,:]
            else:
                 A -= (_np.dot(pix_vec[:,None], slope[None,:]) + intercept[None,:])
                 return A
        else:
            if self.span == 1:
                slope = (A[:, -1] - A[:, 0]) / (pix_vec[-1] - pix_vec[0])
                intercept = A[:,0]
            else:
                slope = ((A[:, -self.span:].mean(axis=1) - A[:, :self.span].mean(axis=1)) /
                          (pix_vec[-self.span:].mean() -
                           pix_vec[:self.span].mean()))
                intercept = (A[:, :self.span] - _np.dot(slope[:,None],
                                                        pix_vec[None,:self.span])).mean(axis=1)

            if self.copy:
                return A - _np.dot(slope[:,None], pix_vec[None,:]) - intercept[:, None]
            else:
                A -= (_np.dot(slope[:,None], pix_vec[None,:]) + intercept[:, None])
                return A

class ConstraintZeroCumSumEndPoints(Constraint):
    """
    Enforce the endpoints of the cumsum (or the mean over a range) is near-zero.
    Note: this is an approximation.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    axis : int
        Axis to operate on

    span : int
        Number of pixels along the ends to average.
    """
    def __init__(self, axis=-1, copy=False):
        """ A must be non-negative"""
        self.copy = copy
        if [0,1,-1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        meaner = A.mean(self.axis)

        if (self.axis == 0):
            if self.copy:
                return A - meaner[None,:]
            else:
                A -= meaner[None,:]
                return A
        else:
            if self.copy:
                return A - meaner[:, None]
            else:
                A -= meaner[:, None]
                return A

class ConstraintNorm(Constraint):
    """
    Normalization constraint.

    Parameters
    ----------
    axis : int
        Which axis of input matrix A to apply normalization acorss.
    fix : list
        Keep fix-axes as-is and normalize the remaining axes based on the
        residual of the fixed axes.
    set_zeros_to_feature : int
        Set all samples which sum-to-zero across axis to 1 for a particular feature
        (See Notes)
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    Notes
    -----

    -   For set_zeros_to_feature, assuming the data represents concentration with
     a matrix [n_samples, n_features] and the axis is across the features, for every
     sample that sums to 0 across axis, would be replaced with a vector [n_features]
     of zeros except at set_zeros_to_feature, which would equal 1. I.e., this pixel is
     now pure substance of index value set_zeros_to_feature.


    """
    def __init__(self, axis=-1, fix=None, copy=False):
        """Normalize along axis"""
        self.copy = copy
        if fix is None:
            self.fix = fix
        elif isinstance(fix, int):
            self.fix = [fix]
        elif isinstance(fix, (list, tuple)):
            self.fix = fix
        elif isinstance(fix, _np.ndarray):
            if _np.issubdtype(fix.dtype, _np.integer):
                self.fix = fix.tolist()
            else:
                raise TypeError('ndarrays must be of dtype int')
        else:
            raise TypeError('Parameter fix must be of type None, int, list, tuple, ndarray')

        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply normalization constraint """
        if self.copy:
            if self.axis == 0:
                if not self.fix:  # No fixed axes
                    return A / A.sum(axis=self.axis)[None, :]
                else:  # Fixed axes
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[0]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[not_fix_locs, :].sum(axis=0)[None, :]
                    div[div == 0] = 1
                    scaler[not_fix_locs, :] = ((1 - A[fix_locs, :].sum(axis=0)[None,:]) / div)

                    return A * scaler
            else:  # Axis = 1 / -1
                if not self.fix:  # No fixed axes
                    return A / A.sum(axis=self.axis)[:, None]
                else:  # Fixed axis
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[-1]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[:, not_fix_locs].sum(axis=-1)[:,None]
                    div[div == 0] = 1
                    scaler[:, not_fix_locs] = ((1 - A[:, fix_locs].sum(axis=-1)[:,None]) / div)

                    return A * scaler
        else:  # Overwrite original data
            if A.dtype != _np.float:
                raise TypeError('A.dtype must be float for in-place math (copy=False)')

            if self.axis == 0:
                if not self.fix:  # No fixed axes
                    A /= A.sum(axis=self.axis)[None, :]
                    return A
                else:  # Fixed axes
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[0]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[not_fix_locs, :].sum(axis=0)[None,:]
                    div[div == 0] = 1
                    scaler[not_fix_locs, :] = ((1 - A[fix_locs, :].sum(axis=0)[None,:]) / div)
                    A *= scaler
                    return A
            else:  # Axis = 1 / -1
                if not self.fix:  # No fixed axes
                    A /= A.sum(axis=self.axis)[:, None]
                    return A
                else:  # Fixed axis
                    fix_locs = self.fix
                    not_fix_locs = [v for v in _np.arange(A.shape[-1]).tolist()
                                    if self.fix.count(v) == 0]
                    scaler = _np.ones(A.shape)
                    div = A[:, not_fix_locs].sum(axis=-1)[:,None]
                    div[div == 0] = 1
                    scaler[:, not_fix_locs] = ((1 - A[:, fix_locs].sum(axis=-1)[:,None]) / div)
                    A *= scaler
                    return A

class ConstraintReplaceZeros(Constraint):
    """
    Samples that sum-to-zero across axis are replaced with a vector of 0's except
    for a 1 at feature if a single value. In a concentration context, e.g., samples with
    no concentration are replaced with 100% concentration of a set feature. If multiple
    features given, equal amounts of each feature (summing to 1) are used.

    Parameters
    ----------
    axis : int
        Which axis of input matrix A to apply normalization acorss.
    feature : int, list, tuple
        Set all samples which sum-to-zero across axis to fval for a particular feature (or fractional)
        for multiple features.
    fval : float
        Value of summation across axis of replacement vector.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    """

    def __init__(self, axis=-1, feature=None, fval=1, copy=False):
        """Replace sum-to-zero samples with new feature vector along axis"""
        self.copy = copy
        self.fval = fval
        if feature is None:
            self.feature = feature
        elif isinstance(feature, int):
            self.feature = [feature]
        elif isinstance(feature, (list, tuple)):
            self.feature = feature
        elif isinstance(feature, _np.ndarray):
            if _np.issubdtype(feature.dtype, _np.integer):
                self.feature = feature.tolist()
            else:
                raise TypeError('ndarrays must be of dtype int')
        else:
            raise TypeError('Parameter feature must be of type None, int, list, tuple, ndarray')

        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply constraint """
        if self.feature:
            replacement = _np.zeros(A.shape[self.axis])
            replacement[self.feature] = self.fval
            replacement /= replacement.sum()
            replacement *= self.fval

            if self.copy:
                A_out = 1*A
                if self.axis == 0:
                    A_out[:, A_out.sum(axis=0)==0] = replacement[:,None]
                else:  # Axis 1 / -1
                    A_out[A_out.sum(axis=-1)==0] = replacement
                return A_out
            else:
                if self.axis == 0:
                    A[:, A.sum(axis=0)==0] = replacement[:,None]
                else:  # Axis 1 / -1
                    A[A.sum(axis=-1)==0] = replacement
                return A
        else:
            return A

class ConstraintCutBelow(Constraint):
    """
    Cut values below (and not-equal to) a certain threshold.

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : int
        If not None, cut below value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, value=0, axis_sumnz=None, copy=False):
        """ """
        self.copy = copy
        self.value = value
        self.axis = axis_sumnz

    def transform(self, A):
        """ Apply cut-below value constraint"""
        if self.axis is None:
            if self.copy:
                return A*(A >= self.value)
            else:
                A *= (A >= self.value)
                return A
        else:
            if self.copy:
                return A*(_np.alltrue(A < self.value, axis=self.axis, keepdims=True) +
                          (A >= self.value))
            else:
                A *= (_np.alltrue(A < self.value, axis=self.axis, keepdims=True) +
                      (A >= self.value))
                return A

class ConstraintCompressBelow(Constraint):
    """
    Compress values below (and not-equal to) a certain threshold (set to value)

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        self.copy = copy
        self.value = value

    def transform(self, A):
        """ Apply compress-below value constraint"""
        if self.copy:
            return A*(A >= self.value) + self.value*(A < self.value)
        else:
            temp = self.value*(A < self.value)
            A *= (A >= self.value)
            A += temp
            return A

class ConstraintCutAbove(Constraint):
    """
    Cut values above (and not-equal to) a certain threshold

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : int
        If not None, cut above value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """
    def __init__(self, value=0, axis_sumnz=None, copy=False):
        """ """
        self.copy = copy
        self.value = value
        self.axis = axis_sumnz

    def transform(self, A):
        """ Apply cut-above value constraint"""
        if self.axis is None:
            if self.copy:
                return A*(A <= self.value)
            else:
                A *= (A <= self.value)
                return A
        else:
            if self.copy:
                return A*(_np.alltrue(A > self.value, axis=self.axis, keepdims=True) +
                          (A <= self.value))
            else:
                A *= (_np.alltrue(A > self.value, axis=self.axis, keepdims=True) +
                      (A <= self.value))
                return A

class ConstraintCompressAbove(Constraint):
    """
    Compress values above (and not-equal to) a certain threshold (set to value)

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        self.copy = copy
        self.value = value

    def transform(self, A):
        """ Apply compress-above value constraint"""
        if self.copy:
            return A*(A <= self.value) + self.value*(A > self.value)
        else:
            temp = self.value*(A > self.value)
            A *= (A <= self.value)
            A += temp
            return A

class ConstraintPlanarize(Constraint):
    """
    Set a particular target to a plane

    Parameters
    ----------

    target : int, list, tuple
        Target numbers to set to a fitted plane
    shape : tuple, list
        Shape of array (M,N) which is (Y,X)
    use_vals_above : float
        Only calculate based on values above (not including)
    use_vals_below : float
        Only calculate based on values below (not including)
    lims_to_plane : bool
        The returned plane will be limited to the range of the optionally supplied
        use_vals_below, use_vals above.
    scaler : float
        A large value that is much bigger than any values in the input array.
        Needed to ensure SVD properly creates plane. If None, auto-calculates.
    recalc_scaler : bool
        Auto-calculate for every new input (does not use previously provided or
        calculated value)
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    Notes
    -----

    -   This uses an SVD to calculate the vector normal to the plane that fits the input data. It
    assumes that the 3rd singular vector is the normal; thus, the x and y vectors for the data need
    be larger than the variance of the input data. Scaler enables this by scaling the auto-generated
    x and y vectors to be much larger than the max-min of the input data

    """

    def __init__(self, target, shape, use_vals_above=None, use_vals_below=None,
                 lims_to_plane=True, scaler=None, recalc_scaler=False, copy=False):
        if isinstance(target, int):
            self.target = [target]
        elif isinstance(target, (list, tuple, _np.ndarray)):
            self.target = target
        else:
            raise TypeError('target must be an int, list, 2D ndarray, or tuple')

        self.shape = shape
        self.copy = copy
        self.scaler = scaler
        self.recalc = recalc_scaler
        self.use_above = use_vals_above
        self.use_below = use_vals_below
        self.lims_to_plane = lims_to_plane

        self._x = None
        self._y = None
        self._X = None
        self._Y = None

        if scaler is not None:
            self._setup_xy(scaler)

    def _setup_xy(self, scaler):

        self.scaler = scaler
        self._x = scaler*_np.arange(self.shape[1], dtype=_np.float)
        self._y = scaler*_np.arange(self.shape[0], dtype=_np.float)

        self._X, self._Y = _np.meshgrid(self._x, self._y)
        self._X = self._X.ravel()
        self._Y = self._Y.ravel()

    def transform(self, A):
        """ Set targets, t, to fit planes """
        if (self.scaler is None) | (self.recalc):
            self._setup_xy(1e3 * _np.abs(A.max() - A.min()))

        if self.copy:
            A2 = 1*A

        for t in self.target:
            X2 = 1*self._X
            Y2 = 1*self._Y
            Z2 = 1*A[:, t]

            Stack = _np.vstack((X2, Y2, Z2))

            if self.use_above is not None:
                X2 = X2[Z2>self.use_above]
                Y2 = Y2[Z2>self.use_above]
                Z2 = Z2[Z2>self.use_above]
                Stack = _np.vstack((X2, Y2, Z2))

            if self.use_below is not None:
                X2 = X2[Z2<self.use_below]
                Y2 = Y2[Z2<self.use_below]
                Z2 = Z2[Z2<self.use_below]
                Stack = _np.vstack((X2, Y2, Z2))

            Stack = Stack - Stack.mean(axis=-1)[:,None]

            U,s,Vh = _np.linalg.svd(Stack, full_matrices=False)
            norm_to_plane = 1*U[:,-1]

            plane = (((-norm_to_plane[0] * (self._X - X2.mean())) - 
                       (norm_to_plane[1] * (self._Y - Y2.mean()))) / norm_to_plane[2]) + Z2.mean()
            
            if self.lims_to_plane:
                if self.use_above is not None:
                    plane[plane < self.use_above] = self.use_above
                if self.use_below is not None:
                    plane[plane > self.use_below] = self.use_below
            if not self.copy:
                A[:, t] = plane
            else:
                A2[:, t] = plane
        if self.copy:
            return A2
        else:
            return A

if __name__ == '__main__':  # pragma: no cover
    C_img = _np.zeros((10, 20, 2))  # Y, X, Target
    x = _np.arange(C_img.shape[1])
    y = _np.arange(C_img.shape[0])
    n_targets = C_img.shape[-1]

    X, Y = _np.meshgrid(x,y)

    C_img[:,:,0] = 0.1*X + 0.3*Y - 2.5
    C_img[:,:,1] = 0.1*X + 0.3*Y - 2.5

    C_ravel = C_img.reshape((-1, n_targets))

    constr = ConstraintPlanarize(0, (10, 20), scaler=None, copy=True, use_vals_above=0, 
                                 use_vals_below=1, lims_to_plane=True)
    out = constr.transform(C_ravel)

    assert C_ravel.min() < 0
    assert C_ravel.max() > 1
    assert out[:,0].min() >= 0
    assert out[:,0].max() <= 1
    assert out[:,1].min() < 0
    assert out[:,1].max() > 1