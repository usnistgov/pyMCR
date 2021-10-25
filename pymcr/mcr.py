""" MCR Main Class for Computation"""
import sys as _sys
import copy as _copy

import numpy as _np
import logging as _logging

from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg
from pymcr.metrics import mse

# create logger for mcr.py and set default level
_logger = _logging.getLogger(__name__)
_logger.setLevel(_logging.INFO)


class McrAR:
    """
    Multivariate Curve Resolution - Alternating Regression

    D = CS^T


    Parameters
    ----------
    c_regr : str, class
        Instantiated regression class (or string, see Notes) for calculating
        the C matrix

    st_regr : str, class
        Instantiated regression class (or string, see Notes) for calculating
        the S^T matrix

    fit_kwargs : dict
        kwargs sent to fit and fit_transform methods

    c_fit_kwargs : dict
        kwargs sent to c_regr.fit method

    st_fit_kwargs : dict
        kwargs sent to st_regr.fit method

    c_constraints : list
        List of constraints applied to calculation of C matrix

    st_constraints : list
        List of constraints applied to calculation of S^T matrix

    max_iter : int
        Maximum number of iterations. One iteration calculates both C and S^T

    err_fcn : function
        Function to calculate error/differences after each least squares
        calculation (ie twice per iteration). Outputs to err attribute.

    tol_increase : float
        Factor increase to allow in err attribute. Set to 0 for no increase
        allowed. E.g., setting to 1.0 means the err can double per iteration.

    tol_n_increase : int
        Number of consecutive iterations for which the err attribute can
        increase

    tol_err_change : float
        If err changes less than tol_err_change, per iteration, break.

    tol_n_above_min : int
        Number of half-iterations that can be performed without reaching a
        new error-minimum

    Attributes
    ----------
    err : list
        List of calculated errors (from err_fcn) after each least squares (ie
        twice per iteration)

    C_ : ndarray [n_samples, n_targets]
        Most recently calculated C matrix (that did not cause a tolerance
        failure)

    ST_ : ndarray [n_targets, n_features]
        Most recently calculated S^T matrix (that did not cause a tolerance
        failure)

    components_ : ndarray [n_targets, n_features]
        Synonym for ST_, providing sklearn like compatibility

    C_opt_ : ndarray [n_samples, n_targets]
        [Optimal] C matrix for lowest err attribute

    ST_opt_ : ndarray [n_targets, n_features]
        [Optimal] ST matrix for lowest err attribute

    n_iter : int
        Total number of iterations performed

    n_features : int
        Total number of features, e.g. spectral frequencies.

    n_samples : int
        Total number of samples (e.g., pixels)

    n_targets : int
        Total number of targets (e.g., pure analytes)

    n_iter_opt : int
        Iteration when optimal C and ST calculated

    exit_max_iter_reached : bool
        Exited iterations due to maximum number of iteration reached (max_iter
        parameter)

    exit_tol_increase : bool
        Exited iterations due to maximum fractional increase in error metric
        (via err_fcn)

    exit_tol_n_increase : bool
        Exited iterations due to maximum number of consecutive increases in
        error metric (via err fcn)

    exit_tol_err_change : bool
        Exited iterations due to error metric change that is smaller than
        tol_err_change

    exit_tol_n_above_min : bool
        Exited iterations due to maximum number of half-iterations for which
        the error metric increased above the minimum error

    Notes
    -----

    -   Built-in regressor classes (str can be used): OLS (ordinary least
        squares), NNLS (non-negatively constrained least squares). See
        mcr.regressors.
    -   Built-in regressor methods can be given as a string to c_regr, st_regr;
        though instantiating an imported class gives more flexibility.
    -   Setting any tolerance to None turns that check off

    """

    def __init__(self, c_regr=OLS(), st_regr=OLS(), fit_kwargs={},
                 c_fit_kwargs={}, st_fit_kwargs={}, c_constraints=[ConstraintNonneg()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=50, err_fcn=mse,
                 tol_increase=0.0, tol_n_increase=10, tol_err_change=None,
                 tol_n_above_min=10
                 ):
        """
        Multivariate Curve Resolution - Alternating Regression
        """

        self.fit_kwargs = fit_kwargs

        self.max_iter = max_iter

        self.tol_increase = tol_increase
        self.tol_n_increase = tol_n_increase
        self.tol_err_change = tol_err_change
        self.tol_n_above_min = tol_n_above_min

        self.err_fcn = err_fcn
        self.err = None

        self.c_constraints = c_constraints
        self.st_constraints = st_constraints

        self.c_regressor = self._check_regr(c_regr)
        self.st_regressor = self._check_regr(st_regr)
        self.c_fit_kwargs = c_fit_kwargs
        self.st_fit_kwargs = st_fit_kwargs

        self.C_ = None
        self.ST_ = None

        self.C_opt_ = None
        self.ST_opt_ = None
        self.n_iter_opt = None

        self.n_iter = None
        self.n_increase = None
        self.n_above_min = None

        self.exit_max_iter_reached = False
        self.exit_tol_increase = False
        self.exit_tol_n_increase = False
        self.exit_tol_err_change = False
        self.exit_tol_n_above_min = False

        # Saving every C or S^T matrix at each iteration
        # Could create huge memory usage
        self._saveall_st = False
        self._saveall_c = False
        self._saved_st = []
        self._saved_c = []

    def _check_regr(self, mth):
        """
            Check regressor method. If acceptable strings, instantiate and
            return object. If instantiated class, make sure it has a fit
            attribute.
        """
        if isinstance(mth, str):
            if mth.upper() == 'OLS':
                return OLS()
            elif mth.upper() == 'NNLS':
                return NNLS()
            else:
                raise ValueError('{} is unknown. Use NNLS or OLS.'.format(mth))
        elif hasattr(mth, 'fit'):
            return mth
        else:
            raise ValueError('Input class '
                             '{} does not have a \'fit\' method'.format(mth))

    @property
    def D_(self):
        """ D matrix with current C and S^T matrices """
        return _np.dot(self.C_, self.ST_)

    @property
    def D_opt_(self):
        """ D matrix with optimal C and S^T matrices """
        return _np.dot(self.C_opt_, self.ST_opt_)

    @property
    def n_features(self):
        """ Number of features """
        if self.ST_ is not None:
            return self.ST_.shape[-1]
        else:
            return None

    @property
    def n_targets(self):
        """ Number of targets """
        if self.C_ is not None:
            return self.C_.shape[1]
        else:
            return None

    @property
    def n_samples(self):
        """ Number of samples """
        if self.C_ is not None:
            return self.C_.shape[0]
        else:
            return None

    def _ismin_err(self, val):
        """ Is the current error the minimum """
        if len(self.err) == 0:
            return True
        else:
            return ([val > x for x in self.err].count(True) == 0)

    def fit(self, D, C=None, ST=None, st_fix=None, c_fix=None, c_first=True,
            verbose=False, post_iter_fcn=None, post_half_fcn=None):
        """
        Perform MCR-AR. D = CS^T. Solve for C and S^T iteratively.

        Parameters
        ----------
        D : ndarray
            D matrix

        C : ndarray
            Initial C matrix estimate. Only provide initial C OR S^T.

        ST : ndarray
            Initial S^T matrix estimate. Only provide initial C OR S^T.

        st_fix : list
            The spectral component numbers to keep fixed.

        c_fix : list
            The concentration component numbers to keep fixed.

        c_first : bool
            Calculate C first when both C and ST are provided. c_fix and st_fix
            must also be provided in this circumstance.

        verbose : bool
            Log iteration and per-least squares err results. See Notes.

        post_iter_fcn : function
            Function to perform after each iteration

        post_half_fcn : function
            Function to perform after half-iteration

        Notes
        -----

        -   Parameters to fit will SUPERCEDE anything in fit_kwargs, if provided during McrAR
            instantiation.
            -   Note that providing C (or ST) to fit_kwargs and providing ST (or C) to fit or
                fit_transform will raise an error.
            -   When in doubt, clear fit_kwargs via self.fit_kwargs = {}
            -   Does not affect verbose or c_first parameters

        -   pyMCR (>= 0.3.1) uses the native Python logging module
            rather than print statements; thus, to see the messages, one will
            need to log-to-file or stream to stdout. More info is available in
            the docs.

        """

        if verbose:
            _logger.setLevel(_logging.DEBUG)
        else:
            _logger.setLevel(_logging.INFO)

        if self.fit_kwargs:
            temp = self.fit_kwargs.get('C')
            if (temp is not None) & (C is None):
                C = temp
                
            temp = self.fit_kwargs.get('ST')
            if (temp is not None) & (ST is None):
                ST = temp
            
            temp = self.fit_kwargs.get('st_fix')
            if (temp is not None) & (st_fix is None):
                st_fix = temp

            temp = self.fit_kwargs.get('c_fix')
            if (temp is not None) & (c_fix is None):
                c_fix = temp

            temp = self.fit_kwargs.get('post_iter_fcn')
            if (temp is not None) & (post_iter_fcn is None):
                post_iter_fcn = temp

            temp = self.fit_kwargs.get('post_half_fcn')
            if (temp is not None) & (post_iter_fcn is None):
                post_half_fcn = temp

        # Ensure only C or ST provided
        if (C is None) & (ST is None):
            raise TypeError('C or ST estimate must be provided')
        elif (C is not None) & (ST is not None) & ((c_fix is None) |
                                                   (st_fix is None)):
            err_str1 = 'Only C or ST estimate must be provided, '
            raise TypeError(
                err_str1 + 'unless c_fix and st_fix are both provided')
        else:
            self.C_ = C
            self.ST_ = ST

        self.n_increase = 0
        self.n_above_min = 0
        self.err = []

        # Both C and ST provided. special_skip_c comes into play below
        both_condition = (self.ST_ is not None) & (self.C_ is not None)

        for num in range(self.max_iter):
            self.n_iter = num + 1

            # Both st and c provided, but c_first is False
            if both_condition & (num == 0) & (not c_first):
                special_skip_c = True
            else:
                special_skip_c = False

            if (self.ST_ is not None) & (not special_skip_c):
                # Debugging feature -- saves every S^T matrix in a list
                # Can create huge memory usage
                if self._saveall_st:
                    self._saved_st.append(self.ST_)

                # * Target is the feature of the regression
                self.c_regressor.fit(self.ST_.T, D.T, **self.c_fit_kwargs)
                C_temp = self.c_regressor.coef_

                # Apply fixed C's
                if c_fix:
                    C_temp[:, c_fix] = self.C_[:, c_fix]

                # Apply c-constraints
                for constr in self.c_constraints:
                    C_temp = constr.transform(C_temp)

                # Apply fixed C's
                if c_fix:
                    C_temp[:, c_fix] = self.C_[:, c_fix]

                D_calc = _np.dot(C_temp, self.ST_)

                err_temp = self.err_fcn(C_temp, self.ST_, D, D_calc)

                if self._ismin_err(err_temp):
                    self.C_opt_ = 1 * C_temp
                    self.ST_opt_ = 1 * self.ST_
                    self.n_iter_opt = num + 1
                    self.n_above_min = 0
                else:
                    self.n_above_min += 1

                if self.tol_n_above_min is not None:
                    if self.n_above_min > self.tol_n_above_min:
                        err_str1 = 'Half-iterated {} times since ' \
                                   'min '.format(self.n_above_min)
                        err_str2 = 'error. Exiting.'
                        _logger.info(err_str1 + err_str2)
                        self.exit_tol_n_above_min = True
                        break

                # Calculate error fcn and check for tolerance increase
                if len(self.err) == 0:
                    self.err.append(1 * err_temp)
                    self.C_ = 1 * C_temp
                elif self.tol_increase is None:
                    self.err.append(1 * err_temp)
                    self.C_ = 1 * C_temp
                elif err_temp <= self.err[-1] * (1 + self.tol_increase):
                    self.err.append(1 * err_temp)
                    self.C_ = 1 * C_temp
                else:
                    err_str1 = 'Error increased above fractional' \
                               'ctol_increase (C iter). Exiting'
                    _logger.info(err_str1)
                    self.exit_tol_increase = True
                    break

                # Check if err went up
                if len(self.err) > 1:
                    if self.err[-1] > self.err[-2]:  # Error increased
                        self.n_increase += 1
                    else:
                        self.n_increase *= 0

                # Break if too many error-increases in a row
                if self.tol_n_increase is not None:
                    if self.n_increase > self.tol_n_increase:
                        out_str1 = 'Maximum error increases reached '
                        _logger.info(
                            out_str1 + '({}) (C iter). '
                                       'Exiting.'.format(self.tol_n_increase))
                        self.exit_tol_n_increase = True
                        break

                _logger.debug('Iter: {} (C)\t{}: '
                              '{:.4e}'.format(self.n_iter,
                                              self.err_fcn.__name__,
                                              err_temp))

                if post_half_fcn is not None:
                    post_half_fcn(self.C_, self.ST_, D, D_calc)

            if self.C_ is not None:

                # Debugging feature -- saves every C matrix in a list
                # Can create huge memory usage
                if self._saveall_c:
                    self._saved_c.append(self.C_)

                # * Target is the feature of the regression
                self.st_regressor.fit(self.C_, D, **self.st_fit_kwargs)
                ST_temp = self.st_regressor.coef_.T

                # Apply fixed ST's
                if st_fix:
                    ST_temp[st_fix] = self.ST_[st_fix]

                # Apply ST-constraints
                for constr in self.st_constraints:
                    ST_temp = constr.transform(ST_temp)

                # Apply fixed ST's
                if st_fix:
                    ST_temp[st_fix] = self.ST_[st_fix]

                D_calc = _np.dot(self.C_, ST_temp)

                err_temp = self.err_fcn(self.C_, ST_temp, D, D_calc)

                # Calculate error fcn and check for tolerance increase
                if self._ismin_err(err_temp):
                    self.ST_opt_ = 1 * ST_temp
                    self.C_opt_ = 1 * self.C_
                    self.n_iter_opt = num + 1
                    self.n_above_min = 0
                else:
                    self.n_above_min += 1

                if self.tol_n_above_min is not None:
                    if self.n_above_min > self.tol_n_above_min:
                        err_str1 = 'Half-iterated {} times ' \
                                   'since min '.format(self.n_above_min)
                        err_str2 = 'error. Exiting.'
                        _logger.info(err_str1 + err_str2)
                        self.exit_tol_n_above_min = True
                        break

                if len(self.err) == 0:
                    self.err.append(1 * err_temp)
                    self.ST_ = 1 * ST_temp
                elif self.tol_increase is None:
                    self.err.append(1 * err_temp)
                    self.ST_ = 1 * ST_temp
                elif err_temp <= self.err[-1] * (1 + self.tol_increase):
                    self.err.append(1 * err_temp)
                    self.ST_ = 1 * ST_temp
                else:
                    err_str1 = 'Error increased above fractional ' \
                               'tol_increase (ST iter). Exiting'
                    _logger.info(err_str1)
                    self.exit_tol_increase = True
                    break

                # Check if err went up
                if len(self.err) > 1:
                    if self.err[-1] > self.err[-2]:  # Error increased
                        self.n_increase += 1
                    else:
                        self.n_increase *= 0

                # Break if too many error-increases in a row
                if self.tol_n_increase is not None:
                    if self.n_increase > self.tol_n_increase:
                        out_str = 'Maximum error increases reached '
                        _logger.info(out_str +
                                     '({}) (ST iter). '
                                     'Exiting.'.format(self.tol_n_increase))
                        self.exit_tol_n_increase = True
                        break

                _logger.debug('Iter: {} (ST)\t{}: '
                              '{:.4e}'.format(self.n_iter,
                                              self.err_fcn.__name__, err_temp))

                if post_half_fcn is not None:
                    post_half_fcn(self.C_, self.ST_, D, D_calc)

                if post_iter_fcn is not None:
                    post_iter_fcn(self.C_, self.ST_, D, D_calc)

            if self.n_iter >= self.max_iter:
                _logger.info('Max iterations reached ({}).'.format(num + 1))
                self.exit_max_iter_reached = True
                break

            self.n_iter = num + 1

            # Check if err changed (absolute value), per iteration, less
            #  than abs(tol_err_change)

            if (self.tol_err_change is not None) & (len(self.err) > 2):
                err_differ = _np.abs(self.err[-1] - self.err[-3])
                if err_differ < _np.abs(self.tol_err_change):
                    _logger.info('Change in err below tol_err_change '
                                 '({:.4e}). Exiting.'.format(err_differ))
                    self.exit_tol_err_change = True
                    break

    def fit_transform(self, D, **kwargs):
        """
        This performs the same purpose as the fit method, but returns the C_ matrix.
        Really, it's just to enable sklearn-expectant APIs compatible with pyMCR.

        It is recommended to use the fit method and retrieve your results from C_ and ST_

        See documentation for the fit method

        Returns
        --------

        C_ : ndarray
            C-matrix is returned

        """

        self.fit(D, **kwargs)

        return self.C_

    @property
    def components_(self):
        """ This is just provided for sklearn-like functionality """

        return self.ST_

if __name__ == '__main__':  # pragma: no cover
    # PyMCR uses the Logging facility to capture messaging
    # Sends logging messages to stdout (prints them)
    stdout_handler = _logging.StreamHandler(stream=_sys.stdout)
    stdout_format = _logging.Formatter('%(message)s')
    stdout_handler.setFormatter(stdout_format)
    _logger.addHandler(stdout_handler)

    M = 21
    N = 21
    P = 101
    n_components = 2

    C_img = _np.zeros((M, N, n_components))
    C_img[..., 0] = _np.dot(_np.ones((M, 1)), _np.linspace(0, 1, N)[None, :])
    C_img[..., 1] = 1 - C_img[..., 0]

    St_known = _np.zeros((n_components, P))
    St_known[0, 40:60] = 1
    St_known[1, 60:80] = 2

    C_known = C_img.reshape((-1, n_components))

    D_known = _np.dot(C_known, St_known)

    mcrar = McrAR()
    mcrar.fit(D_known, ST=St_known, verbose=True)
    # assert_equal(1, mcrar.n_iter_opt)
    assert ((mcrar.D_ - D_known) ** 2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known) ** 2).mean() < 1e-10

    mcrar = McrAR()
    mcrar.fit(D_known, C=C_known)
    # assert_equal(1, mcrar.n_iter_opt)
    assert ((mcrar.D_ - D_known) ** 2).mean() < 1e-10
    assert ((mcrar.D_opt_ - D_known) ** 2).mean() < 1e-10
