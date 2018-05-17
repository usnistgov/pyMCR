""" MCR-ALS Main Class for Computation"""
import numpy as _np

from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from pymcr.metrics import mse

class McrAls:
    """
    Multivariate Curve Resolution - Alternating Least Squares

    D = CS^T

    Parameters
    ----------
    c_regr : str, class
        Instantiated regression class (or string, see Notes) for calculating the
        C matrix

    st_regr : str, class
        Instantiated regression class (or string, see Notes) for calculating the
        S^T matrix

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
        calculaton (ie twice per iteration). Outputs to err attribute.

    tol_increase : float
        Factor increase to allow in err attribute. Set to 0 for no increase
        allowed. E.g., setting to 1.0 means the err can double per iteration.

    tol_n_increase : int
        Number of iterations for which the err attribute can increase

    tol_err_change : float
        If err changes less than tol_err_change, per iteration, break.

    Parameters
    ----------
    err : list
        List of calculated errors (from err_fcn) after each least squares (ie
        twice per iteration)

    C_ : ndarray
        Most recently calculated C matrix (that did not cause a tolerance
        failure)

    ST_ : ndarray
        Most recently calculated S^T matrix (that did not cause a tolerance
        failure)

    C_opt_ : ndarray
        [Optimal] C matrix for lowest err attribute

    ST_opt_ : ndarray
        [Optimal] ST matrix for lowest err attribute

    n_iter : int
        Total number of iterations performed

    n_iter_opt : int
        Iteration when optimal C and ST calculated

    max_iter_reached : bool
        Was the maximum number of iteration reached (max_iter parameter)

    Notes
    -----
    Built-in regressor classes (str can be used): OLS (ordinary least squares),
    NNLS (non-negatively constrained least squares). See mcr.regressors.

    Built-in regressor methods can be given as a string to c_regr, st_regr;
    though instantiating an imported class gives more flexibility.
    """
    def __init__(self, c_regr=OLS(), st_regr=OLS(), c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=50, err_fcn=mse,
                 tol_increase=0.0, tol_n_increase=10, tol_err_change=None
                ):
        """
        Multivariate Curve Resolution - Alternating Regression
        """

        self.max_iter = max_iter
        self.tol_increase = tol_increase
        self.tol_n_increase = tol_n_increase
        self.tol_err_change = tol_err_change

        self.err_fcn = err_fcn
        self.err = []

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
        self.max_iter_reached = False

        # Saving every C or S^T matrix at each iteration
        # Could create huge memory usage
        self._saveall_st = False
        self._saveall_c = False
        self._saved_st = []
        self._saved_c = []

    def _check_regr(self, mth):
        """
            Check regressor method. If accetible strings, instantiate and return
            object. If instantiated class, make sure it has a fit attribute.
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
            raise ValueError('Input class {} does not have a \'fit\' method'.format(mth))


    @property
    def D_(self):
        """ D matrix with current C and S^T matrices """
        return _np.dot(self.C_, self.ST_)

    @property
    def D_opt_(self):
        """ D matrix with optimal C and S^T matrices """
        return _np.dot(self.C_opt_, self.ST_opt_)

    def _ismin_err(self, val):
        """ Is the current error the minimum """
        if len(self.err) == 0:
            return True
        else:
            return ([val > x for x in self.err].count(True) == 0)


    def fit(self, D, C=None, ST=None, st_fix=None, verbose=False,
            post_iter_fcn=None, post_half_fcn=None):
        """
        Perform MCR-ALS. D = CS^T. Solve for C and S^T iteratively.

        Parameters
        ----------
        D : ndarray
            D matrix

        C : ndarray
            Initial C matrix estimate. Only provide initial C OR S^T.

        ST : ndarray
            Initial S^T matrix estimate. Only provide initial C OR S^T.

        st_fix : list
            The component numbers to keep fixed.

        verbose : bool
            Display iteration and per-least squares err results.

        post_iter_fcn : function
            Function to perform after each iteration

        post_half_fcn : function
            Function to perform after half-iteration
        """

        # Ensure only C or ST provided
        if (C is None) & (ST is None):
            raise TypeError('C or ST estimate must be provided')
        elif (C is not None) & (ST is not None):
            raise TypeError('Only C or ST estimate must be provided, only one')
        else:
            self.C_ = C
            self.ST_ = ST

        self.n_increase = 0

        for num in range(self.max_iter):
            self.n_iter = num + 1
            if self.ST_ is not None:
                # Debugging feature -- saves every S^T matrix in a list
                # Can create huge memory usage
                if self._saveall_st:
                    self._saved_st.append(self.ST_)

                self.c_regressor.fit(self.ST_.T, D.T, **self.c_fit_kwargs)
                C_temp = self.c_regressor.coef_

                # Apply c-constraints
                for constr in self.c_constraints:
                    C_temp = constr.transform(C_temp)

                D_calc = _np.dot(C_temp, self.ST_)

                err_temp = self.err_fcn(C_temp, self.ST_, D, D_calc)

                if self._ismin_err(err_temp):
                    self.C_opt_ = 1*C_temp
                    self.ST_opt_ = 1*self.ST_
                    self.n_iter_opt = num + 1

                # Calculate error fcn and check for tolerance increase
                if self.err != 0:
                    self.err.append(1*err_temp)
                    self.C_ = 1*C_temp
                elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                    self.err.append(1*err_temp)
                    self.C_ = 1*C_temp
                else:
                    print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
                    break

                # Check if err went up
                if len(self.err) > 1:
                    if self.err[-1] > self.err[-2]:  # Error increased
                        self.n_increase += 1
                    else:
                        self.n_increase *= 0

                # Break if too many error-increases in a row
                if self.n_increase > self.tol_n_increase:
                    print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                    break


                if verbose:
                    print('Iter: {} (C)\t{}: {:.4e}'.format(self.n_iter, self.err_fcn.__name__, err_temp))

                if post_half_fcn is not None:
                    post_half_fcn(self.C_, self.ST_, D, D_calc)

            if self.C_ is not None:

                # Debugging feature -- saves every C matrix in a list
                # Can create huge memory usage
                if self._saveall_c:
                    self._saved_c.append(self.C_)

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
                    self.ST_opt_ = 1*ST_temp
                    self.C_opt_ = 1*self.C_
                    self.n_iter_opt = num + 1

                if len(self.err) == 0:
                    self.err.append(1*err_temp)
                    self.ST_ = 1*ST_temp
                elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                    self.err.append(1*err_temp)
                    self.ST_ = 1*ST_temp
                else:
                    print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
                    break

                # Check if err went up
                if len(self.err) > 1:
                    if self.err[-1] > self.err[-2]:  # Error increased
                        self.n_increase += 1
                    else:
                        self.n_increase *= 0

                # Break if too many error-increases in a row
                if self.n_increase > self.tol_n_increase:
                    print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                    break

                if verbose:
                    print('Iter: {} (ST)\t{}: {:.4e}'.format(self.n_iter, self.err_fcn.__name__, err_temp))

                if post_half_fcn is not None:
                    post_half_fcn(self.C_, self.ST_, D, D_calc)

                if post_iter_fcn is not None:
                    post_iter_fcn(self.C_, self.ST_, D, D_calc)

            if self.n_iter >= self.max_iter:
                print('Max iterations reached ({}).'.format(num+1))
                self.max_iter_reached = True
                break

            self.n_iter = num + 1

            # Check if err changed (absolute value), per iteration, less
            #  than abs(tol_err_change)

            if ((self.tol_err_change is not None) & (len(self.err) > 2)):
                err_differ = _np.abs(self.err[-1] - self.err[-3])
                if err_differ < _np.abs(self.tol_err_change):
                    print('Change in err below tol_err_change ({:.4e}). Exiting.'.format(err_differ))
                    break


if __name__ == '__main__':  # pragma: no cover

    M = 21
    N = 21
    P = 101
    n_components = 3

    C_img = _np.zeros((M,N,n_components))
    C_img[...,0] = _np.dot(_np.ones((M,1)),_np.linspace(0,1,N)[None,:])
    C_img[...,1] = _np.dot(_np.linspace(0,1,M)[:, None], _np.ones((1,N)))
    C_img[...,2] = 1 - C_img[...,0] - C_img[...,1]
    C_img = C_img / C_img.sum(axis=-1)[:,:,None]

    ST_known = _np.zeros((n_components, P))
    ST_known[0,30:50] = 1.0
    ST_known[1,50:70] = 2.0
    ST_known[2,70:90] = 3.0
    ST_known += 1.0

    C_known = C_img.reshape((-1, n_components))

    D_known = _np.dot(C_known, ST_known)

    mcrals = McrAls(max_iter=50, tol_increase=100, tol_n_increase=10,
                    c_regr='NNLS', st_regr='NNLS',
                    st_constraints=[ConstraintNonneg()],
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                    tol_err_change=1e-8)

    ST_guess = 1 * ST_known
    ST_guess[1, :] = _np.random.rand(P)
    ST_guess[2, :] = _np.random.rand(P)
    mcrals.fit(D_known, ST=ST_guess, st_fix=[0])

    print(mcrals.ST_opt_[0,28:33])
    print(mcrals.ST_opt_[1,48:53])
    print(mcrals.ST_opt_[2,68:73])

    print(_np.allclose(mcrals.C_opt_.sum(axis=-1),1))

    # print(mcrals.err)