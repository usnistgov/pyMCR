"""
Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)
"""

from timeit import default_timer as _timer
import numpy as _np

from pymcr.abstractmcr import AbstractMcrAls

from pymcr.metrics import mse, mean_rel_dif as mrd
from pymcr.npmethods import ols_c, ols_s, nnls_c, nnls_s


class McrAls(AbstractMcrAls):
    """
    Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)
    """
    # alg_list = ['auto', 'pinv', 'cls']

    def __init__(self, tol_dif_spect=1e-6, tol_dif_conc=1e-6, tol_mse=1e-6, 
                 max_iter=50, tol_mrd_st=None, tol_mrd_c=None, **kwargs):
        """
        MCR-ALS: Multivariate Curve Resolution - Alternating Least Squares

        Parameters
        ----------

        tol_dif_spect : float
            Tolerance of difference between retrieved spectra between iterations.

        tol_dif_conc : float
            Tolerance of difference between retrieved concentrations between iterations.

        tol_mse : float
            Tolerance of mean squared error (MSE) value between iterations.

        max_iter : int
            Maximum number of iterations

        tol_mrd_st : float
            Absolute spectral MRD limit. If the |mrd| is greater than this 
            threshold, do not update spectra with new values. If None, always 
            update.
        
        tol_mrd_c : float
            Absolute concentration MRD limit. If the |mrd| is greater than this 
            threshold, do not update concentration with new values. 
            If None, always update.

        kwargs : dict
            Sets and controls the constraints of the ALS algorithm. See Notes.

        Attributes
        ----------

        n_features : int
            Number of features (for a spectrum, this is the number of frequencies)

        n_samples : int
            Number of provided samples (spectra)

        n_components : int
            Number of endmembers/components to solve for

        mse : ndarray (1D)
            Residual sum-of-squares for each iteration

        conc_ : 2D ndarray ([n_samples, n_components])
            Concentration

        spectra_ : 2D ndarray ([n_components, n_features])


        Methods
        -------

        fit : run the MCR-ALS algorithm

        Notes
        -----
        -   Available constraints (bool) are 'nonnegative', 'c_nonnegative', 
            's_nonegative', max_lim', and 'sum_to_one'.
            Additionally, 'max_lim_const' is a modifier if the 'max_lim' = True.
            - If 'nonnegative' is True, 'c_*' and 's_*' - nonnegative must be 
              true. If 'nonnegative' is False, 'c_*' and 's_*' will follow how
              they are set (which may be True and True, respectively)
            - 'nonnegative': both conc_ and spectra_ results must be >= 0
            - 'max_lim': conc_ values above 'max_lim_const' (another constraint keyword)
              are set to ='max_lim_const'. Also known as a **closure** constraint.
            - 'sum_to_one': conc_ values across n_components is 1.0 (i.e, the total
              concentration for a given sample is 1.0)

        """
        super().__init__(tol_dif_spect=tol_dif_spect,
                         tol_dif_conc=tol_dif_conc,
                         tol_mse=tol_mse, 
                         max_iter=max_iter,
                         tol_mrd_st=tol_mrd_st,
                         tol_mrd_c=tol_mrd_c,
                         **kwargs)

        self.alg_c = ols_c
        self.alg_st = ols_s

class McrAls_NNLS(AbstractMcrAls):
    """
    Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)
    using Nonnegative Least Squares approach (CLS)
    """
    # alg_list = ['auto', 'pinv', 'cls']

    def __init__(self, tol_dif_spect=1e-6, tol_dif_conc=1e-6, tol_mse=1e-6, 
                 max_iter=50, tol_mrd_st=None, tol_mrd_c=None, **kwargs):
        """
        MCR-ALS: Multivariate Curve Resolution - Alternating Least Squares

        Parameters
        ----------

        tol_dif_spect : float
            Tolerance of difference between retrieved spectra between iterations.

        tol_dif_conc : float
            Tolerance of difference between retrieved concentrations between iterations.

        tol_mse : float
            Tolerance of mean squared error (MSE) value between iterations.

        max_iter : int
            Maximum number of iterations

        tol_mrd_st : float
            Absolute spectral MRD limit. If the |mrd| is greater than this 
            threshold, do not update spectra with new values. If None, always 
            update.
        
        tol_mrd_c : float
            Absolute concentration MRD limit. If the |mrd| is greater than this 
            threshold, do not update concentration with new values. 
            If None, always update.

        kwargs : dict
            Sets and controls the constraints of the ALS algorithm. See Notes.

        Attributes
        ----------

        n_features : int
            Number of features (for a spectrum, this is the number of frequencies)

        n_samples : int
            Number of provided samples (spectra)

        n_components : int
            Number of endmembers/components to solve for

        mse : ndarray (1D)
            Residual sum-of-squares for each iteration

        conc_ : 2D ndarray ([n_samples, n_components])
            Concentration

        spectra_ : 2D ndarray ([n_components, n_features])


        Methods
        -------

        fit : run the MCR-ALS algorithm

        Notes
        -----
        -   Available constraints (bool) are 'nonnegative', 'c_nonnegative', 
            's_nonegative', max_lim', and 'sum_to_one'.
            Additionally, 'max_lim_const' is a modifier if the 'max_lim' = True.
            - If 'nonnegative' is True, 'c_*' and 's_*' - nonnegative must be 
              true. If 'nonnegative' is False, 'c_*' and 's_*' will follow how
              they are set (which may be True and True, respectively)
            - 'nonnegative': both conc_ and spectra_ results must be >= 0
            - 'max_lim': conc_ values above 'max_lim_const' (another constraint keyword)
              are set to ='max_lim_const'. Also known as a **closure** constraint.
            - 'sum_to_one': conc_ values across n_components is 1.0 (i.e, the total
              concentration for a given sample is 1.0)

        """
        super().__init__(tol_dif_spect=tol_dif_spect, 
                            tol_dif_conc=tol_dif_conc, 
                            tol_mse=tol_mse, 
                            max_iter=max_iter, 
                            tol_mrd_st=tol_mrd_st,
                            tol_mrd_c=tol_mrd_c, 
                            **kwargs)

        self.alg_c = nnls_c
        self.alg_st = nnls_s
        
if __name__ == '__main__':  # pragma: no cover

    n_colors = 200
    wn = _np.linspace(400,2800,n_colors)

    n_components = 3
    sp0 = _np.exp(-(wn-1200)**2/(2*200**2))
    sp1 = _np.exp(-(wn-1600)**2/(2*200**2))
    sp2 = _np.exp(-(wn-2000)**2/(2*200**2))

    M = 80
    N = 120
    n_spectra = M*N

    conc = _np.zeros((M,N,n_components))

    x0 = 25
    y0 = 25

    x1 = 75
    y1 = 25

    x2 = 50
    y2 = 50

    R = 20

    X,Y = _np.meshgrid(_np.arange(N), _np.arange(M))

    conc[...,0] = _np.exp(-(X-x0)**2/(2*R**2))*_np.exp(-(Y-y0)**2/(2*R**2))
    conc[...,1] = _np.exp(-(X-x1)**2/(2*R**2))*_np.exp(-(Y-y1)**2/(2*R**2))
    conc[...,2] = _np.exp(-(X-x2)**2/(2*R**2))*_np.exp(-(Y-y2)**2/(2*R**2))

    conc /= conc.sum(axis=-1)[:,:,None]

    for num in range(n_components):
        idx = _np.unravel_index(conc[...,num].argmax(), conc[...,num].shape)
        tmp = _np.zeros(3)
        tmp[num] = 1
        conc[idx[0],idx[1],:] = 1*tmp
    
    spectra = _np.vstack((sp0, sp1, sp2))
    hsi = _np.dot(conc, spectra)

    # mcrals = McrAls_NNLS(max_iter=10, tol_mse=1e-7, tol_dif_conc=1e-6, tol_dif_spect=1e-8)
    mcrals = McrAls(max_iter=100, tol_mse=1e-7, tol_dif_conc=1e-6, 
                    tol_dif_spect=1e-8, tol_mrd_st=3.2e3)
    # print(mcrals.__dict__)
    mcrals.fit(hsi.reshape((-1,wn.size)), initial_spectra=(spectra*wn))
    # mcrals.fit(hsi.reshape((-1,wn.size)), initial_conc=conc.reshape((-1,n_components)))
    # print(mcrals._c_mrd)
    print(mcrals._st_mrd)
    # print(mcrals.tol_mrd_st)
    # print(mcrals._tmr)


