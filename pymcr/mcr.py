"""
Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)
"""

import numpy as _np

from pymcr.metrics import mse, mean_rel_dif as mrd

class McrAls:
    """Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)"""
    alg_list = ['auto', 'inv', 'cls']

    def __init__(self, tol_dif_spect=1e-6, tol_dif_conc=1e-6, tol_mse=1e-6, max_iter=50,
                 alg='auto', **kwargs):
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

        alg : str
            ALS algorithm type. See Notes

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
        -   Algorithm info goes here...
        -   Available constraints (bool) are 'nonnegative','max_lim', and 'sum_to_one'.
            Additionally, 'max_lim_const' is a modifier if the 'max_lim' = True.
            - 'nonnegative': both conc_ and spectra_ results must be >= 0
            - 'max_lim': conc_ values above 'max_lim_const' (another constraint keyword)
              are set to ='max_lim_const'. Also known as a **closure** constraint.
            - 'sum_to_one': conc_ values across n_components is 1.0 (i.e, the total
              concentration for a given sample is 1.0)

        """

        self.tol_dif_spect = tol_dif_spect
        self.tol_dif_conc = tol_dif_conc
        self.tol = tol_mse
        self.max_iter = max_iter

        self._alg = None

        self._st_last = None
        self._st_now = None

        self._c_last = None
        self._c_now = None

        self._initial_conc = None
        self._initial_spectra = None

        self._n_features = None
        self._n_samples = None
        self._n_components = None

        self.mse = None

        self.constraints = {'nonnegative': True,
                            'max_lim': True,
                            'max_lim_const': 1.0,
                            'sum_to_one': True}

        if len(kwargs):
            self.constraints.update(kwargs)
        else:
            pass

        self.alg = alg

    @property
    def alg(self):
        return self._alg

    @alg.setter
    def alg(self, value):
        if self.alg_list.count(value) != 0:
            self._alg = value
        else:
            raise TypeError('Unknown algorithm {}. Only {} allowed'.format(value, self.alg_list))

        self._set_alg_auto()

    @alg.deleter
    def alg(self):
        print('Cannot delete algorithm. Setting to \'auto\'')
        self.alg = 'auto'

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_components(self):
        return self._n_components

    @property
    def conc_(self):
        return self._c_now

    @property
    def spectra_(self):
        return self._st_now

    @property
    def initial_conc(self):
        return self._initial_conc

    @initial_conc.setter
    def initial_conc(self, value):
        n_rows_conc = value.shape[0]

        if  n_rows_conc == self._n_samples:
            self._initial_spectra = None
            self._initial_conc = 1*value
            self._n_components = self._initial_conc.shape[-1]
        else:
            raise ValueError('initial_conc rows ({}) does not match number of \
samples [data rows] ({})'.format(n_rows_conc, self._n_samples))

    @initial_conc.deleter
    def initial_conc(self):
        self._initial_conc = None

    @property
    def initial_spectra(self):
        return self._initial_spectra

    @initial_spectra.setter
    def initial_spectra(self, value):
        n_cols_spectra = value.shape[-1]

        if  n_cols_spectra == self._n_features:
            self._initial_conc = None
            self._initial_spectra = 1*value
            self._n_components = self._initial_spectra.shape[0]
        else:
            raise ValueError('initial_spectra columns ({}) does not match number of \
features [data cols] ({})'.format(n_cols_spectra, self._n_features))

    @initial_spectra.deleter
    def initial_spectra(self):
        self._initial_spectra = None

    def _set_alg_auto(self):
        """
        If 'auto' is selected for the ALS algorithm, select and explicit
        algorithm (as long as n_components, n_features, and n_components
        are already defined).

        """
        multiplier = 10

        if ((self._alg == 'auto') & (self._n_components is not None) & 
                (self._n_features is not None)):
            if multiplier*self._n_components < self._n_features:
                print('n_components >> n_features: using \'cls\'')
                self.alg = 'cls'
            else:
                self.alg = 'pinv'
                print('n_components is NOT << n_features: using \'pinv\'')

    def _data_info(self, data_shape):
        """
        Set the number of features and samples based on the input data's shape.

        Parameters
        ----------

        data_shape : list, tuple
            Shape of the input data.
        """

        if len(data_shape) == 2:
            self._n_features = data_shape[-1]
            self._n_samples = data_shape[0]
        else:
            raise ValueError('Only 2D data inputs supported [n_samples, n_features]')

        self._set_alg_auto()


    def fit(self, data, initial_conc=None, initial_spectra=None):
        """
        Parameters
        ----------

        data : 2D ndarray ([n_samples, n_features])

        initial_conc : 2D ndarray ([n_samples, n_components])
            Initial concentration guess

        initial_spectra : 2D ndarray ([n_components, n_features])
            Initial spectra guess


        Note
        ----

        - **Either** an initial_conc or initial_spectra must be provided. **Not** both.
        """

        if (initial_conc is None) & (initial_spectra is None):
            raise TypeError('fit() requires either an initial concentration or spectra estimate')
        elif (initial_conc is not None) & (initial_spectra is not None):
            raise TypeError('fit() requires either an initial concentration or spectra \
estimate, NOT both')

        self._data_info(data.shape)

        if initial_conc is not None:
            self.initial_conc = initial_conc
        elif initial_spectra is not None:
            self.initial_spectra = initial_spectra

        self._set_alg_auto()
        assert self.alg_list.count(self._alg) != 0, 'No definite algorithm defined'

        if self._initial_spectra is not None:
            self._st_now = 1*self._initial_spectra
            self._st_last = 1*self._initial_spectra

            self._c_now = _np.zeros((self._n_samples, self._n_components))
            self._c_last = _np.zeros((self._n_samples, self._n_components))
        else:
            self._c_now = 1*self._initial_conc
            self._c_last = 1*self._initial_conc

            self._st_now = _np.zeros((self._n_components, self._n_features))
            self._st_last = _np.zeros((self._n_components, self._n_features))

        self.mse = []
        for num in range(self.max_iter):
            if (num > 0) | ((self._initial_conc is None) & (num == 0)):
                self._c_last *= 0.0
                self._c_last += 1*self._c_now

                self._c_now *= 0.0
                if self._alg == 'cls':
                    self._c_now += _np.dot(_np.dot(data, self._st_now.T), 
                                           _np.linalg.pinv(_np.dot(self._st_now, self._st_now.T)))
                else:
                    self._c_now += _np.dot(data, _np.linalg.pinv(self._st_last))

                if self.constraints['nonnegative']:
                    self._c_now[_np.where(self._c_now < 0.0)] = 0.0

                if self.constraints['max_lim']:
                    self._c_now[_np.where(self._c_now > self.constraints['max_lim_const'])] = \
                        self.constraints['max_lim_const']

                if self.constraints['sum_to_one']:
                    self._c_now /= self._c_now.sum(axis=-1)[:,None]

            self._st_last *= 0
            self._st_last += 1*self._st_now

            self._st_now *= 0
            if self._alg == 'cls':
                self._st_now += _np.dot(_np.dot(_np.linalg.pinv(_np.dot(self._c_now.T, self._c_now)), self._c_now.T),
                                       data)
            else:
                self._st_now += _np.dot(_np.linalg.pinv(self._c_now), data)

            if self.constraints['nonnegative']:
                self._st_now[_np.where(self._st_now < 0.0)] = 0.0

            self.mse.append(mse(data, _np.dot(self._c_now,self._st_now)))
            print('iteration {} : MSE {:.2e}'.format(num+1, self.mse[-1]))

            c_mrd = mrd(self._c_now, self._c_last, only_non_zero=True)
            st_mrd = mrd(self._st_now, self._st_last, only_non_zero=True)
            
            if (self.mse[-1] <= self.tol) & (num > 0):
                print('MSE less than tolerance. Finishing...')
                break

            if c_mrd is not None:
                if (_np.abs(c_mrd) <= self.tol_dif_conc) & (num > 0):
                    print('Mean rel. diff. in concentration less than tolerance. Finishing...')
                    break
            
            if st_mrd is not None:
                if (_np.abs(st_mrd) <= self.tol_dif_spect) & (num > 0):
                    print(st_mrd)
                    print('Mean rel. diff. in spectra less than tolerance. Finishing...')
                    break
        
        self.mse = _np.array(self.mse)


if __name__ == '__main__':

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

    mcrals = McrAls(max_iter=100, tol_mse=1e-7, tol_dif_conc=1e-6, tol_dif_spect=1e-8)
    mcrals.fit(hsi.reshape((-1,wn.size)), initial_spectra=(spectra*wn))


