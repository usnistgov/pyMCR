import numpy as _np

class McrAls:
    alg_list = ['auto', 'inv', 'tinv']
    
    """Simple implementation of Alternating Least-Squares Multivariate Curve Resolution (ALS-MCR)"""
    
    def __init__(self, tol_dif_spect=0.1, tol_dif_conc=0.1, tol_rss=1e-6, max_iter=50, 
                 alg='auto', **kwargs_constraints):
        
        self.tol_dif_spect = tol_dif_spect
        self.tol_dif_conc = tol_dif_conc
        self.tol = tol_rss
        self.max_iter = max_iter
        
        self._alg = None
                
        self._ST_last = None
        self._ST_now = None
        
        self._C_last = None
        self._C_now = None
        
        self._initial_conc = None
        self._initial_spectra = None
        
        self._n_features = None
        self._n_samples = None
        self._n_components = None
        
        self.rss = None
        
        self.constraints = {'nonnegative': True,
                            'max_lim': True,
                            'max_lim_const': 1.0,
                            'sum_to_one': True}
        
        if len(kwargs_constraints):
            self.constraints.update(kwargs_constraints)
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
            
        self.set_alg_auto()
            
    @alg.deleter
    def alg(self):
        print('Cannot delete algorithm. Setting to \'auto\'')
        self.alg = 'auto'
        
#     @property
#     def data(self):
#         return self._data
    
#     @data.setter
#     def data(self, value):
#         if value is not None:
#             if value.ndim == 2:
# #                 self._data = 1*value
#                 self._n_features = self._data.shape[-1]
#                 self._n_samples = self._data.shape[0]
#             else:
#                 raise ValueError('Only 2D data inputs supported [n_samples, n_features]')

#             self.set_alg_auto()
        
#     @data.deleter
#     def data(self):
#         self._data = None
#         self._n_features = None
#         self._n_samples = None
        
    @property
    def n_features(self):
        return self._n_features
    
    @property
    def n_samples(self):
        return self._n_samples
    
    @property
    def n_components(self):
        return self._n_components
    
#     @property
#     def ST_last(self):
#         return self._ST_last
    
#     @property
#     def ST_now(self):
#         return self._ST_now
    
#     @property
#     def C_last(self):
#         return self._C_last
    
#     @property
#     def C_now(self):
#         return self._C_now

    @property
    def conc_(self):
        return self._C_now
    
    @property
    def spectra_(self):
        return self._ST_now
        
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
        
    def set_alg_auto(self):
        multiplier = 10
        
        if (self._alg == 'auto') & (self._n_components is not None) & (self._n_features is not None):
            if multiplier*self._n_components < self._n_features:
                print('n_components >> n_features: using \'tinv\'')
                self.alg = 'tinv'
            else:
                self.alg = 'pinv'
                print('n_components is NOT << n_features: using \'pinv\'')
                
    def data_info(self, data_shape):
        if len(data_shape) == 2:
            self._n_features = data_shape[-1]
            self._n_samples = data_shape[0]
        else:
            raise ValueError('Only 2D data inputs supported [n_samples, n_features]')

        self.set_alg_auto()
        
                
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
            raise TypeError('fit() requires either an initial concentration or spectra estimate, NOT both')
            
        self.data_info(data.shape)
            
        if initial_conc is not None:
            self.initial_conc = initial_conc
        elif initial_spectra is not None:
            self.initial_spectra = initial_spectra
            
        self.set_alg_auto()
        assert self.alg_list.count(self._alg) != 0, 'No definite algorithm defined'
        
        if self._initial_spectra is not None:
            self._ST_now = 1*self._initial_spectra
            self._ST_last = 1*self._initial_spectra
            
            self._C_now = _np.zeros((self._n_samples, self._n_components))
            self._C_last = _np.zeros((self._n_samples, self._n_components))
        else:
            self._C_now = 1*self._initial_conc
            self._C_last = 1*self._initial_conc
            
            self._ST_now = _np.zeros((self._n_components, self._n_features))
            self._ST_last = _np.zeros((self._n_components, self._n_features))
            
        self.rss = []
        for num in range(self.max_iter):
            if (num > 0) | ((self._initial_conc is None) & (num == 0)):
                self._C_last *= 0.0
                self._C_last += 1*self._C_now

                self._C_now *= 0.0
                if self._alg == 'tinv':
                    self._C_now += _np.dot(_np.dot(data, self._ST_now.T), _np.linalg.pinv(_np.dot(self._ST_now, self._ST_now.T)))
                else:
                    self._C_now += _np.dot(data, _np.linalg.pinv(self._ST_last))

                if self.constraints['nonnegative']:    
                    self._C_now[_np.where(self._C_now < 0.0)] = 0.0

                if self.constraints['max_lim']:
                    self._C_now[_np.where(self._C_now > self.constraints['max_lim_const'])] = \
                        self.constraints['max_lim_const']

                if self.constraints['sum_to_one']:
                    self._C_now /= self._C_now.sum(axis=-1)[:,None]

            self._ST_last *= 0
            self._ST_last += 1*self._ST_now
            
            self._ST_now *= 0
            if self._alg == 'tinv':
                self._ST_now += _np.dot(_np.dot(_np.linalg.pinv(_np.dot(self._C_now.T, self._C_now)), self._C_now.T),
                                       data)
            else:
                self._ST_now += _np.dot(_np.linalg.pinv(self._C_now), data)
                
            if self.constraints['nonnegative']:
                self._ST_now[_np.where(self._ST_now < 0.0)] = 0.0

            self.rss.append(_np.sum((data - _np.dot(self._C_now,self._ST_now))**2))
            # /(self._n_features*self._n_samples)
            print('iteration {} : RSS {:.2e}'.format(num+1, self.rss[-1]))
            
            if (self.rss[-1] <= self.tol) & (num > 0):
                break    
        self.rss = _np.array(self.rss)
            
   
        
            
        