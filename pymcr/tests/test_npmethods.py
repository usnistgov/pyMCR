"""
Tests using the npmethods
"""

import numpy as np
import numpy.testing
import pytest

class _phantom:
    def __init__(self):
        self.n_colors = 200
        self.wn = np.linspace(400,2800,self.n_colors)

        self.n_components = 3
        sp0 = np.exp(-(self.wn-1200)**2/(2*200**2))
        sp1 = np.exp(-(self.wn-1600)**2/(2*200**2))
        sp2 = np.exp(-(self.wn-2000)**2/(2*200**2))
        self.spectra = np.vstack((sp0, sp1, sp2))

        self.M = 80
        self.N = 120
        self.n_spectra = self.M * self.N

        self.conc = np.zeros((self.M,self.N,self.n_components))

        x0 = 25
        y0 = 25

        x1 = 75
        y1 = 25

        x2 = 50
        y2 = 50

        R = 20

        X,Y = np.meshgrid(np.arange(self.N), np.arange(self.M))

        self.conc[...,0] = np.exp(-(X-x0)**2/(2*R**2))*np.exp(-(Y-y0)**2/(2*R**2))
        self.conc[...,1] = np.exp(-(X-x1)**2/(2*R**2))*np.exp(-(Y-y1)**2/(2*R**2))
        self.conc[...,2] = np.exp(-(X-x2)**2/(2*R**2))*np.exp(-(Y-y2)**2/(2*R**2))

        self.conc /= self.conc.sum(axis=-1)[:,:,None]

        for num in range(self.n_components):
            idx = np.unravel_index(self.conc[...,num].argmax(), self.conc[...,num].shape)
            tmp = np.zeros(3)
            tmp[num] = 1
            self.conc[idx[0],idx[1],:] = 1*tmp
        
        self.hsi = np.dot(self.conc, self.spectra)

@pytest.fixture
def phantom():
    return _phantom()
    
def test_mcr_spectral_guess(phantom):
    """ Basic test with spectral guessing"""

    from pymcr.mcr import McrAls

    mcrals = McrAls(max_iter=200, tol_mse=1e-7, tol_dif_conc=1e-7, 
                    tol_dif_spect=1e-7)

    initial_guess = (phantom.spectra * phantom.wn)
    mcrals.fit(phantom.hsi.reshape((-1,phantom.wn.size)), 
                                   initial_spectra=initial_guess)

    assert mcrals._n_iters < 60
    assert mcrals.mse[-1] < 1e-7
    assert np.abs(mcrals._c_mrd) > 1e-6
    assert np.abs(mcrals._st_mrd) > 1e-6
    assert np.abs(mcrals._c_mrd) < 1e-2
    assert np.abs(mcrals._st_mrd) < 1e-2
    # np.testing.assert_approx_equal(mcrals._c_mrd, 0.000754519972116, significant=3)
    # np.testing.assert_approx_equal(mcrals._st_mrd, -0.00339178932195, significant=3)

def test_mcr_conc_guess(phantom):
    """ Basic test with concentration guessing"""

    from pymcr.mcr import McrAls

    
    initial_guess = np.zeros((phantom.M,phantom.N,phantom.n_components))
    initial_guess[...,0] = np.dot(np.ones(phantom.M)[:,None], (np.arange(phantom.N,0,-1)/phantom.N)[None,:])
    initial_guess[...,1] = np.dot(np.ones(phantom.M)[:,None], (np.arange(phantom.N)/phantom.N)[None,:])
    initial_guess[...,2] = np.dot((np.arange(phantom.M)/phantom.M)[:,None], np.ones(phantom.N)[None,:])
    initial_guess += np.random.rand(phantom.M,phantom.N,3)

    # reshape
    initial_guess = initial_guess.reshape((-1, phantom.n_components))

    mcrals = McrAls(max_iter=200, tol_mse=1e-7, tol_dif_conc=1e-7, 
                    tol_dif_spect=1e-7)
    mcrals.fit(phantom.hsi.reshape((-1,phantom.wn.size)), 
                                   initial_conc=initial_guess)

    assert mcrals._n_iters < 70
    assert mcrals.mse[-1] < 1e-7
    assert np.abs(mcrals._c_mrd) > 1e-6
    assert np.abs(mcrals._st_mrd) > 1e-6
    assert np.abs(mcrals._c_mrd) < 1e-2
    assert np.abs(mcrals._st_mrd) < 1e-2
