.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |CodeCov|_ |Py34|_ |Py35|_ |Py36|_ |PyPi|_

.. |Travis| image:: https://travis-ci.org/CCampJr/PyMCR.svg?branch=dev
.. _Travis: https://travis-ci.org/CCampJr/PyMCR

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/github/CCampJr/PyMCR?branch=dev&svg=true
.. _AppVeyor: https://ci.appveyor.com/project/CCampJr/PyMCR

.. |CodeCov| image:: https://codecov.io/gh/CCampJr/pyMCR/branch/dev/graph/badge.svg
.. _CodeCov: https://codecov.io/gh/CCampJr/pyMCR

.. |Py34| image:: https://img.shields.io/badge/Python-3.4-blue.svg
.. _Py34: https://www.python.org/downloads/

.. |Py35| image:: https://img.shields.io/badge/Python-3.5-blue.svg
.. _Py35: https://www.python.org/downloads/

.. |Py36| image:: https://img.shields.io/badge/Python-3.6-blue.svg
.. _Py36: https://www.python.org/downloads/

.. |PyPi| image:: https://badge.fury.io/py/PyMCR.svg
.. _PyPi: https://badge.fury.io/py/PyMCR


pyMCR: Multivariate Curve Resolution in Python
===============================================================

pyMCR is a small package for performing multivariate curve resolution.
Currently, it implements a simple alternating least squares method
(i.e., MCR-ALS).

MCR-ALS, in general, is a constrained implementation of alternating
least squares (ALS) nonnegative matrix factorization (NMF). Historically,
other names were used for MCR as well:

-   Self modeling mixture analysis (SMMA)
-   Self modeling curve resolution (SMCR)

Available methods:

-   Ordinary least squares with `Moore-Penrose pseudo-inverse 
    <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.pinv.html>`_ 
    (default, McrAls)
-   Ordinary least squares with `non-negative least squares 
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html>`_ (McrAls_NNLS)

What it **does** do:

-   Approximate the concentration and spectral matrices via minimization routines. 
    This is the core the MCR-ALS methods.
-   Enable the application of certain constraints (currently): sum-to-one, 
    non-negativity, normalization, maximum limits (closure)

What it **does not** do:

-   Estimate the number of components in the sample. This is a bonus feature in 
    some more-advanced MCR-ALS packages.

    - In MATLAB: https://mcrals.wordpress.com/
    - In R: https://cran.r-project.org/web/packages/ALS/index.html

Dependencies
------------

**Note**: These are the developmental system specs. Older versions of certain
packages may work.

-   python >= 3.4
    
    - Tested with 3.4.6, 3.5.4, 3.6.3

-   numpy (1.9.3)
    
    - Tested with 1.12.1, 1.13.1, 1.13.3

-   scipy (1.0.0)
    - Tested with 1.0.0

Known Issues
------------


Installation
------------

Using pip (hard install)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    # Only Python 3.* installed
    pip install pyMCR

    # If you have both Python 2.* and 3.* you may need
    pip3 install pyMCR

Using pip (soft install [can update with git])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::
    
    # Make new directory for pyMCR and enter it
    # Clone from github
    git clone https://github.com/CCampJr/pyMCR

    # Only Python 3.* installed
    pip install -e .

    # If you have both Python 2.* and 3.* you may need instead
    pip3 install -e .

    # To update in the future
    git pull

Using setuptools
~~~~~~~~~~~~~~~~

You will need to `download the repository <https://github.com/CCampJr/pyMCR>`_
or clone the repository with git:

.. code::
    
    # Make new directory for pyMCR and enter it
    # Clone from github
    git clone https://github.com/CCampJr/pyMCR

Perform the install:

.. code::

    python setup.py install

Usage
-----

.. code:: python

    from pymcr.mcr import McrAls
    mcrals = McrAls()
    
    # Data that you will provide
    # data [n_samples, n_features]  # Measurements
    #
    # initial_spectra [n_components, n_features]  ## S^T in the literature
    # OR
    # initial_conc [n_samples, n_components]   ## C in the literature

    # If you have an initial estimate of the spectra
    mcrals.fit(data, initial_spectra=initial_spectra)

    # Otherwise, if you have an initial estimate of the concentrations
    mcrals.fit(data, initial_conc=initial_conc)

Examples
--------

Command line and Jupyter notebook examples are provided in the ``Examples/`` folder.

From ``Examples/Demo.ipynb``:

.. image:: ./Examples/mcr_spectra_retr.png

.. image:: ./Examples/mcr_conc_retr.png
    
References
----------

-   `W. H. Lawton and E. A. Sylvestre, "Self Modeling Curve Resolution", 
    Technometrics **13**, 617–633 (1971). <https://www.jstor.org/stable/1267173>`_
-   https://mcrals.wordpress.com/theory/
-   `J. Jaumot, R. Gargallo, A. de Juan, and R. Tauler, "A graphical user-friendly 
    interface for MCR-ALS: a new tool for multivariate curve resolution in
    MATLAB", Chemometrics and Intelligent Laboratory Systems **76**, 101-110 
    (2005). <http://www.sciencedirect.com/science/article/pii/S0169743904002874>`_
-   `J. Felten, H. Hall, J. Jaumot, R. Tauler, A. de Juan, and A. Gorzsás, 
    "Vibrational spectroscopic image analysis of biological material using 
    multivariate curve resolution–alternating least squares (MCR-ALS)", Nature Protocols 
    **10**, 217-240 (2015). <https://www.nature.com/articles/nprot.2015.008>`_
    

Contact
-------
Charles H Camp Jr: `charles.camp@nist.gov <mailto:charles.camp@nist.gov>`_

Contributors
-------------
Charles H Camp Jr
