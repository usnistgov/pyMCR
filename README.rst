pyMCR: Multivariate Curve Resolution in Python
===============================================================

pyMCR is a small package for performing multivariate curve resolution.
Currently, it implements a simple alternating least squares method
(i.e., MCR-ALS).

Dependencies
------------

**Note**: These are the developmental system specs. Older versions of certain
packages may work.

-   python >= 3.4
    
    - Tested with 3.4.4, 3.5.2, 3.6.1

-   numpy (1.9.3)
    
    - Tested with 1.11.3+mkl

-   scipy

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
    
    # Make new directory for sciplot-pyqt and enter it
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

Perform the install **without building the documentation**:

.. code::

    python setup.py install

Perform the install **and build the documentation** (see dependencies above):

.. code::

    python setup.py build_sphinx
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
    
Contact
-------
Charles H Camp Jr: `charles.camp@nist.gov <mailto:charles.camp@nist.gov>`_

Contributors
-------------
Charles H Camp Jr
