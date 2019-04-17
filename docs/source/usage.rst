Usage and Examples
==================

Some examples are also available in Jupyter Notebook format in the ``Examples/`` directory of the repository.

Basic Usage
-----------

.. code:: python

    from pymcr.mcr import McrAR
    mcrals = McrAR()
    
    # MCR assumes a system of the form: D = CS^T
    #
    # Data that you will provide (hyperspectral context):
    # D [n_pixels, n_frequencies]  # Hyperspectral image unraveled in space (2D)
    #
    # initial_spectra [n_components, n_frequencies]  ## S^T in the literature
    # OR
    # initial_conc [n_pixels, n_components]   ## C in the literature

    # If you have an initial estimate of the spectra
    mcrals.fit(D, ST=initial_spectra)

    # Otherwise, if you have an initial estimate of the concentrations
    mcrals.fit(D, C=initial_conc)


Example: 2D-Gradient with Ordinary Least-Squares and Non-Negativity Constraint
-------------------------------------------------------------------------------

.. code:: python

    from pymcr.mcr import McrAR
    mcrals = McrAR()
    
    # MCR assumes a system of the form: D = CS^T
    #
    # Data that you will provide (hyperspectral context):
    # D [n_pixels, n_frequencies]  # Hyperspectral image unraveled in space (2D)
    #
    # initial_spectra [n_components, n_frequencies]  ## S^T in the literature
    # OR
    # initial_conc [n_pixels, n_components]   ## C in the literature

    # If you have an initial estimate of the spectra
    mcrals.fit(D, ST=initial_spectra)

    # Otherwise, if you have an initial estimate of the concentrations
    mcrals.fit(D, C=initial_conc)

Example: 2D-Gradient with Non-Negatively Constrained Least-Squares
-------------------------------------------------------------------

.. code:: python

    from pymcr.mcr import McrAR
    mcrals = McrAR()
    
    # MCR assumes a system of the form: D = CS^T
    #
    # Data that you will provide (hyperspectral context):
    # D [n_pixels, n_frequencies]  # Hyperspectral image unraveled in space (2D)
    #
    # initial_spectra [n_components, n_frequencies]  ## S^T in the literature
    # OR
    # initial_conc [n_pixels, n_components]   ## C in the literature

    # If you have an initial estimate of the spectra
    mcrals.fit(D, ST=initial_spectra)

    # Otherwise, if you have an initial estimate of the concentrations
    mcrals.fit(D, C=initial_conc)
