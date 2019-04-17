Usage and Examples
==================

Some examples are also available in Jupyter Notebook format in the ``Examples/`` directory of the repository.

Basic Usage
-----------

.. code:: python

    from pymcr.mcr import McrAR
    mcrar = McrAR()
    
    # MCR assumes a system of the form: D = CS^T
    #
    # Data that you will provide (hyperspectral context):
    # D [n_pixels, n_frequencies]  # Hyperspectral image unraveled in space (2D)
    #
    # initial_spectra [n_components, n_frequencies]  ## S^T in the literature
    # OR
    # initial_conc [n_pixels, n_components]   ## C in the literature

    # If you have an initial estimate of the spectra
    mcrar.fit(D, ST=initial_spectra)

    # Otherwise, if you have an initial estimate of the concentrations
    mcrar.fit(D, C=initial_conc)


Default: Ordinary Least-Squares (OLS) with Non-Negativity Constraints
---------------------------------------------------------------------

Using OLS and post-iterative non-negativity constraints is a classic way of performing MCR-ALS. 
OLS algorithms are often faster in practice, but NNLS method may converge MCR faster.

.. code:: python

    from pymcr.mcr import McrAR
    mcrar = McrAR()

    # Equivalent to

    mcrar = McrAR(c_regr='OLS', st_regr='OLS')

    # Equivalent to

    from pymcr.mcr import McrAR
    from pymcr.regressors import OLS
    from pymcr.constraints import ConstraintNonneg

    mcrar = McrAR(c_regr=OLS(), st_regr=OLS(), c_constraints=[ConstraintNonneg()],
                  st_constraints=[ConstraintNonneg()])

    # Then use the fit method with initial guesses of ST or C.

Non-Negative Least-Squares (NNLS), No Constraints
--------------------------------------------------

Using OLS and post-iterative non-negativity constraints is a classic way of performing MCR-ALS. 
OLS algorithms are often faster in practice, but NNLS method may converge MCR faster.

.. code:: python

    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS

    mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), c_constraints=[], st_constraints=[])

    # Then use the fit method with initial guesses of ST or C.

Mix NNLS and OLS Regressors, Non-Negativity and Sum-to-1 (i.e., Norm) Constraints
----------------------------------------------------------------------------------

In this example, NNLS is used to regress the signature/spectrum and OLS is used for
abundance/concentration. Additionally, the ConstraintNorm imposes that after each iteration
the summation of all concentrations for each data point sums-to-one. This is common and
physical for many use-cases such as spectroscopy (as long as all analytes have a signature
[i.e., no 0-intensity signatures])

.. code:: python

    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS, OLS
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm

    # Note constraint order matters
    mcrar = McrAR(max_iter=100, st_regr='NNLS', c_regr='OLS', 
                  c_constraints=[ConstraintNonneg(), ConstraintNorm()])

    # Equivalent to

    # Note constraint order matters
    mcrar = McrAR(max_iter=100, st_regr=NNLS(), c_regr=OLS(), 
                  c_constraints=[ConstraintNonneg(), ConstraintNorm()])

    # Then use the fit method with initial guesses of ST or C.

Ridge-Regression from Scikit-Learn
----------------------------------

In this example, NNLS is used to regress the signatures/spectra but a ridge
regression regressor is imported from `scikit-learn <http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression>`_

.. code:: python

    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm

    from sklearn.linear_model.ridge import Ridge

    # Note that an instance of Ridge can be instantiated (with hyperparameters)
    # within the instantiation of McrAR.
    mcrar = McrAR(max_iter=100, tol_increase=2, tol_err_change=1e-10,
                  c_regr=Ridge(alpha=1), st_regr=NNLS(), 
                  c_constraints=[ConstraintNonneg(), ConstraintNorm()])

    # Then use the fit method with initial guesses of ST or C.
