|
|
.. image:: https://travis-ci.org/CCampJr/pyMCR.svg?branch=0.3.X
    :alt: Travis branch
    :target: https://travis-ci.org/CCampJr/pyMCR

.. image:: https://ci.appveyor.com/api/projects/status/ajld1bj7jo4oweio/branch/0.3.X?svg=true
    :alt: AppVeyor branch
    :target: https://ci.appveyor.com/project/CCampJr/pyMCR

.. image:: https://img.shields.io/codecov/c/github/CCampJr/pyMCR/0.3.X.svg
    :alt: Codecov branch
    :target: https://codecov.io/gh/CCampJr/pyMCR

.. image:: https://img.shields.io/pypi/pyversions/pyMCR.svg
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/pyMCR/

.. image:: https://img.shields.io/pypi/v/pyMCR.svg
    :alt: PyPI
    :target: https://pypi.org/project/pyMCR/

.. image:: https://img.shields.io/badge/License-NIST%20Public%20Domain-green.svg
    :alt: NIST Public Domain
    :target: https://github.com/CCampJr/pyMCR/blob/master/LICENSE.md


pyMCR: Multivariate Curve Resolution in Python
===============================================

pyMCR is a small package for performing multivariate curve resolution.
Currently, it implements a simple alternating least squares method
(i.e., MCR-ALS).

MCR-ALS, in general, is a constrained implementation of alternating
least squares (ALS) nonnegative matrix factorization (NMF). Historically,
other names were used for MCR as well:

-   Self modeling mixture analysis (SMMA)
-   Self modeling curve resolution (SMCR)

Available methods:

-   Regressors:

    -   `Ordinary least squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html>`_ (default)
    -   `Non-negatively constrained least squares 
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html>`_
    -   Native support for `scikit-learn linear model regressors 
        <http://scikit-learn.org/stable/modules/linear_model.html>`_
        (e.g., `LinearRegression <http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares>`_, 
        `RidgeRegression <http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression>`_, 
        `Lasso <http://scikit-learn.org/stable/modules/linear_model.html#lasso>`_)

-   Constraints

    -   Non-negativity
    -   Normalization
    -   Zero end-points
    -   Zero (approx) end-points of cumulative summation (can specify nodes as well)
    -   Non-negativity of cumulative summation
    -   Compress or cut values above or below a threshold value
    -   Replace sum-across-features samples (e.g., 0 concentration) with prescribed target
    -   Enforce a plane ("planarize"). E.g., a concenctration image is a plane.

-   Error metrics / Loss function

    -   Mean-squared error

-   Other options

    -   Fix known targets (C and/or ST, and let others vary)

What it **does** do:

-   Approximate the concentration and spectral matrices via minimization routines. 
    This is the core the MCR-ALS methods.
-   Enable the application of certain constraints in a user-defined order.

What it **does not** do:

-   Estimate the number of components in the sample. This is a bonus feature in 
    some more-advanced MCR-ALS packages.

    - In MATLAB: https://mcrals.wordpress.com/
    - In R: https://cran.r-project.org/web/packages/ALS/index.html


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
