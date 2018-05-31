=========
Changelog
=========

This document records all notable changes to 
`pyMCR <https://github.com/CCampJr/pyMCR>`_.

This project adheres to `PEP 440 -- Version Identification 
and Dependency Specification <https://www.python.org/dev/peps/pep-0440/>`_.

0.3.0
------

-   Perform semi-learning assigning some input ST or C components to be fixed.
-   **Constraints**

    -   Non-negative cumulative summation
    -   Zero end-points
    -   Zero cumulalative summation end-points (approximation)
    -   Compress or cut values above or below a threshold value
    -   Replace sum-across-features samples (e.g., 0 concentration) with prescribed target
    -   Enforce a plane ("planarize"). E.g., a concenctration image is a plane.

0.2.1 (18-05-16)
----------------

-   Improved Demo Notebook documentation

0.2.0 (18-05-02)
----------------

-   **Total re-write** that is incompatible with earlier version
-   Built-in solvers: nonnegative least squares (scipy.optimize.nnls), ordinary 
    least squares (scipy.linalg.lstsq)
-   Native support for scikit-learn estimators as least squares solvers / regressor
-   Can now explicitly list and order constraints.

0.1.1a0 (17-12-18)
------------------

-   Concentration and spectral mean relative distance tracked across
    iterations


0.1.0 (17-12-15)
----------------

-   Initial version
