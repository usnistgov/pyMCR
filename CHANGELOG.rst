=========
Changelog
=========

This document records all notable changes to 
`pyMCR <https://github.com/usnistgov/pyMCR>`_.

This project adheres to `PEP 440 -- Version Identification 
and Dependency Specification <https://www.python.org/dev/peps/pep-0440/>`_.

0.4.0 (21-10-22)
-----------------

- Moved to Github actions for CD
- Logging setup by default upon importing the library.
- Updated Jupyter Notebooks to reflect the change to the logging setup
- Minor bug fixes

0.3.3 (21-10-22)
-----------------
- Moved to Github actions for CI

0.3.2 (19-06-25)
----------------

- Jupyter Notebook in Examples from `JRes NIST publication <https://doi.org/10.6028/jres.124.018>`_.
- Minor tweeks and fixes
- Added Conda-Forge badge


0.3.1 (19-05-17)
-----------------

- Implemented logging and removed print() statements
- Removed Jupyter Notebook from forthcoming pub -- will return in the future with better examples
- Minor fixes to CI


0.3.0 (19-04-22)
-----------------

-   Documentation: https://pages.nist.gov/pyMCR or build locally via Sphinx
-   Added Jupyter Notebook that generates images from forthcoming publication.
-   Perform semi-learning: assigning some input ST or C components to be static in fit method.
-   Main class pymcr.mcr.McrAls renamed to pymcr.mcr.McrAR 
-   **Constraints**

    -   Non-negative cumulative summation
    -   Zero end-points
    -   Zero (approx) cumulative summation end-points (can specify nodes as well)
    -   Compress or cut values above or below a threshold value
    -   Replace sum-across-features samples (e.g., 0 concentration) with prescribed target
    -   Enforce a plane ("planarize"). E.g., a concentration image is a plane.

0.2.1 (18-05-16)
----------------

-   Improved Demo Notebook documentation

0.2.0 (18-05-02)
----------------

-   **Total re-write** that is incompatible with earlier version
-   Built-in solvers: non-negative least squares (scipy.optimize.nnls), ordinary 
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
