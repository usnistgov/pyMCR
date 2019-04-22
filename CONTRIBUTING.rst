.. -*- mode: rst -*-

**Branch: 0.2.X**:

.. image:: https://img.shields.io/travis/CCampJr/pyMCR/0.2.X.svg
    :alt: Travis branch
    :target: https://travis-ci.org/CCampJr/pyMCR

.. image:: https://ci.appveyor.com/api/projects/status/ajld1bj7jo4oweio/branch/0.2.X?svg=true
    :alt: AppVeyor branch
    :target: https://ci.appveyor.com/project/CCampJr/pyMCR

.. image:: https://codecov.io/gh/CCampJr/pyMCR/branch/0.2.X/graph/badge.svg
    :alt: Codecov
    :target: https://codecov.io/gh/CCampJr/pyMCR

**Branch: 0.3.X**:

.. image:: https://img.shields.io/travis/CCampJr/pyMCR/0.3.X.svg
    :alt: Travis branch
    :target: https://travis-ci.org/CCampJr/pyMCR

.. image:: https://ci.appveyor.com/api/projects/status/ajld1bj7jo4oweio/branch/0.3.X?svg=true
    :alt: AppVeyor branch
    :target: https://ci.appveyor.com/project/CCampJr/pyMCR

.. image:: https://codecov.io/gh/CCampJr/pyMCR/branch/0.3.X/graph/badge.svg
    :alt: Codecov
    :target: https://codecov.io/gh/CCampJr/pyMCR

**Branch: 0.4.X**:

.. image:: https://img.shields.io/travis/CCampJr/pyMCR/0.4.X.svg
    :alt: Travis branch
    :target: https://travis-ci.org/CCampJr/pyMCR

.. image:: https://ci.appveyor.com/api/projects/status/ajld1bj7jo4oweio/branch/0.4.X?svg=true
    :alt: AppVeyor branch
    :target: https://ci.appveyor.com/project/CCampJr/pyMCR

.. image:: https://codecov.io/gh/CCampJr/pyMCR/branch/0.4.X/graph/badge.svg
    :alt: Codecov
    :target: https://codecov.io/gh/CCampJr/pyMCR


Contributing
=============

**Thank you for your interest in contributing!** This document is in pre-pre-alpha stage, so feel free to make edits and suggestions through Pull Requests.

-   Until pyMCR gets to v1.0.X; the versioning will increment by 0.1 for every new release that adds new functionality. 
-   Pull-requests that do not add new functionality, will be merged into the **0.3.X** branch prior to being merged into **master**. 
-   Functional changes will be merged into the **0.4.X** branch.
-   **Avoid adding external dependencies** unless absolutely necessary -- try to stick to the SciPy stack.
-   All functional contributions should have associated tests added to the */pymcr/tests* directory.
-   Testing is performed via pytest
-   Test coverage should be > 90%, with some exceptions
-   After contributing, please add your name to the bottom of the README file. No matter how small, all contributors will be recognized.
-   `Gist of the release process <https://gist.github.com/CCampJr/dca856a4322c9640f857956ba08161e6>`_

Style Notes
~~~~~~~~~~~

-   External imports (e.g., import numpy as _np) should have a prepended underscore to prevent
    code completion from identifying numpy as belonging to the pymcr module. (Demos and test files
    exempt).
-   Avoid all-capitalized variables unless constants.
-   For mathematical functions involving matrix math, capitalized and lower-case variables names
    are permissable to distinguish between vectors and matrices.

Commit Messages
~~~~~~~~~~~~~~~
-   I'm working on implementing a standardized notation for commit messages to facilitate auto-changelog. Comments encouraged (send me an email or leave an issue)


Branches
--------

-   **0.2.X**: Old version

    - Non-new functionality
    - Typos in 0.2.X docstrings

-   **0.3.X**: Updates to current **master**

    - Non-new functionality
    - Typos and **master** branch documentation

-   **0.4.X**: Next version of pyMCR

    - New functionality
    - New regressors
    - New constraints
    - New modules
    - Documentation for new features

LICENSE
----------
This software was developed by employees of the National Institute of Standards 
and Technology (NIST), an agency of the Federal Government. Pursuant to 
`title 17 United States Code Section 105 <http://www.copyright.gov/title17/92chap1.html#105>`_, 
works of NIST employees are not subject to copyright protection in the United States and are 
considered to be in the public domain. Permission to freely use, copy, modify, 
and distribute this software and its documentation without fee is hereby granted, 
provided that this notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER 
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY 
THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, 
AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY 
WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE 
FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR 
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED 
WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR 
OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR 
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE 
RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

Contact
-------
Charles H Camp Jr: `charles.camp@nist.gov <mailto:charles.camp@nist.gov>`_
