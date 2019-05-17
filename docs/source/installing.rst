.. _installing:

Installation
============

Dependencies
---------------------
Note: the versions are those that have been tested (at some point), 
but older/newer versions may also work.

- Python (3.4, 3.5, 3.6, 3.7)
- numpy (1.9.3, 1.11.1, 1.11.3+mkl, 1.14.6, 1.16.2)
- scipy (1.0.0, 1.0.1, 1.1.0)
- sklearn, optional (0.20)
- Sphinx, optional (1.4.5, 1.5.2, 1.6.4, 1.8.4) -- only for documentation building

Notes and Known Issues
----------------------


Instructions
------------

Git Dynamic copy
~~~~~~~~~~~~~~~~~~~
::

  # Make new directory for pymcr (DIR)
  # Clone from github
  git clone https://github.com/usnistgov/pyMCR.git ./DIR

  # Within install directory (DIR)
  pip3 install -e .

  # To update installation, from within pymcr directory
  git pull

Git Static copy
~~~~~~~~~~~~~~~~~~~
::

  # Make new directory for pymcr (DIR)
  # Clone from github
  git clone https://github.com/usnistgov/pyMCR.git ./DIR

  # Within install directory (DIR)
  pip3 install .

  # You can now delete the source files you downloaded

(Re)-Building documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The documentation was built using Sphinx.::

  # From within the docs/ directory (Note: on Windows sphinx-apidoc.exe)
  # Exclude setup.py from doc-build
  sphinx-apidoc -f -o ./source/ .. ../setup.py

  make html  
  # On Windows
  make.bat html
