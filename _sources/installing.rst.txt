.. _installing:

Installation
============

Dependencies
---------------------
Note: the versions are those that have been tested, but older/newer
versions may also work.

- Python 3.4, 3.5, 3.6 (3.4.4, 3.5.2, 3.6.1)
- numpy (1.9.3, 1.11.1, 1.11.3+mkl)
- Sphinx (1.4.5, 1.5.2, 1.6.4) -- only for documentation building

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

  # Build all APIs
  # From within the docs/ directory
  sphinx-apidoc -o ./source/ ../pymcr/

  # Build API w/o pyuic5-generated files
  sphinx-apidoc -f -o .\source\ ..\pymcr\ 

  make html  
  # On Windows
  make.bat html
