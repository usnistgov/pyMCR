"""
Setup for pyMCR
"""
import os
import io
from setuptools import setup, find_packages

with io.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

with open('./requirements.txt','r') as f:
        INSTALL_REQUIRES = f.read().splitlines()

# get __version__ from _version.py
ver_file = os.path.join('pymcr', '_version.py')
with open(ver_file) as f:
    exec(f.read())

VERSION = __version__  # noqa: F821

setup(name='pyMCR',
      version = VERSION,
      description = 'Multivariate Curve Resolution in Python',
      long_description = long_description,
      url = 'https://github.com/usnistgov/pyMCR',
      author = 'Charles H. Camp Jr.',
      author_email = 'charles.camp@nist.gov',
      license = 'Public Domain',
      packages = find_packages(),
      zip_safe = False,
      include_package_data = True,
      install_requires=INSTALL_REQUIRES,
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics'])
