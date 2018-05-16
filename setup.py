"""
Setup for pyMCR
"""

from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = f.read()

setup(name='pyMCR',
      version = '0.2.1a0',
      description = 'Multivariate Curve Resolution in Python',
      long_description = long_description,
      url = 'https://github.com/CCampJr/pyMCR',
      author = 'Charles H. Camp Jr.',
      author_email = 'charles.camp@nist.gov',
      license = 'Public Domain',
      packages = find_packages(),
      zip_safe = False,
      include_package_data = True,
      install_requires=['numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics'])