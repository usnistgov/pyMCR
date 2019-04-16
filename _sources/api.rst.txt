.. _api:

API Reference
=============

This is not an exhaustive list of classes and functions,
but rather those most likely to be of interest to users and developer.
See :ref:`genindex` and :ref:`modindex` for a full list.

:mod:`pymcr.condition`: Functions to condition / preprocess data
---------------------------------------------------------------------

.. automodule:: pymcr.condition


Functions
~~~~~~~~~
.. currentmodule:: pymcr.condition

.. autosummary::
    

    pymcr.condition.standardize


:mod:`pymcr.constraints`: Built-in constraints
-----------------------------------------------

.. automodule:: pymcr.constraints

Classes
~~~~~~~
.. currentmodule:: pymcr.constraints

.. autosummary::
    
    pymcr.constraints.ConstraintNonneg
    pymcr.constraints.ConstraintCumsumNonneg
    pymcr.constraints.ConstraintZeroEndPoints
    pymcr.constraints.ConstraintZeroCumSumEndPoints
    pymcr.constraints.ConstraintNorm
    pymcr.constraints.ConstraintReplaceZeros
    pymcr.constraints.ConstraintCutBelow
    pymcr.constraints.ConstraintCompressBelow
    pymcr.constraints.ConstraintCutAbove
    pymcr.constraints.ConstraintCompressAbove
    pymcr.constraints.ConstraintPlanarize

:mod:`pymcr.mcr`: MCR Main Class for Computation
-----------------------------------------------------

.. automodule:: pymcr.mcr

Classes
~~~~~~~
.. currentmodule:: pymcr.mcr

.. autosummary::
    
    pymcr.mcr.McrAls

:mod:`pymcr.metrics`: Metrics used in pyMCR
--------------------------------------------

.. automodule:: pymcr.metrics


Functions
~~~~~~~~~
.. currentmodule:: pymcr.metrics

.. autosummary::
    

    pymcr.metrics.mse

:mod:`pymcr.regressors`: Built-in regression methods
-----------------------------------------------------

.. automodule:: pymcr.regressors

Classes
~~~~~~~
.. currentmodule:: pymcr.regressors

.. autosummary::
    
    pymcr.regressors.LinearRegression
    pymcr.regressors.OLS
    pymcr.regressors.NNLS
