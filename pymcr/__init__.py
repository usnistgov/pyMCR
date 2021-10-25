""" pyMCR: Pythonic Multivariate Curve Resolution - Alternating Least Squares """
from . import mcr
from . import constraints
from . import metrics
from . import regressors
from . import condition

import sys
import logging as _logging

from ._version import __version__

_logger = _logging.getLogger('pymcr')
_logger.setLevel(_logging.DEBUG)

# StdOut is a "stream"; thus, StreamHandler
stdout_handler = _logging.StreamHandler(stream=sys.stdout)

# Set the message format. Simple and removing log level or date info
stdout_format = _logging.Formatter('%(message)s')  # Just a basic message akin to print statements
stdout_handler.setFormatter(stdout_format)

_logger.addHandler(stdout_handler)

