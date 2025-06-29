"""
A package of Python modules, used by the *PyBERT* application.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by:      Mark Marlett <mark.marlett@gmail.com>

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from importlib.metadata import version as _get_version

# Set PEP396 version attribute
try:
    __version__ = _get_version("PipBERT")  # PyPi "PyBERT" package name got stollen. :(
except Exception:  # pylint: disable=broad-exception-caught
    __version__ = "(dev)"

__date__ = "June 22, 2025"
__authors__ = "David Banas & David Patterson"
__copy__ = "Copyright (c) 2014 David Banas, 2019 David Patterson"
