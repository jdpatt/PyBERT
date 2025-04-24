"""A package of Python modules, used by the *PyBERT* application.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by:      Mark Marlett <mark.marlett@gmail.com>

Copyright (c) 2014 by David Banas; All rights reserved World wide.

ToDo:
    1. The docs here: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
        suggest that I've got version determination, below, inverted and that
        there ought to be a way to determine the version dynamically at build time
        from a static definition of `__version__` in this file.
        See this page for more detail: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-version
        ==> Check this out and make appropriate changes.
"""
from importlib.metadata import version as _get_version

# Set PEP396 version attribute
try:
    __version__ = _get_version("PipBERT")  # PyPi "PyBERT" package name got stollen. :(
except Exception:  # pylint: disable=broad-exception-caught
    __version__ = "(dev)"

__date__ = "June 24, 2024"
__authors__ = "David Banas & David Patterson"
__copy__ = "Copyright (c) 2014 David Banas, 2019 David Patterson"
