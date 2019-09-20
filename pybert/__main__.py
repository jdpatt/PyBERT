"""Main entry into the PyBERT GUI."""
from pybert.pybert import PyBERT
from pybert.view import TRAITS_VIEW

PyBERT().configure_traits(view=TRAITS_VIEW)
