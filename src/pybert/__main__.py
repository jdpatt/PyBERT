"""Main entry into PyBERT when called with python -m.

Redirects to the command line interface to either open the application with a GUI or headless.
"""
from pybert.cli import cli


if __name__ == "__main__":
    cli()
