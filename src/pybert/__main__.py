"""Main entry into the PyBERT GUI when called with python -m.

This is now largely for debug or if users want to use the python
-m option since calling `pybert` will instead point to cli.py.
"""
import sys

from PySide6.QtWidgets import QApplication

from pybert.gui import MainWindow
from pybert.pybert import PyBERT
from pybert.utility.logger import setup_logger


def main():
    "Run the PyBERT GUI."
    setup_logger()
    app = QApplication()
    main_window = MainWindow(pybert=PyBERT(), show_debug_console=True)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
