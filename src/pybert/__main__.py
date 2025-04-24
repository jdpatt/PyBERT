"""Main entry into the PyBERT GUI when called with python -m.

This is now largely for debug or if users want to use the python
-m option since calling `pybert` will instead point to cli.py.
"""
import sys
from pybert.gui import MainWindow

from PySide6.QtWidgets import QApplication
from pybert.pybert import PyBERT
from pybert.utility.logger import setup_logger

def main():
    "Run the PyBERT GUI."
    app = QApplication()
    pybert = PyBERT()
    main_window = MainWindow(pybert=pybert)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
