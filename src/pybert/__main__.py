"""Main entry into the PyBERT GUI when called with python -m.

This is now largely for debug or if users want to use the python
-m option since calling `pybert` will instead point to cli.py.
"""

import sys

import pyibisami.ibis.gui
from PySide6.QtWidgets import QApplication

# Monkeypatch setattr in all widgets and tabs so that we can log every change using setattr
import pybert.gui.widgets.channel
import pybert.gui.widgets.jitter_info
import pybert.gui.widgets.rx
import pybert.gui.widgets.rx_equalization
import pybert.gui.widgets.rx_optimization_ctle
import pybert.gui.widgets.rx_optimization_dfe
import pybert.gui.widgets.simulation
import pybert.gui.widgets.tx
import pybert.gui.widgets.tx_equalization
import pybert.gui.widgets.tx_optimization
from pybert.utility.debug import setattr

for mod in [
    pybert.gui.widgets.tx,
    pybert.gui.widgets.tx_optimization,
    pybert.gui.widgets.tx_equalization,
    pybert.gui.widgets.rx_optimization_dfe,
    pybert.gui.widgets.simulation,
    pybert.gui.widgets.rx_optimization_ctle,
    pybert.gui.widgets.rx_equalization,
    pybert.gui.widgets.rx,
    pybert.gui.widgets.jitter_info,
    pybert.gui.widgets.channel,
    pyibisami.ibis.gui,
]:
    mod.setattr = setattr

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
