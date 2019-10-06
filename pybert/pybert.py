#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
import logging
import logging.handlers
import platform
import sys
import traceback
from pathlib import Path

from pybert import __version__ as VERSION

# from pybert.configuration import ConfigurationData
from pybert.defaults import DEBUG, NUM_TAPS
from pybert.logger import ThreadLogHandler, setup_logger
from pybert.sim.simulation import Simulation
from pybert.view.gui import PyBERT_GUI
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QThread

# from pybert.waveform_data import WaveformData


class PyBERT:
    """
    A serial communication link bit error rate tester (BERT) simulator with a GUI interface.

    Useful for exploring the concepts of serial communication link design.
    """

    def __init__(self, run_simulation: bool = True):
        """
        Initial plot setup occurs here.

        In order to populate the data structure we need to
        construct the plots, we must run the simulation.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(PyBERT, self).__init__()

        self.log = setup_logger("pybert", Path(__file__).parent.joinpath("pybert.log"))

        app = QApplication([])
        self.gui = PyBERT_GUI()

        thread_log = ThreadLogHandler()
        thread_log.new_record.connect(self.gui.log_message)
        self.log.addHandler(thread_log)

        self.gui.show()

        self.log.info(
            "System: %s %s Version: %s", platform.system(), platform.release(), platform.version()
        )
        self.log.info("Python Version: %s", platform.python_version())
        self.log.info("PyBERT Version: %s", VERSION)
        self.log.info("Starting PyBERT...")

        # config = ConfigurationData(self)
        # data = WaveformData(self)
        self.sim = Simulation()

        # Slot/Signal Connections
        self.sim.status_update.connect(self.gui.update_statusbar)
        self.sim.sim_done.connect(self.gui.update_plots)
        self.gui.run_act.triggered.connect(self.sim.run_sweeps)
        self.gui.tab_widget.widget(2).tune_buttons.buttonClicked[int].connect(
            self.sim.eq.handler
        )  # EQ Tab
        app.aboutToQuit.connect(self.close_application)

        try:
            if run_simulation:
                # Running the simulation will fill in the required data structure.
                self.sim.run_simulation(initial_run=True)
                # Once the required data structure is filled in, we can create the plots.
                # self.sim.plots.init_plots(self, NUM_TAPS)
                # self.sim.plots.update_eyes()
        except Exception as error:
            self.log.error(traceback.format_exc())
            self.gui.popup_alert(error)

        # Move the simulation to its own thread from the GUI.
        self.sim_thread = QThread(self.gui)
        self.sim.moveToThread(self.sim_thread)
        self.sim_thread.start()

        sys.exit(app.exec_())

    def close_application(self):
        """Close any threads and then kill the application."""
        self.sim_thread.quit()
        self.sim_thread.wait()  # Wait until its done.
        QCoreApplication.instance().quit()


def main():
    """Create a GUI from the view and populate it with all the traits form PyBERT.

    We use the if __name__ == "__main__" so that PyBERT can be used stand-alone,
    or imported, fashion.
    """
    PyBERT()


if __name__ == "__main__":
    main()
