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
from functools import lru_cache
from pathlib import Path

from pybert import __version__ as VERSION

# from pybert.configuration import ConfigurationData
from pybert.defaults import DEBUG, NUM_TAPS
from pybert.simulation import Simulation
from pybert.static import (
    help_menu,
    jitter_rejection_menu,
    performance_menu,
    status_string,
    sweep_results_menu,
)
from pybert.view.gui import PyBERT_GUI
from PySide2.QtWidgets import QApplication

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

        self.log = logging.getLogger("pybert")
        log_file_path = Path(__file__).parent.joinpath("pybert.log")
        # Setup the File Handler.  All messages are logged.
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=10485760, backupCount=5  # 10MB
        )
        fh_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(fh_format)
        file_handler.setLevel(logging.DEBUG)
        self.log.addHandler(file_handler)
        self.log.setLevel(logging.DEBUG)

        self.log.info("Log file created at: %s", log_file_path)

        app = QApplication([])
        gui = PyBERT_GUI()
        gui.show()

        self.log.info(
            "System: %s %s Version: %s", platform.system(), platform.release(), platform.version()
        )
        self.log.info("Python Version: %s", platform.python_version())
        self.log.info("PyBERT Version: %s", VERSION)
        self.log.info("Starting PyBERT...")

        # self.config = ConfigurationData(self)
        # self.data = WaveformData(self)
        self.sim = Simulation()
        self.channel = self.sim.channel

        # self.help_tab = help_menu()

        try:
            if run_simulation:
                # Running the simulation will fill in the required data structure.
                self.sim.run_simulation(initial_run=True)
                # Once the required data structure is filled in, we can create the plots.
                # self.sim.plots.init_plots(self, NUM_TAPS)
                # self.sim.plots.update_eyes()
            else:
                self.channel.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().
        except Exception as error:
            raise
        sys.exit(app.exec_())

    # TODO: The connect slots should move here so that the view (gui) tell's the controller (pybert) to do something with the model (simulation).

    @lru_cache(maxsize=None)
    def _get_sweep_info(self):
        return sweep_results_menu(self.sim.sweep_results)

    @lru_cache(maxsize=None)
    def _get_perf_info(self):
        return performance_menu(
            {key: value * 60.0e-6 for (key, value) in self.sim.performance.items()}
        )

    @lru_cache(maxsize=None)
    def _get_jitter_info(self):
        try:
            jitter_info = jitter_rejection_menu(self.sim.jitter)
        except Exception as error:
            jitter_info = "<H1>Jitter Rejection by Equalization Component</H1>\n"
            # popup_alert("Jitter Calculation Failed", error)
        return jitter_info

    @lru_cache(maxsize=None)
    def _get_status_str(self):
        return status_string(
            self.status,
            self.sim.performance["total"],
            self.channel.chnl_dly,
            self.sim.bit_errors,
            self.sim.tx.rel_power,
            self.sim.jitter["dfe"],
        )


def main():
    """Create a GUI from the view and populate it with all the traits form PyBERT.

    We use the if __name__ == "__main__" so that PyBERT can be used stand-alone,
    or imported, fashion.
    """
    PyBERT()


if __name__ == "__main__":
    main()
