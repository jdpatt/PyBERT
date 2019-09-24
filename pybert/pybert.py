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
from functools import lru_cache
import logging
import logging.handlers
import platform
from pathlib import Path

from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION

# from pybert.configuration import ConfigurationData
from pybert.defaults import DEBUG, NUM_TAPS
from pybert.simulation import Simulation
from pybert.static import (
    about_menu,
    help_menu,
    jitter_rejection_menu,
    performance_menu,
    status_string,
    sweep_results_menu,
)

# from pybert.view import TRAITS_VIEW, popup_alert

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

        self.log = setup_logger(DEBUG)
        self.log.debug("System: %s", platform.system())
        self.log.debug("Python Version: %s", platform.python_version())
        self.log.debug("PyBERT Version: %s", VERSION)

        self.log.info("Starting PyBERT...")

        # self.config = ConfigurationData(self)
        # self.data = WaveformData(self)
        self.sim = Simulation()
        self.channel = self.sim.channel
        self.status = self.sim.status  #: PyBERT status (String).

        # About
        self.about_tab = about_menu(AUTHORS, COPY, DATE, VERSION)
        self.help_tab = help_menu()

        # # Tab Buttons
        # self.btn_rst_eq = Button(label="Reset Eq")
        # self.btn_save_eq = Button(label="Save Eq")
        # self.btn_opt_tx = Button(label="Opt Tx")
        # self.btn_opt_rx = Button(label="Opt Rx")
        # self.btn_coopt = Button(label="Co-Opt")
        # self.btn_abort = Button(label="Abort")
        # self.btn_cfg_tx = Button(label="Configure")
        # self.btn_cfg_rx = Button(label="Configure")

        # # Global Buttons
        # self.run_sim = Button(label="Run")
        # self.stop_sim = Button(label="Stop")
        # self.save_data = Button(label="Save Results")
        # self.load_data = Button(label="Load Results")
        # self.save_cfg = Button(label="Save Config.")
        # self.load_cfg = Button(label="Load Config.")

        try:
            if run_simulation:
                # Running the simulation will fill in the required data structure.
                self.sim.my_run_simulation(initial_run=True)
                # Once the required data structure is filled in, we can create the plots.
                self.sim.plots.init_plots(self, NUM_TAPS)
                self.sim.update_eyes()
            else:
                self.channel.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().
        except Exception as error:
            raise
            # popup_alert(error)

    # Button handlers
    def _btn_rst_eq_fired(self):
        """Reset the equalization."""
        self.sim.eq.reset_equalization()

    def _btn_save_eq_fired(self):
        """Save the equalization."""
        self.sim.eq.save_equalization()

    def _btn_opt_tx_fired(self):
        """Run the tx optimization."""
        self.sim.eq.run_tx_optimization()

    def _btn_opt_rx_fired(self):
        """Run the rx optimization."""
        self.sim.eq.run_rx_optimization()

    def _btn_coopt_fired(self):
        """Run the co-optimization between Tx and Rx."""
        self.sim.eq.run_co_optimization()

    def _btn_abort_fired(self):
        """Kill all the threads that are currently running optimization."""
        self.sim.eq.abort_optimization()

    def _btn_cfg_tx_fired(self):
        """Open the Tx AMI configurator."""
        self.sim.tx.open_config_gui()

    def _btn_cfg_rx_fired(self):
        """Open the Rx AMI configurator."""
        self.sim.rx.open_config_gui()

    def _btn_run_sim_fired(self):
        """Start a new simulation."""
        self.sim.run()

    # def _btn_stop_sim_fired(self):
    #     """Stop the current simulation."""
    #     self.sim.abort()

    # def _btn_save_data_fired(self):
    #     """Save all the waveform data."""
    #     try:
    #         self.data.save()
    #     except Exception as err:
    #         popup_alert("An error occured.  The waveform data was not saved", err)

    # def _btn_load_data_fired(self):
    #     """Load previous waveform data."""
    #     try:
    #         self.data.load()
    #     except Exception as err:
    #         popup_alert("An error occured.  The waveform data could not be loaded.", err)

    # def _btn_save_cfg_fired(self):
    #     """Save all the configuration data."""
    #     try:
    #         self.config.save()
    #     except Exception as err:
    #         popup_alert("An error occured.  The configuration data was not saved", err)

    # def _btn_load_cfg_fired(self):
    #     """Load previous configuration data."""
    #     try:
    #         self.config.load()
    #     except Exception as err:
    #         popup_alert("An error occured.  The configuration data could not be loaded.", err)

    # -----------------------------------------------------------------
    # Changed property handlers.
    def _use_dfe_changed(self, new_value):
        """The user turned on/off DFE."""
        self.sim.eq.toggle_dfe(new_value)

    def _use_dfe_tune_changed(self, new_value):
        """The user turned on/off the tuned DFE."""
        self.sim.eq.toggle_tunded_dfe(new_value)

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
            self.sim.tx.relative_power,
            self.sim.jitter["dfe"],
        )


def setup_logger(debug: bool = False):
    """Setup the logger and return the logging object."""
    logger = logging.getLogger("pybert")
    log_file_path = Path(__file__).parent.joinpath("pybert.log")
    # Setup the File Handler.  All messages are logged.
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=10485760, backupCount=5  # 10MB
    )
    fh_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fh_format)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Setup the Console Handler.  Only INFO or HIGH gets shown
    console_handler = logging.StreamHandler()
    ch_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(ch_format)
    logger.addHandler(console_handler)

    if debug:
        console_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    logger.info("Log file created at: %s", log_file_path)
    return logger


def main():
    """Create a GUI from the view and populate it with all the traits form PyBERT.

    We use the if __name__ == "__main__" so that PyBERT can be used stand-alone,
    or imported, fashion.
    """
    PyBERT()


if __name__ == "__main__":
    main()
