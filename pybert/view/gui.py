"""GUI for the PyBERT Simulator.

Copyright (c) 2019 David Patterson & David Banas; all rights reserved World wide.
"""

import logging
import platform
import webbrowser

import pyqtgraph as pg
from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION
from pybert.view.ui_pybert import Ui_MainWindow
from PySide2.QtCore import *
from PySide2.QtWidgets import *

# I have dark mode enabled in os x which ruins the text colors.  Look into
# stylesheets for different platforms.
if "darwin" in platform.system().lower():
    TEXT_COLOR = {
        "WARNING": "white",
        "INFO": "white",
        "DEBUG": "LightCyan",
        "CRITICAL": "red",
        "ERROR": "red",
    }
else:
    TEXT_COLOR = {
        "WARNING": "black",
        "INFO": "black",
        "DEBUG": "navy",
        "CRITICAL": "red",
        "ERROR": "red",
    }


class PyBERT_GUI(QMainWindow, Ui_MainWindow):
    """Main PyBERT Window"""

    sim_start = Signal()

    def __init__(self):
        QMainWindow.__init__(self)
        self.log = logging.getLogger("pybert.gui")
        self.log.debug("Initializing GUI")
        self.setupUi(self)
        self.setWindowTitle("PyBERT")
        self.create_statusbar()
        self.console.setVisible(True)
        self.init_plots()
        self.connect_actions()
        self.showMaximized()

    def connect_actions(self):
        self.actionE_xit.triggered.connect(QCoreApplication.instance().quit)
        self.actionAbout.triggered.connect(self.about)
        self.actionDocumentation.triggered.connect(open_docs)
        self.actionHelp.triggered.connect(self.help)
        self.actionRun.triggered.connect(self.start_simulation)

    def create_statusbar(self):
        """Create a bar across the bottom for messages.

        This will be where the bit errors and channel delay information live.
        """
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

    @Slot(str)
    def update_statusbar(self, status_str):
        """Update the content of the status bar."""
        self.status_label.setText(status_str)

    def log_message(self, level, msg):
        """Log any logger messages via the slot/signal mechanism so that its thread safe."""
        self.text_edit.ensureCursorVisible()
        if level in TEXT_COLOR:
            self.text_edit.appendHtml(f'<p style="color:{TEXT_COLOR[level]};">{msg}</p>')
        else:
            self.text_edit.appendPlainText(msg)

    def about(self):
        """Popup a Message Box with the About information."""
        QMessageBox.about(
            self,
            self.tr(f"PyBERT v{VERSION}"),
            self.tr(
                f"A serial communication link design tool, written in Python.\n\n"
                f"{AUTHORS}\n"
                f"{DATE}   \n"
                f"{COPY};  \n"
                "All rights reserved World wide."
            ),
        )

    def help(self):
        """Open the generic help menu."""
        QMessageBox.information(self, self.tr("Help"), self.tr(help_menu()))

    def popup_alert(self, error):
        """Popup an alert with the given prompt, log the exception and if debug raise the exception."""
        # if DEBUG:
        #     raise error
        QMessageBox.warning(self, self.tr("PyBERT Error"), str(error))

    def init_plots(self):
        self.cdr_adapt = self.plot_dfe.addPlot(
            row=0, col=0, title="CDR Adaptation", labels={"left": "UI (ps)", "bottom": "Time (ns)"}
        )
        self.dfe_adapt = self.plot_dfe.addPlot(row=0, col=1, title="DFE Adaptation")
        self.dfe_adapt.addLegend()
        self.cdr_histo = self.plot_dfe.addPlot(
            row=1,
            col=0,
            title="CDR Clock Period Histogram",
            labels={"left": "Bin Count", "bottom": "Clock Period (ps)"},
        )
        self.cdr_spect = self.plot_dfe.addPlot(
            row=1,
            col=1,
            title="CDR Adaptation",
            labels={"left": "|H(f)| (dB mean)", "bottom": "Frequency (bit rate)"},
        )
        self.plot_eq.setLabels(
            title="Channel + Tx Pre-emphasis + CTLE + DFE",
            left="Post-CTLE Pulse Response (V)",
            bottom="Time (ns)",
        )
        self.plot_impulse.set_axis_labels(y_axis="Impulse Response (V/ns)", x_axis="Time (ns)")
        self.plot_impulse.enable_legends()
        self.plot_impulse.link_x_axes()
        self.plot_step.set_axis_labels(y_axis="Step Response (V)", x_axis="Time (ns)")
        self.plot_step.enable_legends()
        self.plot_step.link_x_axes()
        self.plot_pulse.set_axis_labels(y_axis="Pulse Response (V)", x_axis="Time (ns)")
        self.plot_pulse.enable_legends()
        self.plot_pulse.link_x_axes()
        self.plot_freq.set_axis_labels(y_axis="Frequency Response (dB)", x_axis="Frequency (GHz)")
        self.plot_freq.enable_legends()
        self.plot_freq.set_x_range(0.01, 1.602)
        self.plot_freq.enable_log_scale(x=True)
        self.plot_freq.set_y_range(-40, 2)
        self.plot_output.set_axis_labels(y_axis="Output (V)", x_axis="Time (ns)")
        self.plot_output.link_x_axes()
        self.plot_eye.set_axis_labels(y_axis="Signal Level (V)", x_axis="Time (ps)")
        self.plot_jitter_dist.set_axis_labels(y_axis="Count", x_axis="Time (ps)")
        self.plot_jitter_dist.enable_legends()
        self.plot_jitter_spect.set_axis_labels(
            y_axis="|FFT(TIE)| (dBui)", x_axis="Frequency (MHz)"
        )
        self.plot_jitter_spect.enable_legends()
        self.plot_jitter_spect.link_x_axes()
        self.plot_bathtub.set_axis_labels(
            y_axis="Log10(P(Transition occurs inside.))", x_axis="Time (ps)"
        )
        self.plot_bathtub.set_y_range(-18, 0)

    @Slot(object, object, object)
    def update_eq_plots(self, t_ns_chnl, ctle_out_h_tune, clocks_tune):
        """Update the EQ Plots during tuning."""
        self.log.debug("Updating EQ Plots")
        self.plot_eq.plot(t_ns_chnl, ctle_out_h_tune, pen="b", clear=True)
        self.plot_eq.plot(t_ns_chnl, clocks_tune)  # Gray by default.

    @Slot(dict, object, object)
    def update_gui_with_results(self, results):
        """Update all the tabs that need updating."""
        self.update_plots(results)
        # self.update_eyes(results)
        self.update_jitter_info(results)

    def update_jitter_info(self, results):
        """Update the jitter rejection tables with the simulation data."""
        jitter_info = results["jitter"]["info"]
        tables = [
            self.jitter_info_tx,
            self.jitter_info_ctle,
            self.jitter_info_dfe,
            self.jitter_info_total,
        ]
        for index, table in enumerate(tables):
            for row in range(0, 4):  # ISI, DCD, PJ and RJ
                for column in range(0, 3):  # Input, Output and Rejection
                    table.setItem(
                        row, column, QTableWidgetItem(jitter_info[index][row][column])                            
                    )

    def update_plots(self, results):
        """Update the plots within the GUI."""
        self.cdr_adapt.plot(results["t_ns"], results["ui_ests"], pen="b", clear=True)
        self.dfe_adapt.clearPlots()
        COLORS = ["r", "m", "y", "g", "b"]
        for index in range(1, results["n_dfe_taps"] + 1):
            self.dfe_adapt.plot(
                results["tap_weight_index"],
                results[f"tap{index}_weights"],
                name=f"tap{index}",
                pen=COLORS[index - 1],
            )
        self.cdr_histo.plot(
            results["clk_per_hist_bins"], results["clk_per_hist_vals"], pen="b", clear=True
        )
        self.cdr_spect.plot(results["clk_freqs"], results["clk_spec"], pen="b", clear=True)
        self.plot_impulse.channel.plot(
            results["t_ns_chnl"],
            results["channel"]["chnl_h"],
            pen="b",
            name="Incremental",
            clear=True,
        )
        self.plot_impulse.channel_tx.plot(
            results["t_ns_chnl"], results["tx"]["out_h"], pen="r", name="Cumulative", clear=True
        )
        self.plot_impulse.channel_ctle.plot(
            results["t_ns_chnl"], results["ctle"]["out_h"], pen="r", name="Cumulative", clear=True
        )
        self.plot_impulse.channel_dfe.plot(
            results["t_ns_chnl"], results["dfe"]["out_h"], pen="r", name="Cumulative", clear=True
        )
        self.plot_step.channel.plot(
            results["t_ns_chnl"],
            results["channel"]["chnl_s"],
            pen="b",
            name="Incremental",
            clear=True,
        )
        self.plot_step.channel_tx.plot(
            results["t_ns_chnl"], results["tx"]["s"], pen="b", name="Incremental", clear=True
        )
        self.plot_step.channel_tx.plot(
            results["t_ns_chnl"], results["tx"]["out_s"], pen="r", name="Cumulative"
        )
        self.plot_step.channel_ctle.plot(
            results["t_ns_chnl"], results["ctle"]["s"], pen="b", name="Incremental", clear=True
        )
        self.plot_step.channel_ctle.plot(
            results["t_ns_chnl"], results["ctle"]["out_s"], pen="r", name="Cumulative"
        )
        self.plot_step.channel_dfe.plot(
            results["t_ns_chnl"], results["dfe"]["s"], pen="b", name="Incremental", clear=True
        )
        self.plot_step.channel_dfe.plot(
            results["t_ns_chnl"], results["dfe"]["out_s"], pen="r", name="Cumulative"
        )
        self.plot_pulse.channel.plot(
            results["t_ns_chnl"], results["channel"]["chnl_p"], pen="b", clear=True
        )
        # self.plot_pulse.channel_tx.plot(results["t_ns_chnl"], results["tx"]["out_p"], pen="b", clear=True)
        # self.plot_pulse.channel_ctle.plot(results["t_ns_chnl"], results["ctle"]["out_p"], pen="b", clear=True)
        # self.plot_pulse.channel_dfe.plot(results["t_ns_chnl"], results["dfe"]["out_p"], pen="b", clear=True)
        self.plot_freq.channel.plot(
            results["f_GHz"],
            results["channel"]["chnl_H"],
            pen="b",
            name="Original Impulse",
            clear=True,
        )
        self.plot_freq.channel.plot(
            results["f_GHz"], results["channel"]["chnl_trimmed_H"], pen="r", name="Trimmed Impulse"
        )
        self.plot_freq.channel_tx.plot(
            results["f_GHz"], results["tx"]["H"], pen="b", name="Incremental", clear=True
        )
        self.plot_freq.channel_tx.plot(
            results["f_GHz"], results["tx"]["out_H"], pen="r", name="Cumulative"
        )
        self.plot_freq.channel_ctle.plot(
            results["f_GHz"], results["ctle"]["H"], pen="b", name="Incremental", clear=True
        )
        self.plot_freq.channel_ctle.plot(
            results["f_GHz"], results["ctle"]["out_H"], pen="r", name="Cumulative"
        )
        self.plot_freq.channel_dfe.plot(
            results["f_GHz"], results["dfe"]["H"], pen="b", name="Incremental", clear=True
        )
        self.plot_freq.channel_dfe.plot(
            results["f_GHz"], results["dfe"]["out_H"], pen="r", name="Cumulative"
        )
        self.plot_output.channel.plot(
            results["t_ns"], results["channel"]["out"], pen="b", clear=True
        )
        self.plot_output.channel_tx.plot(
            results["t_ns"], results["tx"]["out"], pen="b", clear=True
        )
        self.plot_output.channel_ctle.plot(
            results["t_ns"], results["ctle"]["out"], pen="b", clear=True
        )
        self.plot_output.channel_dfe.plot(
            results["t_ns"], results["dfe"]["out"], pen="b", clear=True
        )
        # Only clear the first plotItem in that view to remove old data.
        self.plot_jitter_dist.channel.plot(
            results["jitter"]["channel"].bin_centers,
            results["jitter"]["channel"].hist,
            pen="b",
            clear=True,
            name="Measured",
        )
        self.plot_jitter_dist.channel.plot(
            results["jitter"]["channel"].bin_centers,
            results["jitter"]["channel"].hist_synth,
            pen="r",
            name="Extrapolated",
        )
        self.plot_jitter_dist.channel_tx.plot(
            results["jitter"]["tx"].bin_centers,
            results["jitter"]["tx"].hist,
            pen="b",
            clear=True,
            name="Measured",
        )
        self.plot_jitter_dist.channel_tx.plot(
            results["jitter"]["tx"].bin_centers,
            results["jitter"]["tx"].hist_synth,
            pen="r",
            name="Extrapolated",
        )
        self.plot_jitter_dist.channel_ctle.plot(
            results["jitter"]["ctle"].bin_centers,
            results["jitter"]["ctle"].hist,
            pen="b",
            clear=True,
            name="Measured",
        )
        self.plot_jitter_dist.channel_ctle.plot(
            results["jitter"]["ctle"].bin_centers,
            results["jitter"]["ctle"].hist_synth,
            pen="r",
            name="Extrapolated",
        )
        self.plot_jitter_dist.channel_dfe.plot(
            results["jitter"]["dfe"].bin_centers,
            results["jitter"]["dfe"].hist,
            pen="b",
            clear=True,
            name="Measured",
        )
        self.plot_jitter_dist.channel_dfe.plot(
            results["jitter"]["dfe"].bin_centers,
            results["jitter"]["dfe"].hist_synth,
            pen="r",
            name="Extrapolated",
        )
        self.plot_jitter_spect.channel.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["channel"].jitter_spectrum,
            pen="b",
            clear=True,
            name="Total",
        )
        self.plot_jitter_spect.channel.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["channel"].tie_ind_spectrum,
            pen="r",
            name="Data Independent",
        )
        self.plot_jitter_spect.channel.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["channel"].thresh,
            pen="m",
            name="Pj Threshold",
        )

        self.plot_jitter_spect.channel_tx.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["tx"].jitter_spectrum,
            pen="b",
            clear=True,
            name="Total",
        )
        self.plot_jitter_spect.channel_tx.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["tx"].tie_ind_spectrum,
            pen="r",
            name="Data Independent",
        )
        self.plot_jitter_spect.channel_tx.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["tx"].thresh,
            pen="m",
            name="Pj Threshold",
        )

        self.plot_jitter_spect.channel_ctle.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["ctle"].jitter_spectrum,
            pen="b",
            clear=True,
            name="Total",
        )
        self.plot_jitter_spect.channel_ctle.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["ctle"].tie_ind_spectrum,
            pen="r",
            name="Data Independent",
        )
        self.plot_jitter_spect.channel_ctle.plot(
            results["jitter"]["f_MHz"],
            results["jitter"]["ctle"].thresh,
            pen="m",
            name="Pj Threshold",
        )

        self.plot_jitter_spect.channel_dfe.plot(
            results["jitter"]["f_MHz_dfe"],
            results["jitter"]["dfe"].jitter_spectrum,
            pen="b",
            clear=True,
            name="Total",
        )
        self.plot_jitter_spect.channel_dfe.plot(
            results["jitter"]["f_MHz_dfe"],
            results["jitter"]["dfe"].tie_ind_spectrum,
            pen="r",
            name="Data Independent",
        )
        self.plot_jitter_spect.channel_dfe.plot(
            results["jitter"]["f_MHz_dfe"],
            results["jitter"]["dfe"].thresh,
            pen="m",
            name="Pj Threshold",
        )
        self.plot_bathtub.channel.plot(
            results["jitter"]["channel"].bin_centers, results["bathtub_chnl"], pen="b", clear=True
        )
        self.plot_bathtub.channel_tx.plot(
            results["jitter"]["channel"].bin_centers, results["bathtub_tx"], pen="b", clear=True
        )
        self.plot_bathtub.channel_ctle.plot(
            results["jitter"]["channel"].bin_centers, results["bathtub_ctle"], pen="b", clear=True
        )
        self.plot_bathtub.channel_dfe.plot(
            results["jitter"]["channel"].bin_centers, results["bathtub_dfe"], pen="b", clear=True
        )

    def update_eyes(self, results):
        """Update the heat plots representing the eye diagrams."""
        self.plot_eye.channel.plot(results["eye_index"], results["eye_chnl"])
        self.plot_eye.channel_tx.plot(results["eye_index"], results["eye_tx"])
        self.plot_eye.channel_ctle.plot(results["eye_index"], results["eye_ctle"])
        self.plot_eye.channel_dfe.plot(results["eye_index"], results["eye_dfe"])

    def start_simulation(self):
        """Get the parameters and kick off the simulation."""
        self.sim_start.emit()


def open_docs():
    """Open the documentation on the wiki."""
    # TODO: Redirect to wiki or add readthedoc links.
    webbrowser.open("https://github.com/jdpatt/PyBERT/tree/develop")


def help_menu():
    """Return the content for the help tab of the GUI."""
    return """<H2>PyBERT User's Guide</H2>\n
  <H3>Note to developers</H3>\n
    This is NOT for you. Instead, open 'pybert/doc/_build/html/index.html' in a browser.\n
  <H3>PyBERT User Help Options</H3>\n
    <UL>\n
      <LI>Hover over any user-settable value in the <em>Config.</em> tab, for help message.</LI>\n
      <LI>Peruse the <em>General Tips</em> & <em>Help by Tab</em> section, below.</LI>\n
      <LI>Visit the PyBERT FAQ at: https://github.com/capn-freako/PyBERT/wiki/pybert_faq.</LI>\n
      <LI>Send e-mail to David Banas at capn.freako@gmail.com.</LI>\n
    </UL>\n
  <H3>General Tips</H3>\n
    <H4>Main Window Status Bar</H4>\n
      The status bar, just above the <em>Run</em> button, gives the following information, from left to right:.<p>\n
      (Note: the individual pieces of information are separated by vertical bar, or 'pipe', characters.)\n
        <UL>\n
          <LI>Current state of, and/or activity engaged in by, the program.</LI>\n
          <LI>Simulator performance, in mega-samples per minute. A 'sample' corresponds to a single value in the signal vector being processed.</LI>\n
          <LI>The observed delay in the channel; can be used as a sanity check, if you know your channel.</LI>\n
          <LI>The number of bit errors detected in the last successful simulation run.</LI>\n
          <LI>The average power dissipated by the transmitter, assuming perfect matching to the channel ,no reflections, and a 50-Ohm system impedance.</LI>\n
          <LI>The jitter breakdown for the last run. (Taken at DFE output.)</LI>\n
        </UL>\n
  <H3>Help by Tab</H3>\n
    <H4>Config.</H4>\n
      This tab allows you to configure the simulation.\n
      Hover over any user configurable element for a help message.\n
"""
