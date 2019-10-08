"""GUI for the PyBERT Simulator.

Copyright (c) 2019 David Patterson & David Banas; all rights reserved World wide.
"""

import logging
import platform
import webbrowser

from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION
from pybert.configuration import Configuration
from pybert.view.ui_pybert import Ui_MainWindow
from pybert.waveforms import Waveforms
import pyqtgraph as pg
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
        self.config = Configuration(self)
        self.waveform = Waveforms(self)
        self.connect_actions()
        self.showMaximized()

    def connect_actions(self):
        self.actionE_xit.triggered.connect(QCoreApplication.instance().quit)
        self.actionAbout.triggered.connect(self.about)
        self.actionDocumentation.triggered.connect(open_docs)
        self.actionHelp.triggered.connect(self.help)
        self.actionRun.triggered.connect(self.start_simulation)
        self.actionSave_Configuration.triggered.connect(self.config.save_to_file)
        self.actionLoad_Configuration.triggered.connect(self.config.load_from_file)
        self.actionSave_Waveforms.triggered.connect(self.waveform.save_to_file)
        self.actionLoad_Waveforms.triggered.connect(self.waveform.load_from_file)

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
        self.plot_eq.setLabels(title= "Channel + Tx Pre-emphasis + CTLE + DFE", left="Post-CTLE Pulse Response (V)", bottom="Time (ns)")
        self.plot_impulse.set_axis_labels(y_axis="Impulse Response (V/ns)", x_axis="Time (ns)")
        self.plot_impulse.enable_legends()
        self.plot_step.set_axis_labels(y_axis="Step Response (V)", x_axis="Time (ns)")
        self.plot_step.enable_legends()
        self.plot_pulse.set_axis_labels(y_axis="Pulse Response (V)", x_axis="Time (ns)")
        self.plot_pulse.enable_legends()
        self.plot_freq.set_axis_labels(y_axis="Frequency Response (dB)", x_axis="Frequency (GHz)")
        self.plot_freq.enable_legends()
        self.plot_output.set_axis_labels(y_axis="Output (V)", x_axis="Time (ns)")
        self.plot_eye.set_axis_labels(y_axis="Signal Level (V)", x_axis="Time (ps)")
        self.plot_jitter_dist.set_axis_labels(y_axis="Count", x_axis="Time (ps)")
        self.plot_jitter_dist.enable_legends()
        self.plot_jitter_spect.set_axis_labels(
            y_axis="|FFT(TIE)| (dBui)", x_axis="Frequency (MHz)"
        )
        self.plot_jitter_spect.enable_legends()
        self.plot_bathtub.set_axis_labels(
            y_axis="Log10(P(Transition occurs inside.))", x_axis="Time (ps)"
        )

    @Slot(dict, dict, object)
    def update_gui_with_results(self, results, jitter, t_ns):
        """Update all the tabs that need updating."""
        self.update_plots(results, jitter, t_ns)


    def update_plots(self, results, jitter, t_ns):
        """Update the plots within the GUI."""
        self.plot_pulse.channel.plot(results["t_ns_chnl"], results["channel"]["chnl_p"], pen="b", clear=True)
        # self.plot_pulse.channel_tx.plot(results["t_ns_chnl"], results["tx"]["out_p"], pen="b", clear=True)
        # self.plot_pulse.channel_ctle.plot(results["t_ns_chnl"], results["ctle"]["out_p"], pen="b", clear=True)
        # self.plot_pulse.channel_dfe.plot(results["t_ns_chnl"], results["dfe"]["out_p"], pen="b", clear=True)
        self.plot_output.channel.plot(t_ns, results["channel"]["out"], pen="b", clear=True)
        self.plot_output.channel_tx.plot(t_ns, results["tx"]["out"], pen="b", clear=True)
        self.plot_output.channel_ctle.plot(t_ns, results["ctle"]["out"], pen="b", clear=True)
        self.plot_output.channel_dfe.plot(t_ns, results["dfe"]["out"], pen="b", clear=True)
        # Only clear the first plotItem in that view to remove old data.
        self.plot_jitter_dist.channel.plot(
            jitter["channel"].bin_centers, jitter["channel"].hist, pen="b", clear=True, name="Measured"
        )
        self.plot_jitter_dist.channel.plot(
            jitter["channel"].bin_centers,
            jitter["channel"].hist_synth,
            pen="r",
            name="Extrapolated",
        )
        self.plot_jitter_dist.channel_tx.plot(jitter["tx"].bin_centers, jitter["tx"].hist, pen="b", clear=True, name="Measured")
        self.plot_jitter_dist.channel_tx.plot(
            jitter["tx"].bin_centers, jitter["tx"].hist_synth, pen="r", name="Extrapolated"
        )
        self.plot_jitter_dist.channel_ctle.plot(
            jitter["ctle"].bin_centers, jitter["ctle"].hist, pen="b", clear=True, name="Measured"
        )
        self.plot_jitter_dist.channel_ctle.plot(
            jitter["ctle"].bin_centers, jitter["ctle"].hist_synth, pen="r", name="Extrapolated"
        )
        self.plot_jitter_dist.channel_dfe.plot(
            jitter["dfe"].bin_centers, jitter["dfe"].hist, pen="b", clear=True, name="Measured"
        )
        self.plot_jitter_dist.channel_dfe.plot(
            jitter["dfe"].bin_centers, jitter["dfe"].hist_synth, pen="r", name="Extrapolated"
        )

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
