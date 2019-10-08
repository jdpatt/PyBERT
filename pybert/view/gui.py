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
from pybert.view.static import help_menu
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

    def __init__(self):
        QMainWindow.__init__(self)
        self.log = logging.getLogger("pybert.gui")
        self.log.debug("Initializing GUI")
        self.setupUi(self)
        self.setWindowTitle("PyBERT")
        self.create_statusbar()
        self.connect_actions()
        self.console.setVisible(False)
        self.showMaximized()

    def connect_actions(self):
        self.actionE_xit.triggered.connect(QCoreApplication.instance().quit)
        self.actionAbout.triggered.connect(self.about)
        self.actionDocumentation.triggered.connect(open_docs)
        self.actionHelp.triggered.connect(self.help)

    def create_statusbar(self):
        """Create a bar across the bottom for messages.

        This will be where the bit errors and channel delay information live.
        """
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

    def log_message(self, level, msg):
        """Log any logger messages via the slot/signal mechanism so that its thread safe."""
        self.text_edit.ensureCursorVisible()
        if level in TEXT_COLOR:
            self.text_edit.appendHtml(f'<p style="color:{TEXT_COLOR[level]};">{msg}</p>')
        else:
            self.text_edit.appendPlainText(msg)

    @Slot(str)
    def update_statusbar(self, status_str):
        """Update the content of the status bar."""
        self.status_label.setText(status_str)

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
        self.plot_impulse.set_axis_labels(y_axis="Impulse Response (V/ns)", x_axis="Time (ns)")
        self.plot_step.set_axis_labels(y_axis="Step Response (V)", x_axis="Time (ns)")
        self.plot_pulse.set_axis_labels(y_axis="Pulse Response (V)", x_axis="Time (ns)")
        self.plot_freq.set_axis_labels(y_axis="Frequency Response (dB)", x_axis="Frequency (GHz)")
        self.plot_output.set_axis_labels(y_axis="Output (V)", x_axis="Time (ns)")
        self.plot_eye.set_axis_labels(y_axis="Signal Level (V)", x_axis="Time (ps)")
        self.plot_jitter_dist.set_axis_labels(y_axis="Count", x_axis="Time (ps)")
        self.plot_jitter_spect.set_axis_labels(
            y_axis="|FFT(TIE)| (dBui)", x_axis="Frequency (MHz)"
        )
        self.plot_bathtub.set_axis_labels(
            y_axis="Log10(P(Transition occurs inside.))", x_axis="Time (ps)"
        )


def open_docs():
    """Open the documentation on the wiki."""
    # TODO: Redirect to wiki or add readthedoc links.
    webbrowser.open("https://github.com/jdpatt/PyBERT/tree/remove-traits")
