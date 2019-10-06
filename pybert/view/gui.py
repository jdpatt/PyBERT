"""GUI for the PyBERT Simulator.

Copyright (c) 2019 David Patterson & David Banas; all rights reserved World wide.
"""

import logging
import platform
import webbrowser

import pybert.view.widgets as widgets
from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION
from pybert.view.static import help_menu
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


class PyBERT_GUI(QMainWindow):
    """Main PyBERT Window"""

    def __init__(self):
        super(PyBERT_GUI, self).__init__()
        self.log = logging.getLogger("pybert.gui")
        self.log.debug("Initializing GUI")
        self.setWindowTitle("PyBERT")
        self.create_console_dock()
        self.create_actions()
        self.create_menus()
        self.create_statusbar()
        self.setCentralWidget(self.create_tabs())
        self.showMaximized()

    def create_actions(self):
        """Global Actions for the GUI."""
        self.save_confg_act = QAction(self.tr("&Save Configuration"))
        self.save_confg_act.setShortcut(self.tr("Ctrl+S"))
        self.save_confg_act.setStatusTip(self.tr("Save the current configuration."))

        self.load_confg_act = QAction(self.tr("&Load Configuration"))
        self.load_confg_act.setShortcut(self.tr("Ctrl+N"))
        self.load_confg_act.setStatusTip(self.tr("Load configuration."))

        self.save_data_act = QAction(self.tr("&Save Waveforms"))
        self.save_data_act.setShortcut(self.tr("Ctrl+W"))
        self.save_data_act.setStatusTip(self.tr("Save the current waveform data."))

        self.load_data_act = QAction(self.tr("&Load Waveforms"))
        self.load_data_act.setShortcut(self.tr("Ctrl+M"))
        self.load_data_act.setStatusTip(self.tr("Load waveform data."))

        self.console_act = self.console.toggleViewAction()
        self.console_act.setShortcut(self.tr("Ctrl+`"))
        self.console_act.setStatusTip(self.tr("Toggle Console Visibility"))

        self.preferences_act = QAction(self.tr("&Preferences"), self)
        self.preferences_act.setShortcut(self.tr("Ctrl+,"))

        self.exit_act = QAction(self.tr("E&xit"), self)
        self.exit_act.setShortcut(self.tr("Ctrl+Q"))
        self.exit_act.setStatusTip(self.tr("Exit the application"))
        self.exit_act.triggered.connect(QCoreApplication.instance().quit)

        self.about_act = QAction(self.tr("&About"), self)
        self.about_act.triggered.connect(self.about)

        self.doc_act = QAction(self.tr("&Documentation"), self)
        self.doc_act.triggered.connect(open_docs)

        self.help_act = QAction(self.tr("&Help"), self)
        self.help_act.triggered.connect(self.help)

        self.abort_act = QAction(self.tr("&Abort"), self)
        self.abort_act.setShortcut(self.tr("Ctrl+A"))
        self.run_act = QAction(self.tr("&Run"), self)
        self.run_act.setShortcut(self.tr("Ctrl+R"))

    def create_console_dock(self):
        """Create a dockable toolbar on the bottom.

        Allow it to be un-docked, closed, or floated.  If the user accidentally closes it,
        they can reopen it with the menubar or shortcut.
        """
        self.setDockOptions(QMainWindow.AllowTabbedDocks)
        self.console = QDockWidget()
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(100)
        self.text_edit.setCenterOnScroll(True)
        self.console.setWidget(self.text_edit)
        self.console.setWindowTitle("Console")
        self.console.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)

    def create_menus(self):
        """Create the main menus."""
        self.menubar = QMenuBar()
        self.file_menu = self.menubar.addMenu(self.tr("&File"))
        self.file_menu.addAction(self.save_confg_act)
        self.file_menu.addAction(self.load_confg_act)
        self.file_menu.addAction(self.save_data_act)
        self.file_menu.addAction(self.load_data_act)
        self.file_menu.addAction(self.preferences_act)
        self.file_menu.addAction(self.exit_act)

        self.view_menu = self.menubar.addMenu(self.tr("&View"))
        self.view_menu.addAction(self.console_act)

        self.sim_menu = self.menubar.addMenu(self.tr("&Simulation"))
        self.sim_menu.addAction(self.run_act)
        self.sim_menu.addAction(self.abort_act)

        self.help_menu = self.menubar.addMenu(self.tr("&Help"))
        self.help_menu.addAction(self.doc_act)
        self.help_menu.addAction(self.about_act)
        self.help_menu.addAction(self.help_act)

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

    def create_tabs(self):
        """Create a widget to hold every tab of the GUI."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabShape(QTabWidget.Triangular)

        self.cofig_tab = widgets.ConfigWidget()
        self.dfe_tab = widgets.DFEWidget()
        self.eq_tab = widgets.EQTuneWidget()
        self.impulse_tab = widgets.ImpulseWidget()
        self.step_tab = widgets.StepWidget()
        self.pulse_tab = widgets.PulsesWidget()
        self.freq_tab = widgets.FrequencyWidget()
        self.output_tab = widgets.OutputWidget()
        self.eye_tab = widgets.EyeDiagramWidget()
        self.jitter_dist_tab = widgets.JitterDistributionsWidget()
        self.jitter_spect_tab = widgets.JitterSpectrumsWidget()
        self.bath_tab = widgets.BathtubCurvesWidget()
        self.jitter_info_tab = widgets.JitterInfoWidget()

        self.tab_widget.addTab(self.cofig_tab, self.cofig_tab.title)
        self.tab_widget.addTab(self.dfe_tab, self.dfe_tab.title)
        self.tab_widget.addTab(self.eq_tab, self.eq_tab.title)
        self.tab_widget.addTab(self.impulse_tab, self.impulse_tab.title)
        self.tab_widget.addTab(self.step_tab, self.step_tab.title)
        self.tab_widget.addTab(self.pulse_tab, self.pulse_tab.title)
        self.tab_widget.addTab(self.freq_tab, self.freq_tab.title)
        self.tab_widget.addTab(self.output_tab, self.output_tab.title)
        self.tab_widget.addTab(self.eye_tab, self.eye_tab.title)
        self.tab_widget.addTab(self.jitter_dist_tab, self.jitter_dist_tab.title)
        self.tab_widget.addTab(self.jitter_spect_tab, self.jitter_spect_tab.title)
        self.tab_widget.addTab(self.bath_tab, self.bath_tab.title)
        self.tab_widget.addTab(self.jitter_info_tab, self.jitter_info_tab.title)
        return self.tab_widget

    @Slot(dict, dict, object)
    def update_plots(self, jitter, results, t_ns):
        """Update the plots and any other items in the GUI with the simulation results."""
        self.log.debug("Updating Plots")
        # self.dfe_tab.update_plots()
        # self.impulse_tab.update_plots()
        # self.step_tab.update_plots()
        # self.pulse_tab.update_plots(results)
        # self.freq_tab.update_plots()
        self.output_tab.update_plots(t_ns, results)
        # self.eye_tab.update_plots()
        self.jitter_dist_tab.update_plots(jitter)
        # self.jitter_spect_tab.update_plots()
        # self.bath_tab.update_plots()
        # self.jitter_info_tab.update_plots()
        self.log.debug("Plots Updated")

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


def open_docs():
    """Open the documentation on the wiki."""
    # TODO: Redirect to wiki or add readthedoc links.
    webbrowser.open("https://github.com/jdpatt/PyBERT/tree/remove-traits")
