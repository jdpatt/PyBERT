"""GUI for the PyBERT Simulator.

Copyright (c) 2019 David Patterson & David Banas; all rights reserved World wide.
"""

import inspect
import logging
import traceback
import webbrowser

import pybert.view.widgets as widgets
from pubsub import pub
from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION
from pybert.defaults import DEBUG
from pybert.static import help_menu
from pybert.view.console_logger import QTextEditLogger
from PySide2.QtCore import *
from PySide2.QtWidgets import *


class PyBERT_GUI(QMainWindow):
    """Main PyBERT Window"""

    def __init__(self):
        super(PyBERT_GUI, self).__init__()
        self.log = logging.getLogger("pybert.gui")
        self.log.debug("Initializing GUI")
        self.setWindowTitle("PyBERT")
        self.create_actions()
        self.create_menus()
        self.create_statusbar()
        self.create_console_dock()
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

        self.preferences_act = QAction(self.tr("&Preferences"), self)
        self.preferences_act.setShortcut(self.tr("Ctrl+,"))

        self.exit_act = QAction(self.tr("E&xit"), self)
        self.exit_act.setShortcut(self.tr("Ctrl+Q"))
        self.exit_act.setStatusTip(self.tr("Exit the application"))
        self.exit_act.triggered.connect(QCoreApplication.instance().quit)

        self.about_act = QAction(self.tr("&About"), self)
        self.about_act.triggered.connect(self.about)
        # TODO: These connections should move into pybert so that the view tell's the controller to do something with the model.

        self.doc_act = QAction(self.tr("&Documentation"), self)
        self.doc_act.triggered.connect(self.open_docs)

        self.help_act = QAction(self.tr("&Help"), self)
        self.help_act.triggered.connect(self.help)

        self.run_act = QAction(self.tr("&Start Simulation"), self)
        self.run_act.triggered.connect(self.start_simulation)

    def create_console_dock(self):
        """Create a dockable toolbar on the bottom.

        Allow it to be un-docked, closed, or floated.  If the user accidentally closes it,
        they can reopen it with the menubar or shortcut.
        """
        self.setDockOptions(QMainWindow.AllowTabbedDocks)
        self.console = QDockWidget()
        logging_widget = QTextEditLogger(self)
        logging_widget.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging_widget.setLevel(logging.DEBUG)
        # Attach the widget to the root logger to get all messages.
        logging.getLogger().addHandler(logging_widget)
        self.console.setWidget(logging_widget.widget)
        self.console.setWindowTitle("Console")
        self.console.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)

        self.console_act = self.console.toggleViewAction()
        self.console_act.setShortcut(self.tr("Ctrl+`"))
        self.console_act.setStatusTip(self.tr("Toggle Console Visibility"))
        self.view_menu.addAction(self.console_act)

    def create_menus(self):
        """Create the main menus."""
        self.file_menu = self.menuBar().addMenu(self.tr("&File"))
        self.file_menu.addAction(self.save_confg_act)
        self.file_menu.addAction(self.load_confg_act)
        self.file_menu.addAction(self.save_data_act)
        self.file_menu.addAction(self.load_data_act)
        self.file_menu.addAction(self.preferences_act)
        self.file_menu.addAction(self.exit_act)

        self.view_menu = self.menuBar().addMenu(self.tr("&View"))

        self.menuBar().addAction(self.run_act)

        self.help_menu = self.menuBar().addMenu(self.tr("&Help"))
        self.help_menu.addAction(self.doc_act)
        self.help_menu.addAction(self.about_act)
        self.help_menu.addAction(self.help_act)

    def create_statusbar(self):
        """Create a bar across the bottom for messages.

        This will be where the bit errors and channel delay information live.
        """
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)

    def create_tabs(self):
        """Create a widget to hold every tab of the GUI."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabShape(QTabWidget.Triangular)
        for widget, widget_class in widgets.TABS.items():
            self.widget = widget_class()
            self.tab_widget.addTab(self.widget, self.widget.title)
        return self.tab_widget

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

    def open_docs(self):
        """Open the documentation on the wiki."""
        # TODO: Redirect to wiki or add readthedoc links.
        webbrowser.open("https://github.com/jdpatt/PyBERT/tree/remove-traits")

    def popup_alert(self, error):
        """Popup an alert with the given prompt, log the exception and if debug raise the exception."""
        # if DEBUG:
        #     raise error
        QMessageBox.warning(self, self.tr("PyBERT Error"), str(error))

    def start_simulation(self):
        """Tell pybert to start the simulation."""
        pub.sendMessage("simulation.start")
