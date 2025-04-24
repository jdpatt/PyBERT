"""Main window for PyBERT application using PySide6.

This module implements the main window and overall GUI structure for PyBERT.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QMenuBar,
    QMenu,
    QMessageBox,
    QFileDialog,
)

from pybert import __version__, __authors__, __date__, __copy__
from pybert.configuration import CONFIG_LOAD_WILDCARD, CONFIG_SAVE_WILDCARD
from pybert.constants import GETTING_STARTED_URL

from pybert.gui.widgets import DebugConsoleWidget
from pybert.gui.tabs import ConfigTab, EqualizationTab, OptimizerTab, ResponsesTab, ResultsTab, JitterTab
from pybert.utility.logger import QStatusBarHandler


logger = logging.getLogger("pybert")


class MainWindow(QMainWindow):
    """Main window for the PyBERT application."""

    def __init__(self, pybert=None, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            pybert: PyBERT model instance
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        self.setWindowTitle("PyBERT")
        self.resize(1200, 800)

        # Set window icon
        icon_path = Path(__file__).parent / "images" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create and add tabs
        self.config_tab = ConfigTab()
        self.tab_widget.addTab(self.config_tab, "Setup")

        self.equalization_tab = EqualizationTab()
        self.tab_widget.addTab(self.equalization_tab, "Equalization")

        self.optimizer_tab = OptimizerTab()
        self.tab_widget.addTab(self.optimizer_tab, "Optimizer")

        self.responses_tab = ResponsesTab()
        self.tab_widget.addTab(self.responses_tab, "Responses")

        self.results_tab = ResultsTab()
        self.tab_widget.addTab(self.results_tab, "Results")

        self.jitter_tab = JitterTab()
        self.tab_widget.addTab(self.jitter_tab, "Jitter")

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        logger.addHandler(QStatusBarHandler(self.status_bar))

        # Create the dock widget/debug console
        self.debug_console = DebugConsoleWidget()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.debug_console)
        self.debug_console.hide()

        # Create menus
        self.create_menus()

    def create_menus(self):
        """Create the application menus."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        load_results_action = QAction("Load Results", self)
        load_results_action.triggered.connect(self.load_results)

        save_results_action = QAction("Save Results", self)
        save_results_action.triggered.connect(self.save_results)

        load_config_action = QAction("Open Config...", self)
        load_config_action.setShortcut("Ctrl+O")
        load_config_action.triggered.connect(self.load_config)

        save_config_action = QAction("Save", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_config)

        save_config_as_action = QAction("Save As...", self)
        save_config_as_action.setShortcut("Ctrl+Shift+S")
        save_config_as_action.triggered.connect(self.save_config_as)

        file_menu.addAction(quit_action)
        file_menu.addSeparator()
        file_menu.addAction(load_results_action)
        file_menu.addAction(save_results_action)
        file_menu.addSeparator()
        file_menu.addAction(load_config_action)
        file_menu.addAction(save_config_action)
        file_menu.addAction(save_config_as_action)

        # View menu
        view_menu = self.menuBar().addMenu("&View")

        debug_console_action = QAction("Debug Console", self)
        debug_console_action.setShortcut("Ctrl+`")
        debug_console_action.setCheckable(True)
        debug_console_action.triggered.connect(self.toggle_console_view)

        clear_waveforms_action = QAction("Clear Waveforms", self)
        clear_waveforms_action.triggered.connect(self.clear_waveforms)

        view_menu.addAction(debug_console_action)
        view_menu.addAction(clear_waveforms_action)

        # Simulation menu
        sim_menu = self.menuBar().addMenu("Simulate")

        run_action = QAction("Start", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_simulation)

        abort_action = QAction("Stop", self)
        abort_action.triggered.connect(self.stop_simulation)

        sim_menu.addAction(run_action)
        sim_menu.addAction(abort_action)

        # Optimization menu
        opt_menu = self.menuBar().addMenu("&Optimization")
        use_eq_action = QAction("Use EQ", self)
        use_eq_action.setShortcut("Ctrl+U")
        reset_eq_action = QAction("Reset EQ", self)
        tune_eq_action = QAction("Tune EQ", self)
        tune_eq_action.setShortcut("Ctrl+T")
        stop_tune_action = QAction("Abort", self)
        stop_tune_action.setShortcut("Ctrl+Esc")

        opt_menu.addAction(use_eq_action)
        opt_menu.addAction(reset_eq_action)
        opt_menu.addAction(tune_eq_action)
        opt_menu.addAction(stop_tune_action)

        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")

        # Help menu
        help_menu = self.menuBar().addMenu("&Help")

        getting_started_action = QAction("Getting Started", self)
        getting_started_action.triggered.connect(self.show_getting_started)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)

        help_menu.addAction(getting_started_action)
        help_menu.addAction(about_action)

    # Menu action handlers
    def load_results(self):
        """Load results from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path and self.pybert:
            self.pybert.load_results(Path(file_path))

    def save_results(self):
        """Save results to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path and self.pybert:
            self.pybert.save_results(Path(file_path))

    def load_config(self):
        """Load configuration from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", CONFIG_LOAD_WILDCARD)
        if file_path and self.pybert:
            self.pybert.load_config(Path(file_path))

    def save_config(self):
        """Save configuration to the current file."""
        if self.pybert:
            if self.pybert.config_file:
                self.pybert.save_config()
            else:
                self.save_config_as()

    def save_config_as(self):
        """Save configuration to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", "", CONFIG_SAVE_WILDCARD)
        if file_path and self.pybert:
            self.pybert.save_config_as(Path(file_path))

    def toggle_console_view(self):
        """Toggle the debug console visibility."""
        if self.debug_console.isVisible():
            self.debug_console.hide()
        else:
            self.debug_console.show()

    def clear_waveforms(self):
        """Clear any loaded waveform data."""
        self.pybert.clear_reference_from_plots()

    def run_simulation(self):
        """Start the simulation."""
        logger.info("Starting simulation...")
        self.pybert.simulate()

    def stop_simulation(self):
        """Stop the running simulation."""
        logger.info("Stopping simulation...")
        self.pybert.stop_simulation()

    def show_getting_started(self):
        """Open the getting started guide in the user's default web browser."""
        webbrowser.open(GETTING_STARTED_URL)

    def show_about(self):
        """Open up a dialog box with information about pybert."""
        about_box = QMessageBox()
        about_box.setWindowTitle("PyBERT")
        about_box.setIcon(QMessageBox.Information)
        about_box.setStandardButtons(QMessageBox.Ok)
        about_box.setTextFormat(Qt.RichText)
        about_box.setText(
            "PyBERT is a tool for analyzing and simulating communication systems.<br><br>"
            f"Version: {__version__}<br>"
            f"{__authors__}<br>"
            f"{__date__}<br><br>"
            f"{__copy__}<br>"
            "All rights reserved World wide.<br>"
            "Website: <a href='https://github.com/capn-freako/PyBERT'>https://github.com/capn-freako/PyBERT</a><br>"
        )
        about_box.exec()
