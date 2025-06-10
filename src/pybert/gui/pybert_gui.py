"""Main window for PyBERT application using PySide6.

This module implements the main window and overall GUI structure for
PyBERT.
"""

import logging
import webbrowser
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pybert import __authors__, __copy__, __date__, __version__
from pybert.configuration import CONFIG_LOAD_WILDCARD, CONFIG_SAVE_WILDCARD
from pybert.constants import GETTING_STARTED_URL
from pybert.gui.dialogs import warning_dialog
from pybert.gui.tabs import ConfigTab, OptimizerTab, ResultsTab
from pybert.gui.widgets import DebugConsoleWidget
from pybert.pybert import PyBERT
from pybert.utility.logger import QStatusBarHandler

logger = logging.getLogger("pybert")


class PyBERTSignals(QObject):
    """Signals for PyBERT model changes."""

    model_changed = Signal()  # Emitted when model parameters change
    configuration_loaded = Signal()  # Emitted when new configuration is loaded
    results_loaded = Signal(object, object)  # Emitted when new results are loaded
    reference_results_loaded = Signal(object)  # Emitted when new reference results are loaded


class PyBERTGUI(QMainWindow):
    """Main window for the PyBERT application."""

    def __init__(self, pybert: PyBERT | None = None, show_debug: bool = False, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            pybert: PyBERT model instance
            show_debug: Whether to show additional debug information
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.pybert: PyBERT = pybert
        self._signals = PyBERTSignals()

        self.setWindowTitle(f"PyBERT v{__version__} - Untitled")
        self.resize(1920, 1080)

        # Set window icon
        icon_path = Path(__file__).parent / "images" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create central widget and main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget)

        # Create and add tabs
        self.config_tab = ConfigTab(pybert, self.tab_widget)
        self.tab_widget.addTab(self.config_tab, "Setup")

        self.optimizer_tab = OptimizerTab(pybert, self.tab_widget)
        self.tab_widget.addTab(self.optimizer_tab, "Optimization")

        self.results_tab = ResultsTab(pybert, self.tab_widget)
        self.tab_widget.addTab(self.results_tab, "Results")

        # Create the dock widget/debug console
        self.debug_console = DebugConsoleWidget(self, show_debug)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.debug_console)

        self.create_menus(show_debug)
        self.create_status_bar()

        self.last_config_filepath = None
        self.last_results_filepath = None

        # Connect PyBERT signals if available
        if self.pybert:
            self.connect_signals()
            self.connect_configuration_signals()

    def connect_signals(self):
        """Connect PyBERT signals to status bar update slots."""
        self.pybert.sim_complete.connect(self.update_status_bar)
        self.pybert.status_update.connect(lambda msg: self.status_bar.showMessage(msg))

    def connect_configuration_signals(self) -> None:
        """Connect configuration_loaded signal to all configuration widgets."""
        if not self.pybert:
            return

        # Connect to all configuration widgets
        self._signals.configuration_loaded.connect(self.config_tab.sim_control.update_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.channel_config.update_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.tx_config.update_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.rx_config.update_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.tx_config.tx_equalization.update_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.rx_config.rx_equalization.update_from_model)

        # Connect to all results widgets
        self._signals.results_loaded.connect(self.results_tab.update_results)
        self._signals.reference_results_loaded.connect(self.results_tab.add_reference_plots)

    def create_status_bar(self):
        """Create and setup the status bar with permanent widgets."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create permanent status widgets with styling
        style = "padding: 0px 10px; margin: 2px; border-left: 1px solid #cccccc;"

        self.perf_label = QLabel("Perf: 0.0 Msmpls/min")
        self.perf_label.setStyleSheet(style)

        self.delay_label = QLabel("Channel Delay: 0.0 ns")
        self.delay_label.setStyleSheet(style)

        self.errors_label = QLabel("Bit Errors: 0")
        self.errors_label.setStyleSheet(style)

        self.power_label = QLabel("Tx Power: 0.0 mW")
        self.power_label.setStyleSheet(style)

        # Jitter metrics
        self.isi_label = QLabel("ISI: 0.0 ps")
        self.isi_label.setStyleSheet(style)

        self.dcd_label = QLabel("DCD: 0.0 ps")
        self.dcd_label.setStyleSheet(style)

        self.pj_label = QLabel("Pj: 0.0 ps")
        self.pj_label.setStyleSheet(style)

        self.rj_label = QLabel("Rj: 0.0 ps")
        self.rj_label.setStyleSheet(style)

        # Add permanent widgets to status bar (right-aligned)
        self.status_bar.addPermanentWidget(self.perf_label)
        self.status_bar.addPermanentWidget(self.delay_label)
        self.status_bar.addPermanentWidget(self.errors_label)
        self.status_bar.addPermanentWidget(self.power_label)
        self.status_bar.addPermanentWidget(self.isi_label)
        self.status_bar.addPermanentWidget(self.dcd_label)
        self.status_bar.addPermanentWidget(self.pj_label)
        self.status_bar.addPermanentWidget(self.rj_label)

        # Add the status bar handler for logging
        status_bar_handler = QStatusBarHandler()
        logger.addHandler(status_bar_handler)
        status_bar_handler.new_record.connect(self.status_bar.showMessage)

    def update_status_bar(self, results, perf):
        """Update the status bar with the performance metrics."""
        self.perf_label.setText(f"Perf: {perf.total * 6e-05:6.3f} Msmpls/min")
        self.delay_label.setText(f"Channel Delay: {self.pybert.chnl_dly * 1000000000.0:5.3f} ns")
        self.errors_label.setText(f"Bit Errors: {int(self.pybert.bit_errs)}")
        self.power_label.setText(f"Tx Power: {self.pybert.rel_power * 1e3:3.0f} mW")
        self.isi_label.setText(f"ISI: {self.pybert.isi_dfe* 1.0e12:6.1f} ps")
        self.dcd_label.setText(f"DCD: {self.pybert.dcd_dfe* 1.0e12:6.1f} ps")
        self.pj_label.setText(f"Pj: {self.pybert.pj_dfe* 1.0e12:6.1f} ({self.pybert.pjDD_dfe * 1.0e12:6.1f}) ps")
        self.rj_label.setText(f"Rj: {self.pybert.rj_dfe* 1.0e12:6.1f} ({self.pybert.rjDD_dfe * 1.0e12:6.1f}) ps")

    def create_file_menu(self):
        """Create the file menu for the application with the basic resetting, loading, saving, and quitting."""
        file_menu = self.menuBar().addMenu("&File")

        new_action = QAction("New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_config)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        load_results_action = QAction("Load Results", self)
        load_results_action.triggered.connect(self.load_results)

        load_reference_action = QAction("Load as Reference", self)
        load_reference_action.triggered.connect(self.load_results_as_reference)

        save_results_action = QAction("Save Results", self)
        save_results_action.triggered.connect(self.save_results)

        load_config_action = QAction("Load Config...", self)
        load_config_action.setShortcut("Ctrl+O")
        load_config_action.triggered.connect(self.load_config)

        save_config_action = QAction("Save", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_config)

        save_config_as_action = QAction("Save As...", self)
        save_config_as_action.setShortcut("Ctrl+Shift+S")
        save_config_as_action.triggered.connect(self.save_config_as)

        file_menu.addAction(new_action)
        file_menu.addSeparator()
        file_menu.addAction(load_config_action)
        file_menu.addAction(save_config_action)
        file_menu.addAction(save_config_as_action)
        file_menu.addSeparator()
        file_menu.addAction(load_results_action)
        file_menu.addAction(load_reference_action)
        file_menu.addAction(save_results_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def create_view_menu(self, show_debug: bool = False):
        """Create the view menu with options to manipulate the GUI."""
        view_menu = self.menuBar().addMenu("&View")

        debug_console_action = QAction("Debug Console", self)
        debug_console_action.setShortcut("Ctrl+`")
        debug_console_action.setChecked(show_debug)
        debug_console_action.triggered.connect(self.toggle_console_view)
        view_menu.addAction(debug_console_action)

        # Add logging level submenu
        logging_menu = view_menu.addMenu("Set Logging Level")

        # Create logging level actions
        normal_action = QAction("Normal", self)
        normal_action.setCheckable(True)
        normal_action.triggered.connect(lambda: self.set_logging_level(logging.INFO))

        debug_action = QAction("Debug", self)
        debug_action.setCheckable(True)
        debug_action.setChecked(True) if show_debug else normal_action.setChecked(True)
        debug_action.triggered.connect(lambda: self.set_logging_level(logging.DEBUG))

        # Add actions to logging menu
        logging_menu.addAction(normal_action)
        logging_menu.addAction(debug_action)

        # Store actions for later use
        self._logging_actions = {logging.INFO: normal_action, logging.DEBUG: debug_action}

        view_menu.addSeparator()

        clear_reference_action = QAction("Clear Reference", self)
        clear_reference_action.triggered.connect(self.clear_reference_plots)
        view_menu.addAction(clear_reference_action)

    def create_menus(self, show_debug: bool = False):
        """Create the application menus."""
        self.create_file_menu()
        self.create_view_menu(show_debug)

        # Simulation menu
        sim_menu = self.menuBar().addMenu("Simulate")

        run_action = QAction("Start", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_simulation)

        abort_action = QAction("Abort", self)
        abort_action.triggered.connect(self.stop_simulation)

        sim_menu.addAction(run_action)
        sim_menu.addAction(abort_action)

        # Optimization menu
        opt_menu = self.menuBar().addMenu("&Optimization")
        optimize_eq_action = QAction("Optimize", self)
        optimize_eq_action.setShortcut("Ctrl+T")
        optimize_eq_action.triggered.connect(self.start_optimization)
        opt_menu.addAction(optimize_eq_action)
        abort_optimize_action = QAction("Abort", self)
        abort_optimize_action.triggered.connect(self.stop_optimization)
        opt_menu.addAction(abort_optimize_action)

        opt_menu.addSeparator()

        apply_eq_action = QAction("Apply EQ", self)
        opt_menu.addAction(apply_eq_action)
        apply_eq_action.triggered.connect(self.apply_optimization)
        reset_eq_action = QAction("Reset EQ", self)
        reset_eq_action.triggered.connect(self.reset_optimization)
        opt_menu.addAction(reset_eq_action)

        # Tools menu
        # tools_menu = self.menuBar().addMenu("&Tools")

        # Presets submenu
        # presets_menu = tools_menu.addMenu("Presets")
        # TODO: Add presents like USB3, PCIe, etc.

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
            results = self.pybert.load_results(Path(file_path))
            self.last_results_filepath = Path(file_path)
            self._signals.results_loaded.emit(results.results, results.performance)

    def load_results_as_reference(self):
        """Load results from a file as reference.

        Only impulse, step, pulse, and frequency plots are loaded as reference plots.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Results as Reference", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path and self.pybert:
            results = self.pybert.load_results(Path(file_path))
            self._signals.reference_results_loaded.emit(results.results)

    def clear_reference_plots(self):
        """Clear the reference plots."""
        self.results_tab.clear_reference_plots()

    def save_results(self):
        """Save results to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path and self.pybert:
            self.pybert.save_results(Path(file_path))
            self.last_results_filepath = Path(file_path)

    def load_config(self):
        """Load configuration from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", CONFIG_LOAD_WILDCARD)
        if file_path and self.pybert:
            self.last_config_filepath = Path(file_path)
            self.pybert.load_configuration(self.last_config_filepath)
            self._signals.configuration_loaded.emit()
            self.setWindowTitle(f"PyBERT v{__version__} - {self.last_config_filepath.name}")

    def save_config(self):
        """Save configuration to the current file."""
        if self.pybert and self.last_config_filepath:
            self.pybert.save_configuration(self.last_config_filepath)
        else:
            self.save_config_as()

    def save_config_as(self):
        """Save configuration to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", "", CONFIG_SAVE_WILDCARD)
        if file_path and self.pybert:
            self.last_config_filepath = Path(file_path)
            self.pybert.save_configuration(self.last_config_filepath)
            self.setWindowTitle(f"PyBERT v{__version__} - {self.last_config_filepath.name}")

    def new_config(self):
        """Create a new configuration."""
        self.setWindowTitle(f"PyBERT v{__version__} - Untitled")
        self.last_config_filepath = None
        self.pybert.reset_configuration()
        self._signals.configuration_loaded.emit()

    def toggle_console_view(self):
        """Toggle the debug console visibility."""
        if self.debug_console.isVisible():
            self.debug_console.hide()
        else:
            self.debug_console.show()

    def clear_waveforms(self):
        """Clear any loaded waveform data."""
        self.results_tab.clear_waveforms()

    def run_simulation(self):
        """Start the simulation."""
        self.pybert.simulate()

    def stop_simulation(self):
        """Stop the running simulation."""
        self.pybert.stop_simulation()

    def start_optimization(self):
        """Start the optimization."""
        trials = self.pybert.calculate_optimization_trials()
        if trials > 1_000_000:
            usr_resp = warning_dialog(
                "Large number of trials",
                f"You've opted to run over {trials // 1_000_000} million trials!\nAre you sure?",
            )
            if not usr_resp:
                return
        self.pybert.optimize()

    def stop_optimization(self):
        """Stop the running optimization."""
        self.pybert.stop_optimization()

    def reset_optimization(self):
        """Reset the optimization."""
        self.pybert.reset_optimization()

    def apply_optimization(self):
        """Apply the optimization."""
        self.pybert.apply_optimization()

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

    def set_logging_level(self, level: int):
        """Set the logging level for both the debug console and status bar.

        Args:
            level: The logging level to set (either logging.DEBUG or logging.INFO)
        """
        # Update logger level
        logger.setLevel(level)

        # Update action check states
        for log_level, action in self._logging_actions.items():
            action.setChecked(log_level == level)

        # Update debug console if it exists
        if hasattr(self, "debug_console"):
            self.debug_console.set_logging_level(level)
