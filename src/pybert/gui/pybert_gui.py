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
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pybert import __authors__, __copy__, __date__, __version__
from pybert.configuration import CONFIG_LOAD_WILDCARD, CONFIG_SAVE_WILDCARD
from pybert.constants import GETTING_STARTED_URL
from pybert.gui.dialogs import warning_dialog
from pybert.gui.results_window import ResultsWindow
from pybert.gui.tabs import ConfigTab, OptimizerTab, ResultsTab
from pybert.gui.widgets import DebugConsoleWidget
from pybert.gui.widgets.status_bar import StatusBar
from pybert.pybert import PyBERT
from pybert.results import Results

logger = logging.getLogger("pybert")


class PyBERTSignals(QObject):
    """Signals for PyBERT model changes.

    To keep the GUI responsive and thread safe, we emit signals instead
    of calling methods directly. This allows the GUI to update in the
    main thread without blocking the worker thread. Basically, the
    callback will call a method to emit the signal which will get
    scheduled to run in the main thread.
    """

    configuration_loaded = Signal()  # Emitted when new configuration is loaded
    results_loaded = Signal(object)  # Emitted when new results are loaded
    reference_results_loaded = Signal(object)  # Emitted when new reference results are loaded
    sim_complete = Signal(object)  # Emitted when simulation is complete and new results are available
    opt_complete = Signal(object)  # Emitted when optimization is complete and new results are available
    opt_loop_complete = Signal(
        object
    )  # Emitted when an optimization loop is complete and we need to plot intermediate results
    status_update = Signal(str)  # Emitted when status updates are available. These are in the bottom status bar.


class PyBERTGUI(QMainWindow):
    """Main window for the PyBERT application.

    This class is responsible for creating the main window and managing
    the tabs. It also manages the status bar and the debug console.

    The main window is a QMainWindow that contains a QTabWidget, a
    QStatusBar, and a QDockWidget for the debug console. This requires
    that a QApplication is created before this class is instantiated.
    """

    def __init__(self, pybert: PyBERT, show_debug: bool = False, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            pybert: PyBERT model instance
            show_debug: Whether to show additional debug information and to show the debug console by default
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.pybert = pybert
        self._signals = PyBERTSignals()

        # Track results window state
        self.results_window = None
        self.results_split = False

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
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.debug_console)

        self.create_menus(show_debug)
        self.status_bar = StatusBar(self.pybert, self)
        self.setStatusBar(self.status_bar)

        self.last_config_filepath = None
        self.last_results_filepath = None

        if self.pybert:
            self.connect_callbacks()
            self.connect_signals()

    def connect_callbacks(self):
        """Connect PyBERT callbacks to GUI update methods.

        These callbacks are called by the PyBERT model to maintain
        thread safety.
        """
        self.pybert.add_simulation_callback(self._handle_simulation_complete)
        self.pybert.add_optimization_callback(self._handle_optimization_complete)
        self.pybert.add_optimization_loop_callback(self._handle_optimization_loop)
        self.pybert.add_status_callback(self._handle_status_update)

    def connect_signals(self) -> None:
        """Connect signals to all configuration widgets."""

        # Connect to all configuration widgets -  So they can sync with the model when the configuration is loaded.
        self._signals.configuration_loaded.connect(self.config_tab.sim_config.update_widget_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.channel_config.update_widget_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.tx_config.update_widget_from_model)
        self._signals.configuration_loaded.connect(self.config_tab.rx_config.update_widget_from_model)

        # Connect to all results widgets
        self._signals.results_loaded.connect(self._handle_simulation_results)
        self._signals.reference_results_loaded.connect(self.results_tab.add_reference_plots)

        # Connect all simulation and optimization signals
        self._signals.sim_complete.connect(self._handle_simulation_results)
        self._signals.opt_complete.connect(self._handle_optimization_results)
        self._signals.opt_loop_complete.connect(self.optimizer_tab.update_plot)
        self._signals.status_update.connect(self.status_bar.showMessage)

    # QT Overrides ------------------------------------------------------------
    def closeEvent(self, event):
        """Handle window close event, if the results window is open, close it as well."""
        if self.results_window:
            self.results_window.close()
        super().closeEvent(event)

    # Menu creation ------------------------------------------------------------
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

        # Add split results window action
        self.split_results_action = QAction("Split Results Window", self)
        self.split_results_action.setCheckable(True)
        self.split_results_action.triggered.connect(self.toggle_results_window)
        view_menu.addAction(self.split_results_action)

        # Add logging level submenu
        logging_menu = view_menu.addMenu("Set Logging Level")

        # Create logging level actions
        normal_action = QAction("Normal", self)
        normal_action.setCheckable(True)
        normal_action.triggered.connect(lambda: self._handle_logging_level_change(logging.INFO))

        debug_action = QAction("Debug", self)
        debug_action.setCheckable(True)
        debug_action.setChecked(True) if show_debug else normal_action.setChecked(True)
        debug_action.triggered.connect(lambda: self._handle_logging_level_change(logging.DEBUG))

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

    # Results window management ------------------------------------------------------------
    def toggle_results_window(self):
        """Toggle the results window split state to either split or just a tab in the main window."""
        if self.results_split:
            self.merge_results_window()
        else:
            self.split_results_window()

    def split_results_window(self):
        """Split the results tab into a separate window."""
        if self.results_split:
            return

        # Create the results window
        self.results_window = ResultsWindow(
            self.pybert, self.last_config_filepath.name if self.last_config_filepath else "Untitled", self
        )

        # Connect signals to the results window
        self._signals.results_loaded.connect(self.results_window.update_results)
        self._signals.reference_results_loaded.connect(self.results_window.add_reference_plots)

        # Connect to the results window's close signal for automatic merge
        self.results_window.window_closed.connect(self.merge_results_window)

        # Hide the results tab in the main window
        self.tab_widget.removeTab(self.tab_widget.indexOf(self.results_tab))
        self.tab_widget.setCurrentIndex(0)

        # Show the results window
        self.results_window.show()

        # Update state
        self.results_split = True
        self.split_results_action.setChecked(True)

        # Position the results window next to the main window
        main_geometry = self.geometry()
        self.results_window.move(main_geometry.x() + main_geometry.width() + 10, main_geometry.y())

        # Update the results window with any existing results
        self._update_results_with_current_data(self.results_window)

    def merge_results_window(self):
        """Merge the results window back into the main window."""
        if not self.results_split or not self.results_window:
            return

        # Disconnect signals from the results window
        try:
            self._signals.results_loaded.disconnect(self.results_window.update_results)
            self._signals.reference_results_loaded.disconnect(self.results_window.add_reference_plots)
            self.results_window.window_closed.disconnect(self.merge_results_window)
        except:
            pass  # Signals might not be connected

        # Close the results window
        self.results_window.close()
        self.results_window = None

        # Add the results tab back to the main window
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.setCurrentIndex(0)

        # Update state
        self.results_split = False
        self.split_results_action.setChecked(False)

        # Update the results tab with any existing results
        self._update_results_with_current_data(self.results_tab)

    # Menu action handlers ------------------------------------------------------------
    def load_results(self):
        """Load results from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path:
            path = Path(file_path)
            results = self.pybert.load_results(path)
            self.last_results_filepath = path
            self._signals.results_loaded.emit(results)

    def load_results_as_reference(self):
        """Load results from a file as reference.

        Only impulse, step, pulse, and frequency plots are loaded as
        reference plots.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Results as Reference", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path:
            results = self.pybert.load_results(Path(file_path))
            self._signals.reference_results_loaded.emit(results)

    def clear_reference_plots(self):
        """Clear the reference plots."""
        if self.results_split and self.results_window:
            self.results_window.clear_reference_plots()
        else:
            self.results_tab.clear_reference_plots()

    def save_results(self):
        """Save results to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "PyBERT Results (*.pybert_data);;All Files (*.*)"
        )
        if file_path:
            path = Path(file_path)
            self.pybert.save_results(path)
            self.last_results_filepath = path

    def load_config(self):
        """Load configuration from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", CONFIG_LOAD_WILDCARD)
        if file_path:
            self.last_config_filepath = Path(file_path)
            self.pybert.load_configuration(self.last_config_filepath)
            self._signals.configuration_loaded.emit()
            self._update_window_title(self.last_config_filepath.name)

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
            self._update_window_title(self.last_config_filepath.name)
            if self.results_split and self.results_window:
                self.results_window.update_title(self.last_config_filepath.name)

    def new_config(self):
        """Create a new configuration."""
        self._update_window_title("Untitled")
        self.last_config_filepath = None
        self.pybert.reset_configuration()
        self._signals.configuration_loaded.emit()

    def toggle_console_view(self):
        """Toggle the debug console visibility."""
        if self.debug_console.isVisible():
            self.debug_console.hide()
        else:
            self.debug_console.show()

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
                self,
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
        about_box.setIcon(QMessageBox.Icon.Information)
        about_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        about_box.setTextFormat(Qt.TextFormat.RichText)
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

    def _handle_logging_level_change(self, level: int):
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

    def _handle_simulation_results(self, results: Results):
        """Slot to update all simulation results widgets. Runs on the main GUI thread.

        Args:
            results: The results object containing the simulation results
        """
        self.status_bar.update(results)
        if self.results_split and self.results_window:
            self.results_window.update_results(results)
        else:
            self.results_tab.update_results(results)

    def _handle_optimization_results(self, peak_mag: float):
        """Slot to update optimization result widgets. Runs on the main GUI thread.

        When optimization is complete, we need to update the RX CTLE boost to the final peak magnitude.

        Args:
            peak_mag: The peak magnitude of the optimization result
        """
        self.optimizer_tab.rx_ctle.set_ctle_boost(peak_mag)

    def _update_window_title(self, title: str):
        """Update the window title with the given title.

        Args:
            title: The title to update the window title with
        """
        self.setWindowTitle(f"PyBERT v{__version__} - {title}")

    def _handle_simulation_complete(self, results: Results):
        """Handle simulation completion from the worker thread.

        This must just emit a signal so that the GUI thread can schedule the GUI update without
        blocking the worker thread.

        Args:
            results: The results object containing the simulation results
        """
        self._signals.sim_complete.emit(results)

    def _handle_optimization_complete(self, opt_result: dict):
        """Handle optimization completion from the worker thread.

        This must just emit a signal so that the GUI thread can schedule the GUI update without
        blocking the worker thread.

        Args:
            opt_result: The optimization result
        """
        tx_weights = opt_result.get("tx_weights", [])
        rx_peaking = opt_result.get("rx_peaking", 0)
        for k, tx_weight in enumerate(tx_weights):
            self.pybert.tx_tap_tuners[k].value = tx_weight
        self.pybert.peak_mag_tune = rx_peaking
        self._signals.opt_complete.emit(self.pybert.peak_mag_tune)

    def _handle_optimization_loop(self, opt_loop_result: dict):
        """Handle optimization loop update from the worker thread.

        This must just emit a signal so that the GUI thread can schedule the GUI update without
        blocking the worker thread.

        Args:
            opt_loop_result: The optimization loop result
        """
        self._signals.opt_loop_complete.emit(opt_loop_result)

    def _handle_status_update(self, message: str):
        """Handle status update from the worker thread.

        This must just emit a signal so that the GUI thread can schedule the GUI update without
        blocking the worker thread.

        Args:
            message: The status update message
        """
        self._signals.status_update.emit(message)

    def _update_results_with_current_data(self, results_widget):
        """Update the given widget with current results data.

        Args:
            results_widget: Either self.results_tab or self.results_window
        """
        if not self.pybert.last_results:
            return
        results_widget.update_results(self.pybert.last_results)
