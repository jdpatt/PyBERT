"""If the results window is split, this class will be used to display the results.

Normally, the results is just another tab in the main window.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from pybert import __version__
from pybert.gui.tabs import ResultsTab
from pybert.pybert import PyBERT
from pybert.results import Results


class ResultsWindow(QMainWindow):
    """Separate window for displaying PyBERT results."""

    # Signal emitted when the window is closed
    window_closed = Signal()

    def __init__(self, pybert: PyBERT, name: str = "Untitled", parent: Optional[QWidget] = None):
        """Initialize the results window.

        Args:
            pybert: PyBERT model instance
            name: Name to display in the window title
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        self.update_title(name)
        self.resize(1200, 800)

        # Set window icon
        icon_path = Path(__file__).parent / "images" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create results tab
        self.results_tab = ResultsTab(pybert, central_widget)
        layout.addWidget(self.results_tab)

        # Add quit action with Ctrl+Q shortcut
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

        # Connect signals
        self._connect_signals()

    def closeEvent(self, event):
        """Handle window close event."""
        # Emit signal to notify main window that this window is closing
        self.window_closed.emit()
        super().closeEvent(event)

    def _connect_signals(self):
        """Connect signals to the results tab."""
        if not self.pybert:
            return

        # These signals will be connected from the main window

    def update_results(self, results: Results):
        """Update results in the window."""
        self.results_tab.update_results(results)

    def update_title(self, name: str):
        """Update the window title."""
        self.setWindowTitle(f"PyBERT v{__version__} - {name} Results")

    def add_reference_plots(self, results):
        """Add reference plots to the window."""
        self.results_tab.add_reference_plots(results)

    def clear_reference_plots(self):
        """Clear reference plots from the window."""
        self.results_tab.clear_reference_plots()

    def clear_results(self):
        """Clear the results."""
        self.results_tab.clear_results()
