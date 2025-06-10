from contextlib import contextmanager
from typing import List, Literal, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QWidget,
)


@contextmanager
def block_signals(widget: QWidget):
    """Context manager to temporarily block all signals from a widget and its children."""
    try:
        for child in widget.findChildren(QWidget):
            child.blockSignals(True)
        yield
    finally:
        for child in widget.findChildren(QWidget):
            child.blockSignals(False)


def setup_table(table: QTableWidget, headers: List[str], editable_cols: Optional[List[int]] = None):
    """Set up a QTableWidget with headers and editable columns."""
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    for col in range(len(headers)):
        for row in range(table.rowCount()):
            item = table.item(row, col)
            if item and (editable_cols is None or col not in editable_cols):
                item.setFlags(item.flags() & ~table.Qt.ItemIsEditable)


class StatusIndicator(QLabel):
    """A reusable status indicator widget that shows validation/status states with appropriate styling.

    States:
        - not_loaded: Gray, neutral state
        - valid: Green, success state
        - invalid: Red, error state
        - warning: Yellow, warning state
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Get a temporary button to get the height of the button.
        button = QPushButton()
        button_height = button.sizeHint().height()

        self.setStyleSheet(
            f"""
            QLabel {{
                padding: 0px 8px;
                border-radius: 3px;
                font-weight: bold;
                min-height: {button_height}px;
                max-height: {button_height}px;
            }}
            QLabel[status="valid"] {{
                background-color: #d4edda;
                color: #155724;
            }}
            QLabel[status="invalid"] {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            QLabel[status="warning"] {{
                background-color: #fff3cd;
                color: #856404;
            }}
            QLabel[status="not_loaded"] {{
                background-color: #e2e3e5;
                color: #383d41;
            }}
        """
        )
        self.setAlignment(Qt.AlignCenter)
        self.set_status("not_loaded")

    def set_status(self, status: Literal["valid", "invalid", "warning", "not_loaded"]):
        """Update the status and appearance of the indicator."""
        status_text = {"valid": "Valid", "invalid": "Invalid", "warning": "Warning", "not_loaded": "Not Loaded"}
        self.setText(status_text[status])
        self.setProperty("status", status)
        # Force style update
        self.style().unpolish(self)
        self.style().polish(self)


class FilePickerWidget(QWidget):
    """A reusable widget that combines a label, readonly line edit, and browse button for file selection.

    This widget encapsulates the common pattern of having a label, readonly line edit for displaying
    a file path, and a browse button to open a file dialog.
    """

    file_selected = Signal(str)  # Signal emitted when a file is selected

    def __init__(
        self,
        label_text: str = "File",
        file_filter: str = "All Files (*.*)",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the file picker widget.

        Args:
            label_text: Text to display in the label
            file_filter: Filter string for the file dialog (e.g. "IBIS Files (*.ibs);;All Files (*.*)")
            parent: Parent widget
        """
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Add label
        layout.addWidget(QLabel(label_text))

        # Add readonly line edit
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        layout.addWidget(self.file_edit)

        # Add browse button
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_btn)

        self.file_filter = file_filter

    def _browse_file(self) -> None:
        """Open file dialog and emit signal if file is selected."""
        from pybert.gui.dialogs import select_file_dialog

        filename = select_file_dialog(self, "Select File", self.file_filter)
        if filename:
            self.file_edit.setText(filename)
            self.file_selected.emit(filename)

    def set_text(self, text: str) -> None:
        """Set the text in the line edit."""
        self.file_edit.setText(text)

    def text(self) -> str:
        """Get the current text from the line edit."""
        return self.file_edit.text()
