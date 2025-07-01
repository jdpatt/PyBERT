from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)


class FilePickerWidget(QWidget):
    """A reusable widget that combines a label, readonly line edit, and browse button for file selection.

    This widget encapsulates the common pattern of having a label,
    readonly line edit for displaying a file path, and a browse button
    to open a file dialog.
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
        filename, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)
        if filename:
            self.file_edit.setText(filename)
            self.file_selected.emit(filename)

    def set_filepath(self, text: str) -> None:
        """Set the text in the line edit."""
        self.file_edit.setText(text)
        if text:
            self.file_selected.emit(text)

    def text(self) -> str:
        """Get the current text from the line edit."""
        return self.file_edit.text()

    def clear(self) -> None:
        """Clear the text in the line edit."""
        self.file_edit.clear()
