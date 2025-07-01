"""Shared IBIS-AMI configuration widget for PyBERT GUI.

This widget provides common IBIS-AMI configuration UI elements used by
both transmitter and receiver equalization widgets.
"""

import logging
from pathlib import Path
from typing import Callable, Literal, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.file_picker import FilePickerWidget
from pybert.gui.widgets.status_indicator import StatusIndicator
from pybert.gui.widgets.utils import block_signals
from pybert.models.buffer import Receiver, Transmitter

logger = logging.getLogger("pybert.ibis")


class IbisConfigWidget(QWidget):
    """Widget for configuring IBIS parameters."""

    def __init__(
        self,
        obj_accessor: Callable[[], Transmitter | Receiver],
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the IBIS-AMI configuration widget.

        Args:
            obj_accessor: Function that returns the current Transmitter or Receiver instance
            parent: Parent widget
        """
        super().__init__(parent)
        self._obj_accessor = obj_accessor

        layout = QVBoxLayout()
        self.setLayout(layout)

        # IBIS file selection
        self.file_picker = FilePickerWidget("File", "IBIS Files (*.ibs);;IBIS Files (*.ibis);;All Files (*.*)", self)

        layout.addWidget(self.file_picker)

        # Bottom row with status, checkbox, and configure
        bottom_layout = QHBoxLayout()
        # Status indicator
        bottom_layout.addWidget(QLabel("Status:"))
        self.ibis_valid = StatusIndicator()
        bottom_layout.addWidget(self.ibis_valid)
        bottom_layout.addSpacing(20)  # Add some spacing between status and checkbox

        # On-die S-parameters
        self.use_ts4 = QCheckBox("Use on-die S-parameters")
        self.use_ts4.setEnabled(False)
        bottom_layout.addWidget(self.use_ts4)
        bottom_layout.addStretch()  # Push configure button to the right

        # Configure button
        self.configure_btn = QPushButton("Configure")
        self.configure_btn.setEnabled(False)
        bottom_layout.addWidget(self.configure_btn)

        layout.addLayout(bottom_layout)
        layout.addStretch()

        self.connect_signals()
        self.update_widget_from_model()

    @property
    def obj(self) -> Transmitter | Receiver:
        """Get the current object instance."""
        return self._obj_accessor()

    def connect_signals(self) -> None:
        """Connect signals to PyBERT instance."""
        # Use lambda to capture the current object at signal time
        self.use_ts4.toggled.connect(lambda val: setattr(self.obj, "use_ts4", val))

    def update_widget_from_model(self) -> None:
        """Update the widget from the PyBERT model.

        This method ensures the widget is fully synchronized with the
        PyBERT model, including resetting its state when no model is
        loaded.
        """
        try:
            with block_signals(self):
                current_obj = self.obj  # Get current object
                # Get current state from PyBERT using model methods
                ibis_file = current_obj.get_ibis_file_path()
                ibis_valid = current_obj.is_ibis_loaded()

                # Update file path
                self.file_picker.set_filepath(str(ibis_file) if ibis_file else "")

                # Update status
                self.set_status("valid" if ibis_valid else "not_loaded")

                # Update TS4 checkbox
                self.use_ts4.setEnabled(current_obj.has_ts4)
                self.use_ts4.setChecked(current_obj.use_ts4)
        except Exception as e:
            logger.error(f"Failed to update IBIS config widget from model: {e}")
            self.set_status("invalid")

    def set_status(self, status: Literal["valid", "invalid", "warning", "not_loaded"]) -> None:
        """Set the status of the AMI model."""
        self.ibis_valid.set_status(status)

        if status == "valid":
            self.configure_btn.setEnabled(True)
        elif status == "not_loaded":
            self.configure_btn.setEnabled(False)
            self.set_filepath(None)
        else:
            self.configure_btn.setEnabled(False)
            self.set_filepath(None)
            self.use_ts4.setEnabled(False)

    def set_filepath(self, filepath: str | Path | None) -> None:
        """Set the filepaths of the AMI and DLL files."""
        self.file_picker.set_filepath(str(filepath) if filepath else "")
