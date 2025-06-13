"""Shared IBIS-AMI configuration widget for PyBERT GUI.

This widget provides common IBIS-AMI configuration UI elements used by both
transmitter and receiver equalization widgets.
"""

import logging
from typing import Callable, Optional

from pyibisami import IBISModel
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pybert.gui import dialogs
from pybert.gui.widgets.utils import FilePickerWidget, StatusIndicator, block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.ibis")


class IbisConfigWidget(QWidget):
    """Widget for configuring IBIS parameters."""

    def __init__(
        self,
        pybert: PyBERT,
        is_tx: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the IBIS-AMI configuration widget.

        Args:
            pybert: PyBERT instance
            is_tx: True if the IBIS model is for a transmitter, False if for a receiver
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # TODO: This is temporary until we can reference a Transmitter or Receiver object.
        self.ibis_direction = "tx" if is_tx else "rx"
        self.use_ts4_attr = f"{self.ibis_direction}_use_ts4"
        self.ibis_file_attr = f"{self.ibis_direction}_ibis_file"
        self.ibis_valid_attr = f"{self.ibis_direction}_ibis_valid"
        self.use_ibis_attr = f"{self.ibis_direction}_use_ibis"
        self.has_ts4_attr = f"{self.ibis_direction}_has_ts4"

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
        self.view_btn = QPushButton("Configure")
        self.view_btn.setEnabled(False)
        bottom_layout.addWidget(self.view_btn)

        layout.addLayout(bottom_layout)
        layout.addStretch()

        self.connect_signals()
        self.update_widget_from_model()

    def connect_signals(self) -> None:
        """Connect signals to PyBERT instance."""
        self.use_ts4.toggled.connect(lambda val: setattr(self.pybert, self.use_ts4_attr, val))

    def set_status(self, status: str) -> None:
        """Set the status of the IBIS file."""
        self.ibis_valid.set_status(status)

    def update_widget_from_model(self) -> None:
        """Update the widget from the PyBERT model.

        This method ensures the widget is fully synchronized with the PyBERT model,
        including resetting its state when no model is loaded.
        """
        with block_signals(self):
            # Get current state from PyBERT
            ibis_file = getattr(self.pybert, self.ibis_file_attr, "")
            ibis_valid = getattr(self.pybert, self.ibis_valid_attr, False)
            use_ts4 = getattr(self.pybert, self.use_ts4_attr, False)
            has_ts4 = getattr(self.pybert, self.has_ts4_attr, False)

            # Update file path
            self.file_picker.set_filepath(ibis_file)

            # Update status and button states
            self.set_status("valid" if ibis_valid else "not_loaded")
            self.view_btn.setEnabled(ibis_valid)

            # Update TS4 checkbox
            self.use_ts4.setEnabled(has_ts4)
            self.use_ts4.setChecked(use_ts4)

    def reset(self) -> None:
        """Reset the widget to its initial state."""
        with block_signals(self):
            self.file_picker.set_filepath("")
            self.set_status("not_loaded")
            self.view_btn.setEnabled(False)
            self.use_ts4.setEnabled(False)
            self.use_ts4.setChecked(False)
