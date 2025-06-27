"""Shared IBIS-AMI configuration widget for PyBERT GUI.

This widget provides common IBIS-AMI configuration UI elements used by both
transmitter and receiver equalization widgets.
"""

import logging
from pathlib import Path
from typing import Callable, Literal, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.status_indicator import StatusIndicator
from pybert.gui.widgets.utils import block_signals
from pybert.models.buffer import Receiver, Transmitter

logger = logging.getLogger("pybert.ibis")


class IbisAmiConfigWidget(QWidget):
    """Widget for configuring IBIS-AMI parameters shared between TX and RX."""

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
        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # IBIS-AMI file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("AMI File"))
        self.ami_file = QLineEdit()
        self.ami_file.setReadOnly(True)
        file_layout.addWidget(self.ami_file)
        layout.addLayout(file_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("DLL File"))
        self.dll_file = QLineEdit()
        self.dll_file.setReadOnly(True)
        file_layout.addWidget(self.dll_file)
        layout.addLayout(file_layout)

        # Bottom row with status, checkboxes, and configure
        bottom_layout = QHBoxLayout()

        # Status indicator
        bottom_layout.addWidget(QLabel("Status:"))
        self.ami_model_valid = StatusIndicator()
        bottom_layout.addWidget(self.ami_model_valid)
        bottom_layout.addSpacing(20)

        # GetWave checkbox
        self.use_getwave = QCheckBox("Use GetWave()")
        self.use_getwave.setEnabled(False)
        self.use_getwave.toggled.connect(lambda val: setattr(self.obj, "use_getwave", val))
        bottom_layout.addWidget(self.use_getwave)
        bottom_layout.addSpacing(10)

        # Clocks checkbox (optional)
        # We'll check the object type dynamically
        self.use_clocks = QCheckBox("Use Clocks")
        self.use_clocks.setEnabled(False)
        self.use_clocks.toggled.connect(lambda val: setattr(self.obj, "use_clocks", val))
        bottom_layout.addWidget(self.use_clocks)
        bottom_layout.addSpacing(10)

        bottom_layout.addStretch()

        # Configure button
        self.configure_btn = QPushButton("Configure")
        self.configure_btn.setEnabled(False)
        bottom_layout.addWidget(self.configure_btn)

        layout.addLayout(bottom_layout)
        layout.addStretch()

        self.update_widget_from_model()

    @property
    def obj(self) -> Transmitter | Receiver:
        """Get the current object instance."""
        return self._obj_accessor()

    def update_widget_from_model(self) -> None:
        """Update the widget from the PyBERT model.

        This method ensures the widget is fully synchronized with the PyBERT model,
        including resetting its state when no model is loaded.
        """
        try:
            with block_signals(self):
                current_obj = self.obj  # Get current object
                # Get current state from PyBERT using model methods
                ami_file = current_obj.get_ami_file_path()
                dll_file = current_obj.get_dll_file_path()
                ami_valid = current_obj.is_ami_loaded()
                has_getwave = current_obj.has_getwave
                use_getwave = current_obj.use_getwave

                # Update file paths
                self.set_filepaths(ami_file, dll_file)

                # Update status and button states
                self.ami_model_valid.set_status("valid" if ami_valid else "not_loaded")
                self.configure_btn.setEnabled(ami_valid)

                # Update checkboxes
                self.use_getwave.setEnabled(has_getwave)
                self.use_getwave.setChecked(use_getwave)

                # Update clocks checkbox based on object type
                if isinstance(current_obj, Receiver):
                    self.use_clocks.setVisible(True)
                    self.use_clocks.setEnabled(has_getwave)
                    self.use_clocks.setChecked(current_obj.use_clocks)
                else:
                    self.use_clocks.setVisible(False)
        except Exception as e:
            logger.error(f"Failed to update AMI config widget from model: {e}")
            self.set_status("invalid")

    def set_status(self, status: Literal["valid", "invalid", "warning", "not_loaded"]) -> None:
        """Set the status of the AMI model."""
        self.ami_model_valid.set_status(status)

        if status == "valid":
            current_obj = self.obj  # Get current object
            self.configure_btn.setEnabled(True)
            self.set_filepaths(current_obj.get_ami_file_path(), current_obj.get_dll_file_path())
            if current_obj.has_getwave:
                self.use_getwave.setEnabled(True)
                self.use_getwave.setChecked(current_obj.use_getwave)
                if isinstance(current_obj, Receiver):
                    self.use_clocks.setVisible(True)
                    self.use_clocks.setEnabled(True)
                    self.use_clocks.setChecked(current_obj.use_clocks)
        elif status == "not_loaded":
            self.set_filepaths(None, None)
            self.ami_model_valid.set_status("not_loaded")
            self.configure_btn.setEnabled(False)
            self.use_getwave.setEnabled(False)
            self.use_getwave.setChecked(False)
            self.use_clocks.setEnabled(False)
            self.use_clocks.setChecked(False)
        else:
            self.configure_btn.setEnabled(False)
            self.set_filepaths(None, None)
            self.use_getwave.setEnabled(False)
            self.use_clocks.setEnabled(False)

    def set_filepaths(self, ami_file: str | Path | None, dll_file: str | Path | None) -> None:
        """Set the filepaths of the AMI and DLL files."""
        self.ami_file.setText(str(ami_file) if ami_file else "")
        self.dll_file.setText(str(dll_file) if dll_file else "")
