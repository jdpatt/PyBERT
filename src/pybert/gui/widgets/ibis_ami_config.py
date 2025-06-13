"""Shared IBIS-AMI configuration widget for PyBERT GUI.

This widget provides common IBIS-AMI configuration UI elements used by both
transmitter and receiver equalization widgets.
"""

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.utils import StatusIndicator, block_signals
from pybert.pybert import PyBERT


class IbisAmiConfigWidget(QWidget):
    """Widget for configuring IBIS-AMI parameters shared between TX and RX."""

    def __init__(
        self,
        pybert: PyBERT,
        is_tx: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the IBIS-AMI configuration widget.

        Args:
            pybert: PyBERT instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert
        self.ibis_direction = "tx" if is_tx else "rx"
        self.use_getwave_attr = f"{self.ibis_direction}_use_getwave"
        self.use_clocks_attr = f"{self.ibis_direction}_use_clocks"
        self.ami_file_attr = f"{self.ibis_direction}_ami_file"
        self.dll_file_attr = f"{self.ibis_direction}_dll_file"
        self.ami_valid_attr = f"{self.ibis_direction}_ami_valid"
        self.use_ami_attr = f"{self.ibis_direction}_use_ami"
        self.has_getwave_attr = f"{self.ibis_direction}_has_getwave"
        if not is_tx:
            self.has_clocks_attr = f"{self.ibis_direction}_has_clocks"
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

        # GetWave checkbox (optional)
        self.use_getwave = QCheckBox("Use GetWave()")
        self.use_getwave.setEnabled(False)
        self.use_getwave.toggled.connect(lambda val: setattr(self.pybert, self.use_getwave_attr, val))
        bottom_layout.addWidget(self.use_getwave)
        bottom_layout.addSpacing(10)

        # Clocks checkbox (optional)
        if not is_tx:
            self.use_clocks = QCheckBox("Use Clocks")
            self.use_clocks.setEnabled(False)
            self.use_clocks.toggled.connect(lambda val: setattr(self.pybert, self.use_clocks_attr, val))
            bottom_layout.addWidget(self.use_clocks)
            bottom_layout.addSpacing(10)

        bottom_layout.addStretch()

        # Configure button
        self.configure_btn = QPushButton("Configure")
        self.configure_btn.setEnabled(False)
        bottom_layout.addWidget(self.configure_btn)

        layout.addLayout(bottom_layout)
        layout.addStretch()

        self.update_from_model()

    def update_from_model(self) -> None:
        """Update the widget from the PyBERT model.

        This method ensures the widget is fully synchronized with the PyBERT model,
        including resetting its state when no model is loaded.
        """
        with block_signals(self):
            # Get current state from PyBERT
            use_ami = getattr(self.pybert, self.use_ami_attr, False)
            ami_file = getattr(self.pybert, self.ami_file_attr, "")
            dll_file = getattr(self.pybert, self.dll_file_attr, "")
            ami_valid = getattr(self.pybert, self.ami_valid_attr, False)
            has_getwave = getattr(self.pybert, self.has_getwave_attr, False)
            use_getwave = getattr(self.pybert, self.use_getwave_attr, False)
            if hasattr(self, "use_clocks"):
                use_clocks = getattr(self.pybert, self.use_clocks_attr, False)

            # Update file paths
            self.set_filepaths(ami_file, dll_file)

            # Update status and button states
            self.set_status("valid" if ami_valid else "not_loaded")
            self.configure_btn.setEnabled(ami_valid)

            # Update checkboxes if they exist
            if hasattr(self, "use_getwave"):
                self.use_getwave.setEnabled(has_getwave)
                self.use_getwave.setChecked(use_getwave)
            if hasattr(self, "use_clocks"):
                self.use_clocks.setEnabled(has_getwave)
                self.use_clocks.setChecked(use_clocks)

    def set_status(self, status: str) -> None:
        """Set the status of the AMI model."""
        self.ami_model_valid.set_status(status)

    def set_filepaths(self, ami_file: str | Path | None, dll_file: str | Path | None) -> None:
        """Set the filepaths of the AMI and DLL files."""
        self.ami_file.setText(str(ami_file) if ami_file else "")
        self.dll_file.setText(str(dll_file) if dll_file else "")

    def reset(self) -> None:
        """Reset the widget to its initial state."""
        with block_signals(self):
            self.set_filepaths(None, None)
            self.set_status("not_loaded")
            self.configure_btn.setEnabled(False)
            if hasattr(self, "use_getwave"):
                self.use_getwave.setEnabled(False)
                self.use_getwave.setChecked(False)
            if hasattr(self, "use_clocks"):
                self.use_clocks.setEnabled(False)
                self.use_clocks.setChecked(False)
