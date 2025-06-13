"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS
model selection and native parameters.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pybert.gui import dialogs
from pybert.gui.dialogs import select_file_dialog
from pybert.gui.widgets.ibis_manager import IbisAmiManager
from pybert.gui.widgets.tx_equalization import TxEqualizationWidget
from pybert.gui.widgets.utils import FilePickerWidget, StatusIndicator, block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.tx")


class TxConfigWidget(QWidget):
    """Widget for configuring transmitter parameters."""

    def __init__(self, pybert: PyBERT | None = None, parent: Optional[QWidget] = None) -> None:
        """Initialize the transmitter configuration widget.

        Args:
            pybert: PyBERT model instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.tx_config = QGroupBox("Transmitter")
        tx_layout = QVBoxLayout()
        self.tx_config.setLayout(tx_layout)

        # --- Mode selection radio buttons ---
        mode_layout = QHBoxLayout()
        self.native_radio = QRadioButton("Native")
        self.native_radio.setChecked(True)
        self.ibis_radio = QRadioButton("IBIS")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.ibis_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        tx_layout.addLayout(mode_layout)

        # --- Stacked layout for transmitter config groups ---
        self.stacked_widget = QStackedWidget()

        # Create IBIS-AMI manager
        self.ibis_ami_manager = IbisAmiManager(
            pybert=self.pybert,
            is_tx=True,
            parent=self,
        )

        # Native parameters group
        self.native_group = QWidget(self)
        native_form = QFormLayout()
        self.native_group.setLayout(native_form)

        self.rs = QDoubleSpinBox()
        self.rs.setRange(0.0, 1000.0)
        self.rs.setDecimals(2)
        self.rs.setSuffix(" Ohms")
        native_form.addRow(QLabel("Impedance"), self.rs)

        self.cout = QDoubleSpinBox()
        self.cout.setRange(0.0, 1000.0)
        self.cout.setDecimals(2)
        self.cout.setSuffix(" pF")
        native_form.addRow(QLabel("Capacitance"), self.cout)

        # Add native group to stacked widget
        self.stacked_widget.addWidget(self.native_group)
        self.stacked_widget.addWidget(self.ibis_ami_manager.get_ibis_widget())
        tx_layout.addWidget(self.stacked_widget)

        layout.addWidget(self.tx_config, stretch=1)
        self.tx_equalization = TxEqualizationWidget(pybert=self.pybert, parent=self, ami_manager=self.ibis_ami_manager)
        layout.addWidget(self.tx_equalization, stretch=2)

        if pybert is not None:
            self.update_from_model()
            self.connect_signals(pybert)

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model."""
        with block_signals(self):
            # Update ibis parameters
            self.ibis_radio.setChecked(self.pybert.tx_use_ibis)
            self.native_radio.setChecked(not self.pybert.tx_use_ibis)

            # Update native parameters
            self.rs.setValue(self.pybert.rs)
            self.cout.setValue(self.pybert.cout)

            # Update equalization
            self.tx_equalization.update_from_model()
            self.ibis_ami_manager.update_from_model()
        self._toggle_ibis_native_or_model()

    def connect_signals(self, pybert: "PyBERT") -> None:
        """Connect widget signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(self._toggle_ibis_native_or_model)
        self.rs.valueChanged.connect(lambda val: setattr(pybert, "tx_rs", val))
        self.cout.valueChanged.connect(lambda val: setattr(pybert, "tx_cout", val))
        self.ibis_ami_manager.ibis_changed.connect(self.update_from_model)

    def _toggle_ibis_native_or_model(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked widget."""
        self.stacked_widget.setCurrentIndex(1 if self.ibis_radio.isChecked() else 0)
        setattr(self.pybert, "tx_use_ibis", self.ibis_radio.isChecked())
