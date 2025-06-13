"""Receiver configuration widget for PyBERT GUI.

This widget contains controls for receiver parameters including IBIS
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
from pybert.gui.widgets.rx_equalization import RxEqualizationWidget
from pybert.gui.widgets.utils import FilePickerWidget, StatusIndicator, block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.rx")


class RxConfigWidget(QWidget):
    """Widget for configuring receiver parameters."""

    def __init__(self, pybert: PyBERT | None = None, parent: Optional[QWidget] = None) -> None:
        """Initialize the receiver configuration widget.

        Args:
            pybert: PyBERT instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.rx_config = QGroupBox("Receiver")
        rx_layout = QVBoxLayout()
        self.rx_config.setLayout(rx_layout)

        # --- Mode selection radio buttons ---
        mode_layout = QHBoxLayout()
        self.native_radio = QRadioButton("Native")
        self.ibis_radio = QRadioButton("IBIS")
        self.native_radio.setChecked(True)  # Default to Native
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.ibis_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        rx_layout.addLayout(mode_layout)

        # --- Stacked layout for receiver config groups ---
        self.stacked_widget = QStackedWidget()

        # Create IBIS-AMI manager
        self.ibis_ami_manager = IbisAmiManager(
            pybert=self.pybert,
            is_tx=False,
            parent=self,
        )

        # Native parameters group
        self.native_group = QWidget(self)
        native_form = QFormLayout()
        self.native_group.setLayout(native_form)

        self.rin = QDoubleSpinBox()
        self.rin.setRange(0.0, 1000.0)
        self.rin.setDecimals(2)
        self.rin.setValue(100.0)
        self.rin.setSuffix(" Ohms")
        native_form.addRow(QLabel("Impedance"), self.rin)

        self.cin = QDoubleSpinBox()
        self.cin.setRange(0.0, 1000.0)
        self.cin.setDecimals(2)
        self.cin.setValue(0.5)
        self.cin.setSuffix(" pF")
        native_form.addRow(QLabel("Capacitance"), self.cin)

        self.cac = QDoubleSpinBox()
        self.cac.setRange(0.0, 1000.0)
        self.cac.setDecimals(2)
        self.cac.setValue(1.0)
        self.cac.setSuffix(" uF")
        native_form.addRow(QLabel("AC Coupling"), self.cac)

        # Add native group to stacked widget
        self.stacked_widget.addWidget(self.native_group)
        self.stacked_widget.addWidget(self.ibis_ami_manager.get_ibis_widget())
        rx_layout.addWidget(self.stacked_widget)

        layout.addWidget(self.rx_config, stretch=1)

        self.rx_equalization = RxEqualizationWidget(self.pybert, parent=self, ami_manager=self.ibis_ami_manager)
        layout.addWidget(self.rx_equalization, stretch=2)

        if pybert is not None:
            self.update_from_model()
            self.connect_signals(pybert)

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model."""
        with block_signals(self):
            # Update ibis parameters
            self.ibis_radio.setChecked(self.pybert.rx_use_ibis)
            self.native_radio.setChecked(not self.pybert.rx_use_ibis)
            # self.ibis_ami_manager.update_from_model()

            # Update native parameters
            self.rin.setValue(self.pybert.rin)
            self.cin.setValue(self.pybert.cin)
            self.cac.setValue(self.pybert.cac)

            # Update equalization
            self.rx_equalization.update_from_model()
            self.ibis_ami_manager.update_from_model()
        self._toggle_ibis_native_or_model()

    def connect_signals(self, pybert: "PyBERT") -> None:
        """Connect widget signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(self._toggle_ibis_native_or_model)
        self.ibis_ami_manager.ibis_changed.connect(self.update_from_model)
        self.rin.valueChanged.connect(lambda val: setattr(pybert, "rx_rin", val))
        self.cin.valueChanged.connect(lambda val: setattr(pybert, "rx_cin", val))
        self.cac.valueChanged.connect(lambda val: setattr(pybert, "rx_cac", val))

    def _toggle_ibis_native_or_model(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked widget."""
        self.stacked_widget.setCurrentIndex(1 if self.ibis_radio.isChecked() else 0)
        setattr(self.pybert, "rx_use_ibis", self.ibis_radio.isChecked())
