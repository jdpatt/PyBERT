"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS
model selection and native parameters.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.ibis_config import IbisConfigWidget
from pybert.gui.widgets.ibis_manager import IbisAmiWidgetsManager
from pybert.gui.widgets.tx_equalization import TxEqualizationWidget
from pybert.gui.widgets.utils import block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.tx")


class TxConfigWidget(QWidget):
    """Widget for configuring transmitter parameters."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
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
        self.native_or_ibis_button_group = QButtonGroup(self)
        self.native_or_ibis_button_group.addButton(self.native_radio, 0)
        self.native_or_ibis_button_group.addButton(self.ibis_radio, 1)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        tx_layout.addLayout(mode_layout)

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

        # --- Stacked layout for transmitter config groups ---
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.native_group)
        self.ibis_widget = IbisConfigWidget(obj_accessor=lambda: self.pybert.tx, parent=self.stacked_widget)
        self.stacked_widget.addWidget(self.ibis_widget)
        tx_layout.addWidget(self.stacked_widget)

        layout.addWidget(self.tx_config, stretch=1)
        self.tx_equalization = TxEqualizationWidget(pybert=self.pybert, parent=self)
        layout.addWidget(self.tx_equalization, stretch=2)

        # Create IBIS-AMI manager with accessors
        self.ibis_ami_manager = IbisAmiWidgetsManager(
            obj_accessor=lambda: self.pybert.tx,
            ibis_widget=self.ibis_widget,
            ami_widget=self.tx_equalization.get_ami_widget(),
            ibis_stacked_widget=self.stacked_widget,
            ami_stacked_widget=self.tx_equalization.stacked_widget,
            parent=self,
        )

        self.update_widget_from_model(propagate=False)
        self.connect_signals(self.pybert)

    def connect_signals(self, pybert: PyBERT) -> None:
        """Connect widget signals to PyBERT instance."""
        self.rs.valueChanged.connect(lambda val: setattr(pybert.tx, "impedance", val))
        self.cout.valueChanged.connect(lambda val: setattr(pybert.tx, "capacitance", val))
        self.native_or_ibis_button_group.buttonReleased.connect(self._handle_ibis_radio_toggled)

    def update_widget_from_model(self, propagate: bool = True) -> None:
        """Update all widget values from the PyBERT model."""
        try:
            with block_signals(self):
                # Update mode
                self.native_radio.setChecked(not self.pybert.tx.use_ibis)
                self.ibis_radio.setChecked(self.pybert.tx.use_ibis)

                # Update native parameters
                self.rs.setValue(self.pybert.tx.impedance)
                self.cout.setValue(self.pybert.tx.capacitance)

                # Update stacked widget
                self.stacked_widget.setCurrentIndex(1 if self.pybert.tx.use_ibis else 0)

                # Tell the equalization widget to update its model
                if propagate:
                    self.tx_equalization.update_widget_from_model()
                    self.ibis_ami_manager.update_widget_from_model()
        except Exception as e:
            logger.error(f"Failed to update TX widget from model: {e}")

    def _handle_ibis_radio_toggled(self) -> None:
        """Handle the toggled event of the IBIS radio button."""
        self.stacked_widget.setCurrentIndex(1 if self.ibis_radio.isChecked() else 0)
        setattr(self.pybert.tx, "use_ibis", self.ibis_radio.isChecked())
