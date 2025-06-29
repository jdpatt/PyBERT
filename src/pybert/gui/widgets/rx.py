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
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.file_picker import FilePickerWidget
from pybert.gui.widgets.ibis_config import IbisConfigWidget
from pybert.gui.widgets.ibis_manager import IbisAmiWidgetsManager
from pybert.gui.widgets.rx_equalization import RxEqualizationWidget
from pybert.gui.widgets.status_indicator import StatusIndicator
from pybert.gui.widgets.utils import block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.rx")


class RxConfigWidget(QWidget):
    """Widget for configuring receiver parameters."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
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
        self.mode_group.addButton(self.native_radio, 0)
        self.mode_group.addButton(self.ibis_radio, 1)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        rx_layout.addLayout(mode_layout)

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

        # Viterbi parameters
        self.rx_use_viterbi = QCheckBox("Use Viterbi")
        self.rx_use_viterbi.setChecked(False)
        self.rx_use_viterbi.setToolTip("Apply MLSD to recovered symbols, using Viterbi algorithm.")
        self.rx_viterbi_symbols = QSpinBox()
        self.rx_viterbi_symbols.setRange(1, 10000)
        self.rx_viterbi_symbols.setValue(4)
        self.rx_viterbi_symbols.setSuffix(" Symbols")
        self.rx_viterbi_symbols.setToolTip("Number of symbols to include in MLSD trellis.")
        native_form.addRow(QLabel("Use Viterbi"), self.rx_use_viterbi)
        native_form.addRow(QLabel("# Symbols"), self.rx_viterbi_symbols)

        # --- Stacked layout for receiver config groups ---
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.native_group)
        self.ibis_widget = IbisConfigWidget(obj_accessor=lambda: self.pybert.rx, parent=self)
        self.stacked_widget.addWidget(self.ibis_widget)

        # Add stacked widget to layout
        rx_layout.addWidget(self.stacked_widget)

        layout.addWidget(self.rx_config, stretch=1)

        self.rx_equalization = RxEqualizationWidget(self.pybert, parent=self)
        layout.addWidget(self.rx_equalization, stretch=2)

        # Create IBIS-AMI manager with accessors
        self.ibis_ami_manager = IbisAmiWidgetsManager(
            obj_accessor=lambda: self.pybert.rx,
            ibis_widget=self.ibis_widget,
            ami_widget=self.rx_equalization.get_ami_widget(),
            ibis_stacked_widget=self.stacked_widget,
            ami_stacked_widget=self.rx_equalization.stacked_widget,
            parent=self,
        )

        self.update_widget_from_model(propagate=False)  # Equalization will do this in it's own init
        self.connect_signals()

    def connect_signals(self) -> None:
        """Connect widget signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(self._handle_ibis_radio_toggled)
        self.rin.valueChanged.connect(lambda val: setattr(self.pybert.rx, "impedance", val))
        self.cin.valueChanged.connect(lambda val: setattr(self.pybert.rx, "capacitance", val))
        self.cac.valueChanged.connect(lambda val: setattr(self.pybert.rx, "ac_coupling", val))
        self.rx_use_viterbi.toggled.connect(lambda val: setattr(self.pybert, "rx_use_viterbi", val))
        self.rx_viterbi_symbols.valueChanged.connect(lambda val: setattr(self.pybert, "rx_viterbi_symbols", val))

    def update_widget_from_model(self, propagate: bool = True) -> None:
        """Update all widget values from the PyBERT model."""
        try:
            with block_signals(self):
                # Update mode
                self.native_radio.setChecked(not self.pybert.rx.use_ibis)
                self.ibis_radio.setChecked(self.pybert.rx.use_ibis)

                # Update native parameters
                self.rin.setValue(self.pybert.rx.impedance)
                self.cin.setValue(self.pybert.rx.capacitance)
                self.cac.setValue(self.pybert.rx.ac_coupling)
                self.rx_use_viterbi.setChecked(self.pybert.rx_use_viterbi)
                self.rx_viterbi_symbols.setValue(self.pybert.rx_viterbi_symbols)

                # Update stacked widget
                self.stacked_widget.setCurrentIndex(1 if self.pybert.rx.use_ibis else 0)

                # Tell the equalization widget to update its model
                if propagate:
                    self.rx_equalization.update_widget_from_model()
                    self.ibis_ami_manager.update_widget_from_model()
        except Exception as e:
            logger.error(f"Failed to update RX widget from model: {e}")

    def _handle_ibis_radio_toggled(self) -> None:
        """Handle the toggled event of the IBIS radio button."""
        self.stacked_widget.setCurrentIndex(1 if self.ibis_radio.isChecked() else 0)
        setattr(self.pybert.rx, "use_ibis", self.ibis_radio.isChecked())
