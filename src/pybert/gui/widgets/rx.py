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
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from pybert.gui import dialogs
from pybert.gui.dialogs import select_file_dialog
from pybert.gui.widgets.rx_equalization import RxEqualizationWidget
from pybert.gui.widgets.utils import FilePickerWidget, StatusIndicator, block_signals
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.rx")


class RxConfigWidget(QWidget):
    """Widget for configuring receiver parameters."""

    ibis_has_ami = Signal(bool)

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
        self.stacked_layout = QStackedLayout()

        # IBIS group
        self.ibis_group = QWidget(self)
        ibis_layout = QVBoxLayout()
        self.ibis_group.setLayout(ibis_layout)

        # IBIS file selection
        self.ibis_file = FilePickerWidget(
            "File", "IBIS Files (*.ibs);;IBIS Files (*.ibis);;All Files (*.*)", self.ibis_group
        )
        ibis_layout.addWidget(self.ibis_file)

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

        ibis_layout.addLayout(bottom_layout)
        ibis_layout.addStretch()

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.ibis_group)

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

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.native_group)

        rx_layout.addLayout(self.stacked_layout)

        layout.addWidget(self.rx_config, stretch=1)

        self.rx_equalization = RxEqualizationWidget(self.pybert, parent=self)
        layout.addWidget(self.rx_equalization, stretch=2)

        if pybert is not None:
            self.update_from_model()
            self.connect_signals(pybert)

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        if self.pybert is None:
            return

        with block_signals(self):
            # Update mode
            self.native_radio.setChecked(self.pybert.rx_use_ibis == False)
            self.ibis_radio.setChecked(self.pybert.rx_use_ibis == True)

            # Update IBIS settings
            self.ibis_file.set_text(self.pybert.rx_ibis_file)
            self.use_ts4.setChecked(self.pybert.rx_use_ts4)

            # Update native parameters
            self.rin.setValue(self.pybert.rin)
            self.cin.setValue(self.pybert.cin)
            self.cac.setValue(self.pybert.cac)

            # Update equalization
            self.rx_equalization.update_from_model()
        self._update_mode()

    def connect_signals(self, pybert: "PyBERT") -> None:
        """Connect widget signals to PyBERT instance."""
        self.native_radio.toggled.connect(self._update_mode)
        self.ibis_radio.toggled.connect(self._update_mode)

        self.use_ts4.toggled.connect(lambda val: setattr(pybert, "rx_use_ts4", val))
        self.ibis_file.file_selected.connect(self._new_ibis_file)
        self.mode_group.buttonReleased.connect(
            lambda val: setattr(pybert, "rx_use_ibis", self.native_radio.isChecked() == False)
        )
        self.rin.valueChanged.connect(lambda val: setattr(pybert, "rx_rin", val))
        self.cin.valueChanged.connect(lambda val: setattr(pybert, "rx_cin", val))
        self.cac.valueChanged.connect(lambda val: setattr(pybert, "rx_cac", val))

    def _new_ibis_file(self, filename: str) -> None:
        """Handle IBIS file selection."""
        setattr(self.pybert, "rx_ibis_file", filename)
        ibis = self.pybert.load_new_rx_ibis_file(filename)
        if ibis:
            self.ibis_valid.set_status("valid")
            self.view_btn.setEnabled(True)
            self.view_btn.clicked.connect(ibis.gui)

            if ibis.has_algorithmic_model:
                dialogs.info_dialog(
                    "IBIS Algorithmic Model",
                    "There was an [Algorithmic Model] keyword in this model.\n \
If you wish to use the AMI model associated with this IBIS model,\n \
please, configure it now.",
                )
            else:
                logger.warning(
                    "There was no [Algorithmic Model] keyword for this model or a valid executable for this platform;\n \
PyBERT native equalization modeling being used instead.",
                )
        else:
            self.ibis_valid.set_status("invalid")
            self.view_btn.setEnabled(False)
            self.view_btn.clicked.disconnect()

    def _update_mode(self) -> None:
        """Show only the selected group (Native or IBIS) using stacked layout."""
        if self.native_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.native_group)
            self.pybert.rx_use_ibis = False
        else:
            self.stacked_layout.setCurrentWidget(self.ibis_group)
            self.pybert.rx_use_ibis = True
