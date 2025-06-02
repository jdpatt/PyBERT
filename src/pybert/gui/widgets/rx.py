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
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File"))
        self.ibis_file = QLineEdit()
        self.ibis_file.setReadOnly(True)
        file_layout.addWidget(self.ibis_file)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_ibis)
        file_layout.addWidget(self.browse_btn)

        self.view_btn = QPushButton("Configure")
        self.view_btn.setEnabled(False)
        file_layout.addWidget(self.view_btn)

        ibis_layout.addLayout(file_layout)

        # IBIS valid indicator
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Valid"))
        self.ibis_valid = QCheckBox()
        self.ibis_valid.setEnabled(False)
        info_layout.addWidget(self.ibis_valid)
        info_layout.addStretch()

        # On-die S-parameters
        self.use_ts4 = QCheckBox("Use on-die S-parameters")
        self.use_ts4.setEnabled(False)
        info_layout.addWidget(self.use_ts4)
        info_layout.addStretch()
        ibis_layout.addLayout(info_layout)

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

        self.rx_equalization = RxEqualizationWidget(self)
        layout.addWidget(self.rx_equalization, stretch=2)

        # Connect signals for radio buttons
        self.native_radio.toggled.connect(self._update_mode)
        self.ibis_radio.toggled.connect(self._update_mode)

        # Set initial visibility
        self._update_mode()

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.rx_equalization.connect_signals(pybert)
        self.use_ts4.toggled.connect(lambda val: setattr(pybert, "rx_use_ts4", val))
        self.ibis_file.textChanged.connect(lambda val: setattr(pybert, "rx_ibis_file", val))
        self.mode_group.buttonReleased.connect(
            lambda val: setattr(pybert, "rx_model", "Native" if self.native_radio.isChecked() else "IBIS")
        )
        self.rin.valueChanged.connect(lambda val: setattr(pybert, "rx_rin", val))
        self.cin.valueChanged.connect(lambda val: setattr(pybert, "rx_cin", val))
        self.cac.valueChanged.connect(lambda val: setattr(pybert, "rx_cac", val))
        self.ibis_has_ami.connect(self.rx_equalization.switch_equalization_modes)

    def _browse_ibis(self) -> None:
        """Open file dialog to select IBIS file."""
        filename = select_file_dialog(
            self, "Select IBIS File", "IBIS Files (*.ibs);;IBIS Files (*.ibis);;All Files (*.*)"
        )
        if filename:
            self.ibis_file.setText(filename)
            ibis = self.pybert.load_new_rx_ibis_file(filename)
            if ibis:
                self.ibis_valid.setChecked(True)
                self.view_btn.setEnabled(True)
                self.view_btn.clicked.connect(ibis.gui)

                if ibis.ami_file and ibis.dll_file:
                    dialogs.info_dialog(
                        "IBIS Algorithmic Model",
                        "There was an [Algorithmic Model] keyword in this model.\n \
    If you wish to use the AMI model associated with this IBIS model,\n \
    please, configure it now.",
                    )
                    self.ibis_has_ami.emit(True)
                else:
                    logger.warning(
                        "There was no [Algorithmic Model] keyword for this model or a valid executable for this platform;\n \
    PyBERT native equalization modeling being used instead.",
                    )
            else:
                self.ibis_valid.setChecked(False)
                self.view_btn.setEnabled(False)
                self.view_btn.clicked.disconnect()

    def _update_mode(self) -> None:
        """Show only the selected group (Native or IBIS) using stacked layout."""
        if self.native_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.native_group)
        else:
            self.stacked_layout.setCurrentWidget(self.ibis_group)
