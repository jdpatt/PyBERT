"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS
model selection and native parameters.
"""

from typing import Optional

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

from pybert.gui.dialogs import select_file
from pybert.gui.widgets.tx_equalization import TxEqualizationWidget
from pybert.pybert import PyBERT
from pybert.utility.debug import setattr


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
        self.ibis_radio = QRadioButton("IBIS")
        self.native_radio.setChecked(True)  # Default to Native
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.ibis_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        tx_layout.addLayout(mode_layout)

        # --- Stacked layout for transmitter config groups ---
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
        ibis_layout.addLayout(file_layout)

        # IBIS valid indicator
        valid_layout = QHBoxLayout()
        valid_layout.addWidget(QLabel("Valid"))
        self.ibis_valid = QCheckBox()
        self.ibis_valid.setEnabled(False)
        valid_layout.addWidget(self.ibis_valid)
        valid_layout.addStretch()
        ibis_layout.addLayout(valid_layout)

        # IBIS controls
        controls_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select")
        self.select_btn.setEnabled(False)
        controls_layout.addWidget(self.select_btn)
        self.view_btn = QPushButton("View")
        self.view_btn.setEnabled(False)
        controls_layout.addWidget(self.view_btn)
        ibis_layout.addLayout(controls_layout)

        # On-die S-parameters
        self.use_ts4 = QCheckBox("Use on-die S-parameters")
        self.use_ts4.setEnabled(False)
        ibis_layout.addWidget(self.use_ts4)
        ibis_layout.addStretch()

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.ibis_group)

        # Native parameters group
        self.native_group = QWidget(self)
        native_form = QFormLayout()
        self.native_group.setLayout(native_form)

        self.rs = QDoubleSpinBox()
        self.rs.setRange(0.0, 1000.0)
        self.rs.setDecimals(2)
        self.rs.setValue(100.0)
        self.rs.setSuffix(" Ohms")
        native_form.addRow(QLabel("Impedance"), self.rs)

        self.cout = QDoubleSpinBox()
        self.cout.setRange(0.0, 1000.0)
        self.cout.setDecimals(2)
        self.cout.setValue(1.0)
        self.cout.setSuffix(" pF")
        native_form.addRow(QLabel("Capacitance"), self.cout)

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.native_group)

        tx_layout.addLayout(self.stacked_layout)

        layout.addWidget(self.tx_config, stretch=1)
        self.tx_equalization = TxEqualizationWidget(pybert=self.pybert, parent=self)
        layout.addWidget(self.tx_equalization, stretch=2)

        # Connect signals for radio buttons
        self.ibis_radio.toggled.connect(self._update_mode)
        self.native_radio.toggled.connect(self._update_mode)

        # Set initial visibility
        self._update_mode()

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.tx_equalization.connect_signals(pybert)
        self.use_ts4.toggled.connect(lambda val: setattr(pybert, "tx_use_ts4", val))
        self.rs.valueChanged.connect(lambda val: setattr(pybert, "tx_rs", val))
        self.cout.valueChanged.connect(lambda val: setattr(pybert, "tx_cout", val))
        self.ibis_file.textChanged.connect(lambda val: setattr(pybert, "tx_ibis_file", val))
        self.mode_group.buttonReleased.connect(lambda val: setattr(pybert, "tx_model", "Native" if self.native_radio.isChecked() else "IBIS"))


    def _browse_ibis(self) -> None:
        """Open file dialog to select IBIS file."""
        filename = select_file(self, "Select IBIS File", "IBIS Files (*.ibs);;IBIS Files (*.ibis);;All Files (*.*)")
        if filename:
            self.ibis_file.setText(filename)

    def _update_mode(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)
