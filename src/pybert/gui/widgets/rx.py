"""Receiver configuration widget for PyBERT GUI.

This widget contains controls for receiver parameters including IBIS model
selection and native parameters.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.rx_equalization import RxEqualizationWidget


class RxConfigWidget(QWidget):
    """Widget for configuring receiver parameters."""

    def __init__(self, parent=None):
        """Initialize the receiver configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
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
        self.ibis_group = QWidget()
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
        self.native_group = QWidget()
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

        self.rx_equalization = RxEqualizationWidget()
        layout.addWidget(self.rx_equalization, stretch=2)

        # Connect signals for radio buttons
        self.native_radio.toggled.connect(self._update_mode)
        self.ibis_radio.toggled.connect(self._update_mode)

        # Set initial visibility
        self._update_mode()

    def _browse_ibis(self):
        """Open file dialog to select IBIS file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select IBIS File", "", "IBIS Files (*.ibs);;IBIS Files (*.ibis);;All Files (*.*)"
        )
        if filename:
            self.ibis_file.setText(filename)
            # TODO: Validate IBIS file and enable controls

    def _update_mode(self):
        """Show only the selected group (Native or IBIS) using stacked layout."""
        if self.native_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.native_group)
        else:
            self.stacked_layout.setCurrentWidget(self.ibis_group)
