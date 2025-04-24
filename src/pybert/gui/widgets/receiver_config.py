"""Receiver configuration widget for PyBERT GUI.

This widget contains controls for receiver parameters including IBIS model
selection and native parameters.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Qt


class RxConfigWidget(QGroupBox):
    """Widget for configuring receiver parameters."""

    def __init__(self, parent=None):
        """Initialize the receiver configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Receiver", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # IBIS group
        ibis_group = QGroupBox("IBIS")
        ibis_layout = QVBoxLayout()
        ibis_group.setLayout(ibis_layout)

        # IBIS file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.ibis_file = QLineEdit()
        self.ibis_file.setReadOnly(True)
        file_layout.addWidget(self.ibis_file)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_ibis)
        file_layout.addWidget(self.browse_btn)
        ibis_layout.addLayout(file_layout)

        # IBIS valid indicator
        valid_layout = QHBoxLayout()
        valid_layout.addWidget(QLabel("Valid:"))
        self.ibis_valid = QCheckBox()
        self.ibis_valid.setEnabled(False)
        valid_layout.addWidget(self.ibis_valid)
        valid_layout.addStretch()
        ibis_layout.addLayout(valid_layout)

        # IBIS controls
        controls_layout = QHBoxLayout()
        self.use_ibis = QCheckBox("Use IBIS")
        self.use_ibis.setEnabled(False)
        controls_layout.addWidget(self.use_ibis)
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

        layout.addWidget(ibis_group)

        # Native parameters group
        native_group = QGroupBox("Native")
        native_layout = QVBoxLayout()
        native_group.setLayout(native_layout)

        # Input impedance
        rin_layout = QHBoxLayout()
        rin_layout.addWidget(QLabel("Rx_Rin:"))
        self.rin = QDoubleSpinBox()
        self.rin.setRange(0.0, 200.0)
        self.rin.setValue(50.0)
        self.rin.setSuffix(" Ohms")
        rin_layout.addWidget(self.rin)
        rin_layout.addStretch()
        native_layout.addLayout(rin_layout)

        # Input capacitance
        cin_layout = QHBoxLayout()
        cin_layout.addWidget(QLabel("Rx_Cin:"))
        self.cin = QDoubleSpinBox()
        self.cin.setRange(0.0, 10.0)
        self.cin.setValue(1.0)
        self.cin.setSuffix(" pF")
        cin_layout.addWidget(self.cin)
        cin_layout.addStretch()
        native_layout.addLayout(cin_layout)

        # AC coupling capacitance
        cac_layout = QHBoxLayout()
        cac_layout.addWidget(QLabel("Rx_Cac:"))
        self.cac = QDoubleSpinBox()
        self.cac.setRange(0.0, 10.0)
        self.cac.setValue(1.0)
        self.cac.setSuffix(" uF")
        cac_layout.addWidget(self.cac)
        cac_layout.addStretch()
        native_layout.addLayout(cac_layout)

        layout.addWidget(native_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Connect signals
        self.use_ibis.toggled.connect(self._toggle_native)

    def _browse_ibis(self):
        """Open file dialog to select IBIS file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Select IBIS File", "", "IBIS Files (*.ibs);;All Files (*.*)")
        if filename:
            self.ibis_file.setText(filename)
            # TODO: Validate IBIS file and enable controls

    def _toggle_native(self, use_ibis):
        """Enable/disable native parameters based on IBIS usage."""
        for widget in self.findChildren((QDoubleSpinBox, QSpinBox)):
            if widget.parent().title() == "Native":
                widget.setEnabled(not use_ibis)
