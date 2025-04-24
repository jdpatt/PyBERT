"""Channel configuration widget for PyBERT GUI.

This widget contains controls for channel parameters including file-based
and native (Howard Johnson) channel models.
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


class ChannelConfigWidget(QGroupBox):
    """Widget for configuring channel parameters."""

    def __init__(self, parent=None):
        """Initialize the channel configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Interconnect", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # From File group
        file_group = QGroupBox("From File")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        # File selection
        file_select_layout = QHBoxLayout()
        file_select_layout.addWidget(QLabel("File:"))
        self.channel_file = QLineEdit()
        self.channel_file.setReadOnly(True)
        file_select_layout.addWidget(self.channel_file)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_channel)
        file_select_layout.addWidget(self.browse_btn)
        file_layout.addLayout(file_select_layout)

        # File options
        options_layout = QHBoxLayout()
        self.use_file = QCheckBox("Use file")
        options_layout.addWidget(self.use_file)
        self.renumber = QCheckBox("Fix port numbering")
        options_layout.addWidget(self.renumber)
        options_layout.addStretch()
        file_layout.addLayout(options_layout)

        layout.addWidget(file_group)

        # Native model group
        native_group = QGroupBox("Native")
        native_layout = QVBoxLayout()
        native_group.setLayout(native_layout)

        # Length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Length:"))
        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 10.0)
        self.length.setValue(0.1)
        self.length.setSuffix(" m")
        length_layout.addWidget(self.length)
        length_layout.addStretch()
        native_layout.addLayout(length_layout)

        # Loss tangent
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Tan.:"))
        self.loss_tan = QDoubleSpinBox()
        self.loss_tan.setRange(0.0, 1.0)
        self.loss_tan.setValue(0.02)
        self.loss_tan.setDecimals(3)
        loss_layout.addWidget(self.loss_tan)
        loss_layout.addStretch()
        native_layout.addLayout(loss_layout)

        # Characteristic impedance
        z0_layout = QHBoxLayout()
        z0_layout.addWidget(QLabel("Z0:"))
        self.z0 = QDoubleSpinBox()
        self.z0.setRange(0.0, 200.0)
        self.z0.setValue(100.0)
        self.z0.setSuffix(" Ohms")
        z0_layout.addWidget(self.z0)
        z0_layout.addStretch()
        native_layout.addLayout(z0_layout)

        # Relative velocity
        v0_layout = QHBoxLayout()
        v0_layout.addWidget(QLabel("v_rel:"))
        self.v0 = QDoubleSpinBox()
        self.v0.setRange(0.0, 1.0)
        self.v0.setValue(0.6)
        self.v0.setSuffix(" c")
        v0_layout.addWidget(self.v0)
        v0_layout.addStretch()
        native_layout.addLayout(v0_layout)

        # DC resistance
        rdc_layout = QHBoxLayout()
        rdc_layout.addWidget(QLabel("Rdc:"))
        self.rdc = QDoubleSpinBox()
        self.rdc.setRange(0.0, 100.0)
        self.rdc.setValue(0.0)
        self.rdc.setSuffix(" Ohms")
        rdc_layout.addWidget(self.rdc)
        rdc_layout.addStretch()
        native_layout.addLayout(rdc_layout)

        # Transition frequency
        w0_layout = QHBoxLayout()
        w0_layout.addWidget(QLabel("w0:"))
        self.w0 = QDoubleSpinBox()
        self.w0.setRange(0.0, 1e12)
        self.w0.setValue(0.0)
        self.w0.setSuffix(" rads/s")
        w0_layout.addWidget(self.w0)
        w0_layout.addStretch()
        native_layout.addLayout(w0_layout)

        # Skin effect resistance
        r0_layout = QHBoxLayout()
        r0_layout.addWidget(QLabel("R0:"))
        self.r0 = QDoubleSpinBox()
        self.r0.setRange(0.0, 100.0)
        self.r0.setValue(0.0)
        self.r0.setSuffix(" Ohms")
        r0_layout.addWidget(self.r0)
        r0_layout.addStretch()
        native_layout.addLayout(r0_layout)

        layout.addWidget(native_group)

        # Misc group
        misc_group = QGroupBox("Misc.")
        misc_layout = QHBoxLayout()
        misc_group.setLayout(misc_layout)

        self.use_window = QCheckBox("Apply window")
        self.use_window.setToolTip("Apply raised cosine window to frequency response before FFT()'ing.")
        misc_layout.addWidget(self.use_window)

        layout.addWidget(misc_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Connect signals
        self.use_file.toggled.connect(self._toggle_native)

    def _browse_channel(self):
        """Open file dialog to select channel file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Channel File", "", "All Files (*.*)")
        if filename:
            self.channel_file.setText(filename)

    def _toggle_native(self, use_file):
        """Enable/disable native parameters based on file usage."""
        for widget in self.findChildren((QDoubleSpinBox, QSpinBox)):
            if widget.parent().title() == "Native":
                widget.setEnabled(not use_file)
