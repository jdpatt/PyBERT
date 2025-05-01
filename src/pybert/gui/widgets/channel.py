"""Channel configuration widget for PyBERT GUI.

This widget contains controls for channel parameters including file-based
and native (Howard Johnson) channel models.
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


class ChannelConfigWidget(QGroupBox):
    """Widget for configuring channel parameters."""

    def __init__(self, parent=None):
        """Initialize the channel configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Channel", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Mode selection radio buttons ---
        mode_layout = QHBoxLayout()
        self.native_radio = QRadioButton("Native")
        self.file_radio = QRadioButton("From File")
        self.native_radio.setChecked(True)  # Default to Native
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.file_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- Stacked layout for channel config groups ---
        self.stacked_layout = QStackedLayout()

        # From File group
        self.file_group = QWidget()
        file_layout = QVBoxLayout()
        self.file_group.setLayout(file_layout)

        # File selection
        file_select_layout = QHBoxLayout()
        file_select_layout.addWidget(QLabel("File"))
        self.channel_file = QLineEdit()
        self.channel_file.setReadOnly(True)
        file_select_layout.addWidget(self.channel_file)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_channel)
        file_select_layout.addWidget(self.browse_btn)
        file_layout.addLayout(file_select_layout)

        # File options
        options_layout = QHBoxLayout()
        self.renumber = QCheckBox("Fix port numbering")
        options_layout.addWidget(self.renumber)
        options_layout.addStretch()
        file_layout.addLayout(options_layout)
        file_layout.addStretch()

        # Native model group
        self.native_group = QWidget()
        native_form = QFormLayout()
        self.native_group.setLayout(native_form)

        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 10.0)
        self.length.setValue(0.5)
        self.length.setSuffix(" m")
        native_form.addRow(QLabel("Length"), self.length)

        self.loss_tan = QDoubleSpinBox()
        self.loss_tan.setRange(0.0, 1.0)
        self.loss_tan.setValue(0.02)
        self.loss_tan.setDecimals(3)
        native_form.addRow(
            QLabel(
                "Loss Tangent",
            ),
            self.loss_tan,
        )

        self.z0 = QDoubleSpinBox()
        self.z0.setRange(0.0, 200.0)
        self.z0.setValue(100.0)
        self.z0.setSuffix(" Ohms")
        native_form.addRow(QLabel("Characteristic Impedance"), self.z0)

        self.v0 = QDoubleSpinBox()
        self.v0.setRange(0.0, 1.0)
        self.v0.setValue(0.6)
        self.v0.setSuffix(" c")
        native_form.addRow(QLabel("Relative Velocity"), self.v0)

        self.rdc = QDoubleSpinBox()
        self.rdc.setRange(0.0, 100.0)
        self.rdc.setValue(0.0)
        self.rdc.setSuffix(" Ohms")
        native_form.addRow(QLabel("DC Resistance"), self.rdc)

        self.w0 = QDoubleSpinBox()
        self.w0.setRange(0.0, 1e12)
        self.w0.setValue(0.0)
        self.w0.setSuffix(" rads/s")
        native_form.addRow(QLabel("Transition Frequency"), self.w0)

        self.r0 = QDoubleSpinBox()
        self.r0.setRange(0.0, 100.0)
        self.r0.setValue(0.0)
        self.r0.setSuffix(" Ohms")
        native_form.addRow(QLabel("Skin Effect Resistance"), self.r0)

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.native_group)
        self.stacked_layout.addWidget(self.file_group)
        layout.addLayout(self.stacked_layout, stretch=1)

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

        # Connect signals for radio buttons
        self.native_radio.toggled.connect(self._update_mode)
        self.file_radio.toggled.connect(self._update_mode)

        # Set initial visibility
        self._update_mode()

    def _browse_channel(self):
        """Open file dialog to select channel file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Channel File",
            "",
            "S-parameters (*.s*p);;" "CSV files (*.csv);;" "Text files (*.txt);;" "All files (*.*)",
        )
        if filename:
            self.channel_file.setText(filename)

    def _update_mode(self):
        """Show only the selected group (Native or From File) using stacked layout."""
        if self.native_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.native_group)
        else:
            self.stacked_layout.setCurrentWidget(self.file_group)
