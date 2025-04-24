"""Simulation control widget for PyBERT GUI.

This widget contains controls for simulation parameters like bit rate,
samples per unit interval, modulation type, etc.
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
)
from PySide6.QtCore import Qt


class AnalysisConfigWidget(QGroupBox):
    """Widget for controlling analysis parameters."""

    def __init__(self, parent=None):
        """Initialize the simulation control widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Analysis Parameters", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Impulse Response Length
        impulse_layout = QHBoxLayout()
        impulse_label = QLabel("Impulse Response Length:")
        self.impulse_length = QDoubleSpinBox()
        self.impulse_length.setToolTip("Manual impulse response length override")
        self.impulse_length.setDecimals(3)
        self.impulse_length.setRange(0, 1000)
        impulse_unit = QLabel("ns")
        impulse_layout.addWidget(impulse_label)
        impulse_layout.addWidget(self.impulse_length)
        impulse_layout.addWidget(impulse_unit)
        impulse_layout.addStretch()
        layout.addLayout(impulse_layout)

        # PJ Threshold
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("PJ Threshold:")
        self.thresh = QDoubleSpinBox()
        self.thresh.setToolTip("Threshold for identifying periodic jitter spectral elements. (sigma)")
        self.thresh.setDecimals(3)
        self.thresh.setRange(0, 100)
        thresh_unit = QLabel("sigma")
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh)
        thresh_layout.addWidget(thresh_unit)
        thresh_layout.addStretch()
        layout.addLayout(thresh_layout)

        # Maximum Frequency
        fmax_layout = QHBoxLayout()
        fmax_label = QLabel("fMax:")
        self.f_max = QDoubleSpinBox()
        self.f_max.setToolTip("Maximum frequency used for plotting, modeling, and signal processing. (GHz)")
        self.f_max.setDecimals(3)
        self.f_max.setRange(0, 1000)
        fmax_unit = QLabel("GHz")
        fmax_layout.addWidget(fmax_label)
        fmax_layout.addWidget(self.f_max)
        fmax_layout.addWidget(fmax_unit)
        fmax_layout.addStretch()
        layout.addLayout(fmax_layout)

        # Frequency Step
        fstep_layout = QHBoxLayout()
        fstep_label = QLabel("fStep:")
        self.f_step = QDoubleSpinBox()
        self.f_step.setToolTip("Frequency step used for plotting, modeling, and signal processing. (MHz)")
        self.f_step.setDecimals(3)
        self.f_step.setRange(0, 1000)
        fstep_unit = QLabel("MHz")
        fstep_layout.addWidget(fstep_label)
        fstep_layout.addWidget(self.f_step)
        fstep_layout.addWidget(fstep_unit)
        fstep_layout.addStretch()
        layout.addLayout(fstep_layout)

        # Add stretch to push everything to the top
        layout.addStretch()
