"""Simulation control widget for PyBERT GUI.

This widget contains controls for simulation parameters like bit rate,
samples per unit interval, modulation type, etc.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pybert.utility.debug import setattr


class AnalysisConfigWidget(QGroupBox):
    """Widget for controlling analysis parameters."""

    def __init__(self, pybert=None, parent: Optional[QWidget] = None) -> None:
        """Initialize the simulation control widget.

        Args:
            pybert: PyBERT model instance
            parent: Parent widget
        """
        super().__init__("Analysis Parameters", parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create grid layout for consistent widths
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        # Impulse Response Length
        impulse_label = QLabel("Impulse Response Length")
        self.impulse_length = QDoubleSpinBox()
        self.impulse_length.setToolTip("Manual impulse response length override (Determined automatically, when 0.)")
        self.impulse_length.setDecimals(3)
        self.impulse_length.setValue(0.0)
        self.impulse_length.setFixedWidth(100)
        self.impulse_length.setSuffix(" ns")

        grid_layout.addWidget(impulse_label, 0, 0)
        grid_layout.addWidget(self.impulse_length, 0, 1)

        # PJ Threshold
        thresh_label = QLabel("PJ Threshold")
        self.thresh = QDoubleSpinBox()
        self.thresh.setToolTip("Threshold for identifying periodic jitter spectral elements. (sigma)")
        self.thresh.setDecimals(3)
        self.thresh.setValue(3.0)
        self.thresh.setFixedWidth(100)
        self.thresh.setSuffix(" sigma")

        grid_layout.addWidget(thresh_label, 1, 0)
        grid_layout.addWidget(self.thresh, 1, 1)

        # Maximum Frequency
        fmax_label = QLabel("fMax")
        self.f_max = QDoubleSpinBox()
        self.f_max.setToolTip("Maximum frequency used for plotting, modeling, and signal processing. (GHz)")
        self.f_max.setDecimals(3)
        self.f_max.setValue(40.0)
        self.f_max.setFixedWidth(100)
        self.f_max.setSuffix(" GHz")

        grid_layout.addWidget(fmax_label, 2, 0)
        grid_layout.addWidget(self.f_max, 2, 1)

        # Frequency Step
        fstep_label = QLabel("fStep")
        self.f_step = QDoubleSpinBox()
        self.f_step.setToolTip("Frequency step used for plotting, modeling, and signal processing. (MHz)")
        self.f_step.setDecimals(3)
        self.f_step.setValue(10.0)
        self.f_step.setFixedWidth(100)
        self.f_step.setSuffix(" MHz")

        grid_layout.addWidget(fstep_label, 3, 0)
        grid_layout.addWidget(self.f_step, 3, 1)

        # Add stretch column
        grid_layout.setColumnStretch(3, 1)

        # Add stretch to push everything to the top
        layout.addStretch()

    def connect_signals(self, pybert) -> None:
        self.impulse_length.valueChanged.connect(lambda val: setattr(pybert, "impulse_length", val))
        self.thresh.valueChanged.connect(lambda val: setattr(pybert, "thresh", val))
        self.f_max.valueChanged.connect(lambda val: setattr(pybert, "f_max", val))
        self.f_step.valueChanged.connect(lambda val: setattr(pybert, "f_step", val))
