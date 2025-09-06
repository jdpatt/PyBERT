"""Receiver equalization widget for PyBERT GUI.

This widget contains controls for receiver equalization including CTLE
and DFE.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pybert.optimizer.optimizer import Optimizer


class RxOptimizationCTLEWidget(QGroupBox):
    """Widget for configuring receiver equalization."""

    def __init__(self, optimizer: Optimizer, parent: Optional[QWidget] = None) -> None:
        """Initialize the receiver equalization widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Rx CTLE", parent)
        self.optimizer = optimizer

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # CTLE enable
        self.ctle_enable = QCheckBox("Enable")
        self.ctle_enable.setChecked(True)
        layout.addWidget(self.ctle_enable)

        # CTLE configuration
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)

        # Peaking frequency
        fp_layout = QHBoxLayout()
        fp_layout.addWidget(QLabel("Peaking frequency:"))
        self.peak_freq = QDoubleSpinBox()
        self.peak_freq.setRange(0.1, 50.0)
        self.peak_freq.setValue(3.0)
        self.peak_freq.setSuffix(" GHz")
        fp_layout.addWidget(self.peak_freq)
        config_layout.addLayout(fp_layout)

        # Bandwidth
        bw_layout = QHBoxLayout()
        bw_layout.addWidget(QLabel("Bandwidth:"))
        self.rx_bw = QDoubleSpinBox()
        self.rx_bw.setRange(0.1, 50.0)
        self.rx_bw.setValue(25.0)
        self.rx_bw.setSuffix(" GHz")
        bw_layout.addWidget(self.rx_bw)
        config_layout.addLayout(bw_layout)

        # Min boost
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min. boost:"))
        self.min_boost = QDoubleSpinBox()
        self.min_boost.setRange(-20.0, 20.0)
        self.min_boost.setValue(0.0)
        self.min_boost.setSuffix(" dB")
        min_layout.addWidget(self.min_boost)
        config_layout.addLayout(min_layout)

        # Max boost
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max. boost:"))
        self.max_boost = QDoubleSpinBox()
        self.max_boost.setRange(-20.0, 20.0)
        self.max_boost.setValue(12.0)
        self.max_boost.setSuffix(" dB")
        max_layout.addWidget(self.max_boost)
        config_layout.addLayout(max_layout)

        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step size:"))
        self.step_boost = QDoubleSpinBox()
        self.step_boost.setRange(0.1, 5.0)
        self.step_boost.setValue(1.0)
        self.step_boost.setSuffix(" dB")
        step_layout.addWidget(self.step_boost)
        config_layout.addLayout(step_layout)
        config_layout.addStretch()

        layout.addWidget(config_group)

        # Result group
        result_group = QGroupBox("Result")
        result_layout = QHBoxLayout()
        result_group.setLayout(result_layout)

        result_layout.addWidget(QLabel("Boost:"))
        self.boost_result = QLabel("1.7 dB")
        self.boost_result.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(self.boost_result)
        result_layout.addStretch()

        self.setLayout(result_layout)

        layout.addWidget(result_group)

        # Connect signals
        self.ctle_enable.toggled.connect(self._toggle_ctle)

    def connect_signals(self) -> None:
        """Connect signals to PyBERT instance."""
        self.ctle_enable.toggled.connect(lambda val: setattr(self.optimizer, "ctle_enable_tune", val))
        self.peak_freq.valueChanged.connect(lambda val: setattr(self.optimizer, "peak_freq_tune", val))
        self.rx_bw.valueChanged.connect(lambda val: setattr(self.optimizer, "rx_bw_tune", val))
        self.min_boost.valueChanged.connect(lambda val: setattr(self.optimizer, "min_mag_tune", val))
        self.max_boost.valueChanged.connect(lambda val: setattr(self.optimizer, "max_mag_tune", val))
        self.step_boost.valueChanged.connect(lambda val: setattr(self.optimizer, "step_boost", val))

    def _toggle_ctle(self, enabled: bool) -> None:
        """Enable/disable CTLE controls based on checkbox state."""
        self.peak_freq.setEnabled(enabled)
        self.rx_bw.setEnabled(enabled)
        self.min_boost.setEnabled(enabled)
        self.max_boost.setEnabled(enabled)
        self.step_boost.setEnabled(enabled)

    def set_ctle_boost(self, value: float) -> None:
        """Set the current CTLE boost value.

        Args:
            value: New boost value in dB
        """
        self.boost_result.setText(f"{value:.1f} dB")

    # Independent variable setting intercepts
    # (Primarily, for debugging.)
    # def set_ctle_peak_mag_tune(self, val: float) -> None:
    #     if val > gMaxCTLEPeak or val < 0.0:
    #         logger.error(f"CTLE peak magnitude out of range! {val} dB")
    #     self.peak_mag_tune = val
