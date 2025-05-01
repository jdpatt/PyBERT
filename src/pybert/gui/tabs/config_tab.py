"""Configuration tab for PyBERT GUI.

This module implements the configuration tab which contains simulation control,
channel configuration, and other related settings.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets import (
    AnalysisConfigWidget,
    ChannelConfigWidget,
    RxConfigWidget,
    SimulationControlWidget,
    TxConfigWidget,
)
from pybert.gui.widgets.rx_equalization import RxEqualizationWidget
from pybert.gui.widgets.tx_equalization import TxEqualizationWidget


class ConfigTab(QWidget):
    """Configuration tab containing simulation settings and channel configuration."""

    def __init__(self, parent=None):
        """Initialize the configuration tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create horizontal layout for top section
        top_layout = QHBoxLayout()

        # Add simulation control widget
        self.sim_control = SimulationControlWidget()
        self.sim_control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        top_layout.addWidget(self.sim_control, stretch=2)

        # Add analysis control widget
        self.analysis_control = AnalysisConfigWidget()
        self.analysis_control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        top_layout.addWidget(self.analysis_control, stretch=1)

        # Add top layout to main layout
        layout.addLayout(top_layout)

        # Create horizontal layout for channel section
        interconnect_layout = QHBoxLayout()

        # Add Tx elements
        self.tx_config = TxConfigWidget()
        interconnect_layout.addWidget(self.tx_config)

        # Add Channel configuration
        self.channel_config = ChannelConfigWidget()
        interconnect_layout.addWidget(self.channel_config)

        # Add Rx elements
        self.rx_config = RxConfigWidget()
        interconnect_layout.addWidget(self.rx_config)

        # Add channel layout to main layout with stretch to fill remaining space
        layout.addLayout(interconnect_layout, stretch=1)

        # Add stretch to push everything to the top
        layout.addStretch()
