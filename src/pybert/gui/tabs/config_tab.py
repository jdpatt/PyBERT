"""Configuration tab for PyBERT GUI.

This module implements the configuration tab which contains simulation control,
channel configuration, and other related settings.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Qt

from ..widgets.analysis_config import AnalysisConfigWidget

from ..widgets.simulation_control import SimulationControlWidget
from ..widgets.transmitter_config import TxConfigWidget
from ..widgets.channel_config import ChannelConfigWidget
from ..widgets.receiver_config import RxConfigWidget


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
        top_layout.addWidget(self.sim_control)

        # Add analysis control widget
        self.analysis_control = AnalysisConfigWidget()
        top_layout.addWidget(self.analysis_control)

        # Add top layout to main layout
        layout.addLayout(top_layout)

        # Create horizontal layout for channel section
        channel_layout = QHBoxLayout()

        # Add Tx configuration
        self.tx_config = TxConfigWidget()
        channel_layout.addWidget(self.tx_config)

        # Add Channel configuration
        self.channel_config = ChannelConfigWidget()
        channel_layout.addWidget(self.channel_config)

        # Add Rx configuration
        self.rx_config = RxConfigWidget()
        channel_layout.addWidget(self.rx_config)

        # Add channel layout to main layout
        layout.addLayout(channel_layout)

        # Add stretch to push everything to the top
        layout.addStretch()
