"""Configuration tab for PyBERT GUI.

This module implements the configuration tab which contains simulation
control, channel configuration, and other related settings.
"""

from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
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
from pybert.pybert import PyBERT


class ConfigTab(QWidget):
    """Configuration tab containing simulation settings and channel configuration."""

    def __init__(self, pybert: PyBERT | None = None, parent=None):
        """Initialize the configuration tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create horizontal layout for top section
        top_layout = QHBoxLayout()

        # Add simulation control widget
        self.sim_control = SimulationControlWidget(self)
        self.sim_control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        top_layout.addWidget(self.sim_control, stretch=2)

        # Add analysis control widget
        self.analysis_control = AnalysisConfigWidget(self)
        self.analysis_control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        top_layout.addWidget(self.analysis_control, stretch=1)

        # Add top layout to main layout
        layout.addLayout(top_layout)

        # Create horizontal layout for channel section
        interconnect_layout = QHBoxLayout()

        # Add Tx elements
        self.tx_config = TxConfigWidget(pybert=self.pybert, parent=self)
        interconnect_layout.addWidget(self.tx_config)

        # Add Channel configuration
        self.channel_config = ChannelConfigWidget(pybert, parent=self) # Needs to keep reference to PyBERT instance for channel file changes
        interconnect_layout.addWidget(self.channel_config)

        # Add Rx elements
        self.rx_config = RxConfigWidget(self)
        interconnect_layout.addWidget(self.rx_config)

        # Add channel layout to main layout with stretch to fill remaining space
        layout.addLayout(interconnect_layout, stretch=1)

        # Add stretch to push everything to the top
        layout.addStretch()

    def connect_signals(self, pybert):
        """Connect signals to PyBERT instance."""
        self.sim_control.connect_signals(pybert)
        self.analysis_control.connect_signals(pybert)
        self.tx_config.connect_signals(pybert)
        self.channel_config.connect_signals(pybert)
        self.rx_config.connect_signals(pybert)
