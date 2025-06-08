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
    ChannelConfigWidget,
    RxConfigWidget,
    SimulationConfiglWidget,
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
        self.sim_control = SimulationConfiglWidget(pybert=self.pybert, parent=self)
        self.sim_control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        top_layout.addWidget(self.sim_control, stretch=2)
        layout.addLayout(top_layout)

        self.tx_config = TxConfigWidget(pybert=self.pybert, parent=self)
        self.channel_config = ChannelConfigWidget(pybert, parent=self)
        self.rx_config = RxConfigWidget(pybert=self.pybert, parent=self)

        # Create horizontal layout for tx, channel, and rx
        interconnect_layout = QHBoxLayout()
        interconnect_layout.addWidget(self.tx_config)
        interconnect_layout.addWidget(self.channel_config)
        interconnect_layout.addWidget(self.rx_config)

        layout.addLayout(interconnect_layout, stretch=1)
        layout.addStretch()
