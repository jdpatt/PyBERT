"""Receiver equalization widget for PyBERT GUI.

This widget contains controls for receiver equalization including CTLE
and DFE.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.utils import block_signals
from pybert.models.tx_tap import TxTapTuner
from pybert.pybert import PyBERT


class RxOptimizationDFEWidget(QGroupBox):
    """Widget for configuring receiver equalization."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
        """Initialize the receiver equalization widget.

        Args:
            pybert: PyBERT model instance
            parent: Parent widget
        """
        super().__init__("Rx DFE", parent)
        self.pybert = pybert

        dfe_layout = QVBoxLayout()

        # DFE control buttons
        btn_layout = QHBoxLayout()
        self.disable_btn = QPushButton("Disable All")
        self.enable_btn = QPushButton("Enable All")
        btn_layout.addWidget(self.disable_btn)
        btn_layout.addWidget(self.enable_btn)
        dfe_layout.addLayout(btn_layout)

        # DFE table
        self.dfe_table = QTableWidget()
        self.dfe_table.setColumnCount(5)
        self.dfe_table.setHorizontalHeaderLabels(["Name", "Enabled", "Min", "Max", "Value"])

        # Set default number of taps (can be changed later)
        self.create_table(self.pybert.dfe_tap_tuners)

        # Configure table appearance
        header = self.dfe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Value

        self.dfe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        dfe_layout.addWidget(self.dfe_table)

        self.setLayout(dfe_layout)

        # Connect signals
        self.disable_btn.clicked.connect(self._disable_all_taps)
        self.enable_btn.clicked.connect(self._enable_all_taps)
        self.dfe_table.itemChanged.connect(lambda item: setattr(pybert, "dfe_tap_tuners", self.get_tap_values()))

    def create_table(self, tuners: list[TxTapTuner]) -> None:
        """Set the number of DFE taps.

        Args:
            count: Number of taps to display
        """
        self.dfe_table.setRowCount(len(tuners))

        for i, tuner in enumerate(tuners):
            # Name
            name_item = QTableWidgetItem(tuner.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.dfe_table.setItem(i, 0, name_item)

            # Enabled
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.CheckState.Checked if tuner.enabled else Qt.CheckState.Unchecked)
            self.dfe_table.setItem(i, 1, enabled_item)

            # Min value
            min_item = QTableWidgetItem(f"{tuner.min_val:+.3f}")
            self.dfe_table.setItem(i, 2, min_item)

            # Max value
            max_item = QTableWidgetItem(f"{tuner.max_val:+.3f}")
            self.dfe_table.setItem(i, 3, max_item)

            # Current value
            value_item = QTableWidgetItem(f"{tuner.value:+.3f}")
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.dfe_table.setItem(i, 4, value_item)

        self.dfe_table.resizeRowsToContents()

    def _disable_all_taps(self) -> None:
        """Disable all DFE taps."""
        with block_signals(self.dfe_table):
            for i in range(self.dfe_table.rowCount()):
                self.dfe_table.item(i, 1).setCheckState(Qt.CheckState.Unchecked)
        # Manually trigger the update once
        setattr(self.pybert, "dfe_tap_tuners", self.get_tap_values())

    def _enable_all_taps(self) -> None:
        """Enable all DFE taps."""
        with block_signals(self.dfe_table):
            for i in range(self.dfe_table.rowCount()):
                self.dfe_table.item(i, 1).setCheckState(Qt.CheckState.Checked)
        # Manually trigger the update once
        setattr(self.pybert, "dfe_tap_tuners", self.get_tap_values())

    def get_tap_values(self) -> list[TxTapTuner]:
        """Get the current DFE tap values.

        Returns:
            list: List of TxTapTuner objects
        """
        values = []
        limits = []
        for i in range(self.dfe_table.rowCount()):
            name = self.dfe_table.item(i, 0).text()
            enabled = self.dfe_table.item(i, 1).checkState() == Qt.CheckState.Checked
            min_val = float(self.dfe_table.item(i, 2).text())
            max_val = float(self.dfe_table.item(i, 3).text())
            value = float(self.dfe_table.item(i, 4).text())
            values.append(TxTapTuner(name=name, enabled=enabled, min_val=min_val, max_val=max_val, value=value))
            limits.append((min_val, max_val))

        setattr(self.pybert, "dfe.limits", limits)
        return values

    def set_tap_value(self, tap_index: int, value: float) -> None:
        """Set the value for a specific DFE tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.dfe_table.rowCount():
            self.dfe_table.item(tap_index, 4).setText(f"{value:+.3f}")

    def set_tap_values(self, values: list[float]) -> None:
        """Set the values for enabled taps only.

        Args:
            values: List of float values to set for enabled taps
        """
        value_index = 0
        self.dfe_table.blockSignals(True)
        for i in range(self.dfe_table.rowCount()):
            enabled_item = self.dfe_table.item(i, 1)
            if enabled_item is not None and enabled_item.checkState() == Qt.CheckState.Checked:
                if value_index < len(values):
                    self.set_tap_value(i, values[value_index])
                    value_index += 1
        self.dfe_table.blockSignals(False)

    def connect_signals(self, pybert) -> None:
        self.disable_btn.clicked.connect(lambda: setattr(pybert, "dfe_enable_tune", False))
        self.enable_btn.clicked.connect(lambda: setattr(pybert, "dfe_enable_tune", True))
