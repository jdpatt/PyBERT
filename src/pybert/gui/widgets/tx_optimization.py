"""Transmitter equalization widget for PyBERT GUI.

This widget contains controls for transmitter equalization including FFE
taps.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybert.models.tx_tap import TxTapTuner
from pybert.pybert import PyBERT


class TxOptimizationWidget(QGroupBox):
    """Widget for configuring transmitter equalization."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
        """Initialize the transmitter equalization widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Tx Equalization", parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create FFE table
        self.ffe_table = QTableWidget()
        headers = ["Name", "Enabled", "Min", "Max", "Step", "Value"]
        self.ffe_table.setColumnCount(len(headers))
        self.ffe_table.setHorizontalHeaderLabels(headers)

        # Set default number of taps (can be changed later)
        self.set_taps(self.pybert.tx_tap_tuners)

        # Configure table appearance
        header = self.ffe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Step
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # Value

        self.ffe_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ffe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout.addWidget(self.ffe_table)

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.ffe_table.itemChanged.connect(lambda item: setattr(pybert, "tx_tap_tuners", self.get_tap_values()))

    def set_taps(self, tuners: list[TxTapTuner]) -> None:
        """Set the number of FFE taps.

        Args:
            tuners: List of TxTapTuner objects
        """
        self.ffe_table.setRowCount(len(tuners))

        for i, tuner in enumerate(tuners):
            # Name
            name_item = QTableWidgetItem(tuner.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ffe_table.setItem(i, 0, name_item)

            # Enabled
            enabled_item = QTableWidgetItem()
            flags = enabled_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            flags &= ~Qt.ItemFlag.ItemIsEditable
            enabled_item.setFlags(flags)
            enabled_item.setCheckState(Qt.CheckState.Checked if tuner.enabled else Qt.CheckState.Unchecked)
            enabled_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ffe_table.setItem(i, 1, enabled_item)

            # Min value
            min_item = QTableWidgetItem(f"{tuner.min_val:+.3f}")
            self.ffe_table.setItem(i, 2, min_item)

            # Max value
            max_item = QTableWidgetItem(f"{tuner.max_val:+.3f}")
            self.ffe_table.setItem(i, 3, max_item)

            # Step
            step_item = QTableWidgetItem(f"{tuner.step:+.3f}")
            self.ffe_table.setItem(i, 4, step_item)

            # Current value
            value_item = QTableWidgetItem(f"{tuner.value:+.3f}")
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ffe_table.setItem(i, 5, value_item)

        self.ffe_table.resizeRowsToContents()

    def get_tap_values(self) -> list[TxTapTuner]:
        """Get the current tap values.

        Returns:
            list: List of TxTapTuner objects
        """
        values = []
        for i in range(self.ffe_table.rowCount()):
            name = self.ffe_table.item(i, 0).text()
            enabled = self.ffe_table.item(i, 1).checkState() == Qt.CheckState.Checked
            min_val = float(self.ffe_table.item(i, 2).text())
            max_val = float(self.ffe_table.item(i, 3).text())
            step = float(self.ffe_table.item(i, 4).text())
            value = float(self.ffe_table.item(i, 5).text())
            values.append(
                TxTapTuner(name=name, enabled=enabled, min_val=min_val, max_val=max_val, step=step, value=value)
            )
        return values

    def set_tap_value(self, tap_index: int, value: float) -> None:
        """Set the value for a specific tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.ffe_table.rowCount():
            self.ffe_table.item(tap_index, 5).setText(f"{value:+.3f}")
