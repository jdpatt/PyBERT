"""Transmitter equalization widget for PyBERT GUI.

This widget contains controls for transmitter equalization including FFE taps.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class TxOptimizationWidget(QGroupBox):
    """Widget for configuring transmitter equalization."""

    def __init__(self, parent=None):
        """Initialize the transmitter equalization widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Tx Equalization", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create FFE table
        self.ffe_table = QTableWidget()
        self.ffe_table.setColumnCount(6)
        self.ffe_table.setHorizontalHeaderLabels(["Name", "Enabled", "Min", "Max", "Step", "Value"])

        # Set default number of taps (can be changed later)
        self.set_taps(["Pre-Tap3", "Pre-Tap2", "Pre-Tap1", "Post-Tap1", "Post-Tap2", "Post-Tap3"])

        # Configure table appearance
        header = self.ffe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Step
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Value

        self.ffe_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ffe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(self.ffe_table)

    def set_taps(self, names: list[str]):
        """Set the number of FFE taps.

        Args:
            count: Number of taps to display
        """
        self.ffe_table.setRowCount(len(names))

        for i, name in enumerate(names):
            # Name
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.ffe_table.setItem(i, 0, name_item)

            # Enabled
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.Checked)
            self.ffe_table.setItem(i, 1, enabled_item)

            # Min value
            min_item = QTableWidgetItem("-1.0")
            self.ffe_table.setItem(i, 2, min_item)

            # Max value
            max_item = QTableWidgetItem("1.0")
            self.ffe_table.setItem(i, 3, max_item)

            # Step
            step_item = QTableWidgetItem("0.1")
            self.ffe_table.setItem(i, 4, step_item)

            # Current value
            value_item = QTableWidgetItem("0.0")
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.ffe_table.setItem(i, 5, value_item)

        self.ffe_table.resizeRowsToContents()

    def get_tap_values(self):
        """Get the current tap values.

        Returns:
            list: List of tuples containing (enabled, min, max, step, value) for each tap
        """
        values = []
        for i in range(self.ffe_table.rowCount()):
            enabled = self.ffe_table.item(i, 1).checkState() == Qt.Checked
            min_val = float(self.ffe_table.item(i, 2).text())
            max_val = float(self.ffe_table.item(i, 3).text())
            step = float(self.ffe_table.item(i, 4).text())
            value = float(self.ffe_table.item(i, 5).text())
            values.append((enabled, min_val, max_val, step, value))
        return values

    def set_tap_value(self, tap_index, value):
        """Set the value for a specific tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.ffe_table.rowCount():
            self.ffe_table.item(tap_index, 5).setText(f"{value:+.3f}")
