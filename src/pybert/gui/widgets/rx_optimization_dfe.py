"""Receiver equalization widget for PyBERT GUI.

This widget contains controls for receiver equalization including CTLE and DFE.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RxOptimizationDFEWidget(QGroupBox):
    """Widget for configuring receiver equalization."""

    def __init__(self, parent=None):
        """Initialize the receiver equalization widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Rx DFE", parent)

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
        self.set_tap_count(20)

        # Configure table appearance
        header = self.dfe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Value

        self.dfe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        dfe_layout.addWidget(self.dfe_table)

        self.setLayout(dfe_layout)

        # Connect signals
        self.disable_btn.clicked.connect(self._disable_all_taps)
        self.enable_btn.clicked.connect(self._enable_all_taps)

    def set_tap_count(self, count):
        """Set the number of DFE taps.

        Args:
            count: Number of taps to display
        """
        self.dfe_table.setRowCount(count)

        for i in range(count):
            # Name
            name_item = QTableWidgetItem(f"Tap {i+1}")
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.dfe_table.setItem(i, 0, name_item)

            # Enabled
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.Checked)
            self.dfe_table.setItem(i, 1, enabled_item)

            # Min value
            min_item = QTableWidgetItem("-0.5")
            self.dfe_table.setItem(i, 2, min_item)

            # Max value
            max_item = QTableWidgetItem("0.5")
            self.dfe_table.setItem(i, 3, max_item)

            # Current value
            value_item = QTableWidgetItem("0.0")
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.dfe_table.setItem(i, 4, value_item)

        self.dfe_table.resizeRowsToContents()

    def _disable_all_taps(self):
        """Disable all DFE taps."""
        for i in range(self.dfe_table.rowCount()):
            self.dfe_table.item(i, 1).setCheckState(Qt.Unchecked)

    def _enable_all_taps(self):
        """Enable all DFE taps."""
        for i in range(self.dfe_table.rowCount()):
            self.dfe_table.item(i, 1).setCheckState(Qt.Checked)

    def get_dfe_tap_values(self):
        """Get the current DFE tap values.

        Returns:
            list: List of tuples containing (enabled, min, max, value) for each tap
        """
        values = []
        for i in range(self.dfe_table.rowCount()):
            enabled = self.dfe_table.item(i, 1).checkState() == Qt.Checked
            min_val = float(self.dfe_table.item(i, 2).text())
            max_val = float(self.dfe_table.item(i, 3).text())
            value = float(self.dfe_table.item(i, 4).text())
            values.append((enabled, min_val, max_val, value))
        return values

    def set_dfe_tap_value(self, tap_index, value):
        """Set the value for a specific DFE tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.dfe_table.rowCount():
            self.dfe_table.item(tap_index, 4).setText(f"{value:+.3f}")
