"""Receiver equalization widget for PyBERT GUI.

This widget contains controls for receiver equalization including CTLE and DFE.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
)
from PySide6.QtCore import Qt


class RxEqualizationWidget(QGroupBox):
    """Widget for configuring receiver equalization."""

    def __init__(self, parent=None):
        """Initialize the receiver equalization widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Rx Equalization", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # CTLE group
        ctle_group = QGroupBox("Rx CTLE")
        ctle_layout = QVBoxLayout()
        ctle_group.setLayout(ctle_layout)

        # CTLE enable
        self.ctle_enable = QCheckBox("Enable")
        ctle_layout.addWidget(self.ctle_enable)

        # CTLE configuration
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)

        # Peaking frequency
        fp_layout = QHBoxLayout()
        fp_layout.addWidget(QLabel("fp:"))
        self.peak_freq = QDoubleSpinBox()
        self.peak_freq.setRange(0.1, 50.0)
        self.peak_freq.setValue(3.0)
        self.peak_freq.setSuffix(" GHz")
        fp_layout.addWidget(self.peak_freq)
        config_layout.addLayout(fp_layout)

        # Bandwidth
        bw_layout = QHBoxLayout()
        bw_layout.addWidget(QLabel("BW:"))
        self.rx_bw = QDoubleSpinBox()
        self.rx_bw.setRange(0.1, 50.0)
        self.rx_bw.setValue(25.0)
        self.rx_bw.setSuffix(" GHz")
        bw_layout.addWidget(self.rx_bw)
        config_layout.addLayout(bw_layout)

        # Min boost
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min.:"))
        self.min_boost = QDoubleSpinBox()
        self.min_boost.setRange(-20.0, 20.0)
        self.min_boost.setValue(0.0)
        self.min_boost.setSuffix(" dB")
        min_layout.addWidget(self.min_boost)
        config_layout.addLayout(min_layout)

        # Max boost
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max.:"))
        self.max_boost = QDoubleSpinBox()
        self.max_boost.setRange(-20.0, 20.0)
        self.max_boost.setValue(12.0)
        self.max_boost.setSuffix(" dB")
        max_layout.addWidget(self.max_boost)
        config_layout.addLayout(max_layout)

        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step:"))
        self.step_boost = QDoubleSpinBox()
        self.step_boost.setRange(0.1, 5.0)
        self.step_boost.setValue(1.0)
        self.step_boost.setSuffix(" dB")
        step_layout.addWidget(self.step_boost)
        config_layout.addLayout(step_layout)

        ctle_layout.addWidget(config_group)

        # Result group
        result_group = QGroupBox("Result")
        result_layout = QHBoxLayout()
        result_group.setLayout(result_layout)

        result_layout.addWidget(QLabel("Boost:"))
        self.boost_result = QDoubleSpinBox()
        self.boost_result.setRange(-20.0, 20.0)
        self.boost_result.setValue(0.0)
        self.boost_result.setSuffix(" dB")
        self.boost_result.setReadOnly(True)
        self.boost_result.setButtonSymbols(QDoubleSpinBox.NoButtons)
        result_layout.addWidget(self.boost_result)

        ctle_layout.addWidget(result_group)

        layout.addWidget(ctle_group)

        # DFE group
        dfe_group = QGroupBox("Rx DFE")
        dfe_layout = QVBoxLayout()
        dfe_group.setLayout(dfe_layout)

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
        self.set_tap_count(6)

        # Configure table appearance
        header = self.dfe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Min
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Max
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Value

        self.dfe_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dfe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        dfe_layout.addWidget(self.dfe_table)

        layout.addWidget(dfe_group)

        # Connect signals
        self.ctle_enable.toggled.connect(self._toggle_ctle)
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

    def _toggle_ctle(self, enabled):
        """Enable/disable CTLE controls based on checkbox state."""
        self.peak_freq.setEnabled(enabled)
        self.rx_bw.setEnabled(enabled)
        self.min_boost.setEnabled(enabled)
        self.max_boost.setEnabled(enabled)
        self.step_boost.setEnabled(enabled)

    def _disable_all_taps(self):
        """Disable all DFE taps."""
        for i in range(self.dfe_table.rowCount()):
            self.dfe_table.item(i, 1).setCheckState(Qt.Unchecked)

    def _enable_all_taps(self):
        """Enable all DFE taps."""
        for i in range(self.dfe_table.rowCount()):
            self.dfe_table.item(i, 1).setCheckState(Qt.Checked)

    def get_ctle_settings(self):
        """Get the current CTLE settings.

        Returns:
            tuple: (enabled, peak_freq, rx_bw, min_boost, max_boost, step_boost, current_boost)
        """
        return (
            self.ctle_enable.isChecked(),
            self.peak_freq.value(),
            self.rx_bw.value(),
            self.min_boost.value(),
            self.max_boost.value(),
            self.step_boost.value(),
            self.boost_result.value(),
        )

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

    def set_ctle_boost(self, value):
        """Set the current CTLE boost value.

        Args:
            value: New boost value in dB
        """
        self.boost_result.setValue(value)
