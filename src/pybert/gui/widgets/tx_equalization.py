"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS model
selection and native parameters.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class TxEqualizationWidget(QGroupBox):
    """Widget for configuring transmitter equalization parameters."""

    def __init__(self, parent=None):
        """Initialize the transmitter configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Equalization", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Mode selection radio buttons ---
        mode_layout = QHBoxLayout()
        self.native_radio = QRadioButton("Native")
        self.ibis_radio = QRadioButton("IBIS-AMI")
        self.native_radio.setChecked(True)  # Default to Native
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.ibis_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- Stacked layout for transmitter config groups ---
        self.stacked_layout = QStackedLayout()

        # IBIS-AMI group
        self.ibis_group = QWidget()
        ibis_layout = QVBoxLayout()
        self.ibis_group.setLayout(ibis_layout)

        # IBIS-AMI file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("AMI File"))
        self.ibis_file = QLineEdit()
        self.ibis_file.setReadOnly(True)
        file_layout.addWidget(self.ibis_file)
        ibis_layout.addLayout(file_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("DLL File"))
        self.ibis_file = QLineEdit()
        self.ibis_file.setReadOnly(True)
        file_layout.addWidget(self.ibis_file)
        ibis_layout.addLayout(file_layout)

        # IBIS-AMI valid indicator
        valid_layout = QHBoxLayout()
        valid_layout.addWidget(QLabel("Valid"))
        self.ibis_valid = QCheckBox()
        self.ibis_valid.setEnabled(False)
        valid_layout.addWidget(self.ibis_valid)
        valid_layout.addStretch()
        ibis_layout.addLayout(valid_layout)
        ibis_layout.addStretch()

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.ibis_group)

        # Native parameters group
        self.native_group = QWidget()
        native_layout = QVBoxLayout()
        self.native_group.setLayout(native_layout)

        self.ffe_table = QTableWidget()
        self.ffe_table.setColumnCount(3)
        self.ffe_table.setHorizontalHeaderLabels(["Name", "Enabled", "Value"])

        # Set default number of taps (can be changed later)
        self.set_taps(["Pre-Tap3", "Pre-Tap2", "Pre-Tap1", "Post-Tap1", "Post-Tap2", "Post-Tap3"])

        # Configure table appearance
        header = self.ffe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Enabled
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Value

        self.ffe_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ffe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        native_layout.addWidget(self.ffe_table)

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.native_group)

        layout.addLayout(self.stacked_layout)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Connect signals for radio buttons
        self.ibis_radio.toggled.connect(self._update_mode)
        self.native_radio.toggled.connect(self._update_mode)

        # Set initial visibility
        self._update_mode()

    def _update_mode(self):
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)

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

            # Current value - make it editable
            value_item = QTableWidgetItem("0.0")
            self.ffe_table.setItem(i, 2, value_item)

        self.ffe_table.resizeRowsToContents()

    def get_tap_values(self):
        """Get the current tap values.

        Returns:
            list: List of tuples containing (enabled, min, max, step, value) for each tap
        """
        values = []
        for i in range(self.ffe_table.rowCount()):
            enabled = self.ffe_table.item(i, 1).checkState() == Qt.Checked
            try:
                value = float(self.ffe_table.item(i, 2).text())
            except (ValueError, TypeError):
                value = 0.0  # Default to 0 if invalid input
            values.append((enabled, value))
        return values

    def set_tap_value(self, tap_index, value):
        """Set the value for a specific tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.ffe_table.rowCount():
            self.ffe_table.item(tap_index, 2).setText(f"{value:+.3f}")
