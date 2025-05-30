"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS
model selection and native parameters.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QRadioButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybert.models.tx_tap import TxTapTuner
from pybert.pybert import PyBERT
from pybert.utility.debug import setattr


class TxEqualizationWidget(QGroupBox):
    """Widget for configuring transmitter equalization parameters."""

    def __init__(self, pybert: PyBERT | None = None, parent: Optional[QWidget] = None) -> None:
        """Initialize the transmitter configuration widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Equalization", parent)
        self.pybert = pybert

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
        self.ibis_group = QWidget(self)
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
        self.native_group = QWidget(self)
        native_layout = QVBoxLayout()
        self.native_group.setLayout(native_layout)

        self.ffe_table = QTableWidget()
        headers = ["Name", "Enabled", "Value"]
        self.ffe_table.setColumnCount(len(headers))
        self.ffe_table.setHorizontalHeaderLabels(headers)

        # Set default number of taps (can be changed later)
        self.set_taps(self.pybert.tx_taps)

        # Configure table appearance
        header = self.ffe_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Name
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Enabled
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

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(lambda val: setattr(pybert, "tx_eq", "Native" if self.native_radio.isChecked() else "IBIS"))
        self.ffe_table.itemChanged.connect(lambda item: setattr(pybert, "tx_taps", self.get_tap_values()))

    def _update_mode(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)

    def set_taps(self, tuners: list[TxTapTuner]) -> None:
        """Set the number of FFE taps.

        Args:
            names: List of tap names to display
        """
        self.ffe_table.setRowCount(len(tuners))
        for i, tuner in enumerate(tuners):
            name_item = QTableWidgetItem(tuner.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.ffe_table.setItem(i, 0, name_item)
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.Checked if tuner.enabled else Qt.Unchecked)
            self.ffe_table.setItem(i, 1, enabled_item)
            value_item = QTableWidgetItem(f"{tuner.value:+.3f}")
            self.ffe_table.setItem(i, 2, value_item)
        self.ffe_table.resizeRowsToContents()

    def get_tap_values(self) -> list[TxTapTuner]:
        """Get the current tap values.

        Returns:
            list: List of TxTapTuner objects
        """
        values = []
        for i in range(self.ffe_table.rowCount()):
            name = self.ffe_table.item(i, 0).text()
            enabled = self.ffe_table.item(i, 1).checkState() == Qt.Checked
            try:
                value = float(self.ffe_table.item(i, 2).text())
            except (ValueError, TypeError):
                value = 0.0
            values.append(TxTapTuner(name=name, enabled=enabled, value=value))
        return values

    def set_tap_value(self, tap_index: int, value: float) -> None:
        """Set the value for a specific tap.

        Args:
            tap_index: Index of the tap to update (0-based)
            value: New value for the tap
        """
        if 0 <= tap_index < self.ffe_table.rowCount():
            self.ffe_table.item(tap_index, 2).setText(f"{value:+.3f}")
