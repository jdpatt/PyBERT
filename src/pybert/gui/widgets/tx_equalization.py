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
    QPushButton,
    QRadioButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.dialogs import select_file_dialog
from pybert.gui.widgets.utils import StatusIndicator, block_signals
from pybert.models.tx_tap import TxTapTuner
from pybert.pybert import PyBERT


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
        self.ami_file = QLineEdit()
        self.ami_file.setReadOnly(True)
        file_layout.addWidget(self.ami_file)
        ibis_layout.addLayout(file_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("DLL File"))
        self.dll_file = QLineEdit()
        self.dll_file.setReadOnly(True)
        file_layout.addWidget(self.dll_file)
        ibis_layout.addLayout(file_layout)

        # Bottom row with status, checkbox, and configure
        bottom_layout = QHBoxLayout()
        # Status indicator
        bottom_layout.addWidget(QLabel("Status:"))
        self.ami_model_valid = StatusIndicator()
        bottom_layout.addWidget(self.ami_model_valid)
        bottom_layout.addSpacing(20)  # Add some spacing between status and checkbox

        # GetWave checkbox
        self.use_getwave = QCheckBox("Use GetWave()")
        bottom_layout.addWidget(self.use_getwave)

        bottom_layout.addStretch()  # Push configure button to the right

        # Configure button
        self.ami_configurator = QPushButton("Configure")
        bottom_layout.addWidget(self.ami_configurator)

        ibis_layout.addLayout(bottom_layout)
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

        if pybert:
            self.update_from_model()
            self.connect_signals(pybert)

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.ibis_radio.toggled.connect(self._update_mode)
        self.native_radio.toggled.connect(self._update_mode)

        if pybert is not None:
            pybert.new_tx_model.connect(self._update_ami_view)
            self.mode_group.buttonReleased.connect(
                lambda val: setattr(pybert, "tx_use_ami", self.native_radio.isChecked() == False)
            )
            self.ffe_table.itemChanged.connect(lambda item: setattr(pybert, "tx_taps", self.get_tap_values()))
            self.ami_configurator.clicked.connect(self._open_ami_configurator)
            self.use_getwave.toggled.connect(lambda val: setattr(pybert, "tx_use_getwave", val))

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        if self.pybert is None:
            return

        with block_signals(self):
            # Update mode
            self.native_radio.setChecked(self.pybert.tx_use_ami == False)
            self.ibis_radio.setChecked(self.pybert.tx_use_ami == True)

            # Update AMI settings
            if hasattr(self.pybert, "_tx_ibis") and self.pybert._tx_ibis is not None:
                self.ami_file.setText(str(self.pybert._tx_ibis.ami_file))
                self.dll_file.setText(str(self.pybert._tx_ibis.dll_file))
                self.ami_model_valid.set_status("valid" if self.pybert._tx_ibis.has_algorithmic_model else "invalid")
                self.ami_configurator.setEnabled(self.pybert._tx_ibis.has_algorithmic_model)
                self.use_getwave.setEnabled(self.pybert._tx_ibis.has_algorithmic_model)
                self.use_getwave.setChecked(self.pybert.tx_use_getwave)
            else:
                self.ami_model_valid.set_status("not_loaded")
                self.ami_configurator.setEnabled(False)
                self.use_getwave.setEnabled(False)

            # Update native parameters
            if hasattr(self.pybert, "tx_taps"):
                self.set_taps(self.pybert.tx_taps)
        self._update_mode()

    def _update_mode(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
            self.pybert.tx_use_ami = True
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)
            self.pybert.tx_use_ami = False

    def _update_ami_view(self) -> None:
        """Update the AMI view based on the current IBIS model state."""
        if self.pybert._tx_ibis.has_algorithmic_model:
            self.ibis_radio.setChecked(True)
            self._update_mode()
            self.ami_file.setText(str(self.pybert._tx_ibis.ami_file))
            self.dll_file.setText(str(self.pybert._tx_ibis.dll_file))
            self.ami_model_valid.set_status("valid")
            self.ami_configurator.setEnabled(True)
        else:
            self.ami_model_valid.set_status("invalid")
            self.native_radio.setChecked(True)
            self._update_mode()

    def _open_ami_configurator(self) -> None:
        """Open the AMI configurator."""
        if self.ami_model_valid.property("status") == "valid":
            self.pybert._tx_cfg.gui()

    def switch_equalization_modes(self, has_ami: bool) -> None:
        """Switch the equalization modes based on the presence of an AMI file."""
        self.ibis_radio.setChecked(has_ami)
        self._update_mode()

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

    def _load_ami_model(self) -> None:
        """Load AMI model from file."""
        filename = select_file_dialog(self, "Select AMI Model", "AMI Models (*.dll *.so *.dylib);;All Files (*.*)")
        if filename:
            self.ami_file.setText(filename)
            if self.pybert.load_new_tx_ami_model(filename):
                self.ami_model_valid.set_status("valid")
                self.view_btn.setEnabled(True)
                self.use_getwave.setEnabled(True)
                self.view_btn.clicked.connect(self.pybert.tx_ami_model.gui)
            else:
                self.ami_model_valid.set_status("invalid")
                self.view_btn.setEnabled(False)
                self.use_getwave.setEnabled(False)
                self.view_btn.clicked.disconnect()
