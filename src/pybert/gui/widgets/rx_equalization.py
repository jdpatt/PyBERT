"""Transmitter configuration widget for PyBERT GUI.

This widget contains controls for transmitter parameters including IBIS
model selection and native parameters.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.dialogs import select_file_dialog
from pybert.gui.widgets.utils import StatusIndicator
from pybert.pybert import PyBERT


class RxEqualizationWidget(QGroupBox):
    """Widget for configuring receiver equalization parameters."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
        """Initialize the receiver equalization widget.

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

        # Bottom row with status, checkboxes, and configure
        bottom_layout = QHBoxLayout()
        # Status indicator
        bottom_layout.addWidget(QLabel("Status:"))
        self.ami_model_valid = StatusIndicator()
        bottom_layout.addWidget(self.ami_model_valid)
        bottom_layout.addSpacing(20)  # Add some spacing between status and checkboxes

        # GetWave checkbox
        self.use_getwave = QCheckBox("Use GetWave()")
        bottom_layout.addWidget(self.use_getwave)
        bottom_layout.addSpacing(10)  # Add spacing between checkboxes

        # Clocks checkbox
        self.use_clocks = QCheckBox("Use Clocks")
        bottom_layout.addWidget(self.use_clocks)

        bottom_layout.addStretch()  # Push configure button to the right

        # Configure button
        self.ami_configurator = QPushButton("Configure")
        self.ami_configurator.setEnabled(False)
        bottom_layout.addWidget(self.ami_configurator)

        ibis_layout.addLayout(bottom_layout)
        ibis_layout.addStretch()

        # Add IBIS group to stacked layout
        self.stacked_layout.addWidget(self.ibis_group)

        # Native parameters group
        self.native_group = QWidget(self)
        native_layout = QVBoxLayout()
        self.native_group.setLayout(native_layout)

        # CTLE Group
        ctle_group = QGroupBox("CTLE")
        ctle_layout = QVBoxLayout()

        # CTLE Enable
        self.ctle_enable = QCheckBox("Enable")
        self.ctle_enable.setToolTip("CTLE enable")
        self.ctle_enable.setChecked(True)
        ctle_layout.addWidget(self.ctle_enable)

        # CTLE Mode Selection
        ctle_mode_layout = QHBoxLayout()
        self.ctle_file_radio = QRadioButton("File")
        self.ctle_model_radio = QRadioButton("Model")

        self.ctle_mode_group = QButtonGroup(self)
        self.ctle_mode_group.addButton(self.ctle_file_radio)
        self.ctle_mode_group.addButton(self.ctle_model_radio)
        ctle_mode_layout.addWidget(self.ctle_file_radio)
        ctle_mode_layout.addWidget(self.ctle_model_radio)
        ctle_mode_layout.addStretch()
        ctle_layout.addLayout(ctle_mode_layout)

        # CTLE Stacked Layout
        self.ctle_stacked_layout = QStackedLayout()

        # CTLE File Group
        file_group = QWidget(self)
        file_group_layout = QHBoxLayout()
        file_group.setLayout(file_group_layout)
        self.ctle_file = QLineEdit()
        self.browse_ctle_btn = QPushButton("Browse...")
        self.browse_ctle_btn.clicked.connect(self._browse_ctle)
        file_group_layout.addWidget(QLabel("Filename"))
        file_group_layout.addWidget(self.ctle_file)
        file_group_layout.addWidget(self.browse_ctle_btn)
        self.ctle_stacked_layout.addWidget(file_group)

        # CTLE Model Group
        model_group = QWidget(self)
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)

        # Peak Frequency
        self.peak_freq = QDoubleSpinBox()
        self.peak_freq.setToolTip("CTLE peaking frequency (GHz)")
        self.peak_freq.setDecimals(2)
        self.peak_freq.setSuffix(" GHz")
        model_layout.addWidget(QLabel("Peaking Frequency"))
        model_layout.addWidget(self.peak_freq)

        model_layout.addSpacing(20)  # Add some spacing between controls

        # Bandwidth
        self.rx_bw = QDoubleSpinBox()
        self.rx_bw.setToolTip("Unequalized signal path bandwidth (GHz)")
        self.rx_bw.setDecimals(2)
        self.rx_bw.setSuffix(" GHz")
        model_layout.addWidget(QLabel("Bandwidth"))
        model_layout.addWidget(self.rx_bw)

        model_layout.addSpacing(20)  # Add some spacing between controls

        # Peak Magnitude
        self.peak_mag = QDoubleSpinBox()
        self.peak_mag.setToolTip("CTLE peaking magnitude (dB)")
        self.peak_mag.setDecimals(1)

        self.peak_mag.setSuffix(" dB")
        model_layout.addWidget(QLabel("Boost"))
        model_layout.addWidget(self.peak_mag)
        model_layout.addStretch()

        self.ctle_stacked_layout.addWidget(model_group)
        ctle_layout.addLayout(self.ctle_stacked_layout)
        ctle_group.setLayout(ctle_layout)
        native_layout.addWidget(ctle_group)

        # CDR and DFE Container
        cdr_dfe_layout = QHBoxLayout()

        # CDR Group
        cdr_group = QGroupBox("CDR")
        cdr_layout = QFormLayout()

        # Delta-t
        self.delta_t = QDoubleSpinBox()
        self.delta_t.setToolTip("Magnitude of CDR proportional branch")
        self.delta_t.setDecimals(2)
        self.delta_t.setSuffix(" ps")
        cdr_layout.addRow("Delta-t", self.delta_t)

        # Alpha
        self.alpha = QDoubleSpinBox()
        self.alpha.setToolTip("Relative magnitude of CDR integral branch")
        self.alpha.setDecimals(2)

        cdr_layout.addRow("Alpha", self.alpha)

        # Lock parameters
        self.n_lock_ave = QSpinBox()
        self.n_lock_ave.setToolTip("# of UI estimates to average, when determining lock")
        self.n_lock_ave.setRange(1, 100000)
        cdr_layout.addRow("Lock Nave.", self.n_lock_ave)

        self.rel_lock_tol = QDoubleSpinBox()
        self.rel_lock_tol.setToolTip("Relative tolerance for determining lock")
        self.rel_lock_tol.setDecimals(2)
        cdr_layout.addRow("Lock Tol.", self.rel_lock_tol)

        self.lock_sustain = QSpinBox()
        self.lock_sustain.setToolTip("Length of lock determining hysteresis vector")
        self.lock_sustain.setRange(1, 100000)
        cdr_layout.addRow("Lock Sus.", self.lock_sustain)

        cdr_group.setLayout(cdr_layout)
        cdr_dfe_layout.addWidget(cdr_group)

        # DFE Group
        dfe_group = QGroupBox("DFE")
        dfe_layout = QFormLayout()

        dfe_layout.addRow(QLabel("Use Optimizer tab to configure."))

        # Gain
        self.gain = QDoubleSpinBox()
        self.gain.setToolTip("Error feedback gain")
        self.gain.setDecimals(2)
        dfe_layout.addRow("Gain", self.gain)

        # Nave
        self.n_ave = QSpinBox()
        self.n_ave.setToolTip("# of CDR adaptations per DFE adaptation")
        self.n_ave.setRange(1, 100000)
        dfe_layout.addRow("Nave.", self.n_ave)

        # Decision Level
        self.decision_scaler = QDoubleSpinBox()
        self.decision_scaler.setToolTip("Target output magnitude")
        self.decision_scaler.setDecimals(2)
        level_widget = QWidget()
        level_layout = QHBoxLayout(level_widget)
        level_layout.setContentsMargins(0, 0, 0, 0)
        level_layout.addWidget(self.decision_scaler)
        level_layout.addWidget(QLabel("V"))
        dfe_layout.addRow("Level", level_widget)

        # Bandwidth and Ideal
        self.sum_bw = QDoubleSpinBox()
        self.sum_bw.setToolTip("Summing node bandwidth")
        self.sum_bw.setDecimals(2)
        self.sum_bw.setSuffix(" GHz")
        bw_widget = QWidget()
        bw_layout = QHBoxLayout(bw_widget)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.addWidget(self.sum_bw)
        self.sum_ideal = QCheckBox("Ideal")
        self.sum_ideal.setToolTip("Use ideal DFE. (performance boost)")
        self.sum_ideal.setChecked(True)
        bw_layout.addWidget(self.sum_ideal)
        dfe_layout.addRow("Bandwidth", bw_widget)

        dfe_group.setLayout(dfe_layout)
        cdr_dfe_layout.addWidget(dfe_group)

        native_layout.addLayout(cdr_dfe_layout)
        native_layout.addStretch()

        # Add native group to stacked layout
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
        self.ctle_file_radio.toggled.connect(self._update_ctle_mode)
        self.ctle_model_radio.toggled.connect(self._update_ctle_mode)
        self.ami_configurator.clicked.connect(self._open_ami_configurator)
        self.sum_ideal.toggled.connect(self._update_sum_bw_control)

        pybert.new_rx_model.connect(self._update_ami_view)

        self.mode_group.buttonReleased.connect(
            lambda val: setattr(pybert, "rx_use_ami", self.native_radio.isChecked() == False)
        )
        self.use_getwave.toggled.connect(lambda val: setattr(pybert, "rx_use_getwave", val))
        self.use_clocks.toggled.connect(lambda val: setattr(pybert, "rx_use_clocks", val))
        # CTLE
        self.ctle_enable.toggled.connect(lambda val: setattr(pybert, "ctle_enable", val))
        self.ctle_file.textChanged.connect(lambda val: setattr(pybert, "ctle_file", val))
        # Use buttonReleased instead of toggled to avoid double signals
        self.ctle_mode_group.buttonReleased.connect(
            lambda: setattr(pybert, "rx_ctle_model", "File" if self.ctle_file_radio.isChecked() else "Native")
        )
        self.rx_bw.valueChanged.connect(lambda val: setattr(pybert, "rx_bw", val))
        self.peak_freq.valueChanged.connect(lambda val: setattr(pybert, "peak_freq", val))
        self.peak_mag.valueChanged.connect(lambda val: setattr(pybert, "peak_mag", val))
        # CDR
        self.delta_t.valueChanged.connect(lambda val: setattr(pybert, "delta_t", val))
        self.alpha.valueChanged.connect(lambda val: setattr(pybert, "alpha", val))
        self.n_lock_ave.valueChanged.connect(lambda val: setattr(pybert, "n_lock_ave", val))
        self.rel_lock_tol.valueChanged.connect(lambda val: setattr(pybert, "rel_lock_tol", val))
        self.lock_sustain.valueChanged.connect(lambda val: setattr(pybert, "lock_sustain", val))
        # DFE
        self.gain.valueChanged.connect(lambda val: setattr(pybert, "gain", val))
        self.n_ave.valueChanged.connect(lambda val: setattr(pybert, "n_ave", val))
        self.decision_scaler.valueChanged.connect(lambda val: setattr(pybert, "decision_scaler", val))
        self.sum_bw.valueChanged.connect(lambda val: setattr(pybert, "sum_bw", val))
        self.sum_ideal.toggled.connect(lambda val: setattr(pybert, "sum_ideal", val))

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        if self.pybert is None:
            return

        self.block_signals(True)
        try:
            # Update mode
            self.native_radio.setChecked(self.pybert.rx_use_ami == False)
            self.ibis_radio.setChecked(self.pybert.rx_use_ami == True)

            # Update AMI settings
            if hasattr(self.pybert, "_rx_ibis") and self.pybert._rx_ibis is not None:
                self.ami_file.setText(str(self.pybert._rx_ibis.ami_file))
                self.dll_file.setText(str(self.pybert._rx_ibis.dll_file))
                self.ami_model_valid.set_status("valid" if self.pybert._rx_ibis.has_algorithmic_model else "invalid")
                self.ami_configurator.setEnabled(self.pybert._rx_ibis.has_algorithmic_model)
                self.use_getwave.setEnabled(self.pybert._rx_ibis.has_algorithmic_model)
                self.use_clocks.setEnabled(self.pybert._rx_ibis.has_algorithmic_model)
                self.use_getwave.setChecked(self.pybert.rx_use_getwave)
                self.use_clocks.setChecked(self.pybert.rx_use_clocks)
            else:
                self.ami_model_valid.set_status("not_loaded")
                self.ami_configurator.setEnabled(False)
                self.use_getwave.setEnabled(False)
                self.use_clocks.setEnabled(False)

            # Update CTLE parameters
            self.ctle_enable.setChecked(self.pybert.ctle_enable)
            self.ctle_file_radio.setChecked(self.pybert.use_ctle_file)
            self.ctle_model_radio.setChecked(not self.pybert.use_ctle_file)
            self.peak_freq.setValue(self.pybert.peak_freq)
            self.rx_bw.setValue(self.pybert.rx_bw)
            self.peak_mag.setValue(self.pybert.peak_mag)
            if hasattr(self.pybert, "rx_ctle_file"):
                self.ctle_file.setText(self.pybert.ctle_file)

            # Update CDR parameters
            self.delta_t.setValue(self.pybert.delta_t)
            self.alpha.setValue(self.pybert.alpha)
            self.n_lock_ave.setValue(self.pybert.n_lock_ave)
            self.rel_lock_tol.setValue(self.pybert.rel_lock_tol)
            self.lock_sustain.setValue(self.pybert.lock_sustain)

            # Update DFE parameters
            self.gain.setValue(self.pybert.gain)
            self.n_ave.setValue(self.pybert.n_ave)
            self.decision_scaler.setValue(self.pybert.decision_scaler)
            self.sum_bw.setValue(self.pybert.sum_bw)
            self.sum_ideal.setChecked(self.pybert.sum_ideal)

        finally:
            self._update_mode()
            self._update_ctle_mode()
            self._update_sum_bw_control()
            self.block_signals(False)

    def block_signals(self, block: bool = True) -> None:
        """Block or unblock all widget signals to prevent unnecessary updates.

        Args:
            block: True to block signals, False to unblock
        """
        widgets = [
            self.native_radio,
            self.ibis_radio,
            self.ctle_enable,
            self.ctle_file_radio,
            self.ctle_model_radio,
            self.peak_freq,
            self.rx_bw,
            self.peak_mag,
            self.use_getwave,
            self.use_clocks,
            self.ami_file,
            self.dll_file,
            self.ctle_file,
        ]
        for widget in widgets:
            widget.blockSignals(block)

    def _update_mode(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
            self.pybert.rx_use_ami = True
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)
            self.pybert.rx_use_ami = False

    def _update_ctle_mode(self) -> None:
        """Show only the selected CTLE mode (File or Model) using stacked layout."""
        if self.ctle_file_radio.isChecked():
            self.ctle_stacked_layout.setCurrentWidget(self.ctle_stacked_layout.widget(0))  # File widget
        else:
            self.ctle_stacked_layout.setCurrentWidget(self.ctle_stacked_layout.widget(1))  # Model widget

    def _browse_ctle(self) -> None:
        """Open file dialog to select CTLE file."""
        filename = select_file_dialog(self, "Select CTLE File", "CSV Files (*.csv);;All Files (*.*)")
        if filename:
            self.ctle_file.setText(filename)

    def _update_sum_bw_control(self) -> None:
        """Enable/disable sum bandwidth control based on ideal checkbox."""
        self.sum_bw.setEnabled(not self.sum_ideal.isChecked())

    def _open_ami_configurator(self) -> None:
        """Open the AMI configurator."""
        if self.ami_model_valid.property("status") == "valid":
            self.pybert._rx_cfg.gui()

    def _update_ami_view(self) -> None:
        """Update the AMI view based on the current IBIS model state."""
        if self.pybert._rx_ibis.has_algorithmic_model:
            self.ibis_radio.setChecked(True)
            self._update_mode()
            self.ami_file.setText(str(self.pybert._rx_ibis.ami_file))
            self.dll_file.setText(str(self.pybert._rx_ibis.dll_file))
            self.ami_model_valid.set_status("valid")
            self.ami_configurator.setEnabled(True)
        else:
            self.ami_model_valid.set_status("invalid")
            self.native_radio.setChecked(True)
            self._update_mode()
