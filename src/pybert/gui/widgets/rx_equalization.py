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

from pybert.gui.dialogs import select_file
from pybert.utility.debug import setattr


class RxEqualizationWidget(QGroupBox):
    """Widget for configuring receiver equalization parameters."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the receiver equalization widget.

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
        self.ctle_model_radio.setChecked(True)  # Default to Model
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
        self.peak_freq.setValue(5.0)
        self.peak_freq.setSuffix(" GHz")
        model_layout.addWidget(QLabel("Peaking Frequency"))
        model_layout.addWidget(self.peak_freq)

        model_layout.addSpacing(20)  # Add some spacing between controls

        # Bandwidth
        self.rx_bw = QDoubleSpinBox()
        self.rx_bw.setToolTip("Unequalized signal path bandwidth (GHz)")
        self.rx_bw.setDecimals(2)
        self.rx_bw.setValue(12.0)
        self.rx_bw.setSuffix(" GHz")
        model_layout.addWidget(QLabel("Bandwidth"))
        model_layout.addWidget(self.rx_bw)

        model_layout.addSpacing(20)  # Add some spacing between controls

        # Peak Magnitude
        self.peak_mag = QDoubleSpinBox()
        self.peak_mag.setToolTip("CTLE peaking magnitude (dB)")
        self.peak_mag.setDecimals(1)
        self.peak_mag.setValue(1.7)
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
        self.delta_t.setValue(0.1)
        self.delta_t.setSuffix(" ps")
        cdr_layout.addRow("Delta-t", self.delta_t)

        # Alpha
        self.alpha = QDoubleSpinBox()
        self.alpha.setToolTip("Relative magnitude of CDR integral branch")
        self.alpha.setDecimals(2)
        self.alpha.setValue(0.01)
        cdr_layout.addRow("Alpha", self.alpha)

        # Lock parameters
        self.n_lock_ave = QSpinBox()
        self.n_lock_ave.setToolTip("# of UI estimates to average, when determining lock")
        self.n_lock_ave.setRange(1, 100000)
        self.n_lock_ave.setValue(500)
        cdr_layout.addRow("Lock Nave.", self.n_lock_ave)

        self.rel_lock_tol = QDoubleSpinBox()
        self.rel_lock_tol.setToolTip("Relative tolerance for determining lock")
        self.rel_lock_tol.setDecimals(2)
        self.rel_lock_tol.setValue(0.01)
        cdr_layout.addRow("Lock Tol.", self.rel_lock_tol)

        self.lock_sustain = QSpinBox()
        self.lock_sustain.setToolTip("Length of lock determining hysteresis vector")
        self.lock_sustain.setRange(1, 100000)
        self.lock_sustain.setValue(500)
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
        self.gain.setValue(0.2)
        dfe_layout.addRow("Gain", self.gain)

        # Nave
        self.n_ave = QSpinBox()
        self.n_ave.setToolTip("# of CDR adaptations per DFE adaptation")
        self.n_ave.setRange(1, 100000)
        self.n_ave.setValue(100)
        dfe_layout.addRow("Nave.", self.n_ave)

        # Decision Level
        self.decision_scaler = QDoubleSpinBox()
        self.decision_scaler.setToolTip("Target output magnitude")
        self.decision_scaler.setDecimals(2)
        self.decision_scaler.setValue(0.5)
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
        self.sum_bw.setValue(12.0)
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

        # Connect signals for radio buttons
        self.ibis_radio.toggled.connect(self._update_mode)
        self.native_radio.toggled.connect(self._update_mode)
        self.ctle_file_radio.toggled.connect(self._update_ctle_mode)
        self.ctle_model_radio.toggled.connect(self._update_ctle_mode)

        # Connect signals for enabling/disabling controls
        self.sum_ideal.toggled.connect(self._update_sum_bw_control)

        # Set initial visibility
        self._update_mode()
        self._update_ctle_mode()
        self._update_sum_bw_control()

    def connect_signals(self, pybert) -> None:
        """Connect signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(lambda val: setattr(pybert, "rx_eq", "Native" if self.native_radio.isChecked() else "IBIS"))
        # CTLE
        self.ctle_enable.toggled.connect(lambda val: setattr(pybert, "ctle_enable", val))
        self.ctle_file_radio.toggled.connect(lambda val: setattr(pybert, "rx_ctle_model", "File"))
        self.ctle_model_radio.toggled.connect(lambda val: setattr(pybert, "rx_ctle_model", "Native"))
        self.ctle_file.textChanged.connect(lambda val: setattr(pybert, "ctle_file", val))
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


    def _update_mode(self) -> None:
        """Show only the selected group (IBIS or Native) using stacked layout."""
        if self.ibis_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.ibis_group)
        else:
            self.stacked_layout.setCurrentWidget(self.native_group)

    def _update_ctle_mode(self) -> None:
        """Show only the selected CTLE mode (File or Model) using stacked layout."""
        if self.ctle_file_radio.isChecked():
            self.ctle_stacked_layout.setCurrentWidget(self.ctle_stacked_layout.widget(0))  # File widget
        else:
            self.ctle_stacked_layout.setCurrentWidget(self.ctle_stacked_layout.widget(1))  # Model widget

    def _browse_ctle(self) -> None:
        """Open file dialog to select CTLE file."""
        filename = select_file(self, "Select CTLE File", "CSV Files (*.csv);;All Files (*.*)")
        if filename:
            self.ctle_file.setText(filename)

    def _update_sum_bw_control(self) -> None:
        """Enable/disable sum bandwidth control based on ideal checkbox."""
        self.sum_bw.setEnabled(not self.sum_ideal.isChecked())
