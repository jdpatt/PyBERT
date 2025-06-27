"""Receiver configuration widget for PyBERT GUI.

This widget contains controls for receiver parameters including IBIS
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.file_picker import FilePickerWidget
from pybert.gui.widgets.ibis_ami_config import IbisAmiConfigWidget
from pybert.gui.widgets.status_indicator import StatusIndicator
from pybert.gui.widgets.utils import block_signals
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
        self.ibis_radio.setChecked(False)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio, 0)
        self.mode_group.addButton(self.ibis_radio, 1)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.ibis_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- Stacked layout for transmitter config groups ---
        self.stacked_widget = QStackedWidget()

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
        self.ctle_model_radio.setChecked(True)

        self.ctle_mode_group = QButtonGroup(self)
        self.ctle_mode_group.addButton(self.ctle_file_radio, 0)
        self.ctle_mode_group.addButton(self.ctle_model_radio, 1)
        ctle_mode_layout.addWidget(self.ctle_file_radio)
        ctle_mode_layout.addWidget(self.ctle_model_radio)
        ctle_mode_layout.addStretch()
        ctle_layout.addLayout(ctle_mode_layout)

        # CTLE Stacked Layout
        self.ctle_stacked_widget = QStackedWidget()

        # CTLE File Group
        file_group = QWidget(self)
        file_group_layout = QVBoxLayout()
        file_group.setLayout(file_group_layout)
        self.ctle_file = FilePickerWidget("File", "CSV Files (*.csv);;All Files (*.*)", file_group)
        file_group_layout.addWidget(self.ctle_file)

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

        self.ctle_stacked_widget.addWidget(file_group)
        self.ctle_stacked_widget.addWidget(model_group)
        ctle_layout.addWidget(self.ctle_stacked_widget)
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

        # Add native group to stacked widget
        self.stacked_widget.addWidget(self.native_group)
        self.ami_widget = IbisAmiConfigWidget(obj_accessor=lambda: self.pybert.rx, parent=self)
        self.stacked_widget.addWidget(self.ami_widget)

        layout.addWidget(self.stacked_widget)
        layout.addStretch()

        self.update_widget_from_model()
        self.connect_signals()

    def connect_signals(self) -> None:
        """Connect signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(self._handle_ibis_radio_toggled)
        self.ctle_mode_group.buttonReleased.connect(self._handle_ctle_radio_toggled)
        self.sum_ideal.toggled.connect(self.sum_bw.setEnabled)

        # CTLE
        self.ctle_enable.toggled.connect(lambda val: setattr(self.pybert, "ctle_enable", val))
        self.ctle_file.file_selected.connect(lambda filename: setattr(self.pybert, "ctle_file", filename))
        self.rx_bw.valueChanged.connect(lambda val: setattr(self.pybert, "rx_bw", val))
        self.peak_freq.valueChanged.connect(lambda val: setattr(self.pybert, "peak_freq", val))
        self.peak_mag.valueChanged.connect(lambda val: setattr(self.pybert, "peak_mag", val))
        # CDR
        self.delta_t.valueChanged.connect(lambda val: setattr(self.pybert, "delta_t", val))
        self.alpha.valueChanged.connect(lambda val: setattr(self.pybert, "alpha", val))
        self.n_lock_ave.valueChanged.connect(lambda val: setattr(self.pybert, "n_lock_ave", val))
        self.rel_lock_tol.valueChanged.connect(lambda val: setattr(self.pybert, "rel_lock_tol", val))
        self.lock_sustain.valueChanged.connect(lambda val: setattr(self.pybert, "lock_sustain", val))
        # DFE
        self.gain.valueChanged.connect(lambda val: setattr(self.pybert, "gain", val))
        self.n_ave.valueChanged.connect(lambda val: setattr(self.pybert, "n_ave", val))
        self.decision_scaler.valueChanged.connect(lambda val: setattr(self.pybert, "decision_scaler", val))
        self.sum_bw.valueChanged.connect(lambda val: setattr(self.pybert, "sum_bw", val))
        self.sum_ideal.toggled.connect(lambda val: setattr(self.pybert, "sum_ideal", val))

    def update_widget_from_model(self) -> None:
        """Update all widget values from the PyBERT model."""
        with block_signals(self):
            # Update mode
            self.native_radio.setChecked(not self.pybert.rx.use_ami)
            self.ibis_radio.setChecked(self.pybert.rx.use_ami)

            # Update CTLE parameters
            self.ctle_enable.setChecked(self.pybert.ctle_enable)
            self.ctle_file_radio.setChecked(self.pybert.use_ctle_file)
            self.ctle_model_radio.setChecked(not self.pybert.use_ctle_file)
            self.peak_freq.setValue(self.pybert.peak_freq)
            self.rx_bw.setValue(self.pybert.rx_bw)
            self.peak_mag.setValue(self.pybert.peak_mag)
            if hasattr(self.pybert, "ctle_file"):
                self.ctle_file.set_filepath(self.pybert.ctle_file)

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
            self.sum_bw.setEnabled(not self.sum_ideal.isChecked())

        # Update stacked widget
        self.ctle_stacked_widget.setCurrentIndex(1 if self.pybert.use_ctle_file else 0)
        self.stacked_widget.setCurrentIndex(1 if self.pybert.rx.use_ami else 0)

    def get_ami_widget(self) -> IbisAmiConfigWidget:
        """Get the AMI configuration widget."""
        return self.ami_widget

    def _handle_ctle_radio_toggled(self) -> None:
        """Handle the toggled event of the CTLE radio button."""
        setattr(self.pybert, "use_ctle_file", self.ctle_file_radio.isChecked())
        self.ctle_stacked_widget.setCurrentIndex(1 if self.ctle_model_radio.isChecked() else 0)

    def _handle_ibis_radio_toggled(self) -> None:
        """Handle the toggled event of the IBIS radio button."""
        self.stacked_widget.setCurrentIndex(1 if self.ibis_radio.isChecked() else 0)
        setattr(self.pybert.rx, "use_ami", self.ibis_radio.isChecked())
