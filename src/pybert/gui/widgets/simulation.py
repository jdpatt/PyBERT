"""Simulation control widget for PyBERT GUI.

This widget contains controls for simulation parameters like bit rate,
samples per unit interval, modulation type, etc.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.dialogs import warning_dialog
from pybert.gui.widgets.utils import block_signals
from pybert.models.stimulus import BitPattern, ModulationType
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert")


class SimulationConfiglWidget(QGroupBox):
    """Widget for controlling simulation parameters."""

    def __init__(self, pybert: PyBERT | None = None, parent: Optional[QWidget] = None) -> None:
        """Initialize the simulation control widget.

        Args:
            pybert: PyBERT model instance
            parent: Parent widget
        """
        super().__init__("Stimulus", parent)
        self.pybert = pybert

        # Create main layout
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Rate & Modulation group
        rate_group = QGroupBox("Rate && Modulation")
        rate_layout = QVBoxLayout()
        rate_group.setLayout(rate_layout)

        # Bit Rate
        bit_rate_layout = QHBoxLayout()
        bit_rate_layout.addWidget(QLabel("Bit Rate"))
        self.bit_rate = QDoubleSpinBox()
        self.bit_rate.setRange(0.1, 250.0)
        self.bit_rate.setSuffix(" Gbps")
        bit_rate_layout.addWidget(self.bit_rate)
        rate_layout.addLayout(bit_rate_layout)

        # Samples per UI
        samp_layout = QHBoxLayout()
        samp_layout.addWidget(QLabel("Samples per UI"))
        self.nspui = QSpinBox()
        self.nspui.setRange(2, 256)
        samp_layout.addWidget(self.nspui)
        rate_layout.addLayout(samp_layout)

        # Modulation type
        mod_layout = QHBoxLayout()
        mod_layout.addWidget(QLabel("Modulation"))
        self.modulation = QComboBox()
        self.modulation.addItems([mod.value for mod in ModulationType])
        mod_layout.addWidget(self.modulation)
        rate_layout.addLayout(mod_layout)

        layout.addWidget(rate_group)

        # Test Pattern group
        pattern_group = QGroupBox("Test Pattern")
        pattern_layout = QVBoxLayout()
        pattern_group.setLayout(pattern_layout)

        # Pattern type
        pat_layout = QHBoxLayout()
        pat_layout.addWidget(QLabel("Pattern"))
        self.pattern = QComboBox()
        self.pattern.addItems([pattern.name for pattern in BitPattern])
        pat_layout.addWidget(self.pattern)
        pattern_layout.addLayout(pat_layout)

        # Seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 1000000)
        seed_layout.addWidget(self.seed)
        pattern_layout.addLayout(seed_layout)

        # Number of bits
        nbits_layout = QHBoxLayout()
        nbits_layout.addWidget(QLabel("Nbits"))
        self.nbits = QSpinBox()
        self.nbits.setRange(1000, 10_000_000)
        nbits_layout.addWidget(self.nbits)
        pattern_layout.addLayout(nbits_layout)

        # Eye bits
        eye_layout = QHBoxLayout()
        eye_layout.addWidget(QLabel("EyeBits"))
        self.eye_bits = QSpinBox()
        self.eye_bits.setRange(0, 100000)
        eye_layout.addWidget(self.eye_bits)
        pattern_layout.addLayout(eye_layout)

        layout.addWidget(pattern_group)

        # Tx Level & Noise group
        level_group = QGroupBox("Tx Level && Noise")
        level_layout = QVBoxLayout()
        level_group.setLayout(level_layout)

        # Vod
        vod_layout = QHBoxLayout()
        vod_layout.addWidget(QLabel("Output Voltage"))
        self.vod = QDoubleSpinBox()
        self.vod.setSuffix(" V")
        vod_layout.addWidget(self.vod)
        level_layout.addLayout(vod_layout)

        # Random noise
        rn_layout = QHBoxLayout()
        rn_layout.addWidget(QLabel("Random Noise"))
        self.rn = QDoubleSpinBox()
        self.rn.setSuffix(" V")
        rn_layout.addWidget(self.rn)
        level_layout.addLayout(rn_layout)

        # Periodic noise magnitude
        pn_mag_layout = QHBoxLayout()
        pn_mag_layout.addWidget(QLabel("Periodic Noise"))
        self.pn_mag = QDoubleSpinBox()
        self.pn_mag.setSuffix(" V")
        pn_mag_layout.addWidget(self.pn_mag)
        level_layout.addLayout(pn_mag_layout)

        # Periodic noise frequency
        pn_freq_layout = QHBoxLayout()
        pn_freq_layout.addWidget(QLabel("f(Pn)"))
        self.pn_freq = QDoubleSpinBox()
        self.pn_freq.setSuffix(" MHz")
        pn_freq_layout.addWidget(self.pn_freq)
        level_layout.addLayout(pn_freq_layout)

        layout.addWidget(level_group)

        # --- Analysis Parameters Group ---
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout()
        analysis_group.setLayout(analysis_layout)

        # Impulse Response Length
        impulse_label = QLabel("Impulse Response Length")
        self.impulse_length = QDoubleSpinBox()
        self.impulse_length.setToolTip("Manual impulse response length override (Determined automatically, when 0.)")
        self.impulse_length.setDecimals(3)
        self.impulse_length.setFixedWidth(100)
        self.impulse_length.setSuffix(" ns")
        analysis_layout.addWidget(impulse_label, 0, 0)
        analysis_layout.addWidget(self.impulse_length, 0, 1)

        # PJ Threshold
        thresh_label = QLabel("PJ Threshold")
        self.thresh = QDoubleSpinBox()
        self.thresh.setToolTip("Threshold for identifying periodic jitter spectral elements. (sigma)")
        self.thresh.setDecimals(3)
        self.thresh.setFixedWidth(100)
        self.thresh.setSuffix(" sigma")
        analysis_layout.addWidget(thresh_label, 1, 0)
        analysis_layout.addWidget(self.thresh, 1, 1)

        # Maximum Frequency
        fmax_label = QLabel("fMax")
        self.f_max = QDoubleSpinBox()
        self.f_max.setToolTip("Maximum frequency used for plotting, modeling, and signal processing. (GHz)")
        self.f_max.setDecimals(3)
        self.f_max.setFixedWidth(100)
        self.f_max.setSuffix(" GHz")
        self.f_max.setRange(0.0, 1000.0)
        analysis_layout.addWidget(fmax_label, 2, 0)
        analysis_layout.addWidget(self.f_max, 2, 1)

        # Frequency Step
        fstep_label = QLabel("fStep")
        self.f_step = QDoubleSpinBox()
        self.f_step.setToolTip("Frequency step used for plotting, modeling, and signal processing. (MHz)")
        self.f_step.setDecimals(3)
        self.f_step.setFixedWidth(100)
        self.f_step.setSuffix(" MHz")
        analysis_layout.addWidget(fstep_label, 3, 0)
        analysis_layout.addWidget(self.f_step, 3, 1)

        # Add the analysis group to the main layout
        layout.addWidget(analysis_group)

        # Initialize widget values from model if available
        if self.pybert is not None:
            self.update_from_model()
            self.connect_signals(self.pybert)

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        if self.pybert is None:
            return

        with block_signals(self):
            # Update rate & modulation
            self.bit_rate.setValue(self.pybert.bit_rate)
            self.nspui.setValue(self.pybert.nspui)
            mod_index = {ModulationType.NRZ: 0, ModulationType.DUO: 1, ModulationType.PAM4: 2}[self.pybert.mod_type]
            self.modulation.setCurrentIndex(mod_index)

            # Update pattern settings
            self.pattern.setCurrentText(self.pybert.pattern.name)
            self.seed.setValue(self.pybert.seed)
            self.nbits.setValue(self.pybert.nbits)
            self.eye_bits.setValue(self.pybert.eye_bits)

            # Update level & noise
            self.vod.setValue(self.pybert.vod)
            self.rn.setValue(self.pybert.rn)
            self.pn_mag.setValue(self.pybert.pn_mag)
            self.pn_freq.setValue(self.pybert.pn_freq)

            # Update analysis parameters
            self.impulse_length.setValue(self.pybert.impulse_length)
            self.thresh.setValue(self.pybert.thresh)
            self.f_max.setValue(self.pybert.f_max)
            self.f_step.setValue(self.pybert.f_step)

    def connect_signals(self, pybert: "PyBERT") -> None:
        """Connect widget signals to PyBERT stimulus model."""
        self.bit_rate.valueChanged.connect(lambda val: setattr(pybert, "bit_rate", val))
        self.nspui.valueChanged.connect(lambda val: setattr(pybert, "nspui", val))
        self.modulation.currentTextChanged.connect(lambda val: self.update_modulation(pybert, val))
        self.seed.valueChanged.connect(lambda val: setattr(pybert, "seed", val))
        self.pattern.currentTextChanged.connect(self.update_pattern)
        self.nbits.valueChanged.connect(self.update_nbits)
        self.eye_bits.valueChanged.connect(self.update_eye_bits)
        self.vod.valueChanged.connect(lambda val: setattr(pybert, "vod", val))
        self.rn.valueChanged.connect(lambda val: setattr(pybert, "rn", val))
        self.pn_mag.valueChanged.connect(lambda val: setattr(pybert, "pn_mag", val))
        self.pn_freq.valueChanged.connect(lambda val: setattr(pybert, "pn_freq", val))
        self.impulse_length.valueChanged.connect(lambda val: setattr(pybert, "impulse_length", val))
        self.thresh.valueChanged.connect(lambda val: setattr(pybert, "thresh", val))
        self.f_max.valueChanged.connect(self.update_f_max)
        self.f_step.valueChanged.connect(lambda val: setattr(pybert, "f_step", val))

    def update_modulation(self, pybert: "PyBERT", val: str) -> None:
        """Update the modulation type."""
        if val == ModulationType.NRZ.value:
            setattr(pybert, "mod_type", ModulationType.NRZ)
        elif val == ModulationType.DUO.value:
            setattr(pybert, "mod_type", ModulationType.DUO)
        elif val == ModulationType.PAM4.value:
            setattr(pybert, "mod_type", ModulationType.PAM4)

    def update_nbits(self, nbits: int) -> None:
        self.validate_eye_bits(self.eye_bits.value(), nbits)
        setattr(self.pybert, "nbits", nbits)

    def update_eye_bits(self, eye_bits: int) -> None:
        self.validate_eye_bits(eye_bits, self.nbits.value())
        self.validate_pattern_length(BitPattern[self.pattern.currentText()], eye_bits)
        setattr(self.pybert, "eye_bits", eye_bits)

    def update_pattern(self, text: str) -> None:
        """Update the pattern type."""
        new_pattern = BitPattern[text]
        self.validate_pattern_length(new_pattern, self.eye_bits.value())
        setattr(self.pybert, "pattern", new_pattern)

    def update_f_max(self, new_value: float) -> None:
        fmax = self.nyquist_ghz(
            self.pybert.ui, self.nspui.value()
        )  # Nyquist frequency, given our sampling rate (GHz).
        if new_value > fmax:
            self.f_max.setValue(fmax)
            logger.warning("`fMax` has been held at the Nyquist frequency.")
        else:
            setattr(self.pybert, "f_max", new_value)

    def validate_eye_bits(self, eye_bits: int, nbits: int) -> None:
        """Validate user selected number of eye bits."""
        if eye_bits > nbits:
            logger.warning("`EyeBits` has been held at `Nbits`.")
            self.eye_bits.setValue(nbits)

    def validate_pattern_length(self, pattern: BitPattern, eye_bits: int) -> None:
        """Validate chosen pattern length against number of bits being run."""
        pat_len = 2 * pow(2, max(pattern.value))  # "2 *", to accommodate PAM-4.
        if eye_bits < 5 * pat_len:
            warning_dialog(
                "Configuration Warning",
                "\n".join(
                    [
                        "Accurate jitter decomposition may not be possible with the current configuration!",
                        "Try to keep `EyeBits` > 10 * 2^n, where `n` comes from `PRBS-n`.",
                    ]
                ),
            )

    @staticmethod
    def nyquist_ghz(ui: float, nspui: int) -> float:
        t_step = ui / nspui
        return 0.5e-9 / t_step
