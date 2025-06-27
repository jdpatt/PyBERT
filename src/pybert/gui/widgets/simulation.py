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
    QFormLayout,
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

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None) -> None:
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
        rate_layout = QFormLayout()
        rate_group.setLayout(rate_layout)

        # Bit Rate
        self.bit_rate = QDoubleSpinBox()
        self.bit_rate.setRange(0.1, 250.0)
        self.bit_rate.setSuffix(" Gbps")
        rate_layout.addRow("Bit Rate", self.bit_rate)

        # Samples per UI
        self.nspui = QSpinBox()
        self.nspui.setRange(2, 256)
        rate_layout.addRow("Samples per UI", self.nspui)

        # Modulation type
        self.modulation = QComboBox()
        self.modulation.addItems([mod.value for mod in ModulationType])
        rate_layout.addRow("Modulation", self.modulation)

        layout.addWidget(rate_group)

        # Test Pattern group
        pattern_group = QGroupBox("Test Pattern")
        pattern_layout = QFormLayout()
        pattern_group.setLayout(pattern_layout)

        # Pattern type
        self.pattern = QComboBox()
        self.pattern.addItems([pattern.name for pattern in BitPattern])
        pattern_layout.addRow("Pattern", self.pattern)

        # Seed
        self.seed = QSpinBox()
        self.seed.setRange(0, 1000000)
        pattern_layout.addRow("Seed", self.seed)

        # Number of bits
        self.nbits = QSpinBox()
        self.nbits.setRange(1000, 10_000_000)
        pattern_layout.addRow("Nbits", self.nbits)

        # Eye bits
        self.eye_bits = QSpinBox()
        self.eye_bits.setRange(0, 100000)
        pattern_layout.addRow("EyeBits", self.eye_bits)

        layout.addWidget(pattern_group)

        # Tx Level & Noise group
        level_group = QGroupBox("Tx Level && Noise")
        level_layout = QFormLayout()
        level_group.setLayout(level_layout)

        # Vod
        self.vod = QDoubleSpinBox()
        self.vod.setSuffix(" V")
        level_layout.addRow("Output Voltage", self.vod)

        # Random noise
        self.rn = QDoubleSpinBox()
        self.rn.setSuffix(" V")
        level_layout.addRow("Random Noise", self.rn)

        # Periodic noise magnitude
        self.pn_mag = QDoubleSpinBox()
        self.pn_mag.setSuffix(" V")
        level_layout.addRow("Periodic Noise", self.pn_mag)

        # Periodic noise frequency
        self.pn_freq = QDoubleSpinBox()
        self.pn_freq.setSuffix(" MHz")
        level_layout.addRow("f(Pn)", self.pn_freq)

        layout.addWidget(level_group)

        # --- Analysis Parameters Group ---
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout()
        analysis_group.setLayout(analysis_layout)

        # Impulse Response Length
        self.impulse_length = QDoubleSpinBox()
        self.impulse_length.setToolTip("Manual impulse response length override (Determined automatically, when 0.)")
        self.impulse_length.setDecimals(3)
        self.impulse_length.setSuffix(" ns")
        analysis_layout.addRow("Impulse Response Length", self.impulse_length)

        # PJ Threshold
        self.thresh = QDoubleSpinBox()
        self.thresh.setToolTip("Threshold for identifying periodic jitter spectral elements. (sigma)")
        self.thresh.setDecimals(3)
        self.thresh.setSuffix(" sigma")
        analysis_layout.addRow("PJ Threshold", self.thresh)

        # Maximum Frequency
        self.f_max = QDoubleSpinBox()
        self.f_max.setToolTip("Maximum frequency used for plotting, modeling, and signal processing. (GHz)")
        self.f_max.setDecimals(3)
        self.f_max.setSuffix(" GHz")
        self.f_max.setRange(0.0, 1000.0)
        analysis_layout.addRow("fMax", self.f_max)

        # Frequency Step
        self.f_step = QDoubleSpinBox()
        self.f_step.setToolTip("Frequency step used for plotting, modeling, and signal processing. (MHz)")
        self.f_step.setDecimals(3)
        self.f_step.setSuffix(" MHz")
        analysis_layout.addRow("fStep", self.f_step)

        # Add the analysis group to the main layout
        layout.addWidget(analysis_group)

        # Initialize widget values from model if available
        if self.pybert is not None:
            self.update_widget_from_model()
            self.connect_signals(self.pybert)

    def update_widget_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        with block_signals(self):
            # Update rate & modulation
            self.bit_rate.setValue(self.pybert.bit_rate)
            self.nspui.setValue(self.pybert.nspui)
            # Find the correct index for the current modulation type
            mod_text = self.pybert.mod_type.value
            mod_index = self.modulation.findText(mod_text)
            if mod_index >= 0:
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
            self.impulse_length.setValue(self.pybert.channel.impulse_length)
            self.thresh.setValue(self.pybert.thresh)
            self.f_max.setValue(self.pybert.channel.f_max)
            self.f_step.setValue(self.pybert.channel.f_step)

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
        self.impulse_length.valueChanged.connect(lambda val: setattr(pybert.channel, "impulse_length", val))
        self.thresh.valueChanged.connect(lambda val: setattr(pybert, "thresh", val))
        self.f_max.valueChanged.connect(self.update_f_max)
        self.f_step.valueChanged.connect(lambda val: setattr(pybert.channel, "f_step", val))

    def update_modulation(self, pybert: "PyBERT", val: str) -> None:
        """Update the modulation type."""
        # Map the combo box text to the correct ModulationType
        if val == "NRZ":
            setattr(pybert, "mod_type", ModulationType.NRZ)
        elif val == "Duo-binary":
            setattr(pybert, "mod_type", ModulationType.DUO)
        elif val == "PAM4":
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
            setattr(self.pybert.channel, "f_max", new_value)

    def validate_eye_bits(self, eye_bits: int, nbits: int) -> None:
        """Validate user selected number of eye bits."""
        # TODO: Eventually move validation to the model and have it emit warnings for the GUI to display.
        if eye_bits > nbits:
            logger.warning("`EyeBits` has been held at `Nbits`.")
            self.eye_bits.setValue(nbits)

    def validate_pattern_length(self, pattern: BitPattern, eye_bits: int) -> None:
        """Validate chosen pattern length against number of bits being run."""
        pat_len = 2 * pow(2, max(pattern.value))  # "2 *", to accommodate PAM-4.
        if eye_bits < 5 * pat_len:
            warning_dialog(
                self,
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
