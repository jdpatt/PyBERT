"""Simulation control widget for PyBERT GUI.

This widget contains controls for simulation parameters like bit rate,
samples per unit interval, modulation type, etc.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt


class SimulationControlWidget(QGroupBox):
    """Widget for controlling simulation parameters."""

    def __init__(self, parent=None):
        """Initialize the simulation control widget.

        Args:
            parent: Parent widget
        """
        super().__init__("Simulation Control", parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create horizontal layout for top section
        top_layout = QHBoxLayout()

        # Rate & Modulation group
        rate_group = QGroupBox("Rate && Modulation")
        rate_layout = QVBoxLayout()
        rate_group.setLayout(rate_layout)

        # Bit Rate
        bit_rate_layout = QHBoxLayout()
        bit_rate_layout.addWidget(QLabel("Bit Rate:"))
        self.bit_rate = QDoubleSpinBox()
        self.bit_rate.setRange(0.1, 100.0)
        self.bit_rate.setValue(10.0)
        self.bit_rate.setSuffix(" Gbps")
        bit_rate_layout.addWidget(self.bit_rate)
        rate_layout.addLayout(bit_rate_layout)

        # Samples per UI
        samp_layout = QHBoxLayout()
        samp_layout.addWidget(QLabel("Samples per UI:"))
        self.nspui = QSpinBox()
        self.nspui.setRange(1, 100)
        self.nspui.setValue(32)
        samp_layout.addWidget(self.nspui)
        rate_layout.addLayout(samp_layout)

        # Modulation type
        mod_layout = QVBoxLayout()
        mod_layout.addWidget(QLabel("Modulation:"))
        self.nrz = QCheckBox("NRZ")
        self.nrz.setChecked(True)
        self.duobinary = QCheckBox("Duo-binary")
        self.pam4 = QCheckBox("PAM-4")
        mod_layout.addWidget(self.nrz)
        mod_layout.addWidget(self.duobinary)
        mod_layout.addWidget(self.pam4)
        rate_layout.addLayout(mod_layout)

        top_layout.addWidget(rate_group)

        # Test Pattern group
        pattern_group = QGroupBox("Test Pattern")
        pattern_layout = QVBoxLayout()
        pattern_group.setLayout(pattern_layout)

        # Pattern type
        pat_layout = QHBoxLayout()
        pat_layout.addWidget(QLabel("Pattern:"))
        self.pattern = QLineEdit("PRBS7")
        pat_layout.addWidget(self.pattern)
        pattern_layout.addLayout(pat_layout)

        # Seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 1000000)
        seed_layout.addWidget(self.seed)
        pattern_layout.addLayout(seed_layout)

        # Number of bits
        nbits_layout = QHBoxLayout()
        nbits_layout.addWidget(QLabel("Nbits:"))
        self.nbits = QSpinBox()
        self.nbits.setRange(100, 1000000)
        self.nbits.setValue(10000)
        nbits_layout.addWidget(self.nbits)
        pattern_layout.addLayout(nbits_layout)

        # Eye bits
        eye_layout = QHBoxLayout()
        eye_layout.addWidget(QLabel("EyeBits:"))
        self.eye_bits = QSpinBox()
        self.eye_bits.setRange(100, 100000)
        self.eye_bits.setValue(1000)
        eye_layout.addWidget(self.eye_bits)
        pattern_layout.addLayout(eye_layout)

        top_layout.addWidget(pattern_group)

        # Tx Level & Noise group
        level_group = QGroupBox("Tx Level && Noise")
        level_layout = QVBoxLayout()
        level_group.setLayout(level_layout)

        # Vod
        vod_layout = QHBoxLayout()
        vod_layout.addWidget(QLabel("Vod:"))
        self.vod = QDoubleSpinBox()
        self.vod.setRange(0.1, 2.0)
        self.vod.setValue(0.4)
        self.vod.setSuffix(" V")
        vod_layout.addWidget(self.vod)
        level_layout.addLayout(vod_layout)

        # Random noise
        rn_layout = QHBoxLayout()
        rn_layout.addWidget(QLabel("Rn:"))
        self.rn = QDoubleSpinBox()
        self.rn.setRange(0.0, 0.1)
        self.rn.setValue(0.001)
        self.rn.setSuffix(" V")
        rn_layout.addWidget(self.rn)
        level_layout.addLayout(rn_layout)

        # Periodic noise magnitude
        pn_mag_layout = QHBoxLayout()
        pn_mag_layout.addWidget(QLabel("Pn:"))
        self.pn_mag = QDoubleSpinBox()
        self.pn_mag.setRange(0.0, 0.1)
        self.pn_mag.setValue(0.0)
        self.pn_mag.setSuffix(" V")
        pn_mag_layout.addWidget(self.pn_mag)
        level_layout.addLayout(pn_mag_layout)

        # Periodic noise frequency
        pn_freq_layout = QHBoxLayout()
        pn_freq_layout.addWidget(QLabel("f(Pn):"))
        self.pn_freq = QDoubleSpinBox()
        self.pn_freq.setRange(0.0, 1000.0)
        self.pn_freq.setValue(0.0)
        self.pn_freq.setSuffix(" MHz")
        pn_freq_layout.addWidget(self.pn_freq)
        level_layout.addLayout(pn_freq_layout)

        top_layout.addWidget(level_group)
        layout.addLayout(top_layout)

        # Debug group
        debug_group = QGroupBox("Miscellaneous Options")
        debug_layout = QHBoxLayout()
        debug_group.setLayout(debug_layout)

        self.debug = QCheckBox("Debug")
        debug_layout.addWidget(self.debug)

        layout.addWidget(debug_group)

        # Add stretch to push everything to the top
        layout.addStretch()
