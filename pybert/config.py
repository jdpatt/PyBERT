"""This file holds all the default settings that get populated in the application.

   The configuration class is used to store all the settings of the simulation for
   saving and to encapsulate it if sweeping simulations.
"""
from pathlib import Path
from typing import Union

import yaml

from PySide2.QtWidgets import QFileDialog  # isort:skip
from pybert.sim.utility import CTLE_MODE, MODULATION, TxTapTuner  # isort:skip

# Enables different log levels and infomation if True.
DEBUG = True
DEBUG_OPTIMIZE = False


class Configuration:
    """
    PyBERT simulation configuration data encapsulation class.

    This class is used to encapsulate that subset of the configuration
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Config." button.
    """

    def __init__(self, parent):
        """
        Copy just that subset of the supplied PyBERT instance's
        __dict__, which should be saved.
        """
        super(Configuration, self).__init__()
        self.cfg_file: Union[Path, None] = None
        self.parent = parent

        # Simulation Control ----------------------------------------------------------
        self.bit_rate = 10  # (Gbps)
        self.nbits: int = 8000  # Number of bits to simulate.
        self.pattern_len: int = 127  # PRBS pattern length.
        self.nspb: int = 32  # Signal vector samples per bit.
        self.eye_bits: int = self.nbits // 5  # Number of bits used to form eye. (Default = last 20%)
        self.mod_type = MODULATION.NRZ  # 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
        self.num_sweeps: int = 1  # Number of sweeps to run.
        self.sweep_aves: int = 1  # Number of bit error samples to average, when sweeping.
        self.run_sweeps: bool = False  # Run sweeps? (Default = False)

        # Channel Control ----------------------------------------------------------
        self.use_ch_file: bool = False  # By Deafult, use the built-in model.
        self.padded: bool = False  # Zero pad imported Touchstone data?
        self.windowed: bool = False  # Apply windowing to the Touchstone data?
        self.ch_file: Union[Path, None] = None
        self.impulse_length: float = 0.0  # Impulse response length. (Determined automatically, when 0.)
        self.f_step: float = 10.0  # Frequency step to use when constructing H(f). (MHz)
        self.material: str = "UTP_24Gauge"

        # Tx ----------------------------------------------------------
        self.vod: float = 1.0  # output drive strength (Vp)
        self.output_impedance = 100  # differential source impedance (Ohms)
        self.output_capacitance: float = (
            0.50
        )  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
        self.pn_mag: float = 0.001  # magnitude of periodic noise (V)
        self.pn_freq: float = 0.437  # frequency of periodic noise (MHz)
        self.random_noise = 0.001  # standard deviation of Gaussian random noise (V)
        self.hpf_corner_coupling = (
            1.0e6
        )  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.
        self.tx_use_ami: bool = False  # By Deafult, use the built-in model.
        self.tx_use_getwave: bool = True
        self.tx_ami_file: Union[Path, None] = None
        self.tx_dll_file: Union[Path, None] = None

        # Rx ----------------------------------------------------------
        self.input_impedance = 100  # differential input resistance
        self.input_capacitance: float = (
            0.50
        )  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
        self.ac_capacitance: float = (
            1.0
        )  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)

        self.rx_use_ami: bool = False  # By Deafult, use the built-in model.
        self.rx_use_getwave: bool = True
        self.rx_ami_file: Union[Path, None] = None
        self.rx_dll_file: Union[Path, None] = None

        # EQ ----------------------------------------------------------
        self.max_iter = 50  # EQ optimizer max. # of optimization iterations.
        self.tx_taps = [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]  # List of TxTapTuner objects.
        self.tx_tap_tuners = [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
        self.use_ctle_file: bool = False  # By Deafult, use the built-in model.
        self.ctle_file: Union[Path, None] = None
        self.rx_bw: float = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
        self.peak_freq: float = 5.0  # CTLE peaking frequency (GHz)
        self.peak_mag: float = 10.0  # CTLE peaking magnitude (dB)
        self.ctle_mode = (
            CTLE_MODE.OFF
        )  #: EQ optimizer CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
        self.ctle_offset: float = 0.0  # CTLE d.c. offset (dB)
        self.ctle_mode_tune = CTLE_MODE.OFF
        self.ctle_offset_tune: float = 0.0  # CTLE d.c. offset (dB)

        self.max_ctle_peak: float = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
        self.max_ctle_freq: float = 20.0  # max. allowed CTLE peak frequency (GHz) (when optimizing, only)

        # DFE ----------------------------------------------------------
        self.use_dfe: bool = True  # Include DFE when running simulation.
        self.use_dfe_tune: bool = True  # Use the EQ Tuned DFE
        self.sum_ideal: bool = True  # Use an ideal (i.e. - infinite bandwidth) summing node.
        self.decision_scaler: float = 0.5
        self.gain: float = 0.5
        self.n_ave: int = 100
        self.n_taps: int = 5  # Number of DFE taps
        self.sum_bw = 12.0  # DFE summing node bandwidth when not using ideal DFE (GHz)
        self.sum_num_taps: int = 3  # Number of taps used in summing node filter.

        # CDR ----------------------------------------------------------
        self.delta_t: float = 0.1  # Proportional branch magnitude (ps).
        self.alpha: float = 0.01  # Integral branch magnitude (unitless).
        self.n_lock_ave: int = 500  # Number of averages to take in determining lock.
        self.rel_lock_tol: float = 0.1  # Relative tolerance to use in determining lock.
        self.lock_sustain: int = 500  # Hysteresis to use in determining lock.

        # Analysis ----------------------------------------------------------
        self.thresh: int = 6  # threshold for identifying periodic jitter spectral elements (sigma)
        self.min_bathtub_val = 1.0e-18

    def save_to_file(self):
        """Yaml out the current configuration."""
        if self.cfg_file:
            directory = self.cfg_file
        else:
            directory = ""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent,
            self.parent.tr("Save Configuration"),
            directory,
            self.parent.tr("PyBERT Config Files (*.pybert_cfg)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "w") as config_file:
                yaml.dump(self, config_file)
            self.cfg_file = filename

    def load_from_file(self):
        """Read in the YAML configuration."""
        if self.cfg_file:
            directory = self.cfg_file
        else:
            directory = ""
        filename, _ = QFileDialog.getOpenFileName(
            self.parent,
            self.parent.tr("Load Configuration"),
            directory,
            self.parent.tr("PyBERT Config Files (*.pybert_cfg)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "r") as in_file:
                config = yaml.full_load(in_file)
            if not isinstance(config, Configuration):
                raise Exception("The data structure is NOT a PyBERT Configuration!")
                self.cfg_file = filename
                return config
