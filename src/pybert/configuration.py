"""Simulation configuration data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   5 May 2017

This Python script provides a data structure for encapsulating the
simulation configuration data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
configuration could be saved and later restored.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""

import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from pybert import __version__
from pybert.constants import gPeakFreq, gPeakMag
from pybert.models.stimulus import BitPattern, ModulationType

if TYPE_CHECKING:
    from pybert.pybert import PyBERT

import yaml

from pybert.constants import gPeakFreq, gPeakMag

# TODO: Add a v7 to v8 migration method.


class InvalidConfigFileType(Exception):
    """Raised when a filetype that isn't supported is used when trying to load
    or save files.."""


# These are different for now to allow users to "upgrade" their configuration file.

CONFIG_LOAD_WILDCARD = ";;".join(
    [
        "Yaml Config (*.yaml;*.yml)|*.yaml;*.yml",
        "All files (*)|*",
    ]
)
"""This sets the supported file types in the GUI's loading dialog."""

CONFIG_SAVE_WILDCARD = ";;".join(
    [
        "Yaml Config (*.yaml;*.yml)|*.yaml;*.yml",
        "All files (*)|*",
    ]
)
"""This sets the supported file types in the GUI's save-as dialog."""


class Configuration:  # pylint: disable=too-many-instance-attributes
    """PyBERT simulation configuration data encapsulation class.

    This class is used to encapsulate that subset of the configuration
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Config." button.
    """

    def __init__(
        self, the_PyBERT: "PyBERT", date_created: str = time.asctime(), version: str = __version__
    ):  # pylint: disable=too-many-statements
        """Copy just that subset of the supplied PyBERT instance's __dict__,
        which should be saved."""

        # Generic Information
        self.date_created = date_created
        self.version = version

        # Simulation Control
        self.bit_rate = the_PyBERT.bit_rate
        self.nbits = the_PyBERT.nbits
        self.pattern = the_PyBERT.pattern.name
        self.seed = the_PyBERT.seed
        self.nspui = the_PyBERT.nspui
        self.eye_bits = the_PyBERT.eye_bits
        self.mod_type = the_PyBERT.mod_type.value
        self.f_max = the_PyBERT.f_max
        self.f_step = the_PyBERT.f_step

        # Channel Control
        self.use_ch_file = the_PyBERT.use_ch_file
        self.ch_file = the_PyBERT.ch_file
        self.impulse_length = the_PyBERT.impulse_length
        self.Rdc = the_PyBERT.Rdc
        self.w0 = the_PyBERT.w0
        self.R0 = the_PyBERT.R0
        self.Theta0 = the_PyBERT.Theta0
        self.Z0 = the_PyBERT.Z0
        self.v0 = the_PyBERT.v0
        self.l_ch = the_PyBERT.l_ch
        self.renumber = the_PyBERT.renumber
        self.use_window = the_PyBERT.use_window

        # Tx
        self.vod = the_PyBERT.vod
        self.rs = the_PyBERT.rs
        self.cout = the_PyBERT.cout
        self.pn_mag = the_PyBERT.pn_mag
        self.pn_freq = the_PyBERT.pn_freq
        self.rn = the_PyBERT.rn
        tx_taps = []
        for tap in the_PyBERT.tx_taps:
            tx_taps.append((tap.enabled, tap.value, tap.min_val, tap.max_val))
        self.tx_taps = tx_taps
        self.tx_tap_tuners = []
        for tap in the_PyBERT.tx_tap_tuners:
            self.tx_tap_tuners.append((tap.enabled, tap.pos, tap.min_val, tap.max_val, tap.step))
        self.tx_use_ami = the_PyBERT.tx_use_ami
        self.tx_use_ts4 = the_PyBERT.tx_use_ts4
        self.tx_use_getwave = the_PyBERT.tx_use_getwave
        self.tx_ami_file = the_PyBERT.tx_ami_file
        self.tx_dll_file = the_PyBERT.tx_dll_file
        self.tx_ibis_file = the_PyBERT.tx_ibis_file
        self.tx_use_ibis = the_PyBERT.tx_use_ibis

        # Rx
        self.rin = the_PyBERT.rin
        self.cin = the_PyBERT.cin
        self.cac = the_PyBERT.cac
        self.use_ctle_file = the_PyBERT.use_ctle_file
        self.ctle_file = the_PyBERT.ctle_file
        self.rx_bw = the_PyBERT.rx_bw
        self.peak_freq = the_PyBERT.peak_freq
        self.peak_mag = the_PyBERT.peak_mag
        self.ctle_enable = the_PyBERT.ctle_enable
        self.rx_use_ami = the_PyBERT.rx_use_ami
        self.rx_use_ts4 = the_PyBERT.rx_use_ts4
        self.rx_use_getwave = the_PyBERT.rx_use_getwave
        self.rx_ami_file = the_PyBERT.rx_ami_file
        self.rx_dll_file = the_PyBERT.rx_dll_file
        self.rx_ibis_file = the_PyBERT.rx_ibis_file
        self.rx_use_ibis = the_PyBERT.rx_use_ibis

        # DFE
        self.sum_ideal = the_PyBERT.sum_ideal
        self.decision_scaler = the_PyBERT.decision_scaler
        self.gain = the_PyBERT.gain
        self.n_ave = the_PyBERT.n_ave
        self.sum_bw = the_PyBERT.sum_bw

        # CDR
        self.delta_t = the_PyBERT.delta_t
        self.alpha = the_PyBERT.alpha
        self.n_lock_ave = the_PyBERT.n_lock_ave
        self.rel_lock_tol = the_PyBERT.rel_lock_tol
        self.lock_sustain = the_PyBERT.lock_sustain

        # Analysis
        self.thresh = the_PyBERT.thresh

        # Optimization
        self.rx_bw_tune = the_PyBERT.rx_bw_tune
        self.peak_freq_tune = the_PyBERT.peak_freq_tune
        self.peak_mag_tune = the_PyBERT.peak_mag_tune
        self.min_mag_tune = the_PyBERT.min_mag_tune
        self.max_mag_tune = the_PyBERT.max_mag_tune
        self.step_mag_tune = the_PyBERT.step_mag_tune
        self.ctle_enable_tune = the_PyBERT.ctle_enable_tune
        self.dfe_tap_tuners = []
        for tap in the_PyBERT.dfe_tap_tuners:
            self.dfe_tap_tuners.append((tap.enabled, tap.min_val, tap.max_val))

    @classmethod
    def create_default_config(cls):
        """Create a PyBertCfg instance with default values."""
        config = cls.__new__(cls)

        # Generic Information
        config.date_created = datetime.now().strftime("%c")
        config.version = __version__
        # Simulation Control
        config.bit_rate = 10.0  # Gbps
        config.nbits = 15000
        config.eye_bits = 10160
        config.pattern = BitPattern.PRBS7.name
        config.seed = 1
        config.nspui = 32
        config.mod_type = ModulationType.NRZ.value
        config.f_max = 40.0  # GHz
        config.f_step = 10.0  # MHz
        config.thresh = 3.0
        # Channel Control
        config.use_ch_file = False
        config.ch_file = ""
        config.impulse_length = 0.0
        config.Rdc = 0.1876  # Ohms/m
        config.w0 = 10e6  # rads/s
        config.R0 = 1.452  # Ohms/m
        config.Theta0 = 0.02
        config.Z0 = 100.0  # Ohms
        config.v0 = 0.67  # c
        config.l_ch = 0.5  # m
        config.renumber = False
        config.use_window = False
        # Tx
        config.vod = 1.0  # V
        config.rs = 100  # Ohms
        config.cout = 0.5  # pF
        config.pn_mag = 0.1  # V
        config.pn_freq = 11.0  # MHz
        config.rn = 0.1  # V
        config.tx_taps = [
            (True, 0.0, -0.05, 0.05),  # Pre-tap3
            (True, 0.0, -0.1, 0.1),  # Pre-tap2
            (True, 0.0, -0.2, 0.2),  # Pre-tap1
            (True, 0.0, -0.2, 0.2),  # Post-tap1
            (True, 0.0, -0.1, 0.1),  # Post-tap2
            (True, 0.0, -0.05, 0.05),  # Post-tap3
        ]
        config.tx_tap_tuners = [
            (True, -3, -0.05, 0.05, 0.025),
            (True, -2, -0.1, 0.1, 0.05),
            (True, -1, -0.2, 0.2, 0.1),
            (True, 1, -0.2, 0.2, 0.1),
            (True, 2, -0.1, 0.1, 0.05),
            (True, 3, -0.05, 0.05, 0.025),
        ]
        config.tx_use_ami = False
        config.tx_use_ts4 = False
        config.tx_use_getwave = False
        config.tx_ami_file = ""
        config.tx_dll_file = ""
        config.tx_ibis_file = ""
        config.tx_use_ibis = False
        # Rx
        config.rin = 100  # Ohms
        config.cin = 0.5  # pF
        config.cac = 1.0  # uF
        config.use_ctle_file = False
        config.ctle_file = ""
        config.rx_bw = 12.0  # GHz
        config.peak_freq = gPeakFreq  # GHz
        config.peak_mag = gPeakMag  # dB
        config.ctle_enable = True
        config.rx_use_ami = False
        config.rx_use_ts4 = False
        config.rx_use_getwave = False
        config.rx_ami_file = ""
        config.rx_dll_file = ""
        config.rx_ibis_file = ""
        config.rx_use_ibis = False
        # DFE
        config.sum_ideal = True
        config.decision_scaler = 0.5  # V
        config.gain = 0.2
        config.n_ave = 100
        config.sum_bw = 12.0  # GHz
        # CDR
        config.delta_t = 0.1  # ps
        config.alpha = 0.01
        config.n_lock_ave = 500
        config.rel_lock_tol = 0.1
        config.lock_sustain = 500
        # Optimization
        config.rx_bw_tune = 12.0  # GHz
        config.peak_freq_tune = gPeakFreq  # GHz
        config.peak_mag_tune = gPeakMag  # dB
        config.min_mag_tune = 2.0  # dB
        config.max_mag_tune = 12.0  # dB
        config.step_mag_tune = 1.0  # dB
        config.ctle_enable_tune = True
        config.dfe_tap_tuners = [
            (True, 0.1, 0.4),  # Tap1
            (True, -0.15, 0.15),  # Tap2
            (True, -0.05, 0.1),  # Tap3
            (True, -0.05, 0.1),  # Tap4
            (True, -0.05, 0.1),  # Tap5
            (False, -0.05, 0.1),  # Tap6
            (False, -0.05, 0.1),  # Tap7
            (False, -0.05, 0.1),  # Tap8
            (False, -0.05, 0.1),  # Tap9
            (False, -0.05, 0.1),  # Tap10
            (False, -0.05, 0.1),  # Tap11
            (False, -0.05, 0.1),  # Tap12
            (False, -0.05, 0.1),  # Tap13
            (False, -0.05, 0.1),  # Tap14
            (False, -0.05, 0.1),  # Tap15
            (False, -0.05, 0.1),  # Tap16
            (False, -0.05, 0.1),  # Tap17
            (False, -0.05, 0.1),  # Tap18
            (False, -0.05, 0.1),  # Tap19
            (False, -0.05, 0.1),  # Tap20
        ]
        return config

    @staticmethod
    def apply_default_config(pybert: "PyBERT") -> None:
        """Apply default configuration to a PyBERT instance.

        Args:
            pybert: The PyBERT instance to configure with defaults
        """
        default_config = Configuration.create_default_config()
        Configuration.load_from_config(default_config, pybert)

    @staticmethod
    def load_from_config(config: "Configuration", pybert: "PyBERT") -> None:
        """Apply configuration from a PyBertCfg instance to a PyBERT instance.

        This is similar to load_from_file but works with a config object instead
        of loading from a file.

        Args:
            config: The configuration object to apply
            pybert: The PyBERT instance to configure
        """
        # Reset equalization mode flags first
        pybert.tx_use_ibis = False
        pybert.rx_use_ibis = False
        pybert.tx_use_ami = False
        pybert.rx_use_ami = False
        pybert.tx_ibis_valid = False
        pybert.rx_ibis_valid = False
        pybert.tx_ami_valid = False
        pybert.rx_ami_valid = False
        pybert.tx_dll_valid = False
        pybert.rx_dll_valid = False

        # Apply values back into pybert using `setattr`.
        for prop, value in vars(config).items():
            if prop == "tx_taps":
                for count, (enabled, val, min_val, max_val) in enumerate(value):
                    setattr(pybert.tx_taps[count], "enabled", enabled)
                    setattr(pybert.tx_taps[count], "value", val)
                    setattr(pybert.tx_taps[count], "min_val", min_val)
                    setattr(pybert.tx_taps[count], "max_val", max_val)
            elif prop == "tx_tap_tuners":
                for count, (enabled, pos, min_val, max_val, step) in enumerate(value):
                    setattr(pybert.tx_tap_tuners[count], "enabled", enabled)
                    setattr(pybert.tx_tap_tuners[count], "pos", pos)
                    setattr(pybert.tx_tap_tuners[count], "min_val", min_val)
                    setattr(pybert.tx_tap_tuners[count], "max_val", max_val)
                    setattr(pybert.tx_tap_tuners[count], "step", step)
            elif prop == "dfe_tap_tuners":
                for count, (enabled, min_val, max_val) in enumerate(value):
                    setattr(pybert.dfe_tap_tuners[count], "enabled", enabled)
                    setattr(pybert.dfe_tap_tuners[count], "min_val", min_val)
                    setattr(pybert.dfe_tap_tuners[count], "max_val", max_val)
            elif prop in ("version", "date_created"):
                pass  # Just including it for some good housekeeping.  Not currently used.
            elif prop == "mod_type":
                setattr(pybert, prop, ModulationType(value))
            elif prop == "pattern":
                setattr(pybert, prop, BitPattern[value])
            elif prop == "tx_ibis_file":
                setattr(pybert, prop, value)
                if value:
                    # Load IBIS file
                    ibis = pybert.load_new_tx_ibis_file(value)
                    if ibis:
                        # If we have component/pin/model in the config, set them
                        if hasattr(config, "tx_component") and config.tx_component:
                            ibis.current_component = config.tx_component
                        if hasattr(config, "tx_pin") and config.tx_pin:
                            ibis.current_pin = config.tx_pin
                        if hasattr(config, "tx_model") and config.tx_model:
                            ibis.current_model = config.tx_model
                        setattr(pybert, "tx_ibis", ibis)
            elif prop == "rx_ibis_file":
                setattr(pybert, prop, value)
                if value:
                    # Load IBIS file
                    ibis = pybert.load_new_rx_ibis_file(value)
                    if ibis:
                        # If we have component/pin/model in the config, set them
                        if hasattr(config, "rx_component") and config.rx_component:
                            ibis.current_component = config.rx_component
                        if hasattr(config, "rx_pin") and config.rx_pin:
                            ibis.current_pin = config.rx_pin
                        if hasattr(config, "rx_model") and config.rx_model:
                            ibis.current_model = config.rx_model
                        setattr(pybert, "rx_ibis", ibis)
            else:
                setattr(pybert, prop, value)

    @staticmethod
    def load_from_file(filepath: str | Path, pybert):  # pylint: disable=too-many-branches
        """Apply all of the configuration settings to the pybert instance.

        Confirms that the file actually exists, is the correct extension and
        attempts to set the values back in pybert.

        Args:
            filepath: The full filepath including the extension to save too.
            pybert: instance of the main app
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        # If its a valid extension load it.
        if filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, "r", encoding="UTF-8") as yaml_file:
                user_config = yaml.load(yaml_file, Loader=yaml.Loader)
        elif filepath.suffix == ".pybert_cfg":
            warnings.warn(
                "Using pickle for configuration is not suggested and will be removed in a later release.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(filepath, "rb") as pickle_file:
                user_config = pickle.load(pickle_file)
        else:
            raise InvalidConfigFileType("Pybert does not support this file type.")

        # Right now the loads deserialize back into a `PyBertCfg` class.
        if not isinstance(user_config, Configuration):
            raise ValueError("The data structure read in is NOT of type: PyBertCfg!")

        Configuration.load_from_config(user_config, pybert)

    def save(self, filepath: str | Path):
        """Save out pybert's current configuration to a file.

        The extension must match a yaml file extension or it will still raise
        an invalid file type.  Additional filetypes can be added/supported by
        just adding another if statement and adding to `CONFIG_FILEDIALOG_WILDCARD`.

        Args:
            filepath: The full filepath including the extension to save too.
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, "w", encoding="UTF-8") as yaml_file:
                yaml.dump(self, yaml_file, indent=4, sort_keys=False)
        else:
            raise InvalidConfigFileType("Pybert does not support this file type.")
