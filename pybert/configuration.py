"""
Simulation configuration data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   5 May 2017

This Python script provides a data structure for encapsulating the
simulation configuration data of a PyBERT instance. 

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""
from pathlib import Path

import yaml
from PySide2.QtCore import QObject
from PySide2.QtWidgets import QFileDialog


class Configuration(QObject):
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
        self.cfg_file = None
        self.parent = parent

        # Simulation Control
        # self.bit_rate = simulation.bit_rate
        # self.nbits = simulation.nbits
        # self.pattern_len = simulation.pattern_len
        # self.nspb = simulation.nspb
        # self.eye_bits = simulation.eye_bits
        # self.mod_type = simulation.mod_type
        # self.num_sweeps = simulation.num_sweeps
        # self.sweep_num = simulation.sweep_num
        # self.sweep_aves = simulation.sweep_aves
        # self.do_sweep = simulation.do_sweep

        # # Channel Control
        # self.use_ch_file = simulation.channel.use_ch_file
        # self.ch_file = simulation.channel.ch_file
        # self.impulse_length = simulation.channel.impulse_length
        # self.f_step = simulation.channel.f_step
        # self.Rdc = simulation.channel.material.Rdc
        # self.w0 = simulation.channel.material.w0
        # self.R0 = simulation.channel.material.R0
        # self.Theta0 = simulation.channel.material.Theta0
        # self.Z0 = simulation.channel.material.Z0
        # self.v0 = simulation.channel.material.v0
        # self.l_ch = simulation.channel.material.l_ch

        # # Tx
        # self.vod = simulation.tx.vod
        # self.output_impedance = simulation.tx.output_impedance
        # self.output_capacitance = simulation.tx.output_capacitance
        # self.pn_mag = simulation.tx.pn_mag
        # self.pn_freq = simulation.tx.pn_freq
        # self.random_noise = simulation.tx.random_noise
        # self.tx_use_ami = simulation.tx.use_ami
        # self.tx_use_getwave = simulation.tx.use_getwave
        # self.tx_ami_file = simulation.tx.ami_file
        # self.tx_dll_file = simulation.tx.dll_file

        # # Rx
        # self.input_impedance = simulation.rx.input_impedance
        # self.input_capacitance = simulation.rx.input_capacitance
        # self.cac = simulation.rx.cac
        # self.rx_use_ami = simulation.rx.use_ami
        # self.rx_use_getwave = simulation.rx.use_getwave
        # self.rx_ami_file = simulation.rx.ami_file
        # self.rx_dll_file = simulation.rx.dll_file

        # # EQ
        # tx_taps = []
        # for tap in simulation.tx_taps:
        #     tx_taps.append((tap.enabled, tap.value))
        # self.tx_taps = tx_taps
        # self.tx_tap_tuners = []
        # for tap in simulation.tx_tap_tuners:
        #     self.tx_tap_tuners.append((tap.enabled, tap.value))
        # self.use_ctle_file = simulation.use_ctle_file
        # self.ctle_file = simulation.ctle_file
        # self.rx_bw = simulation.rx_bw
        # self.peak_freq = simulation.peak_freq
        # self.peak_mag = simulation.peak_mag
        # self.ctle_offset = simulation.ctle_offset
        # self.ctle_mode = simulation.ctle_mode
        # self.ctle_mode_tune = simulation.ctle_mode_tune
        # self.ctle_offset_tune = simulation.ctle_offset_tune

        # # DFE
        # self.use_dfe = simulation.use_dfe
        # self.use_dfe_tune = simulation.use_dfe_tune
        # self.sum_ideal = simulation.sum_ideal
        # self.decision_scaler = simulation.decision_scaler
        # self.gain = simulation.gain
        # self.n_ave = simulation.n_ave
        # self.n_taps = simulation.n_taps
        # self.sum_bw = simulation.sum_bw

        # # CDR
        # self.delta_t = simulation.delta_t
        # self.alpha = simulation.alpha
        # self.n_lock_ave = simulation.n_lock_ave
        # self.rel_lock_tol = simulation.rel_lock_tol
        # self.lock_sustain = simulation.lock_sustain

        # # Analysis
        # self.thresh = simulation.thresh

    def save_to_file(self, filename):
        """Yaml out the current configuration."""
        if self.cfg_file:
            directory = self.cfg_file
        else:
            directory = ""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent,
            self.parent.tr("Save Configuration"),
            directory,
            self.parent.tr("PyBERT Config Files (*.cfg.yaml)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "w") as config_file:
                yaml.dump(conf, config_file)
            self.cfg_file = filename

    def load_from_file(self, filename):
        """Read in the YAML configuration."""
        if self.cfg_file:
            directory = self.cfg_file
        else:
            directory = ""
        filename, _ = QFileDialog.getOpenFileName(
            self.parent,
            self.parent.tr("Load Configuration"),
            directory,
            self.parent.tr("PyBERT Config Files (*.cfg.yaml)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "r") as in_file:
                config = yaml.full_load(in_file)
            if not isinstance(config, Configuration):
                raise Exception("The data structure read in is NOT of type: Configuration!")
            for prop, value in vars(config).items():
                if prop == "tx_taps":
                    for count, (enabled, val) in enumerate(value):
                        setattr(self.channel.eq.tx_taps[count], "enabled", enabled)
                        setattr(self.channel.eq.tx_taps[count], "value", val)
                elif prop == "tx_tap_tuners":
                    for count, (enabled, val) in enumerate(value):
                        setattr(self.channel.eq.tx_tap_tuners[count], "enabled", enabled)
                        setattr(self.channel.eq.tx_tap_tuners[count], "value", val)
                else:
                    setattr(self, prop, value)
                self.cfg_file = filename
