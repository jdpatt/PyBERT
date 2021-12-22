"""
Simulation default configuration and configuration loading/saving.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   5 May 2017

This Python script provides a data structure for encapsulating the
simulation configuration data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
configuration could be saved and later restored.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""


class PyBertCfg:
    """PyBERT default and user defined configurations.

    When PyBERT is first started, these are the settings applied to the simulation. When the user
    hits save config, this class handles syncing with the current settings an saving as the
    correct file type.  When loading, this loads this class from a file and applies the changes to
    the simulation.
    """

    # fmt: off

    # - Simulation Control
    bit_rate: float = 10.0  # (Gbps)
    nbits: int = 8000       # number of bits to run
    pattern_len: int = 127  # repeating bit pattern length
    nspb: int = 32          # samples per bit
    num_sweeps: int = 1     # Number of simulations to run.
    sweep_aves: int = 1     # Number of bit error samples to average, when sweeping.

    # - Channel Control
    #     - parameters for Howard Johnson's "Metallic Transmission Model"
    #     - (See "High Speed Signal Propagation", Sec. 3.1.)
    #     - ToDo: These are the values for 24 guage twisted copper pair; need to add other options.
    Rdc: float = 0.1876   # Ohms/m
    w0: float = 10.0e6    # 10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
    R0: float = 1.452     # skin-effect resistance (Ohms/m)log
    Theta0: float = 0.02  # loss tangent
    Z0: float = 100.0     # characteristic impedance in LC region (Ohms)
    v0: float = 0.67      # relative propagation velocity (c)
    l_ch: float = 1.0     # cable length (m)
    rn: float = 0.001     # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)

    # - Tx
    vod: float = 1.0        # output drive strength (Vp)
    rs: float = 100         # differential source impedance (Ohms)
    cout: float = 0.50      # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
    pn_mag: float = 0.001   # magnitude of periodic noise (V)
    pn_freq: float = 0.437  # frequency of periodic noise (MHz)

    # - Rx
    rin: float = 100        # differential input resistance
    cin: float = 0.50       # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
    cac: float = 1.0        # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
    rx_bw: float = 12.0     # Rx signal path bandwidth, assuming no CTLE action. (GHz)
    use_dfe: bool = False   # Include DFE when running simulation.
    sum_ideal: bool = True  # DFE ideal summing node selector

    peak_freq: float = 5.0    # CTLE peaking frequency (GHz)
    peak_mag: float = 10.0    # CTLE peaking magnitude (dB)
    ctle_offset: float = 0.0  # CTLE d.c. offset (dB)

    # - DFE
    decision_scaler: float = 0.5
    n_taps: int = 5
    gain: float = 0.5
    n_ave: int = 100
    sum_bw: float = 12.0  # DFE summing node bandwidth (GHz)

    # - CDR
    delta_t: float = 0.1  # (ps)
    alpha: float = 0.01
    n_lock_ave: int = 500      # number of UI used to average CDR locked status.
    rel_lock_tol: float = 0.1  # relative lock tolerance of CDR.
    lock_sustain: int = 500

    # - Analysis
    thresh: int = 6  # threshold for identifying periodic jitter spectral elements (sigma)
    # fmt: on

    def __init__(self, the_PyBERT, timestamp, version):
        """
        Copy just that subset of the supplied PyBERT instance's
        __dict__, which should be saved during pickling.
        """
        # Generic Information
        self.date_created = timestamp
        self.version = version

        # Simulation Control
        self.bit_rate = the_PyBERT.bit_rate
        self.nbits = the_PyBERT.nbits
        self.pattern_len = the_PyBERT.pattern_len
        self.nspb = the_PyBERT.nspb
        self.eye_bits = the_PyBERT.eye_bits
        self.mod_type = list(the_PyBERT.mod_type)  # See Issue #95 and PR #98
        self.num_sweeps = the_PyBERT.num_sweeps
        self.sweep_num = the_PyBERT.sweep_num
        self.sweep_aves = the_PyBERT.sweep_aves
        self.do_sweep = the_PyBERT.do_sweep
        self.debug = the_PyBERT.debug

        # Channel Control
        self.use_ch_file = the_PyBERT.use_ch_file
        self.ch_file = the_PyBERT.ch_file
        self.impulse_length = the_PyBERT.impulse_length
        self.f_step = the_PyBERT.f_step
        self.Rdc = the_PyBERT.Rdc
        self.w0 = the_PyBERT.w0
        self.R0 = the_PyBERT.R0
        self.Theta0 = the_PyBERT.Theta0
        self.Z0 = the_PyBERT.Z0
        self.v0 = the_PyBERT.v0
        self.l_ch = the_PyBERT.l_ch

        # Tx
        self.vod = the_PyBERT.vod
        self.rs = the_PyBERT.rs
        self.cout = the_PyBERT.cout
        self.pn_mag = the_PyBERT.pn_mag
        self.pn_freq = the_PyBERT.pn_freq
        self.rn = the_PyBERT.rn
        tx_taps = []
        for tap in the_PyBERT.tx_taps:
            tx_taps.append((tap.enabled, tap.value))
        self.tx_taps = tx_taps
        self.tx_tap_tuners = []
        for tap in the_PyBERT.tx_tap_tuners:
            self.tx_tap_tuners.append((tap.enabled, tap.value))
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
        self.ctle_offset = the_PyBERT.ctle_offset
        self.ctle_mode = the_PyBERT.ctle_mode
        self.ctle_mode_tune = the_PyBERT.ctle_mode_tune
        self.ctle_offset_tune = the_PyBERT.ctle_offset_tune
        self.rx_use_ami = the_PyBERT.rx_use_ami
        self.rx_use_ts4 = the_PyBERT.rx_use_ts4
        self.rx_use_getwave = the_PyBERT.rx_use_getwave
        self.rx_ami_file = the_PyBERT.rx_ami_file
        self.rx_dll_file = the_PyBERT.rx_dll_file
        self.rx_ibis_file = the_PyBERT.rx_ibis_file
        self.rx_use_ibis = the_PyBERT.rx_use_ibis

        # DFE
        self.use_dfe = the_PyBERT.use_dfe
        self.use_dfe_tune = the_PyBERT.use_dfe_tune
        self.sum_ideal = the_PyBERT.sum_ideal
        self.decision_scaler = the_PyBERT.decision_scaler
        self.gain = the_PyBERT.gain
        self.n_ave = the_PyBERT.n_ave
        self.n_taps = the_PyBERT.n_taps
        self.sum_bw = the_PyBERT.sum_bw

        # CDR
        self.delta_t = the_PyBERT.delta_t
        self.alpha = the_PyBERT.alpha
        self.n_lock_ave = the_PyBERT.n_lock_ave
        self.rel_lock_tol = the_PyBERT.rel_lock_tol
        self.lock_sustain = the_PyBERT.lock_sustain

        # Analysis
        self.thresh = the_PyBERT.thresh

    def apply(self, pybert):
        """Apply all of the configuration settings to the pybert instance."""
        for prop, value in vars(self).items():
            if prop == "tx_taps":
                for count, (enabled, val) in enumerate(value):
                    setattr(pybert.tx_taps[count], "enabled", enabled)
                    setattr(pybert.tx_taps[count], "value", val)
            elif prop == "tx_tap_tuners":
                for count, (enabled, val) in enumerate(value):
                    setattr(pybert.tx_tap_tuners[count], "enabled", enabled)
                    setattr(pybert.tx_tap_tuners[count], "value", val)
            elif prop in ("version", "timestamp"):
                pass  # Just including it for some good housekeeping.  Not currently used.
            else:
                setattr(pybert, prop, value)
