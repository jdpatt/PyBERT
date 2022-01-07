#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
import itertools
import logging
import platform
import time
from os.path import join
from pathlib import Path
from typing import Optional

import numpy as np
import skrf as rf
from chaco.api import ArrayPlotData, GridPlotContainer, Plot
from numpy import array, convolve, cos, exp, ones, pad, pi, resize, sinc, where, zeros
from numpy.fft import fft, irfft
from numpy.random import randint
from traits.api import (
    Array,
    Bool,
    Button,
    Enum,
    File,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Property,
    Range,
    String,
    cached_property,
)
from traits.etsconfig.api import ETSConfig

from pybert import __authors__, __copy__, __date__, __version__, plot
from pybert.configuration import PyBertCfg
from pybert.content import (
    help_str,
    jitter_info_table,
    performance_info_table,
    status_string,
    sweep_info_table,
)
from pybert.control import my_run_simulation
from pybert.logger import ConsoleTextLogHandler
from pybert.results import PyBertData
from pybert.threads import CoOptThread, RxOptThread, TxOptThread
from pybert.utility import (
    calc_gamma,
    import_channel,
    interp_s2p,
    interp_time,
    lfsr_bits,
    make_ctle,
    pulse_center,
    safe_log10,
    sdd_21,
    trim_impulse,
)
from pyibisami import __version__ as PyAMI_VERSION
from pyibisami.ami import AMIModel, AMIParamConfigurator
from pyibisami.ibis import IBISModel

# fmt: off
# ETSConfig.toolkit = 'qt.celiagg'  # Yields unacceptably small font sizes in plot axis labels.
# ETSConfig.toolkit = 'qt.qpainter'  # Was causing crash on Mac.
# fmt: on


gDebugStatus = False
gDebugOptimize = False
gMaxCTLEPeak = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
gMaxCTLEFreq = 20.0  # max. allowed CTLE peak frequency (GHz) (when optimizing, only)


class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name = String("(noname)")
    enabled = Bool(False)
    min_val = Float(0.0)
    max_val = Float(0.0)
    value = Float(0.0)
    steps = Int(0)  # Non-zero means we want to sweep it.

    def __init__(self, name="(noname)", enabled=False, min_val=0.0, max_val=0.0, value=0.0, steps=0):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.name: str = name
        self.enabled: bool = enabled
        self.min_val: float = min_val
        self.max_val: float = max_val
        self.value: float = value
        self.steps: int = steps

    def sweep_values(self):
        """Return what values should be interated through if sweeping in the main simulation.

        If its enabled either create a list of equally spaced steps or just append the value if
        steps is zero.  If the tap isn't enabled period, append zero.
        """
        values = [0.0]
        if self.enabled:
            if self.steps:
                values = list(np.arange(self.min_val, self.max_val, self.steps))
                values.append(self.max_val)  # We want to the max value to be inclusive.
            else:
                values = [self.value]
        return values


class PyBERT(HasTraits):
    """
    A serial communication link bit error rate tester (BERT) simulator with a GUI interface.

    Useful for exploring the concepts of serial communication link design.
    """

    # Independent variables

    # - Simulation Control
    bit_rate = Range(low=0.1, high=120.0, value=PyBertCfg.bit_rate)  #: (Gbps)
    nbits = Range(low=1000, high=10000000, value=PyBertCfg.nbits)  #: Number of bits to simulate.
    pattern_len = Range(low=7, high=10000000, value=PyBertCfg.pattern_len)  #: PRBS pattern length.
    nspb = Range(low=2, high=256, value=PyBertCfg.nspb)  #: Signal vector samples per bit.
    eye_bits = Int(PyBertCfg.nbits // 5)  #: # of bits used to form eye. (Default = last 20%)
    mod_type = List([0])  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
    num_sweeps = Int(PyBertCfg.num_sweeps)  #: Number of sweeps to run.
    sweep_num = Int(1)
    sweep_aves = Int(PyBertCfg.sweep_aves)
    sweep_sim = Bool(False)  #: Run sweeps? (Default = False)
    debug = Bool(False)  #: Send log messages to terminal, as well as console, when True. (Default = False)

    # - Channel Control
    # fmt: off
    ch_file = File(
        "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
    )                          #: Channel file name.
    use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
    # padded = Bool(False)       #: Zero pad imported Touchstone data? (Default = False)
    # windowed = Bool(False)     #: Apply windowing to the Touchstone data? (Default = False)
    f_step = Float(10)         #: Frequency step to use when constructing H(f). (Default = 10 MHz)
    impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
    Rdc = Float(PyBertCfg.Rdc)            #: Channel d.c. resistance (Ohms/m).
    w0 = Float(PyBertCfg.w0)              #: Channel transition frequency (rads./s).
    R0 = Float(PyBertCfg.R0)              #: Channel skin effect resistance (Ohms/m).
    Theta0 = Float(PyBertCfg.Theta0)      #: Channel loss tangent (unitless).
    Z0 = Float(PyBertCfg.Z0)              #: Channel characteristic impedance, in LC region (Ohms).
    v0 = Float(PyBertCfg.v0)              #: Channel relative propagation velocity (c).
    l_ch = Float(PyBertCfg.l_ch)          #: Channel length (m).
    # fmt: on

    # - EQ Tune
    tx_tap_tuners = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )  #: EQ optimizer list of TxTapTuner objects.
    rx_bw_tune = Float(PyBertCfg.rx_bw)  #: EQ optimizer CTLE bandwidth (GHz).
    peak_freq_tune = Float(PyBertCfg.peak_freq)  #: EQ optimizer CTLE peaking freq. (GHz).
    peak_mag_tune = Float(PyBertCfg.peak_mag)  #: EQ optimizer CTLE peaking mag. (dB).
    ctle_offset_tune = Float(PyBertCfg.ctle_offset)  #: EQ optimizer CTLE d.c. offset (dB).
    ctle_mode_tune = Enum(
        "Off", "Passive", "AGC", "Manual"
    )  #: EQ optimizer CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    use_dfe_tune = Bool(PyBertCfg.use_dfe)  #: EQ optimizer DFE select (Bool).
    n_taps_tune = Int(PyBertCfg.n_taps)  #: EQ optimizer # DFE taps.
    max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.
    tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
    rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
    coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.

    # - Tx
    vod = Float(PyBertCfg.vod)  #: Tx differential output voltage (V)
    rs = Float(PyBertCfg.rs)  #: Tx source impedance (Ohms)
    cout = Range(low=0.001, high=1000, value=PyBertCfg.cout)  #: Tx parasitic output capacitance (pF)
    pn_freq = Float(PyBertCfg.pn_freq)  #: Periodic noise frequency (MHz).
    pn_mag = Float(PyBertCfg.pn_mag)  #: Periodic noise magnitude (V).
    rn = Float(PyBertCfg.rn)  #: Standard deviation of Gaussian random noise (V).
    tx_taps = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )  #: List of TxTapTuner objects.
    rel_power = Float(1.0)  #: Tx power dissipation (W).
    tx_use_ami = Bool(False)  #: (Bool)
    tx_has_ts4 = Bool(False)  #: (Bool)
    tx_use_ts4 = Bool(False)  #: (Bool)
    tx_use_getwave = Bool(False)  #: (Bool)
    tx_has_getwave = Bool(False)  #: (Bool)
    tx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    tx_ami_valid = Bool(False)  #: (Bool)
    tx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    tx_dll_valid = Bool(False)  #: (Bool)
    tx_ibis_file = File(
        "",
        entries=5,
        filter=[
            "IBIS Models (*.ibs)|*.ibs",
        ],
    )  #: (File)
    tx_ibis_valid = Bool(False)  #: (Bool)
    tx_use_ibis = Bool(False)  #: (Bool)

    # - Rx
    rin = Float(PyBertCfg.rin)  #: Rx input impedance (Ohm)
    cin = Range(low=0.0, high=1000, value=PyBertCfg.cin)  #: Rx parasitic input capacitance (pF)
    cac = Float(PyBertCfg.cac)  #: Rx a.c. coupling capacitance (uF)
    use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
    ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
    rx_bw = Float(PyBertCfg.rx_bw)  #: CTLE bandwidth (GHz).
    peak_freq = Float(PyBertCfg.peak_freq)  #: CTLE peaking frequency (GHz)
    peak_mag = Float(PyBertCfg.peak_mag)  #: CTLE peaking magnitude (dB)
    ctle_offset = Float(PyBertCfg.ctle_offset)  #: CTLE d.c. offset (dB)
    ctle_mode = Enum("Off", "Passive", "AGC", "Manual")  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    rx_use_ami = Bool(False)  #: (Bool)
    rx_has_ts4 = Bool(False)  #: (Bool)
    rx_use_ts4 = Bool(False)  #: (Bool)
    rx_use_getwave = Bool(False)  #: (Bool)
    rx_has_getwave = Bool(False)  #: (Bool)
    rx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    rx_ami_valid = Bool(False)  #: (Bool)
    rx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    rx_dll_valid = Bool(False)  #: (Bool)
    rx_ibis_file = File("", entries=5, filter=["*.ibs"])  #: (File)
    rx_ibis_valid = Bool(False)  #: (Bool)
    rx_use_ibis = Bool(False)  #: (Bool)

    # - DFE
    use_dfe = Bool(PyBertCfg.use_dfe)  #: True = use a DFE (Bool).
    sum_ideal = Bool(PyBertCfg.sum_ideal)  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
    decision_scaler = Float(PyBertCfg.decision_scaler)  #: DFE slicer output voltage (V).
    gain = Float(PyBertCfg.gain)  #: DFE error gain (unitless).
    n_ave = Float(PyBertCfg.n_ave)  #: DFE # of averages to take, before making tap corrections.
    n_taps = Int(PyBertCfg.n_taps)  #: DFE # of taps.
    _old_n_taps = n_taps
    sum_bw = Float(PyBertCfg.sum_bw)  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

    # - CDR
    delta_t = Float(PyBertCfg.delta_t)  #: CDR proportional branch magnitude (ps).
    alpha = Float(PyBertCfg.alpha)  #: CDR integral branch magnitude (unitless).
    n_lock_ave = Int(PyBertCfg.n_lock_ave)  #: CDR # of averages to take in determining lock.
    rel_lock_tol = Float(PyBertCfg.rel_lock_tol)  #: CDR relative tolerance to use in determining lock.
    lock_sustain = Int(PyBertCfg.lock_sustain)  #: CDR hysteresis to use in determining lock.

    # - Analysis
    thresh = Int(PyBertCfg.thresh)  #: Threshold for identifying periodic jitter components (sigma).

    # Misc.
    cfg_file = File("", entries=5, filter=["*.pybert_cfg"])  #: PyBERT configuration data storage file (File).
    data_file = File("", entries=5, filter=["*.pybert_data"])  #: PyBERT results data storage file (File).

    # Plots (plot containers, actually)
    plotdata = ArrayPlotData()
    plots_h = Instance(GridPlotContainer)
    plots_s = Instance(GridPlotContainer)
    plots_p = Instance(GridPlotContainer)
    plots_H = Instance(GridPlotContainer)
    plots_dfe = Instance(GridPlotContainer)
    plot_dfe_adapt = Instance(Plot)
    plots_eye = Instance(GridPlotContainer)
    plots_jitter_dist = Instance(GridPlotContainer)
    plots_jitter_spec = Instance(GridPlotContainer)
    plots_bathtub = Instance(GridPlotContainer)

    # Status
    status = String("Ready.")  #: PyBERT status (String).
    jitter_perf = Float(0.0)
    total_perf = Float(0.0)
    sweep_results = List([])
    len_h = Int(0)
    chnl_dly = Float(0.0)  #: Estimated channel delay (s).
    bit_errs = Int(0)  #: # of bit errors observed in last run.
    run_count = Int(0)  # Used as a mechanism to force bit stream regeneration.

    # About
    perf_info = Property(String, depends_on=["total_perf"])
    ident = String(
        f"<H1>PyBERT v{__version__} - a serial communication link design tool, written in Python.</H1>\n\n \
    {__authors__}<BR>\n \
    {__date__}<BR><BR>\n\n \
    {__copy__};<BR>\n \
    All rights reserved World wide."
    )

    # Help
    instructions = help_str

    # Console
    console_log = String("PyBERT Console Log\n\n")

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
    #
    # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
    #         This will slow the GUI down noticeably.
    jitter_info = Property(String, depends_on=["jitter_perf"])
    status_str = Property(String, depends_on=["status"])
    sweep_info = Property(String, depends_on=["sweep_results"])
    tx_h_tune = Property(Array, depends_on=["tx_tap_tuners.value", "nspui"])
    ctle_h_tune = Property(
        Array,
        depends_on=[
            "peak_freq_tune",
            "peak_mag_tune",
            "rx_bw_tune",
            "w",
            "len_h",
            "ctle_mode_tune",
            "ctle_offset_tune",
            "use_dfe_tune",
            "n_taps_tune",
        ],
    )
    ctle_out_h_tune = Property(Array, depends_on=["tx_h_tune", "ctle_h_tune", "chnl_h"])
    cost = Property(Float, depends_on=["ctle_out_h_tune", "nspui"])
    rel_opt = Property(Float, depends_on=["cost"])
    t = Property(Array, depends_on=["ui", "nspb", "nbits"])
    t_ns = Property(Array, depends_on=["t"])
    f = Property(Array, depends_on=["t"])
    w = Property(Array, depends_on=["f"])
    bits = Property(Array, depends_on=["pattern_len", "nbits", "run_count"])
    symbols = Property(Array, depends_on=["bits", "mod_type", "vod"])
    ffe = Property(Array, depends_on=["tx_taps.value", "tx_taps.enabled"])
    ui = Property(Float, depends_on=["bit_rate", "mod_type"])
    nui = Property(Int, depends_on=["nbits", "mod_type"])
    nspui = Property(Int, depends_on=["nspb", "mod_type"])
    eye_uis = Property(Int, depends_on=["eye_bits", "mod_type"])
    dfe_out_p = Array()
    przf_err = Property(Float, depends_on=["dfe_out_p"])

    # Custom buttons, which we'll use in particular tabs.
    # (Globally applicable buttons, such as "Run" and "Ok", are handled more simply, in the View.)
    btn_rst_eq = Button(label="ResetEq")
    btn_save_eq = Button(label="SaveEq")
    btn_opt_tx = Button(label="OptTx")
    btn_opt_rx = Button(label="OptRx")
    btn_coopt = Button(label="CoOpt")
    btn_abort = Button(label="Abort")
    btn_cfg_tx = Button(label="Configure")  # Configure AMI parameters.
    btn_cfg_rx = Button(label="Configure")
    btn_sel_tx = Button(label="Select")  # Select IBIS model.
    btn_sel_rx = Button(label="Select")
    btn_view_tx = Button(label="View")  # View IBIS model.
    btn_view_rx = Button(label="View")

    # Default initialization
    def __init__(self, run_simulation: bool = True, gui: bool = True):
        """
        Initial plot setup occurs here.

        In order to populate the data structure we need to
        construct the plots, we must run the simulation.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
            gui(Bool): Set to `False` for script based usage.
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.has_gui = gui

        self._console_log_handler = ConsoleTextLogHandler(self)
        logging.getLogger().addHandler(self._console_log_handler)

        self._log = logging.getLogger("pybert")
        self.log_system_information()
        self._log.debug("Debug Mode: %s", str(self.debug))

        self._tx_ibis: Optional[IBISModel] = None
        self._tx_cfg: Optional[AMIParamConfigurator] = None
        self._tx_model: Optional[AMIModel] = None
        self._rx_ibis: Optional[IBISModel] = None
        self._rx_cfg: Optional[AMIParamConfigurator] = None
        self._rx_model: Optional[AMIModel] = None

        if run_simulation:
            # Running the simulation will fill in the required data structure.
            my_run_simulation(self, initial_run=True)
            # Once the required data structure is filled in, we can create the plots.
            self.initialize_plots()
        else:
            self.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().

    # Custom button handlers
    def _btn_rst_eq_fired(self):
        """Reset the equalization."""
        for i in range(4):
            self.tx_tap_tuners[i].value = self.tx_taps[i].value
            self.tx_tap_tuners[i].enabled = self.tx_taps[i].enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_mode_tune = self.ctle_mode
        self.ctle_offset_tune = self.ctle_offset
        self.use_dfe_tune = self.use_dfe
        self.n_taps_tune = self.n_taps

    def _btn_save_eq_fired(self):
        """Save the equalization."""
        for i in range(4):
            self.tx_taps[i].value = self.tx_tap_tuners[i].value
            self.tx_taps[i].enabled = self.tx_tap_tuners[i].enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_mode = self.ctle_mode_tune
        self.ctle_offset = self.ctle_offset_tune
        self.use_dfe = self.use_dfe_tune
        self.n_taps = self.n_taps_tune

    def _btn_opt_tx_fired(self):
        if (
            self.tx_opt_thread
            and self.tx_opt_thread.is_alive()
            or not any(self.tx_tap_tuners[i].enabled for i in range(len(self.tx_tap_tuners)))
        ):
            pass
        else:
            self._do_opt_tx()

    def _do_opt_tx(self, update_status=True):
        self.tx_opt_thread = TxOptThread()
        self.tx_opt_thread.pybert = self
        self.tx_opt_thread.update_status = update_status
        self.tx_opt_thread.start()

    def _btn_opt_rx_fired(self):
        if self.rx_opt_thread and self.rx_opt_thread.is_alive() or self.ctle_mode_tune == "Off":
            pass
        else:
            self.rx_opt_thread = RxOptThread()
            self.rx_opt_thread.pybert = self
            self.rx_opt_thread.start()

    def _btn_coopt_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            pass
        else:
            self.coopt_thread = CoOptThread()
            self.coopt_thread.pybert = self
            self.coopt_thread.start()

    def _btn_abort_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            self.coopt_thread.stop()
            self.coopt_thread.join(10)
        if self.tx_opt_thread and self.tx_opt_thread.is_alive():
            self.tx_opt_thread.stop()
            self.tx_opt_thread.join(10)
        if self.rx_opt_thread and self.rx_opt_thread.is_alive():
            self.rx_opt_thread.stop()
            self.rx_opt_thread.join(10)

    def _btn_cfg_tx_fired(self):
        self._tx_cfg()

    def _btn_cfg_rx_fired(self):
        self._rx_cfg()

    def _btn_sel_tx_fired(self):
        """Open the IBIS Component/Model Selector.

        When the selector is closed, update the dll and ami filepaths because they could have
        changed if a new model was selected.
        """
        self._tx_ibis.open_gui()  # Blocks until window is closed.

        self.tx_dll_file = self._tx_ibis.dll_file
        self.tx_ami_file = self._tx_ibis.ami_file

    def _btn_sel_rx_fired(self):
        """Open the IBIS Component/Model Selector.

        When the selector is closed, update the dll and ami filepaths because they could have
        changed if a new model was selected.
        """
        self._rx_ibis.open_gui()  # Blocks until window is closed.

        self.rx_dll_file = self._rx_ibis.dll_file
        self.rx_ami_file = self._rx_ibis.ami_file

    def _btn_view_tx_fired(self):
        """Open the IBIS Model Viewer."""
        self._tx_ibis.model.open_gui()

    def _btn_view_rx_fired(self):
        """Open the IBIS Model Viewer."""
        self._rx_ibis.model.open_gui()

    # Independent variable setting intercepts
    # (Primarily, for debugging.)
    def _set_ctle_peak_mag_tune(self, val):
        if val > gMaxCTLEPeak or val < 0.0:
            raise RuntimeError("CTLE peak magnitude out of range!")
        self.peak_mag_tune = val

    # Dependent variable definitions
    @cached_property
    def _get_t(self):
        """
        Calculate the system time vector, in seconds.

        """

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @cached_property
    def _get_t_ns(self):
        """
        Calculate the system time vector, in ns.
        """

        return self.t * 1.0e9

    @cached_property
    def _get_f(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in Hz.
        # (i.e. - [0, f0, 2 * f0, ... , fN] + [-(fN - f0), -(fN - 2 * f0), ... , -f0]

        Note: Changed to positive freqs. only, in conjunction w/ irfft() usage.
        """
        t = self.t
        npts = len(t)
        f0 = 1.0 / (t[1] * npts)
        half_npts = npts // 2
        return array([i * f0 for i in range(half_npts)])

    @cached_property
    def _get_w(self):
        """
        System frequency vector, in rads./sec.
        """
        return 2 * pi * self.f

    @cached_property
    def _get_bits(self):
        """
        Generate the bit stream.
        """

        pattern_len = self.pattern_len
        nbits = self.nbits
        mod_type = self.mod_type[0]

        bits = []
        seed = randint(128)
        while not seed:  # We don't want to seed our LFSR with zero.
            seed = randint(128)
        bit_gen = lfsr_bits([7, 6], seed)
        for _ in range(pattern_len - 4):
            bits.append(next(bit_gen))

        # The 4-bit prequels, below, are to ensure that the first zero crossing
        # in the actual slicer input signal occurs. This is necessary, because
        # we assume it does, when aligning the ideal and actual signals for
        # jitter calculation.
        #
        # We may want to talk to Mike Steinberger, of SiSoft, about his
        # correlation based approach to this alignment chore. It's
        # probably more robust.
        if mod_type == 1:  # Duo-binary precodes, using XOR.
            return resize(array([0, 0, 1, 0] + bits), nbits)
        return resize(array([0, 0, 1, 1] + bits), nbits)

    @cached_property
    def _get_ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """

        mod_type = self.mod_type[0]
        bit_rate = self.bit_rate * 1.0e9

        ui = 1.0 / bit_rate
        if mod_type == 2:  # PAM-4
            ui *= 2.0

        return ui

    @cached_property
    def _get_nui(self):
        """
        Returns the number of unit intervals in the test vectors.
        """

        mod_type = self.mod_type[0]
        nbits = self.nbits

        nui = nbits
        if mod_type == 2:  # PAM-4
            nui //= 2

        return nui

    @cached_property
    def _get_nspui(self):
        """
        Returns the number of samples per unit interval.
        """

        mod_type = self.mod_type[0]
        nspb = self.nspb

        nspui = nspb
        if mod_type == 2:  # PAM-4
            nspui *= 2

        return nspui

    @cached_property
    def _get_eye_uis(self):
        """
        Returns the number of unit intervals to use for eye construction.
        """

        mod_type = self.mod_type[0]
        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if mod_type == 2:  # PAM-4
            eye_uis //= 2

        return eye_uis

    @cached_property
    def _get_ideal_h(self):
        """
        Returns the ideal link impulse response.
        """

        ui = self.ui
        nspui = self.nspui
        t = self.t
        mod_type = self.mod_type[0]
        ideal_type = self.ideal_type[0]

        t = array(t) - t[-1] / 2.0

        if ideal_type == 0:  # delta
            ideal_h = zeros(len(t))
            ideal_h[len(t) / 2] = 1.0
        elif ideal_type == 1:  # sinc
            ideal_h = sinc(t / (ui / 2.0))
        elif ideal_type == 2:  # raised cosine
            ideal_h = (cos(pi * t / (ui / 2.0)) + 1.0) / 2.0
            ideal_h = where(t < -ui / 2.0, zeros(len(t)), ideal_h)
            ideal_h = where(t > ui / 2.0, zeros(len(t)), ideal_h)
        else:
            raise Exception("PyBERT._get_ideal_h(): ERROR: Unrecognized ideal impulse response type.")

        if mod_type == 1:  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (ideal_h + pad(ideal_h[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)))

        return ideal_h

    @cached_property
    def _get_symbols(self):
        """
        Generate the symbol stream.
        """

        mod_type = self.mod_type[0]
        vod = self.vod
        bits = self.bits

        if mod_type == 0:  # NRZ
            symbols = 2 * bits - 1
        elif mod_type == 1:  # Duo-binary
            symbols = [bits[0]]
            for bit in bits[1:]:  # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = 2 * array(symbols) - 1
        elif mod_type == 2:  # PAM-4
            symbols = []
            for bits in zip(bits[0::2], bits[1::2]):
                if bits == (0, 0):
                    symbols.append(-1.0)
                elif bits == (0, 1):
                    symbols.append(-1.0 / 3.0)
                elif bits == (1, 0):
                    symbols.append(1.0 / 3.0)
                else:
                    symbols.append(1.0)
        else:
            raise Exception("ERROR: _get_symbols(): Unknown modulation type requested!")

        return array(symbols) * vod

    @cached_property
    def _get_ffe(self):
        """
        Generate the Tx pre-emphasis FIR numerator.
        """

        tap_tuners = self.tx_taps

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

        return taps

    @cached_property
    def _get_jitter_info(self):
        try:
            jitter = {
                "isi_chnl": self.isi_chnl * 1.0e12,
                "dcd_chnl": self.dcd_chnl * 1.0e12,
                "pj_chnl": self.pj_chnl * 1.0e12,
                "rj_chnl": self.rj_chnl * 1.0e12,
                "isi_tx": self.isi_tx * 1.0e12,
                "dcd_tx": self.dcd_tx * 1.0e12,
                "pj_tx": self.pj_tx * 1.0e12,
                "rj_tx": self.rj_tx * 1.0e12,
                "isi_ctle": self.isi_ctle * 1.0e12,
                "dcd_ctle": self.dcd_ctle * 1.0e12,
                "pj_ctle": self.pj_ctle * 1.0e12,
                "rj_ctle": self.rj_ctle * 1.0e12,
                "isi_dfe": self.isi_dfe * 1.0e12,
                "dcd_dfe": self.dcd_dfe * 1.0e12,
                "pj_dfe": self.pj_dfe * 1.0e12,
                "rj_dfe": self.rj_dfe * 1.0e12,
            }

            if jitter["isi_tx"]:
                jitter["isi_rej_tx"] = 10.0 * safe_log10(jitter["isi_chnl"] / jitter["isi_tx"])
            else:
                jitter["isi_rej_tx"] = 10.0 * safe_log10(1.0e20)
            if jitter["dcd_tx"]:
                jitter["dcd_rej_tx"] = 10.0 * safe_log10(jitter["dcd_chnl"] / jitter["dcd_tx"])
            else:
                jitter["dcd_rej_tx"] = 10.0 * safe_log10(1.0e20)
            if jitter["isi_ctle"]:
                jitter["isi_rej_ctle"] = 10.0 * safe_log10(jitter["isi_tx"] / jitter["isi_ctle"])
            else:
                jitter["isi_rej_ctle"] = 10.0 * safe_log10(1.0e20)
            if jitter["dcd_ctle"]:
                jitter["dcd_rej_ctle"] = 10.0 * safe_log10(jitter["dcd_tx"] / jitter["dcd_ctle"])
            else:
                jitter["dcd_rej_ctle"] = 10.0 * safe_log10(1.0e20)
            if jitter["pj_ctle"]:
                jitter["pj_rej_ctle"] = 10.0 * safe_log10(jitter["pj_tx"] / jitter["pj_ctle"])
            else:
                jitter["pj_rej_ctle"] = 10.0 * safe_log10(1.0e20)
            if jitter["rj_ctle"]:
                jitter["rj_rej_ctle"] = 10.0 * safe_log10(jitter["rj_tx"] / jitter["rj_ctle"])
            else:
                jitter["rj_rej_ctle"] = 10.0 * safe_log10(1.0e20)
            if jitter["isi_dfe"]:
                jitter["isi_rej_dfe"] = 10.0 * safe_log10(jitter["isi_ctle"] / jitter["isi_dfe"])
            else:
                jitter["isi_rej_dfe"] = 10.0 * safe_log10(1.0e20)
            if jitter["dcd_dfe"]:
                jitter["dcd_rej_dfe"] = 10.0 * safe_log10(jitter["dcd_ctle"] / jitter["dcd_dfe"])
            else:
                jitter["dcd_rej_dfe"] = 10.0 * safe_log10(1.0e20)
            if jitter["pj_dfe"]:
                jitter["pj_rej_dfe"] = 10.0 * safe_log10(jitter["pj_ctle"] / jitter["pj_dfe"])
            else:
                jitter["pj_rej_dfe"] = 10.0 * safe_log10(1.0e20)
            if jitter["rj_dfe"]:
                jitter["rj_rej_dfe"] = 10.0 * safe_log10(jitter["rj_ctle"] / jitter["rj_dfe"])
            else:
                jitter["rj_rej_dfe"] = 10.0 * safe_log10(1.0e20)
            if jitter["isi_dfe"]:
                jitter["isi_rej_total"] = 10.0 * safe_log10(jitter["isi_chnl"] / jitter["isi_dfe"])
            else:
                jitter["isi_rej_total"] = 10.0 * safe_log10(1.0e20)
            if jitter["dcd_dfe"]:
                jitter["dcd_rej_total"] = 10.0 * safe_log10(jitter["dcd_chnl"] / jitter["dcd_dfe"])
            else:
                jitter["dcd_rej_total"] = 10.0 * safe_log10(1.0e20)
            if jitter["pj_dfe"]:
                jitter["pj_rej_total"] = 10.0 * safe_log10(jitter["pj_tx"] / jitter["pj_dfe"])
            else:
                jitter["pj_rej_total"] = 10.0 * safe_log10(1.0e20)
            if jitter["rj_dfe"]:
                jitter["rj_rej_total"] = 10.0 * safe_log10(jitter["rj_tx"] / jitter["rj_dfe"])
            else:
                jitter["rj_rej_total"] = 10.0 * safe_log10(1.0e20)
            jitter_table = jitter_info_table(jitter)
        except Exception as err:
            jitter_table = (
                "<H1>Jitter Rejection by Equalization Component</H1>\n",
                "Sorry, the following error occurred:\n",
                str(err),
            )

        return jitter_table

    @cached_property
    def _get_perf_info(self):
        return performance_info_table(
            self.channel_perf * 6e-05,
            self.tx_perf * 6e-05,
            self.ctle_perf * 6e-05,
            self.dfe_perf * 6e-05,
            self.jitter_perf * 6e-05,
            self.plotting_perf * 6e-05,
            self.total_perf * 60.0e-6,
        )

    @cached_property
    def _get_sweep_info(self):
        return sweep_info_table(self.sweep_results)

    @cached_property
    def _get_status_str(self):
        return status_string(self)

    @cached_property
    def _get_tx_h_tune(self):
        nspui = self.nspui
        tap_tuners = self.tx_tap_tuners

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

        h = sum([[x] + list(zeros(nspui - 1)) for x in taps], [])

        return h

    @cached_property
    def _get_ctle_h_tune(self):
        w = self.w
        len_h = self.len_h
        rx_bw = self.rx_bw_tune * 1.0e9
        peak_freq = self.peak_freq_tune * 1.0e9
        peak_mag = self.peak_mag_tune
        offset = self.ctle_offset_tune
        mode = self.ctle_mode_tune

        _, H = make_ctle(rx_bw, peak_freq, peak_mag, w, mode, offset)
        h = irfft(H)

        return h

    @cached_property
    def _get_ctle_out_h_tune(self):
        chnl_h = self.chnl_h
        tx_h = self.tx_h_tune
        ctle_h = self.ctle_h_tune

        tx_out_h = convolve(tx_h, chnl_h)
        return convolve(ctle_h, tx_out_h)

    @cached_property
    def _get_cost(self):
        nspui = self.nspui
        h = self.ctle_out_h_tune
        mod_type = self.mod_type[0]

        s = h.cumsum()
        p = s - pad(s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        (clock_pos, thresh) = pulse_center(p, nspui)
        if clock_pos == -1:
            return 1.0  # Returning a large cost lets it know it took a wrong turn.
        clocks = thresh * ones(len(p))
        if mod_type == 1:  # Handle duo-binary.
            clock_pos -= nspui // 2
        clocks[clock_pos] = 0.0
        if mod_type == 1:  # Handle duo-binary.
            clocks[clock_pos + nspui] = 0.0

        # Cost is simply ISI minus main lobe amplitude.
        # Note: post-cursor ISI is NOT included in cost, when we're using the DFE.
        isi = 0.0
        ix = clock_pos - nspui
        while ix >= 0:
            clocks[ix] = 0.0
            isi += abs(p[ix])
            ix -= nspui
        ix = clock_pos + nspui
        if mod_type == 1:  # Handle duo-binary.
            ix += nspui
        while ix < len(p):
            clocks[ix] = 0.0
            if not self.use_dfe_tune:
                isi += abs(p[ix])
            ix += nspui
        if self.use_dfe_tune:
            for i in range(self.n_taps_tune):
                if clock_pos + nspui * (1 + i) < len(p):
                    p[int(clock_pos + nspui * (0.5 + i)) :] -= p[clock_pos + nspui * (1 + i)]
        plot_len = len(self.chnl_h)
        self.plotdata.set_data("ctle_out_h_tune", p[:plot_len])
        self.plotdata.set_data("clocks_tune", clocks[:plot_len])

        if mod_type == 1:  # Handle duo-binary.
            return isi - p[clock_pos] - p[clock_pos + nspui] + 2.0 * abs(p[clock_pos + nspui] - p[clock_pos])
        return isi - p[clock_pos]

    @cached_property
    def _get_rel_opt(self):
        return -self.cost

    @cached_property
    def _get_przf_err(self):
        p = self.dfe_out_p
        nspui = self.nspui
        n_taps = self.n_taps

        (clock_pos, _) = pulse_center(p, nspui)
        err = 0
        len_p = len(p)
        for i in range(n_taps):
            ix = clock_pos + (i + 1) * nspui
            if ix < len_p:
                err += p[ix] ** 2

        return err / p[clock_pos] ** 2

    # Changed property handlers.
    def _status_str_changed(self):
        self._log.info(self.status)

    def _debug_changed(self, enable_debug):
        """If the user enables debug, turn up the verbosity of the logger."""
        if enable_debug:
            logging.getLogger().setLevel(logging.DEBUG)
            self._console_log_handler.setLevel(logging.DEBUG)
            self._log.debug("Debug mode enabled.")
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._console_log_handler.setLevel(logging.INFO)

    def _use_dfe_changed(self, enabled: bool):
        """When the checkbox, `use_dfe`, is checked, enable the tx_taps."""
        for tap in self.tx_taps:
            tap.enabled = enabled

    def _use_dfe_tune_changed(self, enabled: bool):
        """When the checkbox, `use_dfe_tune`, is checked, enable the tx_tap_tuners."""
        for tap in self.tx_tap_tuners:
            tap.enabled = enabled

    def read_in_ibis_model(self, ibis_file: Path, is_tx: bool = True):
        """Read in a new ibis file and return an IBISModel or log an error."""
        self.status = f"Parsing IBIS file: {ibis_file}"
        try:
            self._log.info("Parsing IBIS file: %s", ibis_file)
            model = IBISModel(ibis_file, is_tx, debug=self.debug, gui=self.has_gui)
            self._log.info("Result:\n %s", model.ibis_parsing_errors)
            return model
        except Exception as err:
            self._log.error("Failed to open and/or parse IBIS file!\n%s", err, exc_info=True, extra={"alert": True})

    def _tx_ibis_file_changed(self, new_tx_ibis_file):
        """The user changed `tx_ibis_file`, mark the file as invalid and parse the new one."""
        self.tx_ibis_valid = False
        self.tx_use_ami = False
        ibis_model = self.read_in_ibis_model(Path(new_tx_ibis_file), is_tx=True)
        if ibis_model:
            self._tx_ibis = ibis_model
            self.tx_dll_file = ibis_model.dll_file
            self.tx_ami_file = ibis_model.ami_file
            self.tx_ibis_valid = True
        self.status = "Done."

    def _rx_ibis_file_changed(self, new_rx_ibis_file):
        """The user changed `rx_ibis_file`, mark the file as invalid and parse the new one."""
        self.rx_ibis_valid = False
        self.rx_use_ami = False
        ibis_model = self.read_in_ibis_model(Path(new_rx_ibis_file), is_tx=False)
        if ibis_model:
            self._rx_ibis = ibis_model
            self.rx_dll_file = ibis_model.dll_file
            self.rx_ami_file = ibis_model.ami_file
            self.rx_ibis_valid = True
        self.status = "Done."

    def read_in_ami_file(self, new_ami_file: Path):
        """Read in a new ibis ami file and return infomation about the AMI or log an error."""
        try:
            self._log.info("Parsing AMI file, %s", new_ami_file)
            pcfg = AMIParamConfigurator(new_ami_file)
            if pcfg.ami_parsing_errors:
                self._log.warning("Non-fatal parsing errors:\n %s", pcfg.ami_parsing_errors)
            else:
                self._log.info("Success.")
            has_getwave = bool(pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]))
            has_ts4 = bool(pcfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
            return pcfg, has_getwave, has_ts4
        except Exception as error:
            self._log.error("Failed to open and/or parse AMI file!\n%s", error, extra={"alert": True})

    def _tx_ami_file_changed(self, new_tx_ami_file):
        """The user changed `tx_ami_file`, mark the file as invalid and parse the new one."""
        if new_tx_ami_file:
            self.tx_ami_valid = False
            self._tx_cfg, self.tx_has_getwave, self.rx_has_ts4 = self.read_in_ami_file(new_tx_ami_file)
            if self._tx_cfg:
                self.tx_ami_valid = True

    def _rx_ami_file_changed(self, new_rx_ami_file):
        """The user changed `rx_ami_file`, mark the file as invalid and parse the new one."""
        if new_rx_ami_file:
            self.rx_ami_valid = False
            self._rx_cfg, self.rx_has_getwave, self.rx_has_ts4 = self.read_in_ami_file(new_rx_ami_file)
            if self._rx_cfg:
                self.rx_ami_valid = True

    def _tx_dll_file_changed(self, new_tx_dll_file):
        """The user changed `tx_dll_file`, mark the executable as invalid and open the new one."""
        try:
            self.tx_dll_valid = False
            if new_tx_dll_file:
                model = AMIModel(Path(new_tx_dll_file))
                self._tx_model = model
                self.tx_dll_valid = True
        except Exception as err:
            self._log.error("Failed to open DLL/SO file!\n %s", err, extra={"alert": True})

    def _rx_dll_file_changed(self, new_rx_dll_file):
        """The user changed `rx_dll_file`, mark the executable as invalid and open the new one."""
        try:
            self.rx_dll_valid = False
            if new_rx_dll_file:
                model = AMIModel(Path(new_rx_dll_file))
                self._rx_model = model
                self.rx_dll_valid = True
        except Exception as err:
            self._log.info("Failed to open DLL/SO file!\n %s", err, extra={"alert": True})

    def _rx_use_ami_changed(self, rx_ami_is_enabled: bool):
        """If you enable ami for the rx, disable pybert's ideal dfe."""
        if rx_ami_is_enabled:
            self.use_dfe = False

    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    def calc_chnl_h(self):
        """
        Calculates the channel impulse response.

        Also sets, in 'self':
         - chnl_dly:
             group delay of channel
         - start_ix:
             first element of trimmed response
         - t_ns_chnl:
             the x-values, in ns, for plotting 'chnl_h'
         - chnl_H:
             channel frequency response
         - chnl_s:
             channel step response
         - chnl_p:
             channel pulse response

        """

        t = self.t
        f = self.f
        w = self.w
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9
        Rs = self.rs
        Cs = self.cout * 1.0e-12
        RL = self.rin
        Cp = self.cin * 1.0e-12
        CL = self.cac * 1.0e-6

        ts = t[1]
        len_t = len(t)
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.use_ch_file:
            ch_s2p_pre = import_channel(self.ch_file, ts, self.f)
        else:
            # Construct PyBERT default channel model (i.e. - Howard Johnson's UTP model).
            # - Grab model parameters from PyBERT instance.
            l_ch = self.l_ch
            v0 = self.v0 * 3.0e8
            R0 = self.R0
            w0 = self.w0
            Rdc = self.Rdc
            Z0 = self.Z0
            Theta0 = self.Theta0
            # - Calculate propagation constant, characteristic impedance, and transfer function.
            gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            self.Zc = Zc
            H = exp(-l_ch * gamma)
            self.H = H
            # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
            tmp = np.array(list(zip(zip(zeros(len_f), H), zip(H, zeros(len_f)))))
            ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)
            # - And, finally, renormalize to driver impedance.
            ch_s2p_pre.renormalize(Rs)
        ch_s2p_pre.name = "ch_s2p_pre"
        self.ch_s2p_pre = ch_s2p_pre
        ch_s2p = ch_s2p_pre  # In case neither set of on-die S-parameters is being invoked, below.

        # Augment w/ IBIS-AMI on-die S-parameters, if appropriate.
        def add_ondie_s(s2p, ts4f, isRx=False):
            """Add the effect of on-die S-parameters to channel network.

            Args:
                s2p(skrf.Network): initial 2-port network.
                ts4f(string): on-die S-parameter file name.

            KeywordArgs:
                isRx(bool): True when Rx on-die S-params. are being added. (Default = False).

            Returns:
                skrf.Network: Resultant 2-port network.
            """
            ts4N = rf.Network(ts4f)  # Grab the 4-port single-ended on-die network.
            ntwk = sdd_21(ts4N)  # Convert it to a differential, 2-port network.
            ntwk2 = interp_s2p(ntwk, s2p.f)  # Interpolate to system freqs.
            if isRx:
                res = s2p ** ntwk2
            else:  # Tx
                res = ntwk2 ** s2p
            return (res, ts4N, ntwk2)

        if self.tx_use_ibis:
            model = self._tx_ibis.model
            Rs = model.zout * 2
            Cs = model.ccomp[0] / 2  # They're in series.
            self.Rs = Rs  # Primarily for debugging.
            self.Cs = Cs
            if self.tx_use_ts4:
                fname = join(self._tx_ibis_dir, self._tx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname)
                self.ts4N   = ts4N
                self.ntwk   = ntwk
        if self.rx_use_ibis:
            model = self._rx_ibis.model
            RL = model.zin * 2
            Cp = model.ccomp[0] / 2
            self.RL = RL  # Primarily for debugging.
            self.Cp = Cp
            self._log.debug("RL: %d, Cp: %d", RL, Cp)
            if self.rx_use_ts4:
                fname = join(self._rx_ibis_dir, self._rx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
                self.ts4N   = ts4N
                self.ntwk   = ntwk
        ch_s2p.name = "ch_s2p"
        self.ch_s2p = ch_s2p

        # Calculate channel impulse response.
        Zt   = RL / (1 + 1j*w*RL*Cp)    # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:,1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:,1]) / np.sqrt(ch_s2p_term.z0[:,0])
        chnl_h = irfft(chnl_H)
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        temp = chnl_h.copy()
        temp.resize(len(t))
        chnl_trimmed_H = fft(temp)

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    def log_system_information(self):
        """Log the system information."""
        self._log.info("System: %s %s", platform.system(), platform.release())
        self._log.info("Python Version:  %s", platform.python_version())
        self._log.info("PyBERT Version:  %s", __version__)
        self._log.info("PyAMI Version:  %s", PyAMI_VERSION)
        self._log.debug("GUI Toolkit:  %s", ETSConfig.toolkit)
        self._log.debug("Kiva Backend:  %s", ETSConfig.kiva_backend)
        # self._log.debug("Pixel Scale:  %s", self.trait_view().window.base_pixel_scale)

    def load_configuration(self, filepath: Path):
        """Load in a configuration into pybert.

        Support both file formats either yaml or pickle.
        """
        try:
            PyBertCfg.load_from_file(filepath, self)
            self.cfg_file = filepath
            self.status = "Loaded configuration."
        except Exception:
            self._log.error("Failed to load configuration.", exc_info=True, extra={"alert": True})

    def save_configuration(self, filepath: Path):
        """Save out a configuration from pybert.

        Support both file formats either yaml or pickle.
        """
        try:
            PyBertCfg(self, time.asctime(), __version__).save(filepath)
            self.cfg_file = filepath
            self.status = "Configuration saved."
        except Exception:
            self._log.error("Failed to save current user configuration.", exc_info=True, extra={"alert": True})

    def load_results(self, filepath: Path):
        """Load results from a file into pybert."""
        try:
            PyBertData.load_from_file(filepath, self)
            self.data_file = filepath
            self.status = "Loaded results."
        except Exception:
            self._log.error("Failed to load results from file.", exc_info=True, extra={"alert": True})

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file."""
        try:
            PyBertData(self).save(filepath)
            self.data_file = filepath
            self.status = "Saved results."
        except Exception:
            self._log.error("Failed to save results to file.", exc_info=True, extra={"alert": True})

    def initialize_plots(self):
        """Create and initialize all of the plot containers with the default simulation results."""
        self.plots_dfe, self.plot_dfe_adapt = plot.init_dfe_tab_plots(self.plotdata, n_dfe_taps=PyBertCfg.n_taps)
        self.plot_h_tune = plot.init_eq_tune_tab_plots(self.plotdata)
        self.plots_h = plot.init_impulse_tab_plots(self.plotdata)
        self.plots_s = plot.init_step_tab_plots(self.plotdata)
        self.plots_p = plot.init_pulse_tab_plots(self.plotdata)
        self.plots_H = plot.init_frequency_tab_plots(self.plotdata)
        self.plots_out = plot.init_output_tab_plots(self.plotdata)
        self.plots_eye = plot.init_eye_diagram_plots(self.plotdata)
        self.plots_jitter_dist = plot.init_jitter_dist_plots(self.plotdata)
        self.plots_jitter_spec = plot.init_jitter_spec_plots(self.plotdata)
        self.plots_bathtub = plot.init_bathtub_plots(self.plotdata)

        # Regenerate the eye diagrams after they have been populated.
        self.update_eye_diagrams()

    def update_eye_diagrams(self):
        """Update the heat plots representing the eye diagrams."""

        ui = self.ui
        samps_per_ui = self.nspui

        width = 2 * samps_per_ui
        height = 100
        xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)

        for diagram, channel in enumerate((self.chnl_out, self.rx_in, self.ctle_out, self.dfe_out)):
            y_max = 1.1 * max(abs(array(channel)))
            ys = np.linspace(-y_max, y_max, height)
            self.plots_eye.components[diagram].components[0].index.set_data(xs, ys)
            self.plots_eye.components[diagram].x_axis.mapper.range.low = xs[0]
            self.plots_eye.components[diagram].x_axis.mapper.range.high = xs[-1]
            self.plots_eye.components[diagram].y_axis.mapper.range.low = ys[0]
            self.plots_eye.components[diagram].y_axis.mapper.range.high = ys[-1]
            self.plots_eye.components[diagram].invalidate_draw()

        self.plots_eye.request_redraw()

    def simulate(self, initial_run=False, update_plots=True):
        """Run all queued simulations.

        Normally, this is just one simulation unless `sweep_sim` is set.  Then it will run
        `num_sweeps` which is the product of all tx pre/post tap settings multipled by `sweep_aves`.
        If `sweep_aves` is one then its just one run of all pre/post tap combinations.

        When sweeping plots will not be updated, so the plot results is from the prior run.  The
        averaged bit error and standard deviation can be found under Results/Sweep Info.  Otherwise
        just one simulation is run and all the plots are updated.
        """

        if self.sweep_sim:
            # Assemble the list of desired values for each Tx Pre-Emphasis sweepable parameter.
            tap_sweep_values = [tap.sweep_values() for tap in self.tx_taps]
            sweeps = list(itertools.product(*tap_sweep_values))

            # Run the sweep, using the lists assembled, above.
            self.num_sweeps = self.sweep_aves * len(sweeps)
            sweep_results = []
            for sweep_run_number, settings in enumerate(sweeps, start=1):

                # Update the tap settings for this sweep
                for index, tap in enumerate(self.tx_taps):
                    tap.value = settings[index]

                bit_errs = []
                # Run the current settings for the number of sweeps to average
                for sweep_average_number in range(self.sweep_aves):
                    self.sweep_num = sweep_run_number + sweep_average_number
                    my_run_simulation(self, update_plots=False)
                    bit_errs.append(self.bit_errs)

                # Append the averaged results
                sweep_results.append((settings, np.mean(bit_errs), np.std(bit_errs)))

            self.sweep_results = sweep_results
        else:
            my_run_simulation(self, initial_run=initial_run, update_plots=update_plots)
