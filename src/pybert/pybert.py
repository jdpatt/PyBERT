#! /usr/bin/env python

"""Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
import logging
import platform
import time
from os.path import join
from pathlib import Path

import numpy as np
import skrf as rf
from chaco.api import ArrayPlotData, GridPlotContainer
from numpy import array, convolve, cos, ones, pad, pi, sinc, where, zeros
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
    String,
    cached_property,
)

from pybert import __version__ as VERSION
from pybert.configuration import InvalidFileType, PyBertCfg
from pybert.gui.help import help_str
from pybert.gui.plot import make_plots
from pybert.jitter import jitter_html_table
from pybert.logger import TraitsUiConsoleHandler
from pybert.models import Channel, Control, Optimizer, Receiver, Transmitter
from pybert.models.bert import my_run_simulation
from pybert.models.tx_tap import TxTapTuner
from pybert.results import PyBertData
from pybert.utility import (
    interp_s2p,
    lfsr_bits,
    make_ctle,
    pulse_center,
    sdd_21,
    trim_impulse,
)
from pyibisami import __version__ as PyAMI_VERSION

logger = logging.getLogger(__name__)


class PyBERT(HasTraits):
    """A serial communication link bit error rate tester (BERT) simulator with
    a GUI interface.

    Useful for exploring the concepts of serial communication link
    design.
    """

    control: Control = Instance(Control, args=())
    channel: Channel = Instance(Channel, args=())
    tx: Transmitter = Instance(Transmitter, args=())
    rx: Receiver = Instance(Receiver, args=())
    optimizer: Optimizer = Instance(Optimizer, args=())

    rel_power = Float(1.0)  #: Tx power dissipation (W).

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

    cost = Property(Float, depends_on=["ctle_out_h_tune", "nspui"])
    rel_opt = Property(Float, depends_on=["cost"])
    t = Property(Array, depends_on=["ui", "nspb", "nbits"])
    t_ns = Property(Array, depends_on=["t"])
    f = Property(Array, depends_on=["t"])
    w = Property(Array, depends_on=["f"])

    bits = Property(Array, depends_on=["pattern", "nbits", "mod_type", "run_count"])
    symbols = Property(Array, depends_on=["bits", "mod_type", "vod"])
    tx_h_tune = Property(Array, depends_on=["optimizer.tx_tap_tuners.value", "nspui"])
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

    ctle_out_h_tune = Property(Array, depends_on=["optimizer.tx_h_tune", "optimizer.ctle_h_tune", "chnl_h"])
    ffe = Property(Array, depends_on=["tx.taps.value", "tx.taps.enabled"])
    ui = Property(Float, depends_on=["bit_rate", "mod_type"])
    nui = Property(Int, depends_on=["nbits", "mod_type"])
    nspui = Property(Int, depends_on=["nspb", "mod_type"])
    eye_uis = Property(Int, depends_on=["eye_bits", "mod_type"])
    dfe_out_p = Array()
    przf_err = Property(Float, depends_on=["dfe_out_p"])

    # Default initialization
    def __init__(self, run_simulation=True, gui=True):
        """Initial plot setup occurs here.

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

        self._log_console_handler = TraitsUiConsoleHandler(self)
        logger.addHandler(self._log_console_handler)
        log_system_information()

        self.GUI = gui

        if run_simulation:
            self.simulate(initial_run=True)
        else:
            self.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().

    @cached_property
    def _get_tx_h_tune(self):
        nspui = self.nspui
        tap_tuners = self.optimizer.tx_tap_tuners

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

        h = sum([[x] + list(np.zeros(nspui - 1)) for x in taps], [])

        return h

    @cached_property
    def _get_ctle_h_tune(self):
        w = self.w
        len_h = self.len_h
        rx_bw = self.optimizer.rx_bw_tune * 1.0e9
        peak_freq = self.optimizer.peak_freq_tune * 1.0e9
        peak_mag = self.optimizer.peak_mag_tune
        offset = self.optimizer.ctle_offset_tune
        mode = self.optimizer.ctle_mode_tune

        _, H = make_ctle(rx_bw, peak_freq, peak_mag, w, mode, offset)
        h = irfft(H)

        return h

    @cached_property
    def _get_ctle_out_h_tune(self):
        chnl_h = self.chnl_h
        tx_h = self.optimizer.tx_h_tune
        ctle_h = self.optimizer.ctle_h_tune

        tx_out_h = np.convolve(tx_h, chnl_h)
        return np.convolve(ctle_h, tx_out_h)

    # Dependent variable definitions
    @cached_property
    def _get_t(self):
        """Calculate the system time vector, in seconds."""

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @cached_property
    def _get_t_ns(self):
        """Calculate the system time vector, in ns."""

        return self.t * 1.0e9

    @cached_property
    def _get_f(self):
        """Calculate the frequency vector appropriate for indexing non-shifted
        FFT output, in Hz.

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
        """System frequency vector, in rads./sec."""
        return 2 * pi * self.f

    @cached_property
    def _get_bits(self):
        "Generate the bit stream."
        pattern = self.control.pattern_
        seed = self.control.seed
        nbits = self.control.nbits

        if not seed:  # The user sets `seed` to zero when she wants a new random seed generated for each run.
            seed = randint(128)
            while not seed:  # We don't want to seed our LFSR with zero.
                seed = randint(128)
        bit_gen = lfsr_bits(pattern, seed)
        bits = [next(bit_gen) for _ in range(nbits)]
        return array(bits)

    @cached_property
    def _get_ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """

        mod_type = self.control.mod_type[0]
        bit_rate = self.control.bit_rate * 1.0e9

        ui = 1.0 / bit_rate
        if mod_type == 2:  # PAM-4
            ui *= 2.0

        return ui

    @cached_property
    def _get_nui(self):
        """Returns the number of unit intervals in the test vectors."""

        mod_type = self.control.mod_type[0]
        nbits = self.control.nbits

        nui = nbits
        if mod_type == 2:  # PAM-4
            nui //= 2

        return nui

    @cached_property
    def _get_nspui(self):
        """Returns the number of samples per unit interval."""

        mod_type = self.control.mod_type[0]
        nspb = self.control.nspb

        nspui = nspb
        if mod_type == 2:  # PAM-4
            nspui *= 2

        return nspui

    @cached_property
    def _get_eye_uis(self):
        """Returns the number of unit intervals to use for eye construction."""

        mod_type = self.control.mod_type[0]
        eye_bits = self.control.eye_bits

        eye_uis = eye_bits
        if mod_type == 2:  # PAM-4
            eye_uis //= 2

        return eye_uis

    @cached_property
    def _get_ideal_h(self):
        """Returns the ideal link impulse response."""

        ui = self.ui
        nspui = self.nspui
        t = self.t
        mod_type = self.control.mod_type[0]
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
        """Generate the symbol stream."""

        mod_type = self.control.mod_type[0]
        vod = self.control.vod
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
        """Generate the Tx pre-emphasis FIR numerator."""

        tap_tuners = self.tx.taps

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
        return jitter_html_table(self.jitter_channel, self.jitter_tx, self.jitter_ctle, self.jitter_dfe)

    @cached_property
    def _get_perf_info(self):
        info_str = "<H2>Performance by Component</H2>\n"
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += "      <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>\n"
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Channel</TD><TD>{self.channel_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Tx Preemphasis</TD><TD>{self.tx_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">CTLE</TD><TD>{self.ctle_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">DFE</TD><TD>{self.dfe_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Jitter Analysis</TD><TD>{self.jitter_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center"><strong>TOTAL</strong></TD><TD><strong>%6.3f</strong></TD>\n' % (
            self.total_perf * 60.0e-6
        )
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Plotting</TD><TD>{self.plotting_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += "  </TABLE>\n"

        return info_str

    @cached_property
    def _get_sweep_info(self):
        sweep_results = self.sweep_results

        info_str = "<H2>Sweep Results</H2>\n"
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += "      <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>\n"
        info_str += "    </TR>\n"

        for item in sweep_results:
            info_str += '    <TR align="center">\n'
            info_str += str(item)
            # info_str += "      <TD>%+06.3f</TD><TD>%+06.3f</TD><TD>%d</TD><TD>%d</TD>\n" % (
            #     item[0],
            #     item[1],
            #     item[2],
            #     item[3],
            # )
            info_str += "    </TR>\n"

        info_str += "  </TABLE>\n"

        return info_str

    @cached_property
    def _get_status_str(self):
        status_str = "%-20s | Perf. (Msmpls./min.):  %4.1f" % (
            self.status,
            self.total_perf * 60.0e-6,
        )
        dly_str = f"         | ChnlDly (ns):    {self.chnl_dly * 1000000000.0:5.3f}"
        err_str = f"         | BitErrs: {int(self.bit_errs)}"
        pwr_str = f"         | TxPwr (W): {self.rel_power:4.2f}"
        status_str += dly_str + err_str + pwr_str

        try:
            jit_str = "         | Jitter (ps):    ISI=%6.3f    DCD=%6.3f    Pj=%6.3f    Rj=%6.3f" % (
                self.isi_dfe * 1.0e12,
                self.dcd_dfe * 1.0e12,
                self.pj_dfe * 1.0e12,
                self.rj_dfe * 1.0e12,
            )
        except:
            jit_str = "         | (Jitter not available.)"

        status_str += jit_str

        return status_str

    @cached_property
    def _get_cost(self):
        nspui = self.nspui
        h = self.ctle_out_h_tune
        mod_type = self.control.mod_type[0]

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
        n_taps = self.rx.n_taps

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
        logger.info(self.status)

    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    def calc_chnl_h(self):
        """Calculates the channel impulse response.

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
        impulse_length = self.control.impulse_length * 1.0e-9
        Rs = self.tx.get_impedance()
        Cs = self.tx.get_capacitance()
        RL = self.rx.get_impedance()
        Cp = self.rx.get_capacitance()
        CL = self.rx.get_ac_capacitance()

        ts = t[1]
        len_t = len(t)
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.channel.use_ch_file:
            raw_channel_network = self.channel.network_from_file(t[1], self.f)
        else:
            raw_channel_network = self.channel.network_from_native_model(self.f, self.w)
            raw_channel_network.renormalize(self.tx.impedance)  # Renormalize to driver impedance.

        ch_s2p = self.tx.add_characteristics_to_channel(raw_channel_network)
        ch_s2p = self.rx.add_characteristics_to_channel(ch_s2p)
        ch_s2p.name = "ch_s2p"

        # Calculate channel impulse response.
        Zt = RL / (1 + 1j * w * RL * Cp)  # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:, 1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term
        # We take the transfer function, H, to be a ratio of voltages.

        # So, we must normalize our (now generalized) S-parameters.
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:, 1] / ch_s2p_term.z0[:, 0])
        chnl_h = irfft(chnl_H)
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        temp = chnl_h.copy()
        temp.resize(len(t), refcheck=False)
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

    def simulate(self, initial_run=False, update_plots=True):
        """Run all queued simulations."""
        # Running the simulation will fill in the required data structure.
        my_run_simulation(self, initial_run=initial_run, update_plots=update_plots)
        # Once the required data structure is filled in, we can create the plots.
        make_plots(self, n_dfe_taps=self.rx.n_taps)

    def load_configuration(self, filepath: Path):
        """Load in a configuration into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertCfg.load_from_file(filepath, self)
            self.cfg_file = filepath
            self.status = "Loaded configuration."
        except InvalidFileType:
            logger.warning("This filetype is not currently supported.")
        except Exception as exp:
            logger.error("Failed to load configuration. See the console for more detail.")
            logger.exception(exp)

    def save_configuration(self, filepath: Path):
        """Save out a configuration from pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertCfg(self, time.asctime(), VERSION).save(filepath)
            self.cfg_file = filepath
            self.status = "Configuration saved."
        except InvalidFileType:
            logger.warning("This filetype is not currently supported. Please try again as a yaml file.")
        except Exception as exp:
            logger.error("Failed to save current user configuration. See the console for more detail.")
            logger.exception(exp)

    def load_results(self, filepath: Path):
        """Load results from a file into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData.load_from_file(filepath, self)
            self.data_file = filepath
            self.status = "Loaded results."
        except Exception as exp:
            logger.error("Failed to load results from file. See the console for more detail.")
            logger.exception(exp)

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData(self, time.asctime(), VERSION).save(filepath)
            self.data_file = filepath
            self.status = "Saved results."
        except Exception as exp:
            logger.error("Failed to save results to file. See the console for more detail.")
            logger.exception(exp)

    def clear_reference_from_plots(self):
        """If any plots have ref in the name, delete them and then regenerate the plots.

        If we don't actually delete any data, skip regenerating the plots.
        """
        atleast_one_reference_removed = False

        for reference_plot in self.plotdata.list_data():
            if "ref" in reference_plot:
                try:
                    atleast_one_reference_removed = True
                    self.plotdata.del_data(reference_plot)
                except KeyError:
                    pass

        if atleast_one_reference_removed:
            make_plots(self, n_dfe_taps=self.rx.n_taps)


def log_system_information():
    """Log the system information."""
    from traits.etsconfig.api import ETSConfig

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"PyBERT Version: {VERSION}")
    logger.info(f"PyAMI Version: {PyAMI_VERSION}")
    logger.info(f"GUI Toolkit: {ETSConfig.toolkit}")
    logger.info(f"Kiva Backend: {ETSConfig.kiva_backend}")
    # logger.info(f"Pixel Scale: {self.trait_view().window.base_pixel_scale}")
