"""Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.signal as sig
from numpy import (  # type: ignore
    argmax,
    array,
    convolve,
    correlate,
    diff,
    float64,
    histogram,
    linspace,
    mean,
    repeat,
    resize,
    transpose,
    where,
    zeros,
)
from numpy.fft import irfft, rfft  # type: ignore
from numpy.random import normal  # type: ignore
from numpy.typing import NDArray  # type: ignore
from pyibisami.ami.parser import ami_parse
from pyibisami.common import AmiName, AmiNode
from scipy.interpolate import interp1d
from scipy.signal import iirfilter, lfilter

from pybert.models.dfe import DFE
from pybert.models.stimulus import ModulationType
from pybert.stoppable_thread import StoppableThread
from pybert.utility import (
    calc_eye,
    calc_jitter,
    calc_resps,
    find_crossings,
    import_channel,
    make_bathtub,
    make_ctle,
    run_ami_model,
    safe_log10,
    trim_impulse,
)

clock = perf_counter

AmiFloats: TypeAlias = tuple[AmiName, list["float | 'AmiFloats'"]]

DEBUG = False
MIN_BATHTUB_VAL = 1.0e-12
gFc = 1.0e6  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.

logger = logging.getLogger("pybert.sim")


class SimulationThread(StoppableThread):
    """Used to run the simulation in its own thread, in order to preserve GUI
    responsiveness."""

    def __init__(self):
        super().__init__()
        self.pybert = None

    def run(self):
        """Run the simulation(s)."""
        try:
            results, perf = run_simulation(self.pybert, aborted_sim=self.stopped)
            self.pybert.result_queue.put({"type": "simulation_complete", "results": results, "performance": perf})

        except RuntimeError as err:
            logger.critical(f"Error in `pybert.threads.sim.SimulationThread`: {err}")
            raise


@dataclass
class SimulationPerf:
    """Performance metrics for the simulation."""

    start_time: float = 0.0
    end_time: float = 0.0
    channel: float = 0.0
    tx: float = 0.0
    ctle: float = 0.0
    dfe: float = 0.0
    jitter: float = 0.0
    plotting: float = 0.0
    total: float = 0.0

    def __str__(self):
        return (
            f"Performance Metrics: (Msmpls./min) "
            f"Channel: {self.channel * 6e-05:6.3f}  "
            f"Tx Preemphasis: {self.tx * 6e-05:6.3f}  "
            f"CTLE: {self.ctle * 6e-05:6.3f}  "
            f"DFE: {self.dfe * 6e-05:6.3f}  "
            f"Jitter: {self.jitter * 6e-05:6.3f}  "
            f"Plotting: {self.plotting * 6e-05:6.3f}  "
            f"Total: {self.total * 6e-05:6.3f}"
        )


def calculate_plotting_data(self):
    """Calculate all the data needed for plotting results.

    This method calculates all the derived data needed for plotting, including:
    - DFE plots (ui_ests, tap_weights, clock spectrum)
    - Eye diagrams
    - Bathtub curves
    - Jitter distributions and spectra
    - Frequency responses
    """
    # DFE plot calculations
    ui_ests = self.ui_ests
    try:
        tap_weights = np.transpose(np.array(self.adaptation))
    except Exception:
        tap_weights = []

    # Clock spectrum calculations
    (bin_counts, bin_edges) = np.histogram(ui_ests, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    clock_spec = np.fft.rfft(ui_ests)
    t = self.t
    ui = self.ui
    _f0 = 1 / (t[1] * len(t)) if len(t) > 1 else 0
    spec_freqs = np.array([_f0 * k for k in range(len(t) // 2 + 1)])

    # Eye diagram calculations
    samps_per_ui = self.nspui
    eye_uis = self.eye_uis
    num_ui = self.nui
    clock_times = self.clock_times
    ignore_until = (num_ui - eye_uis) * self.ui
    ignore_samps = (num_ui - eye_uis) * samps_per_ui

    width = 2 * samps_per_ui
    xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)
    height = 1000

    # Calculate eye diagrams with tiny noise for better visualization
    tiny_noise = np.random.normal(scale=1e-3, size=len(self.chnl_out[ignore_samps:]))
    chnl_out_noisy = self.chnl_out[ignore_samps:] + tiny_noise
    y_max_chnl = 1.1 * max(abs(np.array(chnl_out_noisy)))
    eye_chnl = calc_eye(self.ui, samps_per_ui, height, chnl_out_noisy, y_max_chnl)

    y_max_rx = 1.1 * max(abs(np.array(self.rx_in[ignore_samps:])))
    eye_tx = calc_eye(self.ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max_rx)

    y_max_ctle = 1.1 * max(abs(np.array(self.ctle_out[ignore_samps:])))
    eye_ctle = calc_eye(self.ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max_ctle)

    y_max_dfe = 1.1 * max(abs(np.array(self.dfe_out[ignore_samps:])))
    i = 0
    len_clock_times = len(clock_times)
    while i < len_clock_times and clock_times[i] < ignore_until:
        i += 1
    if i >= len(clock_times):
        eye_dfe = calc_eye(self.ui, samps_per_ui, height, self.dfe_out[ignore_samps:], y_max_dfe)
    else:
        eye_dfe = calc_eye(
            self.ui,
            samps_per_ui,
            height,
            self.dfe_out[ignore_samps:],
            y_max_dfe,
            np.array(clock_times[i:]) - ignore_until,
        )

    # Bathtub curve calculations
    jitter_bins = self.jitter_bins
    bathtub_chnl = make_bathtub(
        jitter_bins,
        self.jitter_chnl,
        min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_chnl,
        mu_r=self.mu_pos_chnl,
        mu_l=self.mu_neg_chnl,
        extrap=True,
    )
    bathtub_tx = make_bathtub(
        jitter_bins,
        self.jitter_tx,
        min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_tx,
        mu_r=self.mu_pos_tx,
        mu_l=self.mu_neg_tx,
        extrap=True,
    )
    bathtub_ctle = make_bathtub(
        jitter_bins,
        self.jitter_ctle,
        min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_ctle,
        mu_r=self.mu_pos_ctle,
        mu_l=self.mu_neg_ctle,
        extrap=True,
    )
    bathtub_dfe = make_bathtub(
        jitter_bins,
        self.jitter_dfe,
        min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_dfe,
        mu_r=self.mu_pos_dfe,
        mu_l=self.mu_neg_dfe,
        extrap=True,
    )
    len_t = len(self.t_ns)

    # Return all calculated plotting data
    return {
        "t_ns": self.t_ns,
        "t_ns_chnl": self.t_ns_chnl,
        "output_plots": {
            "ideal_signal": self.ideal_signal[:len_t],
            "chnl_out": self.chnl_out[:len_t],
            "rx_in": self.rx_in[:len_t],
            "ctle_out": self.ctle_out[:len_t],
            "dfe_out": self.dfe_out[:len_t],
        },
        "impulse_plots": {
            "chnl_h": self.chnl_h,
            "tx_out_h": self.tx_out_h,
            "ctle_out_h": self.ctle_out_h,
            "dfe_out_h": self.dfe_out_h,
        },
        "step_plots": {
            "chnl_s": self.chnl_s,
            "tx_s": self.tx_s,
            "tx_out_s": self.tx_out_s,
            "ctle_s": self.ctle_s,
            "ctle_out_s": self.ctle_out_s,
            "dfe_s": self.dfe_s,
            "dfe_out_s": self.dfe_out_s,
        },
        "pulse_plots": {
            "chnl_p": self.chnl_p,
            "tx_out_p": self.tx_out_p,
            "ctle_out_p": self.ctle_out_p,
            "dfe_out_p": self.dfe_out_p,
        },
        "freq_plots": {
            "chnl_H": self.chnl_H,
            "chnl_H_raw": self.chnl_H_raw,
            "chnl_trimmed_H": self.chnl_trimmed_H,
            "tx_H": self.tx_H,
            "tx_out_H": self.tx_out_H,
            "ctle_H": self.ctle_H,
            "ctle_out_H": self.ctle_out_H,
            "dfe_H": self.dfe_H,
            "dfe_out_H": self.dfe_out_H,
        },
        # DFE plot data
        "ui_ests": ui_ests,
        "tap_weights": tap_weights,
        "clk_per_hist_bins": bin_centers,
        "clk_per_hist_vals": bin_counts,
        "clk_freqs": spec_freqs[1:] * ui if len(spec_freqs) > 1 else np.zeros(1),
        "clk_spec": safe_log10(np.abs(clock_spec[1:]) / np.abs(clock_spec[1])) if len(clock_spec) > 1 else np.zeros(1),
        # Eye diagram data
        "eye_xs": xs,
        "eye_data": [eye_chnl, eye_tx, eye_ctle, eye_dfe],
        "y_max_values": [y_max_chnl, y_max_rx, y_max_ctle, y_max_dfe],
        # Bathtub data
        "jitter_bins": np.array(jitter_bins) * 1e12,
        "bathtub_data": [
            safe_log10(bathtub_chnl),
            safe_log10(bathtub_tx),
            safe_log10(bathtub_ctle),
            safe_log10(bathtub_dfe),
        ],
        # Jitter data
        "jitter_data": [
            self.jitter_chnl * 1e-12,
            self.jitter_tx * 1e-12,
            self.jitter_ctle * 1e-12,
            self.jitter_dfe * 1e-12,
        ],
        "jitter_ext_data": [
            self.jitter_ext_chnl * 1e-12,
            self.jitter_ext_tx * 1e-12,
            self.jitter_ext_ctle * 1e-12,
            self.jitter_ext_dfe * 1e-12,
        ],
        # Jitter spectrum data
        "f_MHz": self.f_MHz[1:],
        "jitter_spectrum": [
            10.0 * (safe_log10(self.jitter_spectrum_chnl[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_spectrum_tx[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_spectrum_ctle[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_spectrum_dfe[1:]) - safe_log10(self.ui)),
        ],
        "jitter_ind_spectrum": [
            10.0 * (safe_log10(self.jitter_ind_spectrum_chnl[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_ind_spectrum_tx[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_ind_spectrum_ctle[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.jitter_ind_spectrum_dfe[1:]) - safe_log10(self.ui)),
        ],
        "jitter_thresh": [
            10.0 * (safe_log10(self.thresh_chnl[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.thresh_tx[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.thresh_ctle[1:]) - safe_log10(self.ui)),
            10.0 * (safe_log10(self.thresh_dfe[1:]) - safe_log10(self.ui)),
        ],
        # Frequency response data
        "f_GHz": self.f / 1.0e9,
        "freq_responses": {
            "chnl_H": 20.0 * safe_log10(np.abs(self.chnl_H[1:])),
            "chnl_H_raw": 20.0 * safe_log10(np.abs(self.chnl_H_raw[1:])),
            "chnl_trimmed_H": 20.0 * safe_log10(np.abs(self.chnl_trimmed_H[1:])),
            "tx_H": 20.0 * safe_log10(np.abs(self.tx_H[1:])),
            "tx_out_H": 20.0 * safe_log10(np.abs(self.tx_out_H[1:])),
            "ctle_H": 20.0 * safe_log10(np.abs(self.ctle_H[1:])),
            "ctle_out_H": 20.0 * safe_log10(np.abs(self.ctle_out_H[1:])),
            "dfe_H": 20.0 * safe_log10(np.abs(self.dfe_H[1:])),
            "dfe_out_H": 20.0 * safe_log10(np.abs(self.dfe_out_H[1:])),
        },
    }


def run_simulation(self, aborted_sim: Optional[Callable[[], bool]] = None):
    """Runs the simulation.

    Args:
        self: Reference to an instance of the *PyBERT* class.

    Keyword Args:
        aborted_sim: a function that is used to tell the simulation that the user
            has requested to stop/abort the simulation.

    Raises:
        RuntimeError: If the simulation is aborted by the user or cannot continue.

    Notes:
        1. When using IBIS-AMI models, we often need to scale the impulse response
            by the sample interval, or its inverse, because while IBIS-AMI models
            take the impulse response to have units: (V/s), PyBERT uses: (V/sample).
    """

    def _check_sim_status():
        """Checks the status of the simulation thread and if this simulation needs to stop."""
        if aborted_sim and aborted_sim():
            logger.error("Simulation aborted by User.")
            raise RuntimeError("Simulation aborted by User.")

    results = None

    perf = SimulationPerf()
    perf.start_time = clock()
    logger.info("Running channel...")

    # The user sets `seed` to zero to indicate that she wants new bits generated for each run.
    if not self.seed:
        self.run_count += 1  # Force regeneration of bit stream.

    # Pull class variables into local storage, performing unit conversion where necessary.
    t = self.t
    t_irfft = self.t_irfft
    f = self.f
    w = self.w
    bits = self.bits
    symbols = self.symbols
    ffe = self.ffe
    nbits = self.nbits
    nui = self.nui
    eye_bits = self.eye_bits
    eye_uis = self.eye_uis
    nspui = self.nspui
    rn = self.rn
    pn_mag = self.pn_mag
    pn_freq = self.pn_freq * 1.0e6
    pattern = self.pattern.value
    rx_bw = self.rx_bw * 1.0e9
    peak_freq = self.peak_freq * 1.0e9
    peak_mag = self.peak_mag
    ctle_enable = self.ctle_enable
    delta_t = self.delta_t * 1.0e-12
    alpha = self.alpha
    ui = self.ui
    gain = self.gain
    n_ave = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave = self.n_lock_ave
    dfe_tap_tuners = self.dfe_tap_tuners
    rel_lock_tol = self.rel_lock_tol
    lock_sustain = self.lock_sustain
    bandwidth = self.sum_bw * 1.0e9
    rel_thresh = self.thresh
    mod_type = self.mod_type
    impulse_length = self.impulse_length

    # Calculate misc. values.
    Ts = t[1]
    ts = Ts
    fs = 1 / ts
    min_len = 30 * nspui
    max_len = 100 * nspui
    if impulse_length:
        min_len = max_len = round(impulse_length / ts)
    if mod_type == ModulationType.PAM4:  # PAM-4
        nspb = nspui // 2
    else:
        nspb = nspui

    # Generate the ideal over-sampled signal.
    #
    # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
    # impulse response, in order to produce the proper ideal signal.
    x = repeat(symbols, nspui)
    ideal_signal = x
    if mod_type == ModulationType.DUO:  # Handle duo-binary case.
        duob_h = array(([0.5] + [0.0] * (nspui - 1)) * 2)
        ideal_signal = convolve(x, duob_h)[: len(t)]

    # Calculate the channel response, as well as its (hypothetical)
    # solitary effect on the data, for plotting purposes only.
    try:
        split_time = clock()
        chnl_h = self.calc_chnl_h()
        _calc_chnl_time = clock() - split_time
        # Note: We're not using 'self.ideal_signal', because we rely on the system response to
        #       create the duobinary waveform. We only create it explicitly, above,
        #       so that we'll have an ideal reference for comparison.
        split_time = clock()
        chnl_out = convolve(x, chnl_h)[: len(t)]
        _conv_chnl_time = clock() - split_time
        logger.debug(f"Channel calculation time: {_calc_chnl_time}")
        logger.debug(f"Channel convolution time: {_conv_chnl_time}")
        perf.channel = nbits * nspb / (clock() - perf.start_time)
    except Exception as err:
        logger.error(f"Exception: channel: {err}")
        raise
    self.chnl_out = chnl_out

    _check_sim_status()
    split_time = clock()
    logger.info("Running Tx...")

    # Calculate Tx output power dissipation.
    ffe_out = convolve(symbols, ffe)[: len(symbols)]
    if self.use_ch_file:
        self.rel_power = mean(ffe_out**2) / self.rs
    else:
        self.rel_power = mean(ffe_out**2) / self.Z0

    # Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
    # Generate the ideal rectangular aggressor waveform.
    pn_period = 1.0 / pn_freq
    pn_samps = int(pn_period / Ts + 0.5)
    pn = zeros(pn_samps)
    pn[pn_samps // 2 :] = pn_mag
    self.pn_period = pn_period
    self.pn_samps = pn_samps
    pn = resize(pn, len(x))
    # High pass filter it. (Simulating capacitive coupling.)
    (b, a) = iirfilter(2, gFc / (fs / 2), btype="highpass")
    pn = lfilter(b, a, pn)[: len(pn)]
    self.pn = pn

    noise = pn + normal(scale=rn, size=(len(x),))
    self.noise = noise

    # Tx and Rx modeling are not separable in all cases.
    # So, we model each of the 4 possible combinations explicitly.
    # For the purposes of tallying possible combinations,
    # AMI Init() and PyBERT native are equivalent,
    # as both rely on convolving w/ impulse responses.

    def get_ctle_h():
        "Return the impulse response of the PyBERT native CTLE model."
        if self.use_ctle_file:
            # FIXME: The new import_channel() implementation breaks this.
            ctle_h = import_channel(self.ctle_file, ts, f)
            if max(abs(ctle_h)) < 100.0:  # step response?
                ctle_h = diff(ctle_h)  # impulse response is derivative of step response.
            else:
                ctle_h *= ts  # Normalize to (V/sample)
            ctle_h.resize(len(t))
            ctle_H = rfft(ctle_h)  # ToDo: This needs interpolation first.
        else:
            if ctle_enable:
                _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w)
                _ctle_h = irfft(ctle_H)
                krnl = interp1d(t_irfft, _ctle_h, bounds_error=False, fill_value=0)
                ctle_h = krnl(t)
                ctle_h *= sum(_ctle_h) / sum(ctle_h)
                ctle_h, _ = trim_impulse(ctle_h, front_porch=False, min_len=min_len, max_len=max_len)
            else:
                ctle_h = array([1.0] + [0.0 for _ in range(min_len - 1)])
        return ctle_h

    ctle_s = None
    clock_times = None
    try:
        params: list[str] = []
        if self.tx_use_ami and self.tx_use_getwave:
            tx_out, _, tx_h, tx_out_h, msg, _params = run_ami_model(
                self._tx_model, self._tx_cfg, True, ui, ts, chnl_h, x
            )
            params = _params
            logger.info(f"Tx IBIS-AMI model initialization results:\n{msg}")
            tx_getwave_params = list(map(ami_parse, params))
            logger.info(f"Tx IBIS-AMI model GetWave() output parameters:\n{tx_getwave_params}")
            rx_in = convolve(tx_out + noise, chnl_h)[: len(tx_out)]
            tx_s, tx_p, tx_H = calc_resps(t, tx_h, ui, f)
            # Calculate the remaining responses from the impulse responses.
            tx_out_s, tx_out_p, tx_out_H = calc_resps(t, tx_out_h, ui, f)
            perf.tx = nbits * nspb / (clock() - split_time)
            split_time = clock()
            logger.info("Running CTLE...")
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, _, ctle_h, ctle_out_h, msg, _params = run_ami_model(
                    self.rx_model, self.rx_cfg, True, ui, ts, tx_out_h, convolve(tx_out, chnl_h)
                )
                params = _params
                logger.info(f"Rx IBIS-AMI model initialization results:\n{msg}")
                _rx_getwave_params = list(map(ami_parse, params))
                logger.info(f"Rx IBIS-AMI model GetWave() output parameters:\n{_rx_getwave_params}")
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    _, _, ctle_h, ctle_out_h, msg, _ = run_ami_model(
                        self.rx_model, self.rx_cfg, False, ui, ts, chnl_h, tx_out
                    )
                    logger.info(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out = convolve(tx_out, ctle_out_h)[: len(tx_out)]
                else:  # PyBERT native Rx
                    ctle_h = get_ctle_h()
                    ctle_out_h = convolve(ctle_h, tx_out_h)[: len(ctle_h)]
                    ctle_out = convolve(tx_out, convolve(ctle_h, chnl_h))[: len(tx_out)]
        else:  # Tx is either AMI_Init() or PyBERT native.
            if self.tx_use_ami:  # Tx is AMI_Init().
                rx_in, _, tx_h, tx_out_h, msg, _ = run_ami_model(self.tx_model, self.tx_cfg, False, ui, ts, chnl_h, x)
                logger.info(f"Tx IBIS-AMI model initialization results:\n{msg}")
                rx_in += noise
            else:  # Tx is PyBERT native.
                # Using `sum` to concatenate:
                tx_h = array(sum([[x] + list(zeros(nspui - 1)) for x in ffe], []))
                tx_h.resize(len(chnl_h), refcheck=False)  # "refcheck=False", to get around Tox failure.
                tx_out_h = convolve(tx_h, chnl_h)[: len(chnl_h)]
                rx_in = convolve(x, tx_out_h)[: len(x)] + noise
            # Calculate the remaining responses from the impulse responses.
            tx_s, tx_p, tx_H = calc_resps(t, tx_h, ui, f)
            tx_out_s, tx_out_p, tx_out_H = calc_resps(t, tx_out_h, ui, f)
            perf.tx = nbits * nspb / (clock() - split_time)
            split_time = clock()
            logger.info("Running CTLE...")
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, clock_times, ctle_h, ctle_out_h, msg, _params = run_ami_model(
                    self.rx_model, self.rx_cfg, True, ui, ts, tx_out_h, rx_in
                )
                params = _params
                logger.info(f"Rx IBIS-AMI model initialization results:\n{msg}")
                # Time evolution of (<root_name>: AmiName, <param_vals>: list[AmiNode]):
                # (i.e. - There can be no `AmiAtom`s in the root tuple's second member.)
                rx_getwave_params: list[tuple[AmiName, list[AmiNode]]] = list(map(ami_parse, params))
                param_vals = {}

                def isnumeric(x):
                    try:
                        _ = float(x)
                        return True
                    except:  # noqa: E722, pylint: disable=bare-except
                        return False

                def get_numeric_values(prefix: AmiName, node: AmiNode) -> dict[AmiName, list[np.float64]]:
                    "Retrieve all numeric values from an AMI node, encoding hierarchy in key names."
                    pname = node[0]
                    vals = node[1]
                    pname_hier = AmiName(prefix + pname)
                    first_val = vals[0]
                    if isnumeric(first_val):
                        return {pname_hier: list(map(float, vals))}  # type: ignore
                    if type(first_val) == AmiNode:  # noqa: E721, pylint: disable=unidiomatic-typecheck
                        subdicts = list(map(lambda nd: get_numeric_values(pname_hier, nd), vals))  # type: ignore
                        rslt = {}
                        for subdict in subdicts:
                            rslt.update(subdict)
                        return rslt
                    return {}

                for nd in rx_getwave_params[0][1]:
                    param_vals.update(get_numeric_values(AmiName(""), nd))
                for rslt in rx_getwave_params[1:]:
                    for nd in rslt[1]:
                        vals_dict = get_numeric_values(AmiName(""), nd)
                        for pname, pvals in vals_dict.items():
                            param_vals[pname].extend(pvals)

                _tap_weights = []
                dfe_tap_keys: list[AmiName] = list(filter(lambda s: s.tolower().contains("tap"), param_vals.keys()))  # type: ignore
                dfe_tap_keys.sort()
                for dfe_tap_key in dfe_tap_keys:
                    _tap_weights.append(param_vals[dfe_tap_key])
                tap_weights = array(_tap_weights).transpose()
                if "cdr_locked" in param_vals:
                    lockeds: npt.NDArray[np.float64] = array(param_vals[AmiName("cdr_locked")])
                    lockeds = lockeds.repeat(len(t) // len(lockeds))
                    lockeds.resize(len(t))
                else:
                    lockeds = zeros(len(t))
                if "cdr_ui" in param_vals:
                    ui_ests: npt.NDArray[np.float64] = array(param_vals[AmiName("cdr_ui")])
                    ui_ests = ui_ests.repeat(len(t) // len(ui_ests))
                    ui_ests.resize(len(t))
                else:
                    ui_ests = zeros(len(t))
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    ctle_out, _, ctle_h, ctle_out_h, msg, _ = run_ami_model(
                        self.rx_model, self.rx_cfg, False, ui, ts, tx_out_h, x
                    )
                    logger.info(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out += noise
                else:  # PyBERT native Rx
                    if ctle_enable:
                        ctle_h = get_ctle_h()
                        ctle_out_h = convolve(tx_out_h, ctle_h)[: len(tx_out_h)]
                        ctle_out = convolve(x + noise, ctle_out_h)[: len(x)]
                    else:
                        ctle_h = array([1.0] + [0.0] * (min_len - 1))
                        ctle_out_h = tx_out_h
                        ctle_out = rx_in
    except Exception as err:
        logger.error(f"Exception: {err}")
        raise

    # Calculate the remaining responses from the impulse responses.
    if ctle_s is None:
        ctle_s, ctle_p, ctle_H = calc_resps(t, ctle_h, ui, f)
    else:
        _, ctle_p, ctle_H = calc_resps(t, ctle_h, ui, f)
    ctle_out_s, ctle_out_p, ctle_out_H = calc_resps(t, ctle_out_h, ui, f)

    # Calculate convolutional delay.
    ctle_out.resize(len(t), refcheck=False)
    ctle_out_h_main_lobe = where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
    if ctle_out_h_main_lobe.size:
        conv_dly_ix = ctle_out_h_main_lobe[0]
    else:
        conv_dly_ix = int(self.chnl_dly // Ts)
    conv_dly = t[conv_dly_ix]

    # Stash needed intermediate results, as instance variables.
    self.tx_h = tx_h
    self.tx_s = tx_s
    self.tx_p = tx_p
    self.tx_H = tx_H
    self.tx_out_h = tx_out_h
    self.tx_out_s = tx_out_s
    self.tx_out_p = tx_out_p
    self.tx_out_H = tx_out_H
    self.ideal_signal = ideal_signal
    self.rx_in = rx_in
    self.ctle_h = ctle_h
    self.ctle_s = ctle_s
    self.ctle_p = ctle_p
    self.ctle_H = ctle_H
    self.ctle_out_h = ctle_out_h
    self.ctle_out_s = ctle_out_s
    self.ctle_out_p = ctle_out_p
    self.ctle_out_H = ctle_out_H
    self.ctle_out = ctle_out
    self.conv_dly = conv_dly
    self.conv_dly_ix = conv_dly_ix

    perf.ctle = nbits * nspb / (clock() - split_time)
    split_time = clock()
    logger.info("Running DFE/CDR...")

    _check_sim_status()

    # DFE output and incremental/cumulative responses.
    if any(tap.enabled for tap in dfe_tap_tuners):
        _gain = gain
        _ideal = self.sum_ideal
        _n_taps = len(dfe_tap_tuners)
    else:
        _gain = 0.0
        _ideal = True
        _n_taps = 0
    limits = []
    for tuner in self.dfe_tap_tuners:
        if tuner.enabled:
            limits.append((tuner.min_val, tuner.max_val))
        else:
            limits.append((0.0, 0.0))
    dfe = DFE(
        _n_taps,
        _gain,
        delta_t,
        alpha,
        ui,
        nspui,
        decision_scaler,
        mod_type,
        n_ave=n_ave,
        n_lock_ave=n_lock_ave,
        rel_lock_tol=rel_lock_tol,
        lock_sustain=lock_sustain,
        bandwidth=bandwidth,
        ideal=_ideal,
        limits=limits,
    )
    if not (self.rx_use_ami and self.rx_use_getwave):  # Use PyBERT native DFE/CDR.
        (dfe_out, tap_weights, ui_ests, clocks, lockeds, sample_times, bits_out) = dfe.run(t, ctle_out)
    else:  # Process Rx IBIS-AMI GetWave() output.
        # Process any valid clock times returned by Rx IBIS-AMI model's GetWave() function if apropos.
        dfe_out = array(ctle_out)  # In this case, `ctle_out` includes the effects of IBIS-AMI DFE.
        dfe_out.resize(len(t))
        t_ix = 0
        bits_out = []
        clocks = zeros(len(t))
        sample_times = []
        if self.rx_use_clocks and clock_times is not None:
            for clock_time in clock_times:
                if clock_time == -1:  # "-1" is used to flag "no more valid clock times".
                    break
                sample_time = clock_time + ui / 2  # IBIS-AMI clock times are edge aligned.
                while t_ix < len(t) and t[t_ix] < sample_time:
                    t_ix += 1
                if t_ix >= len(t):
                    logger.warning("Went beyond system time vector end searching for next clock time!")
                    break
                _, _bits = dfe.decide(ctle_out[t_ix])
                bits_out.extend(_bits)
                clocks[t_ix] = 1
                sample_times.append(sample_time)
        # Process any remaining output, using inferred sampling instants.
        if t_ix < (len(t) - 5 * nspui / 4):
            # Starting at `nspui/4` handles either case:
            #   - starting at UI boundary, or
            #   - starting at last sampling instant.
            next_sample_ix = (
                t_ix + nspui // 4 + argmax([sum(abs(ctle_out[t_ix + nspui // 4 + k :: nspui])) for k in range(nspui)])
            )
            for t_ix in range(next_sample_ix, len(t), nspui):
                _, _bits = dfe.decide(ctle_out[t_ix])
                bits_out.extend(_bits)
                clocks[t_ix] = 1
                sample_times.append(t[t_ix])
    bits_out = array(bits_out)
    start_ix = max(0, len(bits_out) - eye_bits)
    end_ix = len(bits_out)
    auto_corr = (
        1.0
        * correlate(bits_out[start_ix:end_ix], bits[start_ix:end_ix], mode="same")  # noqa: W504
        / sum(bits[start_ix:end_ix])
    )
    auto_corr = auto_corr[len(auto_corr) // 2 :]
    self.auto_corr = auto_corr
    bit_dly = where(auto_corr == max(auto_corr))[0][0]
    bits_ref = bits[(nbits - eye_bits) :]
    bits_tst = bits_out[(nbits + bit_dly - eye_bits) :]
    if len(bits_ref) > len(bits_tst):
        bits_ref = bits_ref[: len(bits_tst)]
    elif len(bits_tst) > len(bits_ref):
        bits_tst = bits_tst[: len(bits_ref)]
    bit_errs = where(bits_tst ^ bits_ref)[0]
    n_errs = len(bit_errs)
    # if n_errs and False:  # pylint: disable=condition-evals-to-constant
    #     logger.error(f"pybert.models.bert.my_run_simulation(): Bit errors detected at indices: {bit_errs}.")
    self.bit_errs = n_errs

    if len(tap_weights) > 0:
        dfe_h = array(
            [1.0]
            + list(zeros(nspui - 1))  # noqa: W504
            + sum([[-x] + list(zeros(nspui - 1)) for x in tap_weights[-1]], [])
        )  # sum as concat
        dfe_h.resize(len(ctle_out_h), refcheck=False)
    else:
        dfe_h = array([1.0] + list(zeros(nspui - 1)))
    dfe_out_h = convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]

    # Calculate the remaining responses from the impulse responses.
    dfe_s, dfe_p, dfe_H = calc_resps(t, dfe_h, ui, f)
    dfe_out_s, dfe_out_p, dfe_out_H = calc_resps(t, dfe_out_h, ui, f)

    self.dfe_h = dfe_h
    self.dfe_s = dfe_s
    self.dfe_p = dfe_p
    self.dfe_H = dfe_H
    self.dfe_out_h = dfe_out_h
    self.dfe_out_s = dfe_out_s
    self.dfe_out_p = dfe_out_p
    self.dfe_out_H = dfe_out_H
    self.dfe_out = dfe_out
    self.lockeds = lockeds

    perf.dfe = nbits * nspb / (clock() - split_time)
    split_time = clock()
    logger.info("Analyzing jitter...")

    _check_sim_status()

    # Save local variables to class instance for state preservation, performing unit conversion where necessary.
    self.adaptation = tap_weights
    self.ui_ests = array(ui_ests) * 1.0e12  # (ps)
    self.clocks = clocks
    self.lockeds = lockeds
    self.clock_times = sample_times

    # Analyze the jitter.
    self.thresh_tx = array([])
    self.jitter_ext_tx = array([])
    self.jitter_tx = array([])
    self.jitter_spectrum_tx = array([])
    self.jitter_ind_spectrum_tx = array([])
    self.thresh_ctle = array([])
    self.jitter_ext_ctle = array([])
    self.jitter_ctle = array([])
    self.jitter_spectrum_ctle = array([])
    self.jitter_ind_spectrum_ctle = array([])
    self.thresh_dfe = array([])
    self.jitter_ext_dfe = array([])
    self.jitter_dfe = array([])
    self.jitter_spectrum_dfe = array([])
    self.jitter_ind_spectrum_dfe = array([])
    self.f_MHz_dfe = array([])
    self.jitter_rejection_ratio = array([])

    # The pattern length must be doubled in the duo-binary and PAM-4 cases anyway, because:
    #  - in the duo-binary case, the XOR pre-coding can invert every other pattern rep., and
    #  - in the PAM-4 case, the bits are taken in pairs to form the symbols and we start w/ an odd # of bits.
    # So, while it isn't strictly necessary, doubling it in the NRZ case as well provides a certain consistency.
    pattern_len = (pow(2, max(pattern)) - 1) * 2
    len_x_m1 = len(x) - 1
    xing_min_t = (nui - eye_uis) * ui

    def eye_xings(xings, ofst=0) -> NDArray[float64]:
        """
        Return crossings from that portion of the signal used to generate the eye.

        Args:
            xings([float]): List of crossings.

        Keyword Args:
            ofst(float): Time offset to be subtracted from all crossings.

        Returns:
            [float]: Selected crossings, offset and eye-start corrected.
        """
        _xings = array(xings) - ofst
        return _xings[where(_xings > xing_min_t)] - xing_min_t

    try:
        # - ideal
        ideal_xings = find_crossings(t, ideal_signal, decision_scaler, mod_type=mod_type)
        self.ideal_xings = ideal_xings
        ideal_xings_jit = eye_xings(ideal_xings)

        # - channel output
        ofst = (argmax(sig.correlate(chnl_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.t_jitter = t_jitter
        self.isi_chnl = isi
        self.dcd_chnl = dcd
        self.pj_chnl = pj
        self.rj_chnl = rj
        self.pjDD_chnl = pjDD
        self.rjDD_chnl = rjDD
        self.mu_pos_chnl = mu_pos
        self.mu_neg_chnl = mu_neg
        self.thresh_chnl = thresh
        self.jitter_chnl = hist
        self.jitter_ext_chnl = hist_synth
        self.jitter_bins = bin_centers
        self.jitter_spectrum_chnl = jitter_spectrum
        self.jitter_ind_spectrum_chnl = jitter_ind_spectrum
        self.f_MHz = array(spectrum_freqs) * 1.0e-6
        self.ofst_chnl = ofst
        self.tie_chnl = tie
        self.tie_ind_chnl = tie_ind

        # - Tx output
        ofst = (argmax(sig.correlate(rx_in, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_tx = isi
        self.dcd_tx = dcd
        self.pj_tx = pj
        self.rj_tx = rj
        self.pjDD_tx = pjDD
        self.rjDD_tx = rjDD
        self.mu_pos_tx = mu_pos
        self.mu_neg_tx = mu_neg
        self.thresh_tx = thresh
        self.jitter_tx = hist
        self.jitter_ext_tx = hist_synth
        self.jitter_centers_tx = bin_centers
        self.jitter_spectrum_tx = jitter_spectrum
        self.jitter_ind_spectrum_tx = jitter_ind_spectrum
        self.jitter_freqs_tx = spectrum_freqs
        self.t_jitter_tx = t_jitter
        self.tie_tx = tie
        self.tie_ind_tx = tie_ind

        # - CTLE output
        ofst = (argmax(sig.correlate(ctle_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_ctle = isi
        self.dcd_ctle = dcd
        self.pj_ctle = pj
        self.rj_ctle = rj
        self.pjDD_ctle = pjDD
        self.rjDD_ctle = rjDD
        self.mu_pos_ctle = mu_pos
        self.mu_neg_ctle = mu_neg
        self.thresh_ctle = thresh
        self.jitter_ctle = hist
        self.jitter_ext_ctle = hist_synth
        self.jitter_spectrum_ctle = jitter_spectrum
        self.jitter_ind_spectrum_ctle = jitter_ind_spectrum
        self.tie_ctle = tie
        self.tie_ind_ctle = tie_ind

        # - DFE output
        ofst = (argmax(sig.correlate(dfe_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, dfe_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh, dbg_obj=self)
        self.isi_dfe = isi
        self.dcd_dfe = dcd
        self.pj_dfe = pj
        self.rj_dfe = rj
        self.pjDD_dfe = pjDD
        self.rjDD_dfe = rjDD
        self.mu_pos_dfe = mu_pos
        self.mu_neg_dfe = mu_neg
        self.thresh_dfe = thresh
        self.jitter_dfe = hist
        self.jitter_ext_dfe = hist_synth
        self.jitter_spectrum_dfe = jitter_spectrum
        self.jitter_ind_spectrum_dfe = jitter_ind_spectrum
        self.tie_dfe = tie
        self.tie_ind_dfe = tie_ind
        self.f_MHz_dfe = array(spectrum_freqs) * 1.0e-6
        dfe_spec = self.jitter_spectrum_dfe
        self.jitter_rejection_ratio = zeros(len(dfe_spec))

        perf.jitter = nbits * nspb / (clock() - split_time)
    except ValueError as err:
        logger.error(f"The jitter calculation could not be completed, due to the following error:\n{err}")
        # raise

    split_time = clock()
    logger.info("Generating plot data...")

    plotting_data = calculate_plotting_data(self)
    perf.plotting = nbits * nspb / (clock() - split_time)

    perf.total = nbits * nspb / (clock() - perf.start_time)
    perf.end_time = clock()
    logger.info(f"Simulation complete. Duration: {round(perf.end_time - perf.start_time, 3)} s")
    logger.info(str(perf))

    return plotting_data, perf
