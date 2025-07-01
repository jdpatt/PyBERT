#! /usr/bin/env python

# pylint: disable=too-many-lines

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.

ToDo:
    1. Add optional AFE (4th-order Bessel-Thomson).
    2. Add eye contour plots.
"""

import logging
import platform
import queue
import time
from concurrent.futures import Future
from datetime import datetime
from os.path import dirname, join
from pathlib import Path
from typing import Callable, Optional

import numpy as np  # type: ignore
import skrf as rf
from numpy import arange, array, cos, exp, pad, pi, sinc, where, zeros
from numpy.fft import irfft, rfft  # type: ignore
from numpy.random import randint  # type: ignore
from pyibisami import AMIModel, AMIParamConfigurator, IBISModel
from pyibisami import __version__ as PyAMI_VERSION  # type: ignore
from scipy.interpolate import interp1d

from pybert import __version__ as VERSION
from pybert.configuration import Configuration, InvalidConfigFileType
from pybert.constants import gPeakFreq, gPeakMag
from pybert.models.bert import SimulationThread
from pybert.models.buffer import Receiver, Transmitter
from pybert.models.channel import Channel
from pybert.models.stimulus import BitPattern, ModulationType
from pybert.models.tx_tap import TxTapTuner
from pybert.optimization import OptThread
from pybert.results import Results
from pybert.utility import (
    calc_gamma,
    import_channel,
    lfsr_bits,
    raised_cosine,
    safe_log10,
    sdd_21,
    trim_impulse,
)
from pybert.utility.jitter import JitterAnalysis

logger = logging.getLogger("pybert")


class PyBERT:  # pylint: disable=too-many-instance-attributes
    """A serial communication link bit error rate tester (BERT) simulator with a GUI interface.

    Useful for exploring the concepts of serial communication link design.
    """

    def __init__(
        self,
        run_simulation: bool = False,
        tx: Transmitter = Transmitter(),
        channel: Channel = Channel(),
        rx: Receiver = Receiver(),
    ) -> None:
        """Initialize the PyBERT class.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
        """
        # Independent variables
        self.tx = tx
        self.channel = channel
        self.rx = rx

        # - Simulation Control
        self.bit_rate: float = 10.0  #: (Gbps)
        self.nbits: int = 15000  #: Number of bits to simulate.
        self.eye_bits: int = 10160  #: Number of bits used to form eye.
        self.pattern: BitPattern = BitPattern.PRBS7  #: Pattern to use for simulation.
        self.seed: int = 1  # LFSR seed. 0 means regenerate bits, using a new random seed, each run.
        self.nspui: int = 32  #: Signal vector samples per unit interval.
        self.mod_type: ModulationType = ModulationType.NRZ  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
        self.thresh: float = 3.0  #: Spectral threshold for identifying periodic components (sigma). (Default = 3.0)

        # - EQ Tune
        self.tx_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Pre-tap3", pos=-3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
            TxTapTuner(name="Pre-tap2", pos=-2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Pre-tap1", pos=-1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap1", pos=1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap2", pos=2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Post-tap3", pos=3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
        ]  #: EQ optimizer list of TxTapTuner objects.
        self.rx_bw_tune: float = 12.0  #: EQ optimizer CTLE bandwidth (GHz).
        self.peak_freq_tune: float = gPeakFreq  #: EQ optimizer CTLE peaking freq. (GHz).
        self.peak_mag_tune: float = gPeakMag  #: EQ optimizer CTLE peaking mag. (dB).
        self.min_mag_tune: float = 2  #: EQ optimizer CTLE peaking mag. min. (dB).
        self.max_mag_tune: float = 12  #: EQ optimizer CTLE peaking mag. max. (dB).
        self.step_mag_tune: float = 1  #: EQ optimizer CTLE peaking mag. step (dB).
        self.ctle_enable_tune: bool = True  #: EQ optimizer CTLE enable
        self.dfe_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Tap1", enabled=True, min_val=0.1, max_val=0.4, value=0.1),
            TxTapTuner(name="Tap2", enabled=True, min_val=-0.15, max_val=0.15, value=0.0),
            TxTapTuner(name="Tap3", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap4", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap5", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap6", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap7", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap8", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap9", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap10", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap11", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap12", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap13", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap14", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap15", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap16", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap17", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap18", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap19", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap20", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
        ]  #: EQ optimizer list of DFE tap tuner objects.

        # - Tx
        self.vod: float = 1.0  #: Tx differential output voltage (V)
        self.pn_mag: float = 0.1  #: Periodic noise magnitude (V).
        self.pn_freq: float = 11  #: Periodic noise frequency (MHz).
        self.rn: float = 0.1  #: Standard deviation of Gaussian random noise (V).
        self.tx_taps: list[TxTapTuner] = [
            TxTapTuner(name="Pre-tap3", pos=-3, enabled=True, min_val=-0.05, max_val=0.05),
            TxTapTuner(name="Pre-tap2", pos=-2, enabled=True, min_val=-0.1, max_val=0.1),
            TxTapTuner(name="Pre-tap1", pos=-1, enabled=True, min_val=-0.2, max_val=0.2),
            TxTapTuner(name="Post-tap1", pos=1, enabled=True, min_val=-0.2, max_val=0.2),
            TxTapTuner(name="Post-tap2", pos=2, enabled=True, min_val=-0.1, max_val=0.1),
            TxTapTuner(name="Post-tap3", pos=3, enabled=True, min_val=-0.05, max_val=0.05),
        ]  #: List of TxTapTuner objects.
        self.rel_power: float = 1.0  #: Tx power dissipation (W).

        # - Rx
        self.use_ctle_file: bool = False  #: For importing CTLE impulse/step response directly.
        self.ctle_file: str = ""  #: CTLE response file (when use_ctle_file = True).
        self.rx_bw: float = 12.0  #: CTLE bandwidth (GHz).
        self.peak_freq: float = gPeakFreq  #: CTLE peaking frequency (GHz)
        self.peak_mag: float = gPeakMag  #: CTLE peaking magnitude (dB)
        self.ctle_enable: bool = True  #: CTLE enable.
        self.rx_use_viterbi: bool = False  #: Use Viterbi algorithm for MLSD.
        self.rx_viterbi_symbols: int = 4  #: Number of symbols to track in Viterbi decoder.

        # - DFE
        self.sum_ideal: bool = True  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
        self.decision_scaler: float = 0.5  #: DFE slicer output voltage (V).
        self.gain: float = 0.2  #: DFE error gain (unitless).
        self.n_ave: int = 100  #: DFE # of averages to take, before making tap corrections.
        self.sum_bw: float = 12.0  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).
        self.use_agc: bool = False  #: Continuously adjust ``decision_scalar`` when True.

        # - CDR
        self.delta_t: float = 0.1  #: CDR proportional branch magnitude (ps).
        self.alpha: float = 0.01  #: CDR integral branch magnitude (unitless).
        self.n_lock_ave: int = 500  #: CDR # of averages to take in determining lock.
        self.rel_lock_tol: float = 0.1  #: CDR relative tolerance to use in determining lock.
        self.lock_sustain: int = 500  #: CDR hysteresis to use in determining lock.

        # Misc.

        # Status
        self.len_h: float = 0
        self.chnl_dly: float = 0.0  #: Estimated channel delay (s).

        # Initialize jitter analysis objects
        self.chnl_jitter: JitterAnalysis | None = None
        self.tx_jitter: JitterAnalysis | None = None
        self.ctle_jitter: JitterAnalysis | None = None
        self.dfe_jitter: JitterAnalysis | None = None

        # Threading and Processing
        self.simulation_thread: Optional[SimulationThread] = None  # Simulation Thread
        self.opt_thread: Optional[OptThread] = None  #: EQ optimization thread.

        # Add callback-based result handling
        self._simulation_callbacks: list[Callable[[Results], None]] = []
        self._optimization_callbacks: list[Callable[[dict], None]] = []
        self._optimization_loop_callbacks: list[Callable[[dict], None]] = []
        self._status_callbacks: list[Callable[[str], None]] = []

        # Add futures for blocking operations
        self._simulation_future: Optional[Future] = None
        self._optimization_future: Optional[Future] = None

        self.last_results: Results | None = None

        if run_simulation:
            self.simulate()

    def add_simulation_callback(self, callback: Callable[[Results], None]) -> None:
        """Add a callback to be called when simulation completes.

        Args:
            callback: Function that takes Results as arguments
        """
        self._simulation_callbacks.append(callback)

    def add_optimization_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called when optimization completes.

        Args:
            callback: Function that takes optimization result dict as argument
        """
        self._optimization_callbacks.append(callback)

    def add_optimization_loop_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called during optimization loop.

        Args:
            callback: Function that takes optimization loop result dict as argument
        """
        self._optimization_loop_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback to be called for status updates.

        Args:
            callback: Function that takes status message string as argument
        """
        self._status_callbacks.append(callback)

    def _notify_simulation_complete(self, results: Results) -> None:
        """Notify all simulation callbacks with results."""
        # Store results for non-blocking access
        self.last_results = results

        # Resolve the future if it exists (for blocking operations)
        if self._simulation_future and not self._simulation_future.done():
            self._simulation_future.set_result(results)
            self._simulation_future = None

        # Call all callbacks
        for callback in self._simulation_callbacks:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"Error in simulation callback: {e}")

    def _notify_optimization_complete(self, result: dict) -> None:
        """Notify all optimization callbacks with result."""
        # Resolve the future if it exists (for blocking operations)
        if self._optimization_future and not self._optimization_future.done():
            self._optimization_future.set_result(result)
            self._optimization_future = None

        # Call all callbacks
        for callback in self._optimization_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in optimization callback: {e}")

    def _notify_optimization_loop_complete(self, result: dict) -> None:
        """Notify all optimization loop callbacks with result."""
        for callback in self._optimization_loop_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in optimization loop callback: {e}")

    def _notify_status_update(self, message: str) -> None:
        """Notify all status callbacks with message."""
        for callback in self._status_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    # Dependent variable definitions
    @property
    def t(self):
        """Calculate the system time vector, in seconds."""

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @property
    def t_ns(self):
        """Calculate the system time vector, in ns."""

        return self.t * 1.0e9

    @property
    def f(self):
        """
        Calculate the frequency vector for channel model construction.
        """
        fstep = self.channel.f_step * 1e6
        fmax = self.channel.f_max * 1e9
        return arange(0, fmax + fstep, fstep)  # "+fstep", so fmax gets included

    @property
    def w(self):
        """
        Channel modeling frequency vector, in rads./sec.
        """
        return 2 * pi * self.f

    @property
    def t_irfft(self):
        """
        Calculate the time vector appropriate for indexing `irfft()` output.
        """
        f = self.f
        tmax = 1 / f[1]
        tstep = 0.5 / f[-1]
        return arange(0, tmax, tstep)

    @property
    def bits(self):
        "Generate the bit stream."
        pattern = self.pattern.value
        seed = self.seed
        nbits = self.nbits

        if not seed:  # The user sets `seed` to zero when she wants a new random seed generated for each run.
            seed = randint(1, 128)  # Use range to avoid zero directly
        bit_gen = lfsr_bits(pattern, seed)
        bits = np.fromiter((next(bit_gen) for _ in range(nbits)), dtype=int)
        return bits

    @property
    def ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """

        mod_type = self.mod_type
        bit_rate = self.bit_rate * 1.0e9

        ui = 1.0 / bit_rate
        if mod_type == ModulationType.PAM4:  # PAM-4
            ui *= 2.0

        return ui

    @property
    def nui(self):
        """Returns the number of unit intervals in the test vectors."""

        mod_type = self.mod_type
        nbits = self.nbits

        nui = nbits
        if mod_type == ModulationType.PAM4:  # PAM-4
            nui //= 2

        return nui

    @property
    def eye_uis(self):
        """Returns the number of unit intervals to use for eye construction."""

        mod_type = self.mod_type
        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if mod_type == ModulationType.PAM4:  # PAM-4
            eye_uis //= 2

        return eye_uis

    @property
    def ideal_h(self):
        """Returns the ideal link impulse response."""

        ui = self.ui.value
        nspui = self.nspui
        t = self.t
        mod_type = self.mod_type
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
            raise ValueError("PyBERT._get_ideal_h(): ERROR: Unrecognized ideal impulse response type.")

        if (
            mod_type == ModulationType.DUO
        ):  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (ideal_h + pad(ideal_h[: -1 * nspui], (nspui, 0), "constant", constant_values=(0, 0)))

        return ideal_h

    @property
    def symbols(self):
        """Generate the symbol stream."""

        mod_type = self.mod_type
        vod = self.vod
        bits = self.bits

        if mod_type == ModulationType.NRZ:  # NRZ
            symbols = 2 * bits - 1
        elif mod_type == ModulationType.DUO:  # Duo-binary
            symbols = [bits[0]]
            for bit in bits[1:]:  # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = 2 * array(symbols) - 1
        elif mod_type == ModulationType.PAM4:  # PAM-4
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
            raise ValueError("ERROR: _get_symbols(): Unknown modulation type requested!")

        return array(symbols) * vod

    @property
    def ffe(self):
        """Generate the Tx pre-emphasis FIR numerator."""

        tap_tuners = self.tx_taps

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        curs_pos = -tap_tuners[0].pos
        curs_val = 1.0 - sum(abs(array(taps)))
        if curs_pos < 0:
            taps.insert(0, curs_val)
        else:
            taps.insert(curs_pos, curs_val)

        return taps

    # This function has been pulled outside of the standard Traits/UI "depends_on / @property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    # pylint: disable=attribute-defined-outside-init
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

        t = self.t  # This time vector has NO relationship to `f`/`w`!
        t_irfft = self.t_irfft  # This time vector IS related to `f`/`w`.
        f = self.f
        w = self.w
        nspui = self.nspui
        impulse_length = self.channel.impulse_length * 1.0e-9
        Rs = self.tx.impedance
        Cs = self.tx.capacitance * 1.0e-12
        RL = self.rx.impedance
        Cp = self.rx.capacitance * 1.0e-12
        # CL = self.cac * 1.0e-6  # pylint: disable=unused-variable

        ts = t[1]
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        H, ch_s2p_pre = self.channel.form_channel_response(ts, f, w, len_f, self.tx.impedance)

        self.ch_s2p_pre = ch_s2p_pre
        ch_s2p = ch_s2p_pre  # In case neither set of on-die S-parameters is being invoked, below.

        # Augment w/ IBIS-AMI on-die S-parameters, if appropriate.
        def add_ondie_s(s2p, ts4f, isRx=False):
            """Add the effect of on-die S-parameters to channel network.

            Args:
                s2p(skrf.Network): initial 2-port network.
                ts4f(string): on-die S-parameter file name.

            Keyword Args:
                isRx(bool): True when Rx on-die S-params. are being added. (Default = False).

            Returns:
                skrf.Network: Resultant 2-port network.
            """
            ts4N = rf.Network(ts4f)  # Grab the 4-port single-ended on-die network.
            ntwk = sdd_21(ts4N)  # Convert it to a differential, 2-port network.
            # Interpolate to system freqs.
            ntwk2 = (
                ntwk.extrapolate_to_dc()
                .windowed(normalize=False)
                .interpolate(s2p.f, coords="polar", bounds_error=False, fill_value="extrapolate")
            )
            if isRx:
                res = s2p**ntwk2
            else:  # Tx
                res = ntwk2**s2p
            return (res, ts4N, ntwk2)

        if self.tx.use_ibis:
            model = self.tx.ibis.current_model
            Rs = model.impedance * 2
            Cs = model.ccomp[0] / 2  # They're in series.
            self.Rs = Rs  # Primarily for debugging.
            self.Cs = Cs
            if self.tx.use_ts4:
                fname = join(self.tx.ibis_dir, self._tx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname)
                self.ts4N = ts4N
                self.ntwk = ntwk
        if self.rx.use_ibis:
            model = self.rx.ibis.current_model
            RL = model.impedance * 2
            Cp = model.ccomp[0] / 2
            self.RL = RL  # Primarily for debugging.
            self.Cp = Cp
            logger.debug(f"RL: {round(RL, 2)}, Cp: {round(Cp, 2)}")
            if self.rx.use_ts4:
                fname = join(self.rx.ibis_dir, self._rx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
                self.ts4N = ts4N
                self.ntwk = ntwk
        ch_s2p.name = "ch_s2p"
        self.ch_s2p = ch_s2p

        # Calculate channel impulse response.
        Zs = Rs / (1 + 1j * w * Rs * Cs)  # Tx termination impedance
        Zt = RL / (1 + 1j * w * RL * Cp)  # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:, 0] = Zs
        ch_s2p_term_z0[:, 1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term

        # We take the transfer function, H, to be a ratio of voltages.
        # So, we must normalize our (now generalized) S-parameters.
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:, 1] / ch_s2p_term.z0[:, 0])
        if self.channel.use_window:
            chnl_h = irfft(raised_cosine(chnl_H))
        else:
            chnl_h = irfft(chnl_H)
        krnl = interp1d(t_irfft, chnl_h, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        temp = krnl(t)
        chnl_h = temp * t[1] / t_irfft[1]
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = int(impulse_length / ts)
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len, front_porch=True, kept_energy=0.999)
        krnl = interp1d(t[: len(chnl_h)], chnl_h, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        chnl_trimmed_H = rfft(krnl(t_irfft)) * t_irfft[1] / t[1]

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(
            chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)
        )  # pylint: disable=invalid-unary-operand-type

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_H_raw = H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    def load_configuration(self, filepath: Path | str):
        """Load in a configuration into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        Configuration.load_from_file(filepath, self)

    def save_configuration(self, filepath: Path | str):
        """Save out a configuration from pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        Configuration(self).save(filepath)

    def reset_configuration(self) -> None:
        """Reset the PyBERT instance to default configuration values."""
        Configuration.apply_default_config(self)

    def load_results(self, filepath: Path) -> Results | None:
        """Load results from a file into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        self.last_results = Results.load_from_file(filepath)
        return self.last_results

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file.

        Args:
            filepath: A full filepath include the suffix.
        """
        if self.last_results:
            self.last_results.save(filepath)

    def simulate(self, block: bool = False, timeout: int = 180) -> Results | None:
        """Start a simulation of the current configuration in a separate thread.

        Args:
            block: If True, wait for the simulation to complete and return results.
            timeout: Maximum time to wait for simulation completion in seconds.

        Returns:
            If block is True, returns the simulation results. Otherwise returns None.
        """
        if self.simulation_thread and self.simulation_thread.is_alive():
            if block and self._simulation_future:
                # Wait for the current simulation to complete
                return self._simulation_future.result(timeout=timeout)
            return None
        elif self.is_valid_configuration():
            logger.info("Starting simulation.")

            # Create a future for blocking operations
            if block:
                self._simulation_future = Future()

            self.simulation_thread = SimulationThread()
            self.simulation_thread.pybert = self
            self.simulation_thread.start()

            if block:
                # Wait for the future to be resolved by the callback
                return self._simulation_future.result(timeout=timeout)
        return None

    def stop_simulation(self):
        """Stop the running simulation."""
        logger.info("Stopping simulation.")
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.stop()
            self.simulation_thread.join(10)

    def calculate_optimization_trials(self):
        """Calculate the number of trials for the optimization."""
        n_trials = int((self.max_mag_tune - self.min_mag_tune) / self.step_mag_tune)
        for tuner in self.tx_tap_tuners:
            n_trials *= int((tuner.max_val - tuner.min_val) / tuner.step)
        return n_trials

    def optimize(self, block: bool = False, timeout: int = 180):
        """Start the optimization process using the tuner values.

        Args:
            block: If True, wait for the optimization to complete and return results.
            timeout: Maximum time to wait for optimization completion in seconds.

        Returns:
            If block is True, returns the optimization results. Otherwise returns None.
        """
        if self.opt_thread and self.opt_thread.is_alive():
            if block and self._optimization_future:
                return self._optimization_future.result(timeout=timeout)
            return None
        elif self.is_valid_configuration():
            logger.info("Starting optimization.")

            # Create a future for blocking operations
            if block:
                self._optimization_future = Future()

            self.opt_thread = OptThread()
            self.opt_thread.pybert = self
            self.opt_thread.start()

            if block:
                return self._optimization_future.result(timeout=timeout)
        return None

    def stop_optimization(self):
        """Stop the running optimization."""
        logger.info("Stopping optimization.")
        if self.opt_thread and self.opt_thread.is_alive():
            self.opt_thread.stop()
            self.opt_thread.join(10)

    def reset_optimization(self):
        """Reset the optimization back to what the current configuration is."""
        logger.info("Resetting optimization.")
        for i, tap in enumerate(self.tx_taps):
            self.tx_tap_tuners[i].value = tap.value
            self.tx_tap_tuners[i].enabled = tap.enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_enable_tune = self.ctle_enable

    def apply_optimization(self):
        """Apply the optimization to the current configuration."""
        logger.info("Applying optimization.")
        for i, tap in enumerate(self.tx_tap_tuners):
            self.tx_taps[i].value = tap.value
            self.tx_taps[i].enabled = tap.enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_enable = self.ctle_enable_tune

    def is_valid_configuration(self):
        """Validate that the user has selected a valid configuration for simulation or optimization."""
        if not self.channel.elements and self.channel.use_ch_file:
            logger.error("No channel file selected. Please select a channel file.")
            return False
        if not self.tx.ibis and self.tx.use_ibis:
            logger.error("No Tx IBIS file selected. Please select a Tx IBIS file.")
            return False
        if not self.rx.ibis and self.rx.use_ibis:
            logger.error("No Rx IBIS file selected. Please select a Rx IBIS file.")
            return False
        if not self.tx.ami and self.tx.model and self.tx.use_ami:
            logger.error("No Tx AMI loaded or configured.")
            return False
        if not self.rx.ami and self.rx.model and self.rx.use_ami:
            logger.error("No Tx AMI loaded or configured.")
            return False
        return True
