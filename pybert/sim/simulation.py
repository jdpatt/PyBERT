"""Where all the magic happens."""
from collections import defaultdict
from functools import lru_cache
from logging import getLogger
from time import clock

import numpy as np
from numpy.fft import fft, ifft
from numpy.random import normal, randint
from pybert.defaults import (
    BIT_RATE,
    HPF_CORNER_COUPLING,
    MIN_BATHTUB_VAL,
    MODULATION,
    NUM_AVG,
    NUM_BITS,
    NUM_TAPS,
    PATTERN_LEN,
    SAMPLES_PER_BIT,
    THRESHOLD,
)
from pybert.sim.buffer import Receiver, Transmitter
from pybert.sim.channel import Channel
from pybert.sim.dfe import DFE
from pybert.sim.equalization import Equalization
from pybert.sim.jitter import Jitter, calculate_jitter_rejection
from pybert.sim.utility import (
    calc_eye,
    calc_G,
    calc_gamma,
    find_crossings,
    import_channel,
    lfsr_bits,
    make_ctle,
    pulse_center,
    status_string,
    trim_impulse,
)
from PySide2.QtCore import QObject, Signal, Slot
from scipy.signal import iirfilter, lfilter
from scipy.signal.windows import hann


class Simulation(QObject):
    """The main object in pybert containing the simulator."""

    eq_results = Signal(object, object, object)
    sweep_results = Signal()
    status_update = Signal(str)
    sim_done = Signal(dict)
    metrics = Signal(dict)

    def __init__(self):
        super(Simulation, self).__init__()
        self.log = getLogger("pybert.simulation")
        self.log.debug("Initializing Simulation")
        self._status = "Ready"

        self.bit_rate = BIT_RATE  #: (Gbps)
        self.nbits = NUM_BITS  #: Number of bits to simulate.
        self.pattern_len = PATTERN_LEN  #: PRBS pattern length.
        self.nspb = SAMPLES_PER_BIT  #: Signal vector samples per bit.
        self.eye_bits = NUM_BITS // 5  #: # of bits used to form eye. (Default = last 20%)
        self.mod_type = MODULATION.NRZ  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
        self.num_sweeps = 1  #: Number of sweeps to run.
        self.sweep_num = 1
        self.sweep_aves = NUM_AVG
        self.do_sweep = False  #: Run sweeps? (Default = False)

        self.run_count = 0  # Used as a mechanism to force bit stream regeneration.
        self.thresh = THRESHOLD

        self.channel = Channel()
        self.tx = Transmitter(self.channel.material.random_noise)
        self.rx = Receiver()
        self.eq = Equalization()

        # The outputs of the simulation
        self.performance = {}
        self.results = defaultdict(dict)
        self.sweep_results = []

        # Variables that got defined randomly throughout the simulation.  TODO: Clean up data structure.
        self.x = np.array([])
        self.rx_in = np.array([])
        self.auto_corr = np.array([])
        self.ideal_xings = np.array([])
        self.ideal_signal = np.array([])
        self.conv_dly = np.array([])
        self.conv_dly_ix = np.array([])
        self.adaptation = np.array([])
        self.ui_ests = np.array([])
        self.clocks = np.array([])
        self.lockeds = np.array([])
        self.clock_times = np.array([])

        # Dependent variables
        # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
        #
        # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
        #         This will slow the GUI down noticeably.
        self.jitter_info = ""
        self.perf_info = ""
        self.status_str = ""
        self.sweep_info = ""
        self._cost: float = 0.0
        self._rel_opt: float = 0.0
        self._t = np.array([])
        self._t_ns = np.array([])
        self._f = 0.0
        self._w = 0.0
        self._bits = np.array([])
        self._symbols = np.array([])
        self._ffe = np.array([])
        self._ui: float = 0.0
        self._nui: int = 0
        self._nspui: int = 0
        self._eye_uis: int = 0
        self._przf_err: float = 0.0

        self.chnl_dly = np.array([])
        self.start_ix = np.array([])
        self.len_h = np.array([])

    @property
    def status(self):
        """Return the status string."""
        return self._status

    @status.setter
    def status(self, message):
        """Override the status setter so that we can log all messages."""
        self._status = message
        self.log.info(message)
        self.status_update.emit(self.get_status_str())

    # Dependent variable definitions
    @property
    @lru_cache(maxsize=None)
    def t(self):
        """
        Calculate the system time vector, in seconds.

        """

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return np.array([i * t0 for i in range(npts)])

    @property
    @lru_cache(maxsize=None)
    def t_ns(self):
        """
        Calculate the system time vector, in ns.
        """

        return self.t * 1.0e9

    @property
    @lru_cache(maxsize=None)
    def f(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in Hz.
        # (i.e. - [0, f0, 2 * f0, ... , fN] + [-(fN - f0), -(fN - 2 * f0), ... , -f0]
        """

        t = self.t

        npts = len(t)
        f0 = 1.0 / (t[1] * npts)
        half_npts = npts // 2

        return np.array(
            [i * f0 for i in range(half_npts + 1)]
            + [(half_npts - i) * -f0 for i in range(1, half_npts)]
        )

    @property
    @lru_cache(maxsize=None)
    def w(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in rads./sec.
        """

        return 2 * np.pi * self.f

    @property
    @lru_cache(maxsize=None)
    def bits(self):
        """
        Generate the bit stream.
        """

        pattern_len = self.pattern_len
        nbits = self.nbits

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
        if self.mod_type == MODULATION.DUO:  # Use XOR.
            return np.resize(np.array([0, 0, 1, 0] + bits), nbits)
        return np.resize(np.array([0, 0, 1, 1] + bits), nbits)

    @property
    @lru_cache(maxsize=None)
    def ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """
        ui = 1.0 / (self.bit_rate * 1.0e9)
        if self.mod_type == MODULATION.PAM4:
            ui *= 2.0

        return ui

    @property
    @lru_cache(maxsize=None)
    def nui(self):
        """
        Returns the number of unit intervals in the test vectors.
        """

        nbits = self.nbits

        nui = nbits
        if self.mod_type == MODULATION.PAM4:
            nui //= 2

        return nui

    @property
    @lru_cache(maxsize=None)
    def nspui(self):
        """
        Returns the number of samples per unit interval.
        """

        nspb = self.nspb

        nspui = nspb
        if self.mod_type == MODULATION.PAM4:
            nspui *= 2

        return nspui

    @property
    @lru_cache(maxsize=None)
    def eye_uis(self):
        """
        Returns the number of unit intervals to use for eye construction.
        """

        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if self.mod_type == MODULATION.PAM4:
            eye_uis //= 2

        return eye_uis

    @property
    @lru_cache(maxsize=None)
    def symbols(self):
        """
        Generate the symbol stream.
        """

        vod = self.tx.vod
        bits = self.bits

        if self.mod_type == MODULATION.NRZ:
            symbols = 2 * bits - 1
        elif self.mod_type == MODULATION.DUO:
            symbols = [bits[0]]
            for bit in bits[1:]:  # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = 2 * np.array(symbols) - 1
        elif self.mod_type == MODULATION.PAM4:
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
            raise ValueError("Unknown modulation type requested!")
        return np.array(symbols) * vod

    @property
    @lru_cache(maxsize=None)
    def cost(self):
        nspui = self.nspui
        h = self.eq.ctle_out_h_tune

        s = h.cumsum()
        p = s - np.pad(s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        (clock_pos, thresh) = pulse_center(p, nspui)
        if clock_pos == -1:
            return 1.0  # Returning a large cost lets it know it took a wrong turn.
        clocks = thresh * np.ones(len(p))
        if self.mod_type == MODULATION.DUO:
            clock_pos -= nspui // 2
        clocks[clock_pos] = 0.0
        if self.mod_type == MODULATION.DUO:
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
        if self.mod_type == MODULATION.DUO:
            ix += nspui
        while ix < len(p):
            clocks[ix] = 0.0
            if not self.eq.use_dfe_tune:
                isi += abs(p[ix])
            ix += nspui
        if self.eq.use_dfe_tune:
            for i in range(self.eq.n_taps_tune):
                if clock_pos + nspui * (1 + i) < len(p):
                    p[int(clock_pos + nspui * (0.5 + i)) :] -= p[clock_pos + nspui * (1 + i)]

        # Signal the GUI that there is new data to plot
        self.eq_results.emit(self.results["t_ns_chnl"], p, clocks)

        if self.mod_type == MODULATION.DUO:
            return (
                isi
                - p[clock_pos]
                - p[clock_pos + nspui]
                + 2.0 * abs(p[clock_pos + nspui] - p[clock_pos])
            )
        return isi - p[clock_pos]

    @property
    @lru_cache(maxsize=None)
    def rel_opt(self):
        return -self.cost

    @property
    @lru_cache(maxsize=None)
    def przf_err(self):
        p = self.results["dfe"]["out_p"]
        nspui = self.nspui
        n_taps = self.eq.n_taps

        (clock_pos, _) = pulse_center(p, nspui)
        err = 0
        for i in range(n_taps):
            err += p[clock_pos + (i + 1) * nspui] ** 2

        return err / p[clock_pos] ** 2

    @lru_cache(maxsize=None)
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
        w = self.w
        nspui = self.nspui
        ts = t[1]
        impulse_length = self.channel.impulse_length * 1.0e-9

        if self.channel.use_ch_file:
            chnl_h = import_channel(
                self.channel.filename, ts, self.channel.padded, self.channel.windowed
            )
            if chnl_h[-1] > (max(chnl_h) / 2.0):  # step response?
                chnl_h = np.diff(chnl_h)  # impulse response is derivative of step response.
            chnl_h /= sum(chnl_h)  # Normalize d.c. to one.
            chnl_dly = t[np.where(chnl_h == max(chnl_h))[0][0]]
            chnl_h.resize(len(t))
            chnl_H = fft(chnl_h)
        else:
            l_ch = self.channel.material.channel_length
            rel_velocity = self.channel.material.rel_velocity * 3.0e8
            skin_effect_resistance = self.channel.material.skin_effect_resistance
            w_transition_freq = self.channel.material.w_transition_freq
            dc_resistance_per_meter = self.channel.material.dc_resistance_per_meter
            characteristic_impedance = self.channel.material.characteristic_impedance
            loss_tangent = self.channel.material.loss_tangent
            output_impedance = self.tx.output_impedance
            output_capacitance = self.tx.output_capacitance * 1.0e-12
            input_impedance = self.rx.input_impedance
            input_capacitance = self.rx.input_capacitance * 1.0e-12
            CL = self.rx.cac * 1.0e-6

            chnl_dly = l_ch / rel_velocity
            gamma, Zc = calc_gamma(
                skin_effect_resistance,
                w_transition_freq,
                dc_resistance_per_meter,
                characteristic_impedance,
                rel_velocity,
                loss_tangent,
                w,
            )
            H = np.exp(-l_ch * gamma)
            chnl_H = 2.0 * calc_G(
                H,
                output_impedance,
                output_capacitance,
                Zc,
                input_impedance,
                input_capacitance,
                CL,
                w,
            )  # Compensating for nominal /2 divider action.
            chnl_h = np.real(ifft(chnl_H))

        min_len = 10 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        chnl_h /= sum(chnl_h)  # a temporary crutch.
        temp = chnl_h.copy()
        temp.resize(len(t))
        chnl_trimmed_H = fft(temp)

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - np.pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        len_h = len(chnl_h)
        t_ns_chnl = np.array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9

        self.chnl_dly = chnl_dly
        self.start_ix = start_ix
        self.results["channel"]["chnl_trimmed_H"] = chnl_trimmed_H
        self.results["t_ns_chnl"] = t_ns_chnl
        self.results["channel"]["chnl_h"] = chnl_h
        self.results["channel"]["chnl_H"] = chnl_H
        self.results["channel"]["chnl_s"] = chnl_s
        self.results["channel"]["chnl_p"] = chnl_p
        self.len_h = len_h
        return chnl_h

    def get_status_str(self):
        return status_string(
            self.status,
            self.performance.get("total", 0.0) * 60.0e-6,
            self.channel.chnl_dly,
            self.results.get("bit_errors", 0),
            self.tx.rel_power,
            self.results["jitter"].get("dfe", None),
        )

    @lru_cache(maxsize=None)
    def _get_perf_info(self):
        return {key: value * 60.0e-6 for (key, value) in self.performance.items()}

    @property
    @lru_cache(maxsize=None)
    def przf_err(self):
        p = self.results["dfe"]["out_p"]
        nspui = self.nspui
        n_taps = self.eq.n_taps

        (clock_pos, _) = pulse_center(p, nspui)
        err = 0
        for i in range(n_taps):
            err += p[clock_pos + (i + 1) * nspui] ** 2

        return err / p[clock_pos] ** 2

    def run_sweeps(self):
        """
        Runs the simulation sweeps.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.

        """

        self.log.debug("Simulation Started")
        sweep_aves = self.sweep_aves
        do_sweep = self.do_sweep
        tx_taps = self.eq.tx_taps

        if do_sweep:
            # Assemble the list of desired values for each sweepable parameter.
            sweep_vals = []
            for tap in tx_taps:
                if tap.enabled:
                    if tap.steps:
                        sweep_vals.append(
                            list(
                                np.arange(
                                    tap.min_val,
                                    tap.max_val,
                                    (tap.max_val - tap.min_val) / tap.steps,
                                )
                            )
                        )
                    else:
                        sweep_vals.append([tap.value])
                else:
                    sweep_vals.append([0.0])
            # Run the sweep, using the lists assembled, above.
            sweeps = [
                [w, x, y, z]
                for w in sweep_vals[0]
                for x in sweep_vals[1]
                for y in sweep_vals[2]
                for z in sweep_vals[3]
            ]
            num_sweeps = sweep_aves * len(sweeps)
            self.num_sweeps = num_sweeps
            sweep_results = []
            sweep_num = 1
            for sweep in sweeps:
                for i in range(4):
                    self.eq.tx_taps[i].value = sweep[i]
                bit_errs = []
                for i in range(sweep_aves):
                    self.sweep_num = sweep_num
                    self.run_simulation(update_plots=False)
                    bit_errs.append(self.results["bit_errors"])
                    sweep_num += 1
                sweep_results.append((sweep, np.mean(bit_errs), np.std(bit_errs)))
            self.sweep_results = sweep_results
        else:
            self.run_simulation()

    def run_simulation(self, update_plots: bool = True):
        """
        Runs the simulation.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.
            update_plots(Bool): If True, update the plots, after simulation
                completes. This option can be used by larger scripts, which
                import *pybert*, in order to avoid graphical back-end
                conflicts and speed up this function's execution time.
                (Optional; default = True.)
        """
        self.log.info("Starting Simulation")
        num_sweeps = self.num_sweeps
        sweep_num = self.sweep_num

        start_time = clock()
        self.status = f"Running Channel...(Sweep {sweep_num} of {num_sweeps})"
        self.run_count += 1  # Force regeneration of bit stream.

        # Pull class variables into local storage, performing unit conversion where necessary.
        t = self.t
        w = self.w
        bits = self.bits
        symbols = self.symbols
        ffe = self.eq.ffe
        nbits = self.nbits
        nui = self.nui
        bit_rate = self.bit_rate * 1.0e9
        eye_bits = self.eye_bits
        eye_uis = self.eye_uis
        nspb = self.nspb
        nspui = self.nspui
        rn = self.tx.random_noise
        pn_mag = self.tx.pn_mag
        pn_freq = self.tx.pn_freq * 1.0e6
        pattern_len = self.pattern_len
        rx_bw = self.eq.rx_bw * 1.0e9
        peak_freq = self.eq.peak_freq * 1.0e9
        peak_mag = self.eq.peak_mag
        ctle_offset = self.eq.ctle_offset
        ctle_mode = self.eq.ctle_mode
        delta_t = self.eq.delta_t * 1.0e-12
        alpha = self.eq.alpha
        ui = self.ui
        n_taps = self.eq.n_taps
        gain = self.eq.gain
        n_ave = self.eq.n_ave
        decision_scaler = self.eq.decision_scaler
        n_lock_ave = self.eq.n_lock_ave
        rel_lock_tol = self.eq.rel_lock_tol
        lock_sustain = self.eq.lock_sustain
        bandwidth = self.eq.sum_bw * 1.0e9
        rel_thresh = self.thresh

        try:
            # Calculate misc. values.
            fs = bit_rate * nspb
            Ts = t[1]
            ts = Ts

            # Generate the ideal over-sampled signal.
            #
            # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
            # impulse response, in order to produce the proper ideal signal.
            x = np.repeat(symbols, nspui)
            self.x = x
            if self.mod_type == MODULATION.DUO:  # Handle duo-binary case.
                duob_h = np.array(([0.5] + [0.0] * (nspui - 1)) * 2)
                x = np.convolve(x, duob_h)[: len(t)]
            self.ideal_signal = x

            # Find the ideal crossing times, for subsequent jitter analysis of transmitted signal.
            ideal_xings = find_crossings(
                t, x, decision_scaler, min_delay=(ui / 2.0), mod_type=self.mod_type
            )
            self.ideal_xings = ideal_xings

            # Calculate the channel output.
            # -------------------------------------------------------------------------------------
            #
            # Note: We're not using 'self.ideal_signal', because we rely on the system response to
            #       create the duobinary waveform. We only create it explicitly, above,
            #       so that we'll have an ideal reference for comparison.
            chnl_h = self.calc_chnl_h()
            self.log.debug("Channel impulse response is %d samples long.", len(chnl_h))
            chnl_out = np.convolve(self.x, chnl_h)[: len(t)]

            self.performance["channel"] = nbits * nspb / (clock() - start_time)
            split_time = clock()
            self.status = f"Running Tx...(sweep {sweep_num} of {num_sweeps})"
        except Exception as error:
            self.status = "Exception: channel"
            raise

        self.results["channel"]["out"] = chnl_out
        self.results["channel"]["out_H"] = fft(chnl_out)

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the Tx.
        # -----------------------------------------------------------------------------------------
        try:
            if self.tx.use_ami:
                try:
                    # Start with a delta function, to capture the model's impulse response.

                    self.log.info("Tx IBIS-AMI Model Initializing...")
                    tx_model = self.tx.initialize_model(
                        ts, [1.0 / ts] + [0.0] * (len(chnl_h) - 1), ui
                    )
                    tx_h = np.array(tx_model.initOut) * ts
                except ValueError:
                    self.status = "Simulation Error."
                    raise
                except TypeError:
                    self.status = "Simulation Error."
                    raise
                if self.tx.use_getwave:
                    # For GetWave, use a step to extract the model's native properties.
                    # Position the input edge at the center of the vector, in
                    # order to minimize high frequency artifactual energy
                    # introduced by frequency domain processing in some models.
                    half_len = len(chnl_h) // 2
                    tx_s = tx_model.getWave(np.array([0.0] * half_len + [1.0] * half_len))
                    # Shift the result back to the correct location, extending the last sample.
                    tx_s = np.pad(tx_s[half_len:], (0, half_len), "edge")
                    tx_h = np.diff(
                        np.concatenate((np.array([0.0]), tx_s))
                    )  # Without the leading 0, we miss the pre-tap.
                    tx_out = tx_model.getWave(self.x)
                else:  # Init()-only.
                    tx_s = tx_h.cumsum()
                    tx_out = np.convolve(tx_h, self.x)
            else:
                # - Generate the ideal, post-preemphasis signal.
                # To consider: use 'scipy.interp()'. This is what Mark does, in order to induce jitter in the Tx output.
                ffe_out = np.convolve(symbols, ffe)[: len(symbols)]
                self.tx.rel_power = np.mean(
                    ffe_out ** 2
                )  # Store the relative average power dissipated in the Tx.
                tx_out = np.repeat(ffe_out, nspui)  # oversampled output

                # - Calculate the responses.
                # - (The Tx is unique in that the calculated responses aren't used to form the output.
                #    This is partly due to the out of order nature in which we combine the Tx and channel,
                #    and partly due to the fact that we're adding noise to the Tx output.)
                tx_h = np.array(
                    sum([[x] + list(np.zeros(nspui - 1)) for x in ffe], [])
                )  # Using sum to concatenate.
                tx_h.resize(len(chnl_h))
                tx_s = tx_h.cumsum()
            tx_out.resize(len(t))
            temp = tx_h.copy()
            temp.resize(len(w))
            tx_H = fft(temp)
            tx_H *= tx_s[-1] / abs(tx_H[0])

            # - Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
            #   - Generate the ideal rectangular aggressor waveform.
            pn_period = 1.0 / pn_freq
            pn_samps = int(pn_period / Ts + 0.5)
            pn = np.zeros(pn_samps)
            pn[pn_samps // 2 :] = pn_mag
            pn = np.resize(pn, len(tx_out))
            #   - High pass filter it. (Simulating capacitive coupling.)
            (b, a) = iirfilter(2, HPF_CORNER_COUPLING / (fs / 2), btype="highpass")
            pn = lfilter(b, a, pn)[: len(pn)]

            # - Add the uncorrelated periodic and random noise to the Tx output.
            tx_out += pn
            tx_out += normal(scale=rn, size=(len(tx_out),))

            # - Convolve w/ channel.
            tx_out_h = np.convolve(tx_h, chnl_h)[: len(chnl_h)]
            temp = tx_out_h.copy()
            temp.resize(len(w))
            tx_out_H = fft(temp)
            rx_in = np.convolve(tx_out, chnl_h)[: len(tx_out)]

            self.results["tx"]["s"] = tx_s
            self.results["tx"]["out"] = tx_out
            self.rx_in = rx_in
            self.results["tx"]["out_s"] = tx_out_h.cumsum()
            self.results["tx"]["out_p"] = (
                self.results["tx"]["out_s"][nspui:] - self.results["tx"]["out_s"][:-nspui]
            )
            self.results["tx"]["H"] = tx_H
            self.results["tx"]["h"] = tx_h
            self.results["tx"]["out_H"] = tx_out_H
            self.results["tx"]["out_h"] = tx_out_h

            self.performance["tx"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running CTLE...(sweep {sweep_num} of {num_sweeps})"
        except Exception as error:
            self.status = "Exception: Tx"
            raise

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the CTLE.
        # -----------------------------------------------------------------------------------------
        try:
            if self.rx.use_ami:
                try:
                    self.log.info("Rx IBIS-AMI Model Initializing...")
                    rx_model = self.rx.initialize_model(ts, tx_out_h / ts, ui)
                    ctle_out_h = np.array(rx_model.initOut) * ts
                except ValueError:
                    self.status = "Simulation Error."
                    raise
                except TypeError:
                    self.status = "Simulation Error."
                    raise
                if self.rx.use_getwave:
                    ctle_out, clock_times = rx_model.getWave(rx_in, len(rx_in))
                    self.log.info(rx_model.ami_params_out)

                    ctle_H = fft(ctle_out * hann(len(ctle_out))) / fft(rx_in * hann(len(rx_in)))
                    ctle_h = np.real(ifft(ctle_H)[: len(chnl_h)])
                    ctle_out_h = np.convolve(ctle_h, tx_out_h)[: len(chnl_h)]
                else:  # Init() only.
                    ctle_out_h_padded = np.pad(
                        ctle_out_h,
                        (nspb, len(rx_in) - nspb - len(ctle_out_h)),
                        "linear_ramp",
                        end_values=(0.0, 0.0),
                    )
                    tx_out_h_padded = np.pad(
                        tx_out_h,
                        (nspb, len(rx_in) - nspb - len(tx_out_h)),
                        "linear_ramp",
                        end_values=(0.0, 0.0),
                    )
                    ctle_H = fft(ctle_out_h_padded) / fft(tx_out_h_padded)
                    ctle_h = np.real(ifft(ctle_H)[: len(chnl_h)])
                    ctle_out = np.convolve(rx_in, ctle_h)
                ctle_s = ctle_h.cumsum()
            else:
                if self.eq.use_ctle_file:
                    ctle_h = import_channel(self.eq.ctle_file, ts)
                    if max(abs(ctle_h)) < 100.0:  # step response?
                        ctle_h = np.diff(
                            ctle_h
                        )  # impulse response is derivative of step response.
                    else:
                        ctle_h *= ts  # Normalize to (V/sample)
                    ctle_h.resize(len(t))
                    ctle_H = fft(ctle_h)
                    ctle_H *= sum(ctle_h) / ctle_H[0]
                else:
                    _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w, ctle_mode, ctle_offset)
                    ctle_h = np.real(ifft(ctle_H))[: len(chnl_h)]
                    ctle_h *= abs(ctle_H[0]) / sum(ctle_h)
                ctle_out = np.convolve(rx_in, ctle_h)
                ctle_out -= np.mean(ctle_out)  # Force zero mean.
                if self.eq.ctle_mode == "AGC":  # Automatic gain control engaged?
                    ctle_out *= 2.0 * decision_scaler / ctle_out.ptp()
                ctle_s = ctle_h.cumsum()
                ctle_out_h = np.convolve(tx_out_h, ctle_h)[: len(tx_out_h)]
            ctle_out.resize(len(t))
            self.results["ctle"]["s"] = ctle_s
            ctle_out_h_main_lobe = np.where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
            if ctle_out_h_main_lobe.size:
                conv_dly_ix = ctle_out_h_main_lobe[0]
            else:
                conv_dly_ix = self.chnl_dly / Ts
            conv_dly = t[conv_dly_ix]
            ctle_out_s = ctle_out_h.cumsum()
            temp = ctle_out_h.copy()
            temp.resize(len(w))
            ctle_out_H = fft(temp)
            # - Store local variables to class instance.
            self.results["ctle"]["out_s"] = ctle_out_s
            # Consider changing this; it could be sensitive to insufficient "front porch" in the CTLE output step response.
            self.results["ctle"]["out_p"] = ctle_out_s[nspui:] - ctle_out_s[:-nspui]
            self.results["ctle"]["H"] = ctle_H
            self.results["ctle"]["h"] = ctle_h
            self.results["ctle"]["out_H"] = ctle_out_H
            self.results["ctle"]["out_h"] = ctle_out_h
            self.results["ctle"]["out"] = ctle_out
            self.conv_dly = conv_dly
            self.conv_dly_ix = conv_dly_ix

            self.performance["ctle"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running DFE/CDR...(sweep {sweep_num} of {num_sweeps})"
        except Exception as error:
            self.status = "Exception: Rx"
            raise

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the DFE.
        # -----------------------------------------------------------------------------------------
        try:
            if self.eq.use_dfe:
                dfe = DFE(
                    n_taps,
                    gain,
                    delta_t,
                    alpha,
                    ui,
                    nspui,
                    decision_scaler,
                    self.mod_type,
                    n_ave=n_ave,
                    n_lock_ave=n_lock_ave,
                    rel_lock_tol=rel_lock_tol,
                    lock_sustain=lock_sustain,
                    bandwidth=bandwidth,
                    ideal=self.eq.sum_ideal,
                )
            else:
                dfe = DFE(
                    n_taps,
                    0.0,
                    delta_t,
                    alpha,
                    ui,
                    nspui,
                    decision_scaler,
                    self.mod_type,
                    n_ave=n_ave,
                    n_lock_ave=n_lock_ave,
                    rel_lock_tol=rel_lock_tol,
                    lock_sustain=lock_sustain,
                    bandwidth=bandwidth,
                    ideal=True,
                )
            (dfe_out, tap_weights, ui_ests, clocks, lockeds, clock_times, bits_out) = dfe.run(
                t, ctle_out
            )
            dfe_out = np.array(dfe_out)
            dfe_out.resize(len(t))
            bits_out = np.array(bits_out)
            auto_corr = (
                1.0
                * np.correlate(
                    bits_out[(nbits - eye_bits) :], bits[(nbits - eye_bits) :], mode="same"
                )
                / sum(bits[(nbits - eye_bits) :])
            )
            auto_corr = auto_corr[len(auto_corr) // 2 :]
            self.auto_corr = auto_corr
            bit_dly = np.where(auto_corr == max(auto_corr))[0][0]
            bits_ref = bits[(nbits - eye_bits) :]
            bits_tst = bits_out[(nbits + bit_dly - eye_bits) :]
            if len(bits_ref) > len(bits_tst):
                bits_ref = bits_ref[: len(bits_tst)]
            elif len(bits_tst) > len(bits_ref):
                bits_tst = bits_tst[: len(bits_ref)]
            bit_errs = np.where(bits_tst ^ bits_ref)[0]
            self.results["bit_errors"] = len(bit_errs)

            dfe_h = np.array(
                [1.0]
                + list(np.zeros(nspb - 1))
                + sum([[-x] + list(np.zeros(nspb - 1)) for x in tap_weights[-1]], [])
            )
            dfe_h.resize(len(ctle_out_h))
            temp = dfe_h.copy()
            temp.resize(len(w))
            dfe_H = fft(temp)
            self.results["dfe"]["s"] = dfe_h.cumsum()
            dfe_out_H = ctle_out_H * dfe_H
            dfe_out_h = np.convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]
            dfe_out_s = dfe_out_h.cumsum()
            self.results["dfe"]["out_p"] = dfe_out_s - np.pad(
                dfe_out_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)
            )
            self.results["dfe"]["H"] = dfe_H
            self.results["dfe"]["h"] = dfe_h
            self.results["dfe"]["out_H"] = dfe_out_H
            self.results["dfe"]["out_h"] = dfe_out_h
            self.results["dfe"]["out_s"] = dfe_out_s
            self.results["dfe"]["out"] = dfe_out

            self.performance["dfe"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Analyzing Jitter...(sweep {sweep_num} of {num_sweeps})"
        except Exception as error:
            self.status = "Exception: DFE"
            raise

        # Save local variables to class instance for state preservation, performing unit conversion where necessary.
        self.adaptation = tap_weights
        self.ui_ests = np.array(ui_ests) * 1.0e12  # (ps)
        self.clocks = clocks
        self.lockeds = lockeds
        self.clock_times = clock_times

        # Analyze the jitter.
        # -----------------------------------------------------------------------------------------

        try:
            if self.mod_type == MODULATION.DUO:  # Handle duo-binary case.
                pattern_len *= 2  # Because, the XOR pre-coding can invert every other pattern rep.
            if self.mod_type == MODULATION.PAM4:  # Handle PAM-4 case.
                if pattern_len % 2:
                    pattern_len *= 2  # Because, the bits are taken in pairs, to form the symbols.

            # - channel output
            actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=self.mod_type)
            self.results["jitter"]["channel"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )
            self.results["jitter"]["f_MHz"] = (
                np.array(self.results["jitter"]["channel"].spectrum_freqs) * 1.0e-6
            )

            # - Tx output
            actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=self.mod_type)
            self.results["jitter"]["tx"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )

            # - CTLE output
            actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=self.mod_type)
            self.results["jitter"]["ctle"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )

            # - DFE output
            ignore_until = (
                nui - eye_uis
            ) * ui + 0.75 * ui  # 0.5 was causing an occasional misalignment.
            ideal_xings = np.array([x for x in list(ideal_xings) if x > ignore_until])
            min_delay = ignore_until + conv_dly
            actual_xings = find_crossings(
                t,
                dfe_out,
                decision_scaler,
                min_delay=min_delay,
                mod_type=self.mod_type,
                rising_first=False,
            )
            self.results["jitter"]["dfe"] = Jitter.calc_jitter(
                ui, eye_uis, pattern_len, ideal_xings, actual_xings, rel_thresh
            )
            self.results["jitter"]["f_MHz_dfe"] = (
                np.array(self.results["jitter"]["dfe"].spectrum_freqs) * 1.0e-6
            )
            self.results["jitter"]["rejection_ratio"] = np.zeros(
                len(self.results["jitter"]["dfe"].jitter_spectrum)
            )

            self.performance["jitter"] = nbits * nspb / (clock() - split_time)
            self.performance["total"] = nbits * nspb / (clock() - start_time)
            split_time = clock()
        except Exception as error:
            self.status = "Exception: Jitter"
            raise

        # Analyze the results.
        # -----------------------------------------------------------------------------------------
        self.status = f"Cleaning Results...(sweep {sweep_num} of {num_sweeps})"
        self.parse_and_clean_results()

        # Simulation Done
        # -----------------------------------------------------------------------------------------
        try:
            if update_plots:
                self.status = f"Updating plots...(sweep {sweep_num} of {num_sweeps})"
                # Tell the GUI that the simulation is done and to update everything.
                self.sim_done.emit(self.results)
            self.performance["plot"] = nbits * nspb / (clock() - split_time)
            self.metrics.emit(self.performance)
            self.log.debug("Performance: %s", self._get_perf_info())
            self.status = "Ready"
        except Exception as error:
            self.status = "Exception: plotting"
            raise

    def parse_and_clean_results(self):
        """Transform, prune or clean any results from the simulation to pass to the GUI."""

        ui = self.ui
        samps_per_ui = self.nspui
        eye_uis = self.eye_uis
        num_ui = self.nui
        clock_times = self.clock_times
        f = self.f
        t = self.t
        n_taps = self.eq.n_taps

        Ts = t[1]
        ignore_until = (num_ui - eye_uis) * ui
        ignore_samps = (num_ui - eye_uis) * samps_per_ui

        # Misc.
        f_GHz = f[: len(f) // 2] / 1.0e9
        len_f_GHz = len(f_GHz)
        self.results["f_GHz"] = f_GHz[1:]
        self.results["t_ns"] = self.t_ns

        # DFE.
        self.results["n_dfe_taps"] = NUM_TAPS
        tap_weights = np.transpose(np.array(self.adaptation))
        for index, tap_weight in enumerate(tap_weights, 1):
            self.results[f"tap{index}_weights"] = tap_weight
        self.results["tap_weight_index"] = list(range(len(tap_weight)))
        if self.eq._old_n_taps != n_taps:
            self.results["n_dfe_taps"] = self.eq.n_taps
            self.eq._old_n_taps = n_taps

        clock_pers = np.diff(clock_times)
        start_t = t[np.where(self.lockeds)[0][0]]
        start_ix = np.where(clock_times > start_t)[0][0]
        (bin_counts, bin_edges) = np.histogram(clock_pers[start_ix:], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        clock_spec = fft(clock_pers[start_ix:])
        clock_spec = abs(clock_spec[: len(clock_spec) // 2])
        spec_freqs = np.arange(len(clock_spec)) / (
            2.0 * len(clock_spec)
        )  # In this case, fNyquist = half the bit rate.
        clock_spec /= clock_spec[1:].mean()  # Normalize the mean non-d.c. value to 0 dB.
        self.results["clk_per_hist_bins"] = bin_centers * 1.0e12  # (ps)
        self.results["clk_per_hist_vals"] = bin_counts
        self.results["clk_spec"] = 10.0 * np.log10(clock_spec[1:])  # Omit the d.c. value.
        self.results["clk_freqs"] = spec_freqs[1:]
        self.results["ui_ests"] = self.ui_ests
        self.results["clocks"] = self.clocks
        self.results["lockeds"] = self.lockeds

        # Impulse responses
        # Re-normalize to (V/ns), for plotting.
        self.results["channel"]["chnl_h"] = self.results["channel"]["chnl_h"] * 1.0e-9 / Ts
        self.results["tx"]["h"] = self.results["tx"]["h"] * 1.0e-9 / Ts
        self.results["tx"]["out_h"] = self.results["tx"]["out_h"] * 1.0e-9 / Ts
        self.results["ctle"]["h"] = self.results["ctle"]["h"] * 1.0e-9 / Ts
        self.results["ctle"]["out_h"] = self.results["ctle"]["out_h"] * 1.0e-9 / Ts
        self.results["dfe"]["h"] = self.results["dfe"]["h"] * 1.0e-9 / Ts
        self.results["dfe"]["out_h"] = self.results["dfe"]["out_h"] * 1.0e-9 / Ts

        # Frequency responses
        self.results["channel"]["chnl_H"] = 20.0 * np.log10(
            abs(self.results["channel"]["chnl_H"][1:len_f_GHz])
        )
        self.results["channel"]["chnl_trimmed_H"] = 20.0 * np.log10(
            abs(self.results["channel"]["chnl_trimmed_H"][1:len_f_GHz])
        )
        self.results["tx"]["H"] = 20.0 * np.log10(abs(self.results["tx"]["H"][1:len_f_GHz]))
        self.results["tx"]["out_H"] = 20.0 * np.log10(
            abs(self.results["tx"]["out_H"][1:len_f_GHz])
        )
        self.results["ctle"]["H"] = 20.0 * np.log10(abs(self.results["ctle"]["H"][1:len_f_GHz]))
        self.results["ctle"]["out_H"] = 20.0 * np.log10(
            abs(self.results["ctle"]["out_H"][1:len_f_GHz])
        )
        self.results["dfe"]["H"] = 20.0 * np.log10(abs(self.results["dfe"]["H"][1:len_f_GHz]))
        self.results["dfe"]["out_H"] = 20.0 * np.log10(
            abs(self.results["dfe"]["out_H"][1:len_f_GHz])
        )

        self.results["jitter"]["channel"].bin_centers = (
            np.array(self.results["jitter"]["channel"].bin_centers) * 1.0e12
        )
        self.results["jitter_chnl"] = self.results["jitter"]["channel"].hist
        self.results["jitter_ext_chnl"] = self.results["jitter"]["channel"].hist_synth
        self.results["jitter_tx"] = self.results["jitter"]["tx"].hist
        self.results["jitter_ext_tx"] = self.results["jitter"]["tx"].hist_synth
        self.results["jitter_ctle"] = self.results["jitter"]["ctle"].hist
        self.results["jitter_ext_ctle"] = self.results["jitter"]["ctle"].hist_synth
        self.results["jitter_dfe"] = self.results["jitter"]["dfe"].hist
        self.results["jitter_ext_dfe"] = self.results["jitter"]["dfe"].hist_synth

        # Jitter spectrums
        log10_ui = np.log10(ui)
        self.results["jitter"]["f_MHz"] = self.results["jitter"]["f_MHz"][1:]
        self.results["jitter"]["f_MHz_dfe"] = self.results["jitter"]["f_MHz_dfe"][1:]
        self.results["jitter"]["channel"].jitter_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["channel"].jitter_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["channel"].tie_ind_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["channel"].tie_ind_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["channel"].thresh = 10.0 * (
            np.log10(self.results["jitter"]["channel"].thresh[1:]) - log10_ui
        )
        self.results["jitter"]["tx"].jitter_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["tx"].jitter_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["tx"].tie_ind_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["tx"].tie_ind_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["tx"].thresh = 10.0 * (
            np.log10(self.results["jitter"]["tx"].thresh[1:]) - log10_ui
        )
        self.results["jitter"]["ctle"].jitter_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["ctle"].jitter_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["ctle"].tie_ind_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["ctle"].tie_ind_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["ctle"].thresh = 10.0 * (
            np.log10(self.results["jitter"]["ctle"].thresh[1:]) - log10_ui
        )
        self.results["jitter"]["dfe"].jitter_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["dfe"].jitter_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["dfe"].tie_ind_spectrum = 10.0 * (
            np.log10(self.results["jitter"]["dfe"].tie_ind_spectrum[1:]) - log10_ui
        )
        self.results["jitter"]["dfe"].thresh = 10.0 * (
            np.log10(self.results["jitter"]["dfe"].thresh[1:]) - log10_ui
        )
        self.results["jitter_rejection_ratio"] = self.results["jitter"]["rejection_ratio"][1:]

        # Bathtubs
        half_len = len(self.results["jitter"]["channel"].hist_synth) // 2
        #  - Channel
        bathtub_chnl = list(
            np.cumsum(self.results["jitter"]["channel"].hist_synth[-1 : -(half_len + 1) : -1])
        )
        bathtub_chnl.reverse()
        bathtub_chnl = np.array(
            bathtub_chnl
            + list(np.cumsum(self.results["jitter"]["channel"].hist_synth[: half_len + 1]))
        )
        bathtub_chnl = np.where(
            bathtub_chnl < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_chnl)),
            bathtub_chnl,
        )  # To avoid Chaco log scale plot wierdness.
        self.results["bathtub_chnl"] = np.log10(bathtub_chnl)
        #  - Tx
        bathtub_tx = list(
            np.cumsum(self.results["jitter"]["tx"].hist_synth[-1 : -(half_len + 1) : -1])
        )
        bathtub_tx.reverse()
        bathtub_tx = np.array(
            bathtub_tx + list(np.cumsum(self.results["jitter"]["tx"].hist_synth[: half_len + 1]))
        )
        bathtub_tx = np.where(
            bathtub_tx < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_tx)),
            bathtub_tx,
        )  # To avoid Chaco log scale plot wierdness.
        self.results["bathtub_tx"] = np.log10(bathtub_tx)
        #  - CTLE
        bathtub_ctle = list(
            np.cumsum(self.results["jitter"]["ctle"].hist_synth[-1 : -(half_len + 1) : -1])
        )
        bathtub_ctle.reverse()
        bathtub_ctle = np.array(
            bathtub_ctle
            + list(np.cumsum(self.results["jitter"]["ctle"].hist_synth[: half_len + 1]))
        )
        bathtub_ctle = np.where(
            bathtub_ctle < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_ctle)),
            bathtub_ctle,
        )  # To avoid Chaco log scale plot weirdness.
        self.results["bathtub_ctle"] = np.log10(bathtub_ctle)
        #  - DFE
        bathtub_dfe = list(
            np.cumsum(self.results["jitter"]["dfe"].hist_synth[-1 : -(half_len + 1) : -1])
        )
        bathtub_dfe.reverse()
        bathtub_dfe = np.array(
            bathtub_dfe + list(np.cumsum(self.results["jitter"]["dfe"].hist_synth[: half_len + 1]))
        )
        bathtub_dfe = np.where(
            bathtub_dfe < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_dfe)),
            bathtub_dfe,
        )  # To avoid Chaco log scale plot weirdness.
        self.results["bathtub_dfe"] = np.log10(bathtub_dfe)

        # Eyes
        width = 2 * samps_per_ui
        xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)
        height = 100
        y_max = 1.1 * max(abs(np.array(self.results["channel"]["out"])))
        eye_chnl = calc_eye(
            ui, samps_per_ui, height, self.results["channel"]["out"][ignore_samps:], y_max
        )
        y_max = 1.1 * max(abs(np.array(self.rx_in)))
        eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(np.array(self.results["ctle"]["out"])))
        eye_ctle = calc_eye(
            ui, samps_per_ui, height, self.results["ctle"]["out"][ignore_samps:], y_max
        )
        i = 0
        while clock_times[i] <= ignore_until:
            i += 1
            assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
        y_max = 1.1 * max(abs(np.array(self.results["dfe"]["out"])))
        eye_dfe = calc_eye(
            ui, samps_per_ui, height, self.results["dfe"]["out"], y_max, clock_times[i:]
        )
        self.results["eye_index"] = xs
        self.results["eye_chnl"] = eye_chnl
        self.results["eye_tx"] = eye_tx
        self.results["eye_ctle"] = eye_ctle
        self.results["eye_dfe"] = eye_dfe
