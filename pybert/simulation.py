"""Where all the magic happens."""
from logging import getLogger
from threading import Thread
from time import clock

import numpy as np
from chaco.api import Plot
from chaco.tools.api import PanTool, ZoomTool
from numpy import (
    arange,
    array,
    concatenate,
    convolve,
    correlate,
    cos,
    cumsum,
    diff,
    histogram,
    linspace,
    log10,
    mean,
    ones,
    pad,
    pi,
    real,
    repeat,
    resize,
    sinc,
    std,
    transpose,
    where,
    zeros,
)
from numpy.fft import fft, ifft
from numpy.random import normal, randint
from pybert.buffer import Receiver, Transmitter
from pybert.channel import Channel
from pybert.defaults import (
    BIT_RATE,
    HPF_CORNER_COUPLING,
    MIN_BATHTUB_VAL,
    NUM_AVG,
    NUM_BITS,
    PATTERN_LEN,
    SAMPLES_PER_BIT,
    THRESHOLD,
)
from pybert.dfe import DFE
from pybert.equalization import Equalization
from pybert.jitter import Jitter
from pybert.plot import Plots
from pybert.utility import (
    calc_eye,
    find_crossings,
    import_channel,
    lfsr_bits,
    make_ctle,
    pulse_center,
)
from pybert.view import popup_alert
from scipy.signal import iirfilter, lfilter
from scipy.signal.windows import hann
from traits.api import (
    HTML,
    Array,
    Bool,
    Dict,
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


class RunSimThread(Thread):
    """Used to run the simulation in its own thread, in order to preserve GUI responsiveness."""

    def run(self):
        """Run the simulation(s)."""
        self.my_run_sweeps(self.the_pybert)


class Simulation(HasTraits):
    """docstring for Simulation"""

    def __init__(self):
        super(Simulation, self).__init__()
        self.log = getLogger("pybert.simulation")
        self.log.debug("Creating Simulation Object")
        self.status = String("Ready.")

        self.bit_rate = Range(low=0.1, high=120.0, value=BIT_RATE)  #: (Gbps)
        self.nbits = Range(low=1000, high=10000000, value=NUM_BITS)  #: Number of bits to simulate.
        self.pattern_len = Range(low=7, high=10000000, value=PATTERN_LEN)  #: PRBS pattern length.
        self.nspb = Range(
            low=2, high=256, value=SAMPLES_PER_BIT
        )  #: Signal vector samples per bit.
        self.eye_bits = Int(NUM_BITS // 5)  #: # of bits used to form eye. (Default = last 20%)
        self.mod_type = List([0])  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
        self.num_sweeps = Int(1)  #: Number of sweeps to run.
        self.sweep_num = Int(1)
        self.sweep_aves = Int(NUM_AVG)
        self.do_sweep = Bool(False)  #: Run sweeps? (Default = False)

        self.performance = Dict({})
        self.sweep_results = List([])
        self.bit_errors = Int(0)  #: # of bit errors observed in last run.
        self.run_count = Int(0)  # Used as a mechanism to force bit stream regeneration.
        self.thresh = THRESHOLD

        self.channel = Channel()
        self.tx = Transmitter()
        self.rx = Receiver()
        self.eq = Equalization()

        self.jitter = {}
        self.plots = Plots()

        # Dependent variables
        # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
        #
        # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
        #         This will slow the GUI down noticeably.
        self.jitter_info = Property(HTML, depends_on=["performance['jitter']"])
        self.perf_info = Property(HTML, depends_on=["performance['total']"])
        self.status_str = Property(String, depends_on=["status"])
        self.sweep_info = Property(HTML, depends_on=["sweep_results"])
        self.cost = Property(Float, depends_on=["ctle_out_h_tune", "nspui"])
        self.rel_opt = Property(Float, depends_on=["cost"])
        self.t = Property(Array, depends_on=["ui", "nspb", "nbits"])
        self.t_ns = Property(Array, depends_on=["t"])
        self.f = Property(Array, depends_on=["t"])
        self.w = Property(Array, depends_on=["f"])
        self.bits = Property(Array, depends_on=["pattern_len", "nbits", "run_count"])
        self.symbols = Property(Array, depends_on=["bits", "mod_type", "vod"])
        self.ffe = Property(Array, depends_on=["tx_taps.value", "tx_taps.enabled"])
        self.ui = Property(Float, depends_on=["bit_rate", "mod_type"])
        self.nui = Property(Int, depends_on=["nbits", "mod_type"])
        self.nspui = Property(Int, depends_on=["nspb", "mod_type"])
        self.eye_uis = Property(Int, depends_on=["eye_bits", "mod_type"])
        self.dfe_out_p = Array()
        self.przf_err = Property(Float, depends_on=["dfe_out_p"])

        self.run_sim_thread = Instance(RunSimThread)

    def run(self):
        """Spawn a simulation thread and run with the current settings."""
        if self.run_sim_thread and self.run_sim_thread.isAlive():
            pass
        else:
            self.run_sim_thread = RunSimThread()
            self.run_sim_thread.the_pybert = the_pybert
            self.log.debug("Simulation Started")
            self.run_sim_thread.start()

    def abort(self):
        """Kill the simulation thread."""
        if self.run_sim_thread and self.run_sim_thread.isAlive():
            self.run_sim_thread.stop()
            self.log.warning("Simulation Aborted")

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, message):
        """Override the status setter so that we can log all as debug messages."""
        self.log.debug(message)
        self.__status = message

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
        """

        t = self.t

        npts = len(t)
        f0 = 1.0 / (t[1] * npts)
        half_npts = npts // 2

        return array(
            [i * f0 for i in range(half_npts + 1)]
            + [(half_npts - i) * -f0 for i in range(1, half_npts)]
        )

    @cached_property
    def _get_w(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in rads./sec.
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
            nui /= 2

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
            eye_uis /= 2

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
            popup_alert("Unrecognized ideal impulse response type.", ValueError())
        if (
            mod_type == 1
        ):  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (
                ideal_h + pad(ideal_h[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))
            )

        return ideal_h

    @cached_property
    def _get_symbols(self):
        """
        Generate the symbol stream.
        """

        mod_type = self.mod_type[0]
        vod = self.tx.vod
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
            popup_alert("Unknown modulation type requested!", ValueError())
        return array(symbols) * vod

    @cached_property
    def _get_cost(self):
        nspui = self.nspui
        h = self.eq.ctle_out_h_tune
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
            if not self.eq.use_dfe_tune:
                isi += abs(p[ix])
            ix += nspui
        if self.eq.use_dfe_tune:
            for i in range(self.eq.n_taps_tune):
                if clock_pos + nspui * (1 + i) < len(p):
                    p[int(clock_pos + nspui * (0.5 + i)) :] -= p[clock_pos + nspui * (1 + i)]

        self.plots.update_data("ctle_out_h_tune", p)
        self.plots.update_data("clocks_tune", clocks)

        if mod_type == 1:  # Handle duo-binary.
            return (
                isi
                - p[clock_pos]
                - p[clock_pos + nspui]
                + 2.0 * abs(p[clock_pos + nspui] - p[clock_pos])
            )
        return isi - p[clock_pos]

    @cached_property
    def _get_rel_opt(self):
        return -self.cost

    @cached_property
    def _get_przf_err(self):
        p = self.dfe_out_p
        nspui = self.nspui
        n_taps = self.eq.n_taps

        (clock_pos, _) = pulse_center(p, nspui)
        err = 0
        for i in range(n_taps):
            err += p[clock_pos + (i + 1) * nspui] ** 2

        return err / p[clock_pos] ** 2

    def my_run_sweeps(self):
        """
        Runs the simulation sweeps.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.

        """

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
                                arange(
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
                    self.my_run_simulation(update_plots=False)
                    bit_errs.append(self.bit_errs)
                    sweep_num += 1
                sweep_results.append((sweep, mean(bit_errs), std(bit_errs)))
            self.sweep_results = sweep_results
        else:
            self.my_run_simulation()

    def my_run_simulation(self, initial_run=False, update_plots=True):
        """
        Runs the simulation.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.
            initial_run(Bool): If True, don't update the eye diagrams, since
                they haven't been created, yet. (Optional; default = False.)
            update_plots(Bool): If True, update the plots, after simulation
                completes. This option can be used by larger scripts, which
                import *pybert*, in order to avoid graphical back-end
                conflicts and speed up this function's execution time.
                (Optional; default = True.)
        """

        num_sweeps = self.num_sweeps
        sweep_num = self.sweep_num

        start_time = clock()
        self.status = f"Running channel...(sweep {sweep_num} of {num_sweeps})"
        self.log.debug(type(self.run_count))
        self.log.debug(type(1))
        self.run_count += 1  # Force regeneration of bit stream.

        # Pull class variables into local storage, performing unit conversion where necessary.
        t = self.t
        w = self.w
        bits = self.bits
        symbols = self.symbols
        ffe = self.ffe
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
        mod_type = self.mod_type[0]

        try:
            # Calculate misc. values.
            fs = bit_rate * nspb
            Ts = t[1]
            ts = Ts

            # Generate the ideal over-sampled signal.
            #
            # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
            # impulse response, in order to produce the proper ideal signal.
            x = repeat(symbols, nspui)
            self.x = x
            if mod_type == 1:  # Handle duo-binary case.
                duob_h = array(([0.5] + [0.0] * (nspui - 1)) * 2)
                x = convolve(x, duob_h)[: len(t)]
            self.ideal_signal = x

            # Find the ideal crossing times, for subsequent jitter analysis of transmitted signal.
            ideal_xings = find_crossings(
                t, x, decision_scaler, min_delay=(ui / 2.0), mod_type=mod_type
            )
            self.ideal_xings = ideal_xings

            # Calculate the channel output.
            #
            # Note: We're not using 'self.ideal_signal', because we rely on the system response to
            #       create the duobinary waveform. We only create it explicitly, above,
            #       so that we'll have an ideal reference for comparison.
            chnl_h = self.channel.calc_chnl_h()
            self.log.debug("Channel impulse response is %d samples long.", len(chnl_h))
            chnl_out = convolve(self.x, chnl_h)[: len(t)]

            self.performance["channel"] = nbits * nspb / (clock() - start_time)
            split_time = clock()
            self.status = f"Running Tx...(sweep {sweep_num} of {num_sweeps})"
        except Exception:
            self.status = "Exception: channel"
            raise

        self.chnl_out = chnl_out
        self.chnl_out_H = fft(chnl_out)

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the Tx.
        try:
            if self.tx.use_ami:
                try:
                    # Start with a delta function, to capture the model's impulse response.
                    tx_model = self.tx.initialize_model(
                        ts, [1.0 / ts] + [0.0] * (len(chnl_h) - 1), ui
                    )
                    tx_h = array(tx_model.initOut) * ts
                except ValueError as error:
                    popup_alert(
                        "Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!", error
                    )
                    self.status = "Simulation Error."
                    return
                except TypeError as error:
                    popup_alert(
                        "You have elected not to use GetWave for a model, which does not \
                        return an impulse response! Aborting... Please, select 'Use GetWave'",
                        error,
                    )
                    self.status = "Simulation Error."
                    return
                if self.tx.use_getwave:
                    # For GetWave, use a step to extract the model's native properties.
                    # Position the input edge at the center of the vector, in
                    # order to minimize high frequency artifactual energy
                    # introduced by frequency domain processing in some models.
                    half_len = len(chnl_h) // 2
                    tx_s = tx_model.getWave(array([0.0] * half_len + [1.0] * half_len))
                    # Shift the result back to the correct location, extending the last sample.
                    tx_s = pad(tx_s[half_len:], (0, half_len), "edge")
                    tx_h = diff(
                        concatenate((array([0.0]), tx_s))
                    )  # Without the leading 0, we miss the pre-tap.
                    tx_out = tx_model.getWave(self.x)
                else:  # Init()-only.
                    tx_s = tx_h.cumsum()
                    tx_out = convolve(tx_h, self.x)
            else:
                # - Generate the ideal, post-preemphasis signal.
                # To consider: use 'scipy.interp()'. This is what Mark does, in order to induce jitter in the Tx output.
                ffe_out = convolve(symbols, ffe)[: len(symbols)]
                self.rel_power = mean(
                    ffe_out ** 2
                )  # Store the relative average power dissipated in the Tx.
                tx_out = repeat(ffe_out, nspui)  # oversampled output

                # - Calculate the responses.
                # - (The Tx is unique in that the calculated responses aren't used to form the output.
                #    This is partly due to the out of order nature in which we combine the Tx and channel,
                #    and partly due to the fact that we're adding noise to the Tx output.)
                tx_h = array(
                    sum([[x] + list(zeros(nspui - 1)) for x in ffe], [])
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
            pn = zeros(pn_samps)
            pn[pn_samps // 2 :] = pn_mag
            pn = resize(pn, len(tx_out))
            #   - High pass filter it. (Simulating capacitive coupling.)
            (b, a) = iirfilter(2, HPF_CORNER_COUPLING / (fs / 2), btype="highpass")
            pn = lfilter(b, a, pn)[: len(pn)]

            # - Add the uncorrelated periodic and random noise to the Tx output.
            tx_out += pn
            tx_out += normal(scale=rn, size=(len(tx_out),))

            # - Convolve w/ channel.
            tx_out_h = convolve(tx_h, chnl_h)[: len(chnl_h)]
            temp = tx_out_h.copy()
            temp.resize(len(w))
            tx_out_H = fft(temp)
            rx_in = convolve(tx_out, chnl_h)[: len(tx_out)]

            self.tx_s = tx_s
            self.tx_out = tx_out
            self.rx_in = rx_in
            self.tx_out_s = tx_out_h.cumsum()
            self.tx_out_p = self.tx_out_s[nspui:] - self.tx_out_s[:-nspui]
            self.tx_H = tx_H
            self.tx_h = tx_h
            self.tx_out_H = tx_out_H
            self.tx_out_h = tx_out_h

            self.performance["tx"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running CTLE...(sweep {sweep_num} of {num_sweeps})"
        except Exception:
            self.status = "Exception: Tx"
            raise

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the CTLE.
        try:
            if self.rx.use_ami:
                try:
                    rx_model = self.rx.initialize_model(ts, tx_out_h / ts, ui)
                    ctle_out_h = np.array(rx_model.initOut) * ts
                except ValueError as error:
                    popup_alert(
                        "Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!", error
                    )
                    self.status = "Simulation Error."
                    return
                except TypeError as error:
                    popup_alert(
                        "You have elected not to use GetWave for a model, which does not \
                        return an impulse response! Aborting... Please, select 'Use GetWave'",
                        error,
                    )
                    self.status = "Simulation Error."
                    return
                if self.rx.use_getwave:
                    ctle_out, clock_times = rx_model.getWave(rx_in, len(rx_in))
                    self.log.info(rx_model.ami_params_out)

                    ctle_H = fft(ctle_out * hann(len(ctle_out))) / fft(rx_in * hann(len(rx_in)))
                    ctle_h = real(ifft(ctle_H)[: len(chnl_h)])
                    ctle_out_h = convolve(ctle_h, tx_out_h)[: len(chnl_h)]
                else:  # Init() only.
                    ctle_out_h_padded = pad(
                        ctle_out_h,
                        (nspb, len(rx_in) - nspb - len(ctle_out_h)),
                        "linear_ramp",
                        end_values=(0.0, 0.0),
                    )
                    tx_out_h_padded = pad(
                        tx_out_h,
                        (nspb, len(rx_in) - nspb - len(tx_out_h)),
                        "linear_ramp",
                        end_values=(0.0, 0.0),
                    )
                    ctle_H = fft(ctle_out_h_padded) / fft(tx_out_h_padded)
                    ctle_h = real(ifft(ctle_H)[: len(chnl_h)])
                    ctle_out = convolve(rx_in, ctle_h)
                ctle_s = ctle_h.cumsum()
            else:
                if self.eq.use_ctle_file:
                    ctle_h = import_channel(self.eq.ctle_file, ts)
                    if max(abs(ctle_h)) < 100.0:  # step response?
                        ctle_h = diff(ctle_h)  # impulse response is derivative of step response.
                    else:
                        ctle_h *= ts  # Normalize to (V/sample)
                    ctle_h.resize(len(t))
                    ctle_H = fft(ctle_h)
                    ctle_H *= sum(ctle_h) / ctle_H[0]
                else:
                    _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w, ctle_mode, ctle_offset)
                    ctle_h = real(ifft(ctle_H))[: len(chnl_h)]
                    ctle_h *= abs(ctle_H[0]) / sum(ctle_h)
                ctle_out = convolve(rx_in, ctle_h)
                ctle_out -= mean(ctle_out)  # Force zero mean.
                if self.eq.ctle_mode == "AGC":  # Automatic gain control engaged?
                    ctle_out *= 2.0 * decision_scaler / ctle_out.ptp()
                ctle_s = ctle_h.cumsum()
                ctle_out_h = convolve(tx_out_h, ctle_h)[: len(tx_out_h)]
            ctle_out.resize(len(t))
            self.ctle_s = ctle_s
            ctle_out_h_main_lobe = where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
            if ctle_out_h_main_lobe.size:
                conv_dly_ix = ctle_out_h_main_lobe[0]
            else:
                conv_dly_ix = self.channel.chnl_dly / Ts
            conv_dly = t[conv_dly_ix]
            ctle_out_s = ctle_out_h.cumsum()
            temp = ctle_out_h.copy()
            temp.resize(len(w))
            ctle_out_H = fft(temp)
            # - Store local variables to class instance.
            self.ctle_out_s = ctle_out_s
            # Consider changing this; it could be sensitive to insufficient "front porch" in the CTLE output step response.
            self.ctle_out_p = self.ctle_out_s[nspui:] - self.ctle_out_s[:-nspui]
            self.ctle_H = ctle_H
            self.ctle_h = ctle_h
            self.ctle_out_H = ctle_out_H
            self.ctle_out_h = ctle_out_h
            self.ctle_out = ctle_out
            self.conv_dly = conv_dly
            self.conv_dly_ix = conv_dly_ix

            self.performance["ctle"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running DFE/CDR...(sweep {sweep_num} of {num_sweeps})"
        except Exception:
            self.status = "Exception: Rx"
            raise

        # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the DFE.
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
                    mod_type,
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
                    mod_type,
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
            dfe_out = array(dfe_out)
            dfe_out.resize(len(t))
            bits_out = array(bits_out)
            auto_corr = (
                1.0
                * correlate(
                    bits_out[(nbits - eye_bits) :], bits[(nbits - eye_bits) :], mode="same"
                )
                / sum(bits[(nbits - eye_bits) :])
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
            self.bit_errors = len(bit_errs)

            dfe_h = array(
                [1.0]
                + list(zeros(nspb - 1))
                + sum([[-x] + list(zeros(nspb - 1)) for x in tap_weights[-1]], [])
            )
            dfe_h.resize(len(ctle_out_h))
            temp = dfe_h.copy()
            temp.resize(len(w))
            dfe_H = fft(temp)
            self.dfe_s = dfe_h.cumsum()
            dfe_out_H = ctle_out_H * dfe_H
            dfe_out_h = convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]
            dfe_out_s = dfe_out_h.cumsum()
            self.dfe_out_p = dfe_out_s - pad(
                dfe_out_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)
            )
            self.dfe_H = dfe_H
            self.dfe_h = dfe_h
            self.dfe_out_H = dfe_out_H
            self.dfe_out_h = dfe_out_h
            self.dfe_out_s = dfe_out_s
            self.dfe_out = dfe_out

            self.performance["dfe"] = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Analyzing jitter...(sweep {sweep_num} of {num_sweeps})"
        except Exception:
            self.status = "Exception: DFE"
            raise

        # Save local variables to class instance for state preservation, performing unit conversion where necessary.
        self.adaptation = tap_weights
        self.ui_ests = array(ui_ests) * 1.0e12  # (ps)
        self.clocks = clocks
        self.lockeds = lockeds
        self.clock_times = clock_times

        # Analyze the jitter.
        # -------------------------------------------------------------------------------------------

        try:
            if mod_type == 1:  # Handle duo-binary case.
                pattern_len *= 2  # Because, the XOR pre-coding can invert every other pattern rep.
            if mod_type == 2:  # Handle PAM-4 case.
                if pattern_len % 2:
                    pattern_len *= 2  # Because, the bits are taken in pairs, to form the symbols.

            # - channel output
            actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=mod_type)
            self.jitter["channel"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )
            self.jitter["f_MHz"] = array(self.jitter["channel"].spectrum_freqs) * 1.0e-6

            # - Tx output
            actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=mod_type)
            self.jitter["tx"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )

            # - CTLE output
            actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=mod_type)
            self.jitter["ctle"] = Jitter.calc_jitter(
                ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh
            )

            # - DFE output
            ignore_until = (
                nui - eye_uis
            ) * ui + 0.75 * ui  # 0.5 was causing an occasional misalignment.
            ideal_xings = array([x for x in list(ideal_xings) if x > ignore_until])
            min_delay = ignore_until + conv_dly
            actual_xings = find_crossings(
                t,
                dfe_out,
                decision_scaler,
                min_delay=min_delay,
                mod_type=mod_type,
                rising_first=False,
            )
            self.jitter["dfe"] = Jitter.calc_jitter(
                ui, eye_uis, pattern_len, ideal_xings, actual_xings, rel_thresh
            )
            self.jitter["f_MHz_dfe"] = array(self.jitter["dfe"].spectrum_freqs) * 1.0e-6
            self.jitter["rejection_ratio"] = zeros(len(self.jitter["dfe"].jitter_spectrum))

            self.performance["jitter"] = nbits * nspb / (clock() - split_time)
            self.performance["total"] = nbits * nspb / (clock() - start_time)
            split_time = clock()
            self.status = f"Updating plots...(sweep {sweep_num} of {num_sweeps})"
        except Exception:
            self.status = "Exception: jitter"

        # Update plots.
        # -------------------------------------------------------------------------------------------
        try:
            if update_plots:
                self.update_results()
                if not initial_run:
                    self.update_eyes()

            self.performance["plot"] = nbits * nspb / (clock() - split_time)
            self.status = "Ready."
        except Exception:
            self.status = "Exception: plotting"
            raise

    def update_results(self):
        """
        Updates all plot data used by GUI.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.

        """

        # Copy globals into local namespace.
        ui = self.ui
        samps_per_ui = self.nspui
        eye_uis = self.eye_uis
        num_ui = self.nui
        clock_times = self.clock_times
        f = self.f
        t = self.t
        t_ns = self.t_ns
        t_ns_chnl = self.channel.t_ns_chnl
        n_taps = self.eq.n_taps

        Ts = t[1]
        ignore_until = (num_ui - eye_uis) * ui
        ignore_samps = (num_ui - eye_uis) * samps_per_ui

        # Misc.
        f_GHz = f[: len(f) // 2] / 1.0e9
        len_f_GHz = len(f_GHz)
        self.plots.update_data("f_GHz", f_GHz[1:])
        self.plots.update_data("t_ns", t_ns)
        self.plots.update_data("t_ns_chnl", t_ns_chnl)

        # DFE.
        tap_weights = transpose(array(self.adaptation))
        i = 1
        for tap_weight in tap_weights:
            self.plots.update_data("tap%d_weights" % i, tap_weight)
            i += 1
        self.plots.update_data("tap_weight_index", list(range(len(tap_weight))))
        if self.eq._old_n_taps != n_taps:
            new_plot = Plot(
                self.plots.data,
                auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
                padding_left=75,
            )
            for i in range(self.eq.n_taps):
                new_plot.plot(
                    ("tap_weight_index", "tap%d_weights" % (i + 1)),
                    type="line",
                    color="auto",
                    name="tap%d" % (i + 1),
                )
            new_plot.title = "DFE Adaptation"
            new_plot.tools.append(
                PanTool(new_plot, constrain=True, constrain_key=None, constrain_direction="x")
            )
            zoom9 = ZoomTool(new_plot, tool_mode="range", axis="index", always_on=False)
            new_plot.overlays.append(zoom9)
            new_plot.legend.visible = True
            new_plot.legend.align = "ul"
            self.plots.plots_dfe.remove(self.plots._dfe_plot)
            self.plots.plots_dfe.insert(1, new_plot)
            self.plots._dfe_plot = new_plot
            self.eq._old_n_taps = n_taps

        clock_pers = diff(clock_times)
        start_t = t[where(self.lockeds)[0][0]]
        start_ix = where(clock_times > start_t)[0][0]
        (bin_counts, bin_edges) = histogram(clock_pers[start_ix:], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        clock_spec = fft(clock_pers[start_ix:])
        clock_spec = abs(clock_spec[: len(clock_spec) // 2])
        spec_freqs = arange(len(clock_spec)) / (
            2.0 * len(clock_spec)
        )  # In this case, fNyquist = half the bit rate.
        clock_spec /= clock_spec[1:].mean()  # Normalize the mean non-d.c. value to 0 dB.
        self.plots.update_data("clk_per_hist_bins", bin_centers * 1.0e12)  # (ps)
        self.plots.update_data("clk_per_hist_vals", bin_counts)
        self.plots.update_data("clk_spec", 10.0 * log10(clock_spec[1:]))  # Omit the d.c. value.
        self.plots.update_data("clk_freqs", spec_freqs[1:])
        self.plots.update_data("dfe_out", self.dfe_out)
        self.plots.update_data("ui_ests", self.ui_ests)
        self.plots.update_data("clocks", self.clocks)
        self.plots.update_data("lockeds", self.lockeds)

        # Impulse responses
        self.plots.update_data(
            "chnl_h", self.channel.chnl_h * 1.0e-9 / Ts
        )  # Re-normalize to (V/ns), for plotting.
        self.plots.update_data("tx_h", self.tx_h * 1.0e-9 / Ts)
        self.plots.update_data("tx_out_h", self.tx_out_h * 1.0e-9 / Ts)
        self.plots.update_data("ctle_h", self.ctle_h * 1.0e-9 / Ts)
        self.plots.update_data("ctle_out_h", self.ctle_out_h * 1.0e-9 / Ts)
        self.plots.update_data("dfe_h", self.dfe_h * 1.0e-9 / Ts)
        self.plots.update_data("dfe_out_h", self.dfe_out_h * 1.0e-9 / Ts)

        # Step responses
        self.plots.update_data("chnl_s", self.channel.chnl_s)
        self.plots.update_data("tx_s", self.tx_s)
        self.plots.update_data("tx_out_s", self.tx_out_s)
        self.plots.update_data("ctle_s", self.ctle_s)
        self.plots.update_data("ctle_out_s", self.ctle_out_s)
        self.plots.update_data("dfe_s", self.dfe_s)
        self.plots.update_data("dfe_out_s", self.dfe_out_s)

        # Pulse responses
        self.plots.update_data("chnl_p", self.channel.chnl_p)
        self.plots.update_data("tx_out_p", self.tx_out_p)
        self.plots.update_data("ctle_out_p", self.ctle_out_p)
        self.plots.update_data("dfe_out_p", self.dfe_out_p)

        # Outputs
        self.plots.update_data("ideal_signal", self.ideal_signal)
        self.plots.update_data("chnl_out", self.chnl_out)
        self.plots.update_data("tx_out", self.rx_in)
        self.plots.update_data("ctle_out", self.ctle_out)
        self.plots.update_data("dfe_out", self.dfe_out)
        self.plots.update_data("auto_corr", self.auto_corr)

        # Frequency responses
        self.plots.update_data("chnl_H", 20.0 * log10(abs(self.channel.chnl_H[1:len_f_GHz])))
        self.plots.update_data(
            "chnl_trimmed_H", 20.0 * log10(abs(self.channel.chnl_trimmed_H[1:len_f_GHz]))
        )
        self.plots.update_data("tx_H", 20.0 * log10(abs(self.tx_H[1:len_f_GHz])))
        self.plots.update_data("tx_out_H", 20.0 * log10(abs(self.tx_out_H[1:len_f_GHz])))
        self.plots.update_data("ctle_H", 20.0 * log10(abs(self.ctle_H[1:len_f_GHz])))
        self.plots.update_data("ctle_out_H", 20.0 * log10(abs(self.ctle_out_H[1:len_f_GHz])))
        self.plots.update_data("dfe_H", 20.0 * log10(abs(self.dfe_H[1:len_f_GHz])))
        self.plots.update_data("dfe_out_H", 20.0 * log10(abs(self.dfe_out_H[1:len_f_GHz])))

        self.plots.update_data("jitter_bins", array(self.jitter["channel"].jitter_bins) * 1.0e12)
        self.plots.update_data("jitter_chnl", self.jitter["channel"].hist)
        self.plots.update_data("jitter_ext_chnl", self.jitter["channel"].hist_synth)
        self.plots.update_data("jitter_tx", self.jitter["tx"].hist)
        self.plots.update_data("jitter_ext_tx", self.jitter["tx"].hist_synth)
        self.plots.update_data("jitter_ctle", self.jitter["ctle"].hist)
        self.plots.update_data("jitter_ext_ctle", self.jitter["ctle"].hist_synth)
        self.plots.update_data("jitter_dfe", self.jitter["dfe"].hist)
        self.plots.update_data("jitter_ext_dfe", self.jitter["dfe"].hist_synth)

        # Jitter spectrums
        log10_ui = log10(ui)
        self.plots.update_data("f_MHz", self.jitter["f_MHz"][1:])
        self.plots.update_data("f_MHz_dfe", self.jitter["f_MHz_dfe"][1:])
        self.plots.update_data(
            "jitter_spectrum_chnl",
            10.0 * (log10(self.jitter["channel"].jitter_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "jitter_ind_spectrum_chnl",
            10.0 * (log10(self.jitter["channel"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "thresh_chnl", 10.0 * (log10(self.jitter["channel"].thresh[1:]) - log10_ui)
        )
        self.plots.update_data(
            "jitter_spectrum_tx", 10.0 * (log10(self.jitter["tx"].jitter_spectrum[1:]) - log10_ui)
        )
        self.plots.update_data(
            "jitter_ind_spectrum_tx",
            10.0 * (log10(self.jitter["tx"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "thresh_tx", 10.0 * (log10(self.jitter["tx"].thresh[1:]) - log10_ui)
        )
        self.plots.update_data(
            "jitter_spectrum_ctle",
            10.0 * (log10(self.jitter["ctle"].jitter_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "jitter_ind_spectrum_ctle",
            10.0 * (log10(self.jitter["ctle"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "thresh_ctle", 10.0 * (log10(self.jitter["ctle"].thresh[1:]) - log10_ui)
        )
        self.plots.update_data(
            "jitter_spectrum_dfe",
            10.0 * (log10(self.jitter["dfe"].jitter_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "jitter_ind_spectrum_dfe",
            10.0 * (log10(self.jitter["dfe"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.plots.update_data(
            "thresh_dfe", 10.0 * (log10(self.jitter["dfe"].thresh[1:]) - log10_ui)
        )
        self.plots.update_data("jitter_rejection_ratio", self.jitter["rejection_ratio"][1:])

        # Bathtubs
        half_len = len(self.jitter["channel"].hist_synth) // 2
        #  - Channel
        bathtub_chnl = list(cumsum(self.jitter["channel"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_chnl.reverse()
        bathtub_chnl = array(
            bathtub_chnl + list(cumsum(self.jitter["channel"].hist_synth[: half_len + 1]))
        )
        bathtub_chnl = where(
            bathtub_chnl < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_chnl)),
            bathtub_chnl,
        )  # To avoid Chaco log scale plot wierdness.
        self.plots.update_data("bathtub_chnl", log10(bathtub_chnl))
        #  - Tx
        bathtub_tx = list(cumsum(self.jitter["tx"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_tx.reverse()
        bathtub_tx = array(bathtub_tx + list(cumsum(self.jitter["tx"].hist_synth[: half_len + 1])))
        bathtub_tx = where(
            bathtub_tx < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_tx)), bathtub_tx
        )  # To avoid Chaco log scale plot wierdness.
        self.plots.update_data("bathtub_tx", log10(bathtub_tx))
        #  - CTLE
        bathtub_ctle = list(cumsum(self.jitter["ctle"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_ctle.reverse()
        bathtub_ctle = array(
            bathtub_ctle + list(cumsum(self.jitter["ctle"].hist_synth[: half_len + 1]))
        )
        bathtub_ctle = where(
            bathtub_ctle < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_ctle)),
            bathtub_ctle,
        )  # To avoid Chaco log scale plot weirdness.
        self.plots.update_data("bathtub_ctle", log10(bathtub_ctle))
        #  - DFE
        bathtub_dfe = list(cumsum(self.jitter["dfe"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_dfe.reverse()
        bathtub_dfe = array(
            bathtub_dfe + list(cumsum(self.jitter["dfe"].hist_synth[: half_len + 1]))
        )
        bathtub_dfe = where(
            bathtub_dfe < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_dfe)),
            bathtub_dfe,
        )  # To avoid Chaco log scale plot weirdness.
        self.plots.update_data("bathtub_dfe", log10(bathtub_dfe))

        # Eyes
        width = 2 * samps_per_ui
        xs = linspace(-ui * 1.0e12, ui * 1.0e12, width)
        height = 100
        y_max = 1.1 * max(abs(array(self.chnl_out)))
        eye_chnl = calc_eye(ui, samps_per_ui, height, self.chnl_out[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(array(self.rx_in)))
        eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(array(self.ctle_out)))
        eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max)
        i = 0
        while clock_times[i] <= ignore_until:
            i += 1
            assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
        y_max = 1.1 * max(abs(array(self.dfe_out)))
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out, y_max, clock_times[i:])
        self.plots.update_data("eye_index", xs)
        self.plots.update_data("eye_chnl", eye_chnl)
        self.plots.update_data("eye_tx", eye_tx)
        self.plots.update_data("eye_ctle", eye_ctle)
        self.plots.update_data("eye_dfe", eye_dfe)

    def update_eyes(self):
        """
        Update the heat plots representing the eye diagrams.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.

        """

        ui = self.ui
        samps_per_ui = self.nspui

        width = 2 * samps_per_ui
        height = 100
        xs = linspace(-ui * 1.0e12, ui * 1.0e12, width)

        y_max = 1.1 * max(abs(array(self.chnl_out)))
        ys = linspace(-y_max, y_max, height)
        self.plots.eyes.components[0].components[0].index.set_data(xs, ys)
        self.plots.eyes.components[0].x_axis.mapper.range.low = xs[0]
        self.plots.eyes.components[0].x_axis.mapper.range.high = xs[-1]
        self.plots.eyes.components[0].y_axis.mapper.range.low = ys[0]
        self.plots.eyes.components[0].y_axis.mapper.range.high = ys[-1]
        self.plots.eyes.components[0].invalidate_draw()

        y_max = 1.1 * max(abs(array(self.rx_in)))
        ys = linspace(-y_max, y_max, height)
        self.plots.eyes.components[1].components[0].index.set_data(xs, ys)
        self.plots.eyes.components[1].x_axis.mapper.range.low = xs[0]
        self.plots.eyes.components[1].x_axis.mapper.range.high = xs[-1]
        self.plots.eyes.components[1].y_axis.mapper.range.low = ys[0]
        self.plots.eyes.components[1].y_axis.mapper.range.high = ys[-1]
        self.plots.eyes.components[1].invalidate_draw()

        y_max = 1.1 * max(abs(array(self.dfe_out)))
        ys = linspace(-y_max, y_max, height)
        self.plots.eyes.components[3].components[0].index.set_data(xs, ys)
        self.plots.eyes.components[3].x_axis.mapper.range.low = xs[0]
        self.plots.eyes.components[3].x_axis.mapper.range.high = xs[-1]
        self.plots.eyes.components[3].y_axis.mapper.range.low = ys[0]
        self.plots.eyes.components[3].y_axis.mapper.range.high = ys[-1]
        self.plots.eyes.components[3].invalidate_draw()

        self.plots.eyes.components[2].components[0].index.set_data(xs, ys)
        self.plots.eyes.components[2].x_axis.mapper.range.low = xs[0]
        self.plots.eyes.components[2].x_axis.mapper.range.high = xs[-1]
        self.plots.eyes.components[2].y_axis.mapper.range.low = ys[0]
        self.plots.eyes.components[2].y_axis.mapper.range.high = ys[-1]
        self.plots.eyes.components[2].invalidate_draw()

        self.plots.eyes.request_redraw()
