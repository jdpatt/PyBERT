"""
Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import logging
from time import perf_counter

import numpy as np
import skrf as rf
from numpy import array, exp, pad, where, zeros
from numpy.fft import fft, irfft
from numpy.random import normal
from scipy.signal import iirfilter, lfilter
from scipy.signal.windows import hann

from pybert.sim.dfe import DFE
from pybert.sim.jitter import calc_jitter, find_crossings
from pybert.utility import (
    add_ondie_s,
    calc_gamma,
    getwave_step_resp,
    import_channel,
    make_ctle,
    trim_impulse,
)
from pyibisami.ami import AMIModelInitializer

gFc = 1.0e6  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.

log = logging.getLogger("pybert.sim")

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
    if self.tx_use_ibis:
        model = self._tx_ibis.model
        Rs = model.zout * 2
        Cs = model.ccomp[0] / 2  # They're in series.
        self.Rs = Rs  # Primarily for debugging.
        self.Cs = Cs
        if self.tx_use_ts4:
            fname = self._tx_ibis_dir.joinpath(self._tx_ami_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
            ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname)
            self.ts4N = ts4N
            self.ntwk = ntwk
    if self.rx_use_ibis:
        model = self._rx_ibis.model
        RL = model.zin * 2
        Cp = model.ccomp[0] / 2
        self.RL = RL  # Primarily for debugging.
        self.Cp = Cp
        self._log.debug("RL: %d, Cp: %d", RL, Cp)
        if self.rx_use_ts4:
            fname = self._rx_ibis_dir.joinpath(self._rx_ami_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
            ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
            self.ts4N = ts4N
            self.ntwk = ntwk
    ch_s2p.name = "ch_s2p"
    self.ch_s2p = ch_s2p

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
    # t_h, chnl_h = ch_s2p_term.s21.impulse_response()
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


def my_run_simulation(self):
    """
    Runs the simulation.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
    """
    num_sweeps = self.num_sweeps
    sweep_num = self.sweep_num

    start_time = perf_counter()
    self.status = f"Running channel...(sweep {sweep_num} of {num_sweeps})"

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
    rn = self.rn
    pn_mag = self.pn_mag
    pn_freq = self.pn_freq * 1.0e6
    pattern_len = self.pattern_len
    rx_bw = self.rx_bw * 1.0e9
    peak_freq = self.peak_freq * 1.0e9
    peak_mag = self.peak_mag
    ctle_offset = self.ctle_offset
    ctle_mode = self.ctle_mode
    delta_t = self.delta_t * 1.0e-12
    alpha = self.alpha
    ui = self.ui
    n_taps = self.n_taps
    gain = self.gain
    n_ave = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave = self.n_lock_ave
    rel_lock_tol = self.rel_lock_tol
    lock_sustain = self.lock_sustain
    bandwidth = self.sum_bw * 1.0e9
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
        x = np.repeat(symbols, nspui)
        self.x = x
        if mod_type == 1:  # Handle duo-binary case.
            duob_h = np.array(([0.5] + [0.0] * (nspui - 1)) * 2)
            x = np.convolve(x, duob_h)[: len(t)]
        self.ideal_signal = x

        # Find the ideal crossing times, for subsequent jitter analysis of transmitted signal.
        ideal_xings = find_crossings(t, x, decision_scaler, min_delay=(ui / 2.0), mod_type=mod_type)
        self.ideal_xings = ideal_xings

        # Calculate the channel output.
        #
        # Note: We're not using 'self.ideal_signal', because we rely on the system response to
        #       create the duobinary waveform. We only create it explicitly, above,
        #       so that we'll have an ideal reference for comparison.
        chnl_h = calc_chnl_h(self)
        chnl_out = np.convolve(self.x, chnl_h)[: len(t)]

        self.channel_perf = nbits * nspb / (perf_counter() - start_time)
    except Exception:
        self.status = "Exception: channel"
        raise

    self.chnl_out = chnl_out
    self.chnl_out_H = fft(chnl_out)

    split_time = perf_counter()
    self.status = f"Running Tx...(sweep {sweep_num} of {num_sweeps})"

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the Tx.
    try:
        if self.tx_use_ami:
            # Note: Within the PyBERT computational environment, we use normalized impulse responses,
            #       which have units of (V/ts), where 'ts' is the sample interval. However, IBIS-AMI models expect
            #       units of (V/s). So, we have to scale accordingly, as we transit the boundary between these two worlds.
            tx_cfg = self._tx_ami_cfg  # Grab the 'AMIParamConfigurator' instance for this model.
            # Get the model invoked and initialized, except for 'channel_response', which
            # we need to do several different ways, in order to gather all the data we need.
            tx_param_dict = tx_cfg.input_ami_params
            tx_model_init = AMIModelInitializer(tx_param_dict)
            tx_model_init.sample_interval = ts  # Must be set, before 'channel_response'!
            # Start with a delta function, to capture the model's impulse response.
            tx_model_init.channel_response = [1.0 / ts] + [0.0] * (len(chnl_h) - 1)
            tx_model_init.bit_time = ui
            tx_model = self._tx_ami_model
            tx_model.initialize(tx_model_init)
            log.info(
                "Tx IBIS-AMI model initialization results:\nInput parameters: %s\nOutput parameters: %s\nMessage: %s",
                tx_model.ami_params_in.decode("utf-8"),
                tx_model.ami_params_out.decode("utf-8"),
                tx_model.msg.decode("utf-8"),
            )
            if tx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]):
                tx_h = np.array(tx_model.initOut) * ts
            elif not tx_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
                self.status = "Simulation Error."
                log.error(
                    "Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!\n \
I cannot continue.\nYou will have to select a different model.",
                    extra={"alert": True},
                )
                return
            elif not self.tx_use_getwave:
                self.status = "Simulation Error."
                log.error(
                    "You have elected not to use GetWave for a model, which does not return an impulse response!\n \
I cannot continue.\nPlease, select 'Use GetWave' and try again.",
                    extra={"alert": True},
                )
                return
            if self.tx_use_getwave:
                try:
                    tx_s = getwave_step_resp(tx_model)
                except RuntimeError as err:
                    self.status = "Tx GetWave() Error."
                    self.log("ERROR: Never saw a rising step come out of Tx GetWave()!", alert=True)
                    return
                tx_h, _ = trim_impulse(np.diff(tx_s))
                tx_out, _ = tx_model.getWave(self.x)
            else:  # Init()-only.
                tx_out = np.convolve(tx_h, self.x)
                tx_s = tx_h.cumsum()
            self.tx_model = tx_model
        else:
            # - Generate the ideal, post-preemphasis signal.
            # To consider: use 'scipy.interp()'. This is what Mark does, in order to induce jitter in the Tx output.
            ffe_out = np.convolve(symbols, ffe)[: len(symbols)]
            if self.use_ch_file:
                self.rel_power = np.mean(ffe_out**2) / self.rs
            else:
                self.rel_power = np.mean(ffe_out**2) / self.Z0
            tx_out = np.repeat(ffe_out, nspui)  # oversampled output

            # - Calculate the responses.
            # - (The Tx is unique in that the calculated responses aren't used to form the output.
            #    This is partly due to the out of order nature in which we combine the Tx and channel,
            #    and partly due to the fact that we're adding noise to the Tx output.)
            tx_h = np.array(sum([[x] + list(np.zeros(nspui - 1)) for x in ffe], []))  # Using sum to concatenate.
            tx_h.resize(len(chnl_h))
            tx_s = tx_h.cumsum()
        tx_out.resize(len(t))
        temp = tx_h.copy()
        temp.resize(len(t))
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
        (b, a) = iirfilter(2, gFc / (fs / 2), btype="highpass")
        pn = lfilter(b, a, pn)[: len(pn)]

        # - Add the uncorrelated periodic and random noise to the Tx output.
        tx_out += pn
        tx_out += normal(scale=rn, size=(len(tx_out),))

        # - Convolve w/ channel.
        tx_out_h = np.convolve(tx_h, chnl_h)[: len(chnl_h)]
        temp = tx_out_h.copy()
        temp.resize(len(t))
        tx_out_H = fft(temp)
        rx_in = np.convolve(tx_out, chnl_h)[: len(tx_out)]

        self.tx_s = tx_s
        self.tx_out = tx_out
        self.rx_in = rx_in
        self.tx_out_s = tx_out_h.cumsum()
        self.tx_out_p = self.tx_out_s[nspui:] - self.tx_out_s[:-nspui]
        self.tx_H = tx_H
        self.tx_h = tx_h
        self.tx_out_H = tx_out_H
        self.tx_out_h = tx_out_h

        self.tx_perf = nbits * nspb / (perf_counter() - split_time)
    except Exception:
        self.status = "Exception: Tx"
        raise

    split_time = perf_counter()
    self.status = f"Running CTLE...(sweep {sweep_num} of {num_sweeps})"
    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the CTLE.
    try:
        if self.rx_use_ami:
            rx_cfg = self._rx_ami_cfg  # Grab the 'AMIParamConfigurator' instance for this model.
            # Get the model invoked and initialized, except for 'channel_response', which
            # we need to do several different ways, in order to gather all the data we need.
            rx_param_dict = rx_cfg.input_ami_params
            rx_model_init = AMIModelInitializer(rx_param_dict)
            rx_model_init.sample_interval = ts  # Must be set, before 'channel_response'!
            rx_model_init.channel_response = tx_out_h / ts
            rx_model_init.bit_time = ui
            rx_model = self._rx_ami_model
            rx_model.initialize(rx_model_init)
            log.info(
                "Rx IBIS-AMI model initialization results:\nInput parameters: %s\nMessage: %s\nOutput parameters: %s",
                rx_model.ami_params_in.decode("utf-8"),
                rx_model.msg.decode("utf-8"),
                rx_model.ami_params_out.decode("utf-8"),
            )
            if rx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]):
                ctle_out_h = np.array(rx_model.initOut) * ts
            elif not rx_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
                self.status = "Simulation Error."
                log.error(
                    "Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!\n \
I cannot continue.\nYou will have to select a different model.",
                    extra={"alert": True},
                )
                return
            elif not self.rx_use_getwave:
                self.status = "Simulation Error."
                log.error(
                    "You have elected not to use GetWave for a model, which does not return an impulse response!\n \
I cannot continue.\nPlease, select 'Use GetWave' and try again.",
                    extra={"alert": True},
                )
                return
            if self.rx_use_getwave:
                try:
                    ctle_s = getwave_step_resp(rx_model)
                except RuntimeError as err:
                    self.status = "Rx GetWave() Error."
                    self.log("ERROR: Never saw a rising step come out of Rx GetWave()!", alert=True)
                    return
                ctle_h = np.diff(ctle_s)
                temp = ctle_h.copy()
                temp.resize(len(t))
                ctle_H = fft(temp)
                ctle_h.resize(len(chnl_h))
                ctle_out_h = np.convolve(ctle_h, tx_out_h)[: len(chnl_h)]
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
                ctle_H = fft(ctle_out_h_padded) / fft(tx_out_h_padded)  # ToDo: I think this is wrong.
                ctle_h = irfft(ctle_H)  # I shouldn't be sending the output of `fft()` into `irfft()`, should I?
                ctle_h.resize(len(chnl_h))
            ctle_s = ctle_h.cumsum()
            ctle_out = np.convolve(rx_in, ctle_h)
        else:
            if self.use_ctle_file:
                # FIXME: The new import_channel() implementation breaks this:
                ctle_h = import_channel(self.ctle_file, ts, self.f)
                if max(abs(ctle_h)) < 100.0:  # step response?
                    ctle_h = np.diff(ctle_h)  # impulse response is derivative of step response.
                else:
                    ctle_h *= ts  # Normalize to (V/sample)
                ctle_h.resize(len(t))
                ctle_H = fft(ctle_h)
                ctle_H *= sum(ctle_h) / ctle_H[0]
            else:
                _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w, ctle_mode, ctle_offset)
                ctle_h = irfft(ctle_H)
            ctle_h.resize(len(chnl_h))
            ctle_out = np.convolve(rx_in, ctle_h)
            ctle_out -= np.mean(ctle_out)  # Force zero mean.
            if self.ctle_mode == "AGC":  # Automatic gain control engaged?
                ctle_out *= 2.0 * decision_scaler / ctle_out.ptp()
            ctle_s = ctle_h.cumsum()
            ctle_out_h = np.convolve(tx_out_h, ctle_h)[: len(tx_out_h)]
        ctle_out.resize(len(t))
        ctle_out_h_main_lobe = np.where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
        if ctle_out_h_main_lobe.size:
            conv_dly_ix = ctle_out_h_main_lobe[0]
        else:
            conv_dly_ix = int(self.chnl_dly // Ts)
        conv_dly = t[conv_dly_ix]  # Keep this line only.
        ctle_out_s = ctle_out_h.cumsum()
        temp = ctle_out_h.copy()
        temp.resize(len(t))
        ctle_out_H = fft(temp)
        # - Store local variables to class instance.
        # Consider changing this; it could be sensitive to insufficient "front porch" in the CTLE output step response.
        self.ctle_out_p = ctle_out_s[nspui:] - ctle_out_s[:-nspui]
        self.ctle_H = ctle_H
        self.ctle_h = ctle_h
        self.ctle_s = ctle_s
        self.ctle_out_H = ctle_out_H
        self.ctle_out_h = ctle_out_h
        self.ctle_out_s = ctle_out_s
        self.ctle_out = ctle_out
        self.conv_dly = conv_dly
        self.conv_dly_ix = conv_dly_ix

        self.ctle_perf = nbits * nspb / (perf_counter() - split_time)
    except Exception:
        self.status = "Exception: Rx"
        raise

    split_time = perf_counter()
    self.status = f"Running DFE/CDR...(sweep {sweep_num} of {num_sweeps})"
    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of the DFE.
    try:
        if self.use_dfe:
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
                ideal=self.sum_ideal,
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
        (dfe_out, tap_weights, ui_ests, clocks, lockeds, clock_times, bits_out) = dfe.run(t, ctle_out)
        dfe_out = np.array(dfe_out)
        dfe_out.resize(len(t))
        bits_out = np.array(bits_out)
        auto_corr = (
            1.0
            * np.correlate(bits_out[(nbits - eye_bits) :], bits[(nbits - eye_bits) :], mode="same")
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
        self.bit_errs = len(bit_errs)

        dfe_h = np.array(
            [1.0] + list(np.zeros(nspb - 1)) + sum([[-x] + list(np.zeros(nspb - 1)) for x in tap_weights[-1]], [])
        )
        dfe_h.resize(len(ctle_out_h))
        temp = dfe_h.copy()
        temp.resize(len(t))
        dfe_H = fft(temp)
        self.dfe_s = dfe_h.cumsum()
        dfe_out_H = ctle_out_H * dfe_H
        dfe_out_h = np.convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]
        dfe_out_s = dfe_out_h.cumsum()
        self.dfe_out_p = dfe_out_s - np.pad(dfe_out_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))
        self.dfe_H = dfe_H
        self.dfe_h = dfe_h
        self.dfe_out_H = dfe_out_H
        self.dfe_out_h = dfe_out_h
        self.dfe_out_s = dfe_out_s
        self.dfe_out = dfe_out

        self.dfe_perf = nbits * nspb / (perf_counter() - split_time)
    except Exception:
        self.status = "Exception: DFE"
        raise

    # Save local variables to class instance for state preservation, performing unit conversion where necessary.
    self.adaptation = tap_weights
    self.ui_ests = np.array(ui_ests) * 1.0e12  # (ps)
    self.clocks = clocks
    self.lockeds = lockeds
    self.clock_times = clock_times

    # Analyze the jitter.
    split_time = perf_counter()
    self.status = f"Analyzing jitter...(sweep {sweep_num} of {num_sweeps})"
    jitter = {}

    try:
        if mod_type == 1:  # Handle duo-binary case.
            pattern_len *= 2  # Because, the XOR pre-coding can invert every other pattern rep.
        elif mod_type == 2:  # Handle PAM-4 case.
            if pattern_len % 2:
                pattern_len *= 2  # Because, the bits are taken in pairs, to form the symbols.

        # - channel output
        actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=mod_type)
        jitter.update({"chnl": calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)})
        jitter.update({"f_MHz": np.array(jitter["chnl"].spectrum_freqs) * 1.0e-6})

        # - Tx output
        actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=mod_type)
        jitter.update({"tx": calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)})

        # - CTLE output
        actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=mod_type)
        jitter.update({"ctle": calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)})

        # - DFE output
        ignore_until = (nui - eye_uis) * ui + 0.75 * ui  # 0.5 was causing an occasional misalignment.
        ideal_xings = np.array([x for x in list(ideal_xings) if x > ignore_until])
        min_delay = ignore_until + conv_dly
        actual_xings = find_crossings(
            t, dfe_out, decision_scaler, min_delay=min_delay, mod_type=mod_type, rising_first=False
        )
        jitter.update({"dfe": calc_jitter(ui, eye_uis, pattern_len, ideal_xings, actual_xings, rel_thresh)})
        jitter.update({"f_MHz_dfe": np.array(jitter["dfe"].spectrum_freqs) * 1.0e-6})

        self.jitter = jitter
        self.jitter_rejection_ratio = np.zeros(len(jitter["dfe"].jitter_spectrum))

        self.jitter_perf = nbits * nspb / (perf_counter() - split_time)
        self.total_perf = nbits * nspb / (perf_counter() - start_time)
    except Exception as error:
        self.status = "Exception: jitter"
        log.debug(error)

    log.info("Simulation Complete.")
