"""General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 27, 2014 (Copied from pybert_cntrl.py.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import re
from functools import reduce
from logging import getLogger
from pathlib import Path

import numpy as np
import skrf as rf
from scipy.signal import freqs, get_window, invres


def moving_average(a, n=3):
    """
    Calculates a sliding average over the input vector.

    Args:
        a([float]): Input vector to be averaged.
        n(int): Width of averaging window, in vector samples. (Optional;
            default = 3.)

    Returns: the moving average of the input vector, leaving the input
        vector unchanged.
    """

    ret = np.cumsum(a, dtype=np.float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.insert(ret[n - 1 :], 0, ret[n - 1] * np.ones(n - 1)) / n


def find_crossing_times(
    t,
    x,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev: float = 0.1,
    thresh: float = 0.0,
):
    """
    Finds the threshold crossing times of the input signal.

    Args:
        t([float]): Vector of sample times. Intervals do NOT need to be
            uniform.
        x([float]): Sampled input vector.
        min_delay(float): Minimum delay required, before allowing
            crossings. (Helps avoid false crossings at beginning of
            signal.) (Optional; default = 0.)
        rising_first(bool): When True, start with the first rising edge
            found. (Optional; default = True.) When this option is True,
            the first rising edge crossing is the first crossing returned.
            This is the desired behavior for PyBERT, because we always
            initialize the bit stream with [0, 0, 1, 1], in order to
            provide a known synchronization point for jitter analysis.
        min_init_dev(float): The minimum initial deviation from zero,
            which must be detected, before searching for crossings.
            Normalized to maximum input signal magnitude.
            (Optional; default = 0.1.)
        thresh(float): Vertical crossing threshold.

    Returns: an array of signal threshold crossing times.
    """

    if len(t) != len(x):
        raise ValueError("len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x)))

    t = np.array(t)
    x = np.array(x)

    try:
        max_mag_x = max(abs(x))
    except Exception as error:
        log = getLogger()
        log.error("len(x): %d", len(x))
        raise error
    min_mag_x = min_init_dev * max_mag_x
    i = 0
    while abs(x[i]) < min_mag_x:
        i += 1
        assert i < len(x), "Input signal minimum deviation not detected!"
    x = x[i:] - thresh
    t = t[i:]

    sign_x = np.sign(x)
    sign_x = np.where(sign_x, sign_x, np.ones(len(sign_x)))  # "0"s can produce duplicate xings.
    diff_sign_x = np.diff(sign_x)
    xing_ix = np.where(diff_sign_x)[0]
    xings = [t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) for i in xing_ix]

    if not xings:
        return np.array([])

    i = 0
    if min_delay:
        assert (
            min_delay < xings[-1]
        ), "min_delay ({}) must be less than last crossing time ({}).".format(min_delay, xings[-1])
        while xings[i] < min_delay:
            i += 1

    try:
        if rising_first and diff_sign_x[xing_ix[i]] < 0.0:
            i += 1
    except Exception as error:
        log = getLogger()
        log.error("len(diff_sign_x): %d", len(diff_sign_x))
        log.error("len(xing_ix): %d", len(xing_ix))
        log.error("i: %d", i)
        raise

    return np.array(xings[i:])


def find_crossings(
    t,
    x,
    amplitude,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev=0.1,
    mod_type=0,
):
    """
    Finds the crossing times in a signal, according to the modulation type.

    Args:
        t([float]): The times associated with each signal sample.
        x([float]): The signal samples.
        amplitude(float): The nominal signal amplitude. (Used for
            determining thresholds, in the case of some modulation
            types.)
        min_delay(float): The earliest possible sample time we want
            returned. (Optional; default = 0.)
        rising_first(bool): When True, start with the first rising edge
            found. When this option is True, the first rising edge
            crossing is the first crossing returned. This is the desired
            behavior for PyBERT, because we always initialize the bit
            stream with [0, 1, 1], in order to provide a known
            synchronization point for jitter analysis.
            (Optional; default = True.)
        min_init_dev(float): The minimum initial deviation from zero,
            which must be detected, before searching for crossings.
            Normalized to maximum input signal magnitude.
            (Optional; default = 0.1.)
        mod_type(int): The modulation type. Allowed values are:
            {0: NRZ, 1: Duo-binary, 2: PAM-4}
            (Optional; default = 0.)

    Returns: The signal threshold crossing times.
    """

    xings = []
    if mod_type == 0:  # NRZ
        xings.append(
            find_crossing_times(
                t, x, min_delay=min_delay, rising_first=rising_first, min_init_dev=min_init_dev
            )
        )
    elif mod_type == 1:  # Duo-binary
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(-0.5 * amplitude),
            )
        )
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(0.5 * amplitude),
            )
        )
    elif (
        mod_type == 2
    ):  # PAM-4 (Enabling the +/-0.67 cases yields multiple ideal crossings at the same edge.)
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(0.0 * amplitude),
            )
        )
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")

    return np.sort(np.concatenate(xings))


def calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, ws):
    """
    Calculates propagation constant from cross-sectional parameters.

    The formula's applied are taken from Howard Johnson's "Metallic Transmission Model"
    (See "High Speed Signal Propagation", Sec. 3.1.)

    Inputs:
      - R0          skin effect resistance (Ohms/m)
      - w0          cross-over freq.
      - Rdc         d.c. resistance (Ohms/m)
      - Z0          characteristic impedance in LC region (Ohms)
      - v0          propagation velocity (m/s)
      - Theta0      loss tangent
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = np.array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    Rac = R0 * np.sqrt(2 * 1j * w / w0)  # AC resistance vector
    R = np.sqrt(np.power(Rdc, 2) + np.power(Rac, 2))  # total resistance vector
    L0 = Z0 / v0  # "external" inductance per unit length (H/m)
    C0 = 1.0 / (Z0 * v0)  # nominal capacitance per unit length (F/m)
    C = C0 * np.power(
        (1j * w / w0), (-2.0 * Theta0 / np.pi)
    )  # complex capacitance per unit length (F/m)
    gamma = np.sqrt((1j * w * L0 + R) * (1j * w * C))  # propagation constant (nepers/m)
    Zc = np.sqrt((1j * w * L0 + R) / (1j * w * C))  # characteristic impedance (Ohms)

    return (gamma, Zc)


def calc_G(H, Rs, Cs, Zc, RL, Cp, CL, ws):
    """
    Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance
      - Cs    source parallel (parasitic) capacitance
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - CL    load series (d.c. blocking) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     frequency dependent transfer function of channel
    """

    w = np.array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rs / (1.0 + 1j * w * Rs * Cs)
    # Rx load impedance is 2 series, a.c.-coupling capacitors, in series w/ parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = 2.0 * 1.0 / (1j * w * CL) + RL / (1.0 + 1j * w * RL * Cp / 2)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Zc / (1.0 + 1j * w * Zc * Cs)
    A = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1 = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2 = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    G = A * H * (1 + R1) / (1 - R1 * R2 * H ** 2)
    G = G * (
        ((RL / (1j * w * Cp / 2)) / (RL + 1 / (1j * w * Cp / 2))) / ZL
    )  # Corrected for divider action.
    # (i.e. - We're interested in what appears across RL.)
    return G


def calc_eye(ui, samps_per_ui, height, ys, y_max, clock_times=None):
    """
    Calculates the "eye" diagram of the input signal vector.

    Args:
        ui(float): unit interval (s)
        samps_per_ui(int): # of samples per unit interval
        height(int): height of output image data array
        ys([float]): signal vector of interest
        y_max(float): max. +/- vertical extremity of plot

    Keyword Args:
        clock_times([float]): (optional) vector of clock times to use
            for eye centers. If not provided, just use mean
            zero-crossing and assume constant UI and no phase jumps.
            (This allows the same function to be used for eye diagram
            creation, for both pre and post-CDR signals.)

    Returns: The "heat map" representing the eye diagram. Each grid
        location contains a value indicating the number of times the
        signal passed through that location.
    """

    # List/array necessities.
    ys = np.array(ys)

    # Intermediate variable calculation.
    tsamp = ui / samps_per_ui

    # Adjust the scaling.
    width = 2 * samps_per_ui
    y_scale = height // (2 * y_max)  # (pixels/V)
    y_offset = height // 2  # (pixels)

    # Generate the "heat" picture array.
    img_array = np.zeros([height, width])
    if clock_times:
        for clock_time in clock_times:
            start_time = clock_time - ui
            start_ix = int(start_time / tsamp)
            if start_ix + 2 * samps_per_ui > len(ys):
                break
            interp_fac = (start_time - start_ix * tsamp) // tsamp
            i = 0
            for (samp1, samp2) in zip(
                ys[start_ix : start_ix + 2 * samps_per_ui],
                ys[start_ix + 1 : start_ix + 1 + 2 * samps_per_ui],
            ):
                y = samp1 + (samp2 - samp1) * interp_fac
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                i += 1
    else:
        start_ix = np.where(np.diff(np.sign(ys)))[0][0] + samps_per_ui // 2
        last_start_ix = len(ys) - 2 * samps_per_ui
        while start_ix < last_start_ix:
            i = 0
            for y in ys[start_ix : start_ix + 2 * samps_per_ui]:
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                i += 1
            start_ix += samps_per_ui

    return img_array


def make_ctle(rx_bw, peak_freq, peak_mag, w, mode="Passive", dc_offset=0):
    """
    Generate the frequency response of a continuous time linear
    equalizer (CTLE), given the:

    - signal path bandwidth,
    - peaking specification
    - list of frequencies of interest, and
    - operational mode/offset.

    We use the 'invres()' function from scipy.signal, as it suggests
    itself as a natural approach, given our chosen use model of having
    the user provide the peaking frequency and degree of peaking.

    That is, we define our desired frequency response using one zero
    and two poles, where:

    - The pole locations are equal to:
       - the signal path natural bandwidth, and
       - the user specified peaking frequency.

    - The zero location is chosen, so as to provide the desired degree
      of peaking.

    Inputs:

      - rx_bw        The natural (or, unequalized) signal path bandwidth (Hz).

      - peak_freq    The location of the desired peak in the frequency
                     response (Hz).

      - peak_mag     The desired relative magnitude of the peak (dB). (mag(H(0)) = 1)

      - w            The list of frequencies of interest (rads./s).

      - mode         The operational mode; must be one of:
                       - 'Off'    : CTLE is disengaged.
                       - 'Passive': Maximum frequency response has magnitude one.
                       - 'AGC'    : Automatic gain control. (Handled by calling routine.)
                       - 'Manual' : D.C. offset is set manually.

      - dc_offset    The d.c. offset of the CTLE gain curve (dB).
                     (Only valid, when 'mode' = 'Manual'.)

    Outputs:

      - w, H         The resultant complex frequency response, at the
                     given frequencies.

    """

    if mode == "Off":
        return (w, np.ones(len(w)))

    p2 = -2.0 * np.pi * rx_bw
    p1 = -2.0 * np.pi * peak_freq
    z = p1 / pow(10.0, peak_mag / 20.0)
    if p2 != p1:
        r1 = (z - p1) / (p2 - p1)
        r2 = 1 - r1
    else:
        r1 = -1.0
        r2 = z - p1
    b, a = invres([r1, r2], [p1, p2], [])
    w, H = freqs(b, a, w)

    if mode == "Passive":
        H /= max(abs(H))
    elif mode in ("Manual", "AGC"):
        H *= pow(10.0, dc_offset / 20.0) / abs(H[0])  # Enforce d.c. offset.
    else:
        raise RuntimeError(
            "pybert_util.make_ctle(): Unrecognized value for 'mode' parameter: {}.".format(mode)
        )

    return (w, H)


def trim_impulse(g, min_len=0, max_len=1_000_000):
    """
    Trim impulse response, for more useful display, by:
      - clipping off the tail, after 99.8% of the total power has been
        captured (Using 99.9% was causing problems; I don't know why.), and
      - setting the "front porch" length equal to 20% of the total length.

    Inputs:

      - g         impulse response

      - min_len   (optional) minimum length of returned vector

      - max_len   (optional) maximum length of returned vector

    Outputs:

      - g_trim    trimmed impulse response

      - start_ix  index of first returned sample

    """

    # Trim off potential FFT artifacts from the end and capture peak location.
    g = np.array(g[: int(0.9 * len(g))])
    max_ix = np.argmax(g)

    # Capture 99.8% of the total energy.
    Pt = 0.998 * sum(g ** 2)
    i = 0
    P = 0
    while P < Pt:
        P += g[i] ** 2
        i += 1
    stop_ix = min(max_ix + max_len, max(i, max_ix + min_len))

    # Set "front porch" to 20%, guarding against negative start index.
    start_ix = max(0, max_ix - (stop_ix - max_ix) // 4)

    return (g[start_ix:stop_ix], start_ix)


def import_channel(filename, sample_per, padded=False, windowed=False):
    """
    Read in a channel file.

    Args:
        filename(str): Name of file from which to import channel description.
        sample_per(float): Sample period of signal vector (s).
        padded(Bool): (Optional) Zero pad s4p data, such that fmax >= 1/(2*sample_per)? (Default = False)
        windowed(Bool): (Optional) Window s4p data, before converting to time domain? (Default = False)

    Returns: Imported channel impulse, or step, response.
    """

    channel = Path(filename)
    if channel.suffix in ("s4p", "S4P"):
        return import_freq(channel, sample_per, padded=padded, windowed=windowed)
    return import_time(channel, sample_per)


def interp_time(ts, xs, sample_per):
    """
    Resample time domain data, using linear interpolation.

    Args:
        ts([float]): Original time values.
        xs([float]): Original signal values.
        sample_per(float): System sample period.

    Returns: Resampled waveform.
    """

    tmax = ts[-1]
    res = []
    t = 0.0
    i = 0
    while t < tmax:
        while ts[i] <= t:
            i = i + 1
        res.append(xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ts[i - 1]) / (ts[i] - ts[i - 1]))
        t += sample_per

    return np.array(res)


def import_time(filename, sample_per):
    """
    Read in a time domain waveform file, resampling as
    appropriate, via linear interpolation.

    Args:
        filename(str): Name of waveform file to read in.
        sample_per(float): New sample interval

    Returns: Resampled waveform.
    """

    # Read in original data from file.
    ts = []
    xs = []
    with open(filename, mode="rU") as file:
        for line in file:
            try:
                tmp = list(map(np.float, [_f for _f in re.split("[, ;:]+", line) if _f][0:2]))
            except Exception:
                continue
            ts.append(tmp[0])
            xs.append(tmp[1])

    return interp_time(ts, xs, sample_per)


def sdd_21(ntwk):
    """
    Given a 4-port single-ended network, return its differential throughput.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    Returns: Sdd[2,1].
    """

    if np.real(ntwk.s21.s[0, 0, 0]) < 0.5:  # 1 ==> 3 port numbering?
        ntwk.renumber((2, 3), (3, 2))

    return 0.5 * (ntwk.s21 - ntwk.s23 + ntwk.s43 - ntwk.s41)


def import_freq(filename, sample_per, padded=False, windowed=False, f_step=10e6):
    """
    Read in a single ended 4-port Touchstone file, and extract the
    differential throughput step response, resampling as
    appropriate, via linear interpolation.

    Args:
        filename(str): Name of Touchstone file to read in.
        sample_per(float): New sample interval
        padded(Bool): (Optional) Zero pad s4p data, such that fmax >= 1/(2*sample_per)? (Default = False)
        windowed(Bool): (Optional) Window s4p data, before converting to time domain? (Default = False)

    Returns: Resampled step response waveform.
    """

    ntwk = rf.Network(filename)

    # Form frequency vector.
    f = ntwk.f
    # fmin = f[0]
    # if(fmin == 0):  # Correct, if d.c. point was included in original data.
    #     fmin = f[1]
    fmin = f_step
    fmax = f[-1]
    f = np.arange(fmin, fmax + fmin, fmin)
    F = rf.Frequency.from_f(
        f / 1e9
    )  # skrf.Frequency.from_f() expects its argument to be in units of GHz.

    # Form impulse response from frequency response.
    H = sdd_21(ntwk).interpolate_from_f(F).s[:, 0, 0]
    # ntwk = ntwk.interpolate_from_f(F)
    # H = np.concatenate((H, np.conj(np.flipud(H[:-1]))))  # Forming the vector that fft() would've outputted.
    H = np.pad(H, (1, 0), "constant", constant_values=1.0)  # Presume d.c. value = 1.
    if windowed:
        window = get_window(6.0, 2 * len(H))[len(H) :]
        H *= window
    # h = np.real(np.fft.ifft(H))
    if padded:
        h = np.fft.irfft(H, int(1.0 / (fmin * sample_per)) + 1)
        fmax = 1.0 / (2.0 * sample_per)
    else:
        h = np.fft.irfft(H)
    h /= np.abs(h.sum())  # Equivalent to assuming that step response settles at 1.

    # Form step response from impulse response.
    s = np.cumsum(h)

    # Form time vector.
    t0 = 1.0 / (2.0 * fmax)  # Sampling interval = 1 / (2 fNyquist).
    t = np.array([n * t0 for n in range(len(h))])

    return interp_time(t, s, sample_per)


def lfsr_bits(taps, seed):
    """
    Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps([int]): The set of fed back taps.
                     (Largest determines order of generator.)
        seed(int): The initial value of the shift register.

    Returns:
        A PRBS generator object with a next() method, for retrieving
        the next bit in the sequence.
    """

    val = int(seed)
    num_taps = max(taps)
    mask = (1 << num_taps) - 1

    while True:
        xor_res = reduce(lambda x, b: x ^ b, [bool(val & (1 << (tap - 1))) for tap in taps])
        val = (val << 1) & mask  # Just to keep 'val' from growing without bound.
        if xor_res:
            val += 1
        yield val & 1


def safe_log10(value):
    """Guards against pesky 'Divide by 0' error messages."""

    if hasattr(value, "__len__"):
        value = np.where(value == 0, 1.0e-20 * np.ones(len(value)), value)
    elif value == 0:
        value = 1.0e-20
    return np.log10(value)


def pulse_center(p, nspui):
    """
    Determines the center of the pulse response, using the "Hula Hoop"
    algorithm (See SiSoft/Tellian's DesignCon 2016 paper.)

    Args:
        p([Float]): The single bit pulse response.
        nspui(Int): The number of vector elements per unit interval.

    Returns:
        clock_pos(Int): The estimated index at which the clock will
                        sample the main lobe.
        thresh(Float):  The vertical threshold at which the main lobe is
                        UI wide.
    """

    div = 2.0
    p_max = p.max()
    thresh = p_max / div
    main_lobe_ixs = np.where(p > thresh)[0]
    if not main_lobe_ixs.size:  # Sometimes, the optimizer really whacks out.
        return (-1, 0)  # Flag this, by returning an impossible index.

    err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui
    while err and div < 5000:
        div *= 2.0
        if err > 0:
            thresh += p_max / div
        else:
            thresh -= p_max / div
        main_lobe_ixs = np.where(p > thresh)[0]
        err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui

    clock_pos = int(np.mean([main_lobe_ixs[0], main_lobe_ixs[-1]]))
    return (clock_pos, thresh)
