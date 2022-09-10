"""
General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 27, 2014 (Copied from control.py.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import importlib
import logging
import os.path
import pkgutil
import re
from cmath import phase, rect
from functools import reduce

import numpy as np
import skrf as rf
from numpy import array, diff, log10, mean, ones, pi, power, sign, sqrt, where, zeros
from numpy.fft import fft
from scipy.linalg import inv
from scipy.signal import freqs, invres

log = logging.getLogger("pybert.utility")


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

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    Rac = R0 * sqrt(2 * 1j * w / w0)  # AC resistance vector
    R = sqrt(power(Rdc, 2) + power(Rac, 2))  # total resistance vector
    L0 = Z0 / v0  # "external" inductance per unit length (H/m)
    C0 = 1.0 / (Z0 * v0)  # nominal capacitance per unit length (F/m)
    C = C0 * power((1j * w / w0), (-2.0 * Theta0 / pi))  # complex capacitance per unit length (F/m)
    gamma = sqrt((1j * w * L0 + R) * (1j * w * C))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L0 + R) / (1j * w * C))  # characteristic impedance (Ohms)
    Zc[0] = Z0  # d.c. impedance blows up and requires correcting.

    return (gamma, Zc)


def calc_gamma_RLGC(R, L, G, C, ws):
    """
    Calculates propagation constant from R, L, G, and C.

    Inputs:
      - R           resistance per unit length (Ohms/m)
      - L           inductance per unit length (Henrys/m)
      - G           conductance per unit length (Siemens/m)
      - C           capacitance per unit length (Farads/m)
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    gamma = sqrt((1j * w * L + R) * (1j * w * C + G))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L + R) / (1j * w * C + G))  # characteristic impedance (Ohms)

    return (gamma, Zc)


def calc_G(H, Rs, Cs, Zc, RL, Cp, ws):
    """
    Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance (differential)
      - Cs    source parallel (parasitic) capacitance (single ended)
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     transfer function of fully loaded channel
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12
    if Cp == 0:
        Cp = 1e-18

    def Rpar2C(R, C):
        """Calculates the impedance of the parallel combination of
        `R` with two `C`s in series.
        """
        return R / (1.0 + 1j * w * R * C / 2)

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rpar2C(Rs, Cs)  # The parasitic capacitances are in series.
    # Rx load impedance is parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = Rpar2C(RL, Cp)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Rpar2C(Zc, Cs)
    Y = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1 = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2 = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    return Y * H * (1 + R1) / (1 - R1 * R2 * H**2)


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

    Returns:
        2D *NumPy* array: The "heat map" representing the eye diagram. Each grid
            location contains a value indicating the number of times the
            signal passed through that location.
    """

    # List/array necessities.
    ys = array(ys)

    # Intermediate variable calculation.
    tsamp = ui / samps_per_ui

    # Adjust the scaling.
    width = 2 * samps_per_ui
    y_scale = height // (2 * y_max)  # (pixels/V)
    y_offset = height // 2  # (pixels)

    # Generate the "heat" picture array.
    img_array = zeros([height, width])
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
        start_ix = where(diff(sign(ys)))[0][0] + samps_per_ui // 2
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
        return (w, ones(len(w)))

    p2 = -2.0 * pi * rx_bw
    p1 = -2.0 * pi * peak_freq
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
        raise RuntimeError(f"utility.make_ctle(): Unrecognized value for 'mode' parameter: {mode}.")

    return (w, H)


def trim_impulse(g, min_len=0, max_len=1000000):
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
    g = array(g[: int(0.9 * len(g))])
    max_ix = np.argmax(g)

    # Capture 99.8% of the total energy.
    Pt = 0.998 * sum(g**2)
    i = 0
    P = 0
    while P < Pt:
        P += g[i] ** 2
        i += 1
    stop_ix = min(max_ix + max_len, max(i, max_ix + min_len))

    # Set "front/back porch" to 20%, doing appropriate bounds checking.
    length = stop_ix - max_ix
    porch = length // 3
    start_ix = max(0, max_ix - porch)
    stop_ix = min(len(g), stop_ix + porch)
    return (g[start_ix:stop_ix].copy(), start_ix)


def H_2_s2p(H, Zc, fs, Zref=50):
    """Convert transfer function to 2-port network.

    Args:
        H([complex]): transfer function of medium alone.
        Zc([complex]): complex impedance of medium.
        fs([real]): frequencies at which `H` and `Zc` were sampled (Hz).

    KeywordArgs:
        Zref(real): reference (i.e. - port) impedance to be used in constructing the network (Ohms). (Default: 50)

    Returns:
        skrf.Network: 2-port network representing the channel to which `H` and `Zc` pertain.
    """
    ws = 2 * pi * fs
    G = calc_G(H, Zref, 0, Zc, Zref, 0, ws)  # See `calc_G()` docstring.
    R1 = (Zc - Zref) / (Zc + Zref)  # reflection coefficient looking into medium from port
    T1 = 1 + R1  # transmission coefficient looking into medium from port
    # Z2   = Zc * (1 - R1*H**2)         # impedance looking into port 2, with port 1 terminated into Zref
    # R2   = (Z2 - Zc) / (Z2 + Zc)      # reflection coefficient looking out of port 2
    # R2   = 0
    # Z1   = Zc * (1 + R2*H**2)         # impedance looking into port 1, with port 2 terminated into Z2
    # Calculate the one-way transfer function of medium capped w/ ports of the chosen impedance.
    # G    = calc_G(H, Zref, 0, Zc, Zc, 0, 2*pi*fs)  # See `calc_G()` docstring.
    # R2   = -R1                        # reflection coefficient looking into ref. impedance
    S21 = G
    # S11  = 2*(R1 + H*R2*G)
    tmp = np.array(list(zip(zip(S11, S21), zip(S21, S11))))
    return rf.Network(s=tmp, f=fs / 1e9, z0=[Zref, Zref])  # `f` is presumed to have units: GHz.


def import_channel(filename, sample_per, fs, zref=100):
    """
    Read in a channel description file.

    Args:
        filename(str): Name of file from which to import channel description.
        sample_per(real): Sample period of system signal vector.
        fs([real]): (Positive only) frequency values being used by caller.

    KeywordArgs:
        zref(real): Reference impedance (Ohms), for time domain files. (Default = 100)

    Returns:
        skrf.Network: 2-port network description of channel.

    Notes:
        1. When a time domain (i.e. - impulse or step response) file is being imported,
        we have little choice but to use the given reference impedance as the channel
        characteristic impedance, for all frequencies. This implies two things:

            1. Importing time domain descriptions of channels into PyBERT
            yields necessarily lower fidelity results than importing Touchstone descriptions;
            probably not a surprise to those skilled in the art.

            2. The user should take care to ensure that the reference impedance value
            in the GUI is equal to the nominal characteristic impedance of the channel
            being imported when using time domain channel description files.
    """
    extension = os.path.splitext(filename)[1][1:]
    if re.search(r"^s\d+p$", extension, re.ASCII | re.IGNORECASE):  # Touchstone file?
        ts2N = interp_s2p(import_freq(filename), fs)
    else:  # simple 2-column time domain description (impulse or step).
        h = import_time(filename, sample_per)
        # Fixme: an a.c. coupled channel breaks this naive approach!
        if h[-1] > (max(h) / 2.0):  # step response?
            h = diff(h)  # impulse response is derivative of step response.
        Nf = len(fs)
        h.resize(2 * Nf)
        H = fft(h * sample_per)[:Nf]  # Keep the positive frequencies only.
        ts2N = H_2_s2p(H, zref * ones(len(H)), fs, Zref=zref)
    return ts2N


def interp_time(ts, xs, sample_per):
    """
    Resample time domain data, using linear interpolation.

    Args:
        ts([float]): Original time values.
        xs([float]): Original signal values.
        sample_per(float): System sample period.

    Returns:
        [float]: Resampled waveform.
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

    return array(res)


def import_time(filename, sample_per):
    """
    Read in a time domain waveform file, resampling as
    appropriate, via linear interpolation.

    Args:
        filename(str): Name of waveform file to read in.
        sample_per(float): New sample interval

    Returns:
        [float]: Resampled waveform.
    """
    ts = []
    xs = []
    tmp = []
    with open(filename, mode="r", encoding="UTF-8") as file:
        for line in file:
            try:
                vals = [_f for _f in re.split("[, ;:]+", line) if _f]
                tmp = list(map(float, vals[0:2]))
                ts.append(tmp[0])
                xs.append(tmp[1])
            except:
                # self._log.error(f"vals: {vals}; tmp: {tmp}; len(ts): {len(ts)}")
                continue

    return interp_time(ts, xs, sample_per)


def sdd_21(ntwk, norm=0.5):
    """
    Given a 4-port single-ended network, return its differential 2-port network.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    KeywordArgs:
        norm(real): Normalization factor. (Default = 0.5)

    Returns:
        skrf.Network: Sdd (2-port).
    """
    mm = se2mm(ntwk)
    return rf.Network(frequency=ntwk.f / 1e9, s=mm.s[:, 0:2, 0:2], z0=mm.z0[:, 0:2])


def se2mm(ntwk, norm=0.5):
    """
    Given a 4-port single-ended network, return its mixed mode equivalent.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    KeywordArgs:
        norm(real): Normalization factor. (Default = 0.5)

    Returns:
        skrf.Network: Mixed mode equivalent network, in the following format:
            Sdd11  Sdd12  Sdc11  Sdc12
            Sdd21  Sdd22  Sdc21  Sdc22
            Scd11  Scd12  Scc11  Scc12
            Scd21  Scd22  Scc21  Scc22
    """
    # Confirm correct network dimmensions.
    (_, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 4, "Touchstone file must have 4 ports!"

    # Detect/correct "1 => 3" port numbering.
    ix = ntwk.s.shape[0] // 5  # So as not to be fooled by d.c. blocking.
    if abs(ntwk.s21.s[ix, 0, 0]) < abs(ntwk.s31.s[ix, 0, 0]):  # 1 ==> 3 port numbering?
        ntwk.renumber((1, 2), (2, 1))

    # Convert S-parameter data.
    s = np.zeros(ntwk.s.shape, dtype=complex)
    s[:, 0, 0] = norm * (ntwk.s11 - ntwk.s13 - ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 0, 1] = norm * (ntwk.s12 - ntwk.s14 - ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 0, 2] = norm * (ntwk.s11 + ntwk.s13 - ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 0, 3] = norm * (ntwk.s12 + ntwk.s14 - ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 1, 0] = norm * (ntwk.s21 - ntwk.s23 - ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 1, 1] = norm * (ntwk.s22 - ntwk.s24 - ntwk.s42 + ntwk.s44).s.flatten()
    s[:, 1, 2] = norm * (ntwk.s21 + ntwk.s23 - ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 1, 3] = norm * (ntwk.s22 + ntwk.s24 - ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 2, 0] = norm * (ntwk.s11 - ntwk.s13 + ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 2, 1] = norm * (ntwk.s12 - ntwk.s14 + ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 2, 2] = norm * (ntwk.s11 + ntwk.s13 + ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 2, 3] = norm * (ntwk.s12 + ntwk.s14 + ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 3, 0] = norm * (ntwk.s21 - ntwk.s23 + ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 3, 1] = norm * (ntwk.s22 - ntwk.s24 + ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 3, 2] = norm * (ntwk.s21 + ntwk.s23 + ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 3, 3] = norm * (ntwk.s22 + ntwk.s24 + ntwk.s42 + ntwk.s44).s.flatten()

    # Convert port impedances.
    f = ntwk.f
    z = np.zeros((len(f), 4), dtype=complex)
    z[:, 0] = ntwk.z0[:, 0] + ntwk.z0[:, 2]
    z[:, 1] = ntwk.z0[:, 1] + ntwk.z0[:, 3]
    z[:, 2] = (ntwk.z0[:, 0] + ntwk.z0[:, 2]) / 2
    z[:, 3] = (ntwk.z0[:, 1] + ntwk.z0[:, 3]) / 2

    return rf.Network(frequency=f / 1e9, s=s, z0=z)


def import_freq(filename):
    """
    Read in a 1, 2, or 4-port Touchstone file,
    and return an equivalent 2-port network.

    Args:
        filename(str): Name of Touchstone file to read in.

    Returns:
        skrf.Network: 2-port network.

    Raises:
        ValueError: If Touchstone file is not 1, 2, or 4-port.

    Notes:
        1. A 4-port Touchstone file is assumed single-ended,
        and the "DD" quadrant of its mixed-mode equivalent gets returned.
    """
    # Import and sanity check the Touchstone file.
    ntwk = rf.Network(filename)
    (_, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs in (1, 2, 4), "Touchstone file must have 1, 2, or 4 ports!"

    # Convert to a 2-port network.
    if rs == 4:  # 4-port Touchstone files are assumed single-ended!
        two_port_network = sdd_21(ntwk)
    elif rs == 2:
        two_port_network = ntwk
    else:  # rs == 1
        two_port_network = rf.network.one_port_2_two_port(ntwk)

    return two_port_network


def lfsr_bits(taps, seed):
    """
    Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps([int]): The set of fed back taps.
                     (Largest determines order of generator.)
        seed(int): The initial value of the shift register.

    Returns:
        generator: A PRBS generator object with a next() method, for retrieving
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


def safe_log10(x):
    """Guards against pesky 'Divide by 0' error messages."""

    if hasattr(x, "__len__"):
        x = where(x == 0, 1.0e-20 * ones(len(x)), x)
    else:
        if x == 0:
            x = 1.0e-20

    return log10(x)


def pulse_center(p, nspui):
    """
    Determines the center of the pulse response, using the "Hula Hoop"
    algorithm (See SiSoft/Tellian's DesignCon 2016 paper.)

    Args:
        p([Float]): The single bit pulse response.
        nspui(Int): The number of vector elements per unit interval.

    Returns:
        (Int, float): The estimated index at which the clock will
            sample the main lobe, and the vertical threshold at which
            the main lobe is UI wide.
    """
    div = 2.0
    p_max = p.max()
    thresh = p_max / div
    main_lobe_ixs = where(p > thresh)[0]
    if not main_lobe_ixs.size:  # Sometimes, the optimizer really whacks out.
        return (-1, 0)  # Flag this, by returning an impossible index.

    err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui
    while err and div < 5000:
        div *= 2.0
        if err > 0:
            thresh += p_max / div
        else:
            thresh -= p_max / div
        main_lobe_ixs = where(p > thresh)[0]
        err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui

    clock_pos = int(mean([main_lobe_ixs[0], main_lobe_ixs[-1]]))
    return (clock_pos, thresh)


def submodules(package):
    """Find all sub-modules of a package."""
    rst = {}

    for _, name, _ in pkgutil.iter_modules(package.__path__):
        fullModuleName = f"{package.__name__}.{name}"
        mod = importlib.import_module(fullModuleName, package=package.__path__)
        rst[name] = mod

    return rst


def cap_mag(zs, maxMag=1.0):
    """Cap the magnitude of a list of complex values,
    leaving the phase unchanged.

    Args:
        zs([complex]): The complex values to be capped.

    KeywordArgs:
        maxMag(real): The maximum allowed magnitude. (Default = 1)

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    # orig_shape = zs.shape
    zs_flat = zs.flatten()
    subs = [rect(maxMag, phase(z)) for z in zs_flat]
    return where(abs(zs_flat) > maxMag, subs, zs_flat).reshape(zs.shape)


def mon_mag(zs):
    """Enforce monotonically decreasing magnitude in list of complex values,
    leaving the phase unchanged.

    Args:
        zs([complex]): The complex values to be adjusted.

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    # orig_shape = zs.shape
    zs_flat = zs.flatten()
    for ix in range(1, len(zs_flat)):
        zs_flat[ix] = rect(min(abs(zs_flat[ix - 1]), abs(zs_flat[ix])), phase(zs_flat[ix]))
    return zs_flat.reshape(zs.shape)


def interp_s2p(ntwk, f):
    """Safely interpolate a 2-port network, by applying certain
    constraints to any necessary extrapolation.

    Args:
        ntwk(skrf.Network): The 2-port network to be interpolated.
        f([real]): The list of new frequency sampling points.

    Returns:
        skrf.Network: The interpolated/extrapolated 2-port network.

    Raises:
        ValueError: If `ntwk` is _not_ a 2-port network.
    """
    (_, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 2, "Touchstone file must have 2 ports!"

    extrap = ntwk.interpolate(f / 1e9, fill_value="extrapolate", coords="polar", assume_sorted=True)
    s11 = cap_mag(extrap.s[:, 0, 0])
    s22 = cap_mag(extrap.s[:, 1, 1])
    s12 = ntwk.s12.interpolate(
        f / 1e9, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True
    ).s.flatten()
    s21 = ntwk.s21.interpolate(
        f / 1e9, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True
    ).s.flatten()
    s = np.array(list(zip(zip(s11, s12), zip(s21, s22))))
    if ntwk.name is None:
        ntwk.name = "s2p"
    return rf.Network(f=f / 1e9, s=s, z0=extrap.z0, name=(ntwk.name + "_interp"))


def renorm_s2p(ntwk, zs):
    """Renormalize a simple 2-port network to a new set of port impedances.

    This function was originally written as a check on the
    `skrf.Network.renormalize()` function, which I was attempting to use
    to model the Rx termination when calculating the channel impulse
    response. (See lines 1640-1650'ish of `pybert.py`.)

    In my original specific case, I was attempting to model an open
    circuit termination. And when I did the magnitude of my resultant
    S21 dropped from 0 to -44 dB!
    I didn't think that could possibly be correct.
    So, I wrote this function as a check on that.

    Args:
        ntwk(skrf.Network): A 2-port network, which must use the same
        (singular) impedance at both ports.

        zs(complex array-like): The set of new port impedances to be
        used. This set of frequencies may be unique for each port and at
        each frequency.

    Returns:
        skrf.Network: The renormalized 2-port network.
    """
    (Nf, Nr, Nc) = ntwk.s.shape
    assert Nr == 2 and Nc == 2, "May only be used to renormalize a 2-port network!"
    assert all(ntwk.z0[:, 0] == ntwk.z0[0, 0]) and all(
        ntwk.z0[:, 0] == ntwk.z0[:, 1]
    ), f"May only be used to renormalize a network with equal (singular) reference impedances! z0: {ntwk.z0}"
    assert zs.shape == (2,) or zs.shape == (
        len(ntwk.f),
        2,
    ), "The list of new impedances must have shape (2,) or (len(ntwk.f), 2)!"

    if zs.shape == (2,):
        zt = zs.repeat(len(Nf))
    else:
        zt = np.array(zs)
    z0 = ntwk.z0[0, 0]
    S = ntwk.s
    I = np.identity(2)
    Z = []
    for s in S:
        Z.append(inv(I - s).dot(I + s))  # Resultant values are normalized to z0.
    Z = np.array(Z)
    Zn = []
    for (z, zn) in zip(Z, zt):  # Iteration is over frequency and yields: (2x2 array, 2-element vector).
        Zn.append(z.dot(z0 / zn))
    Zn = np.array(Zn)
    Sn = []
    for z in Zn:
        Sn.append(inv(z + I).dot(z - I))
    return rf.Network(s=Sn, f=ntwk.f / 1e9, z0=zs)


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
        res = s2p**ntwk2
    else:  # Tx
        res = ntwk2**s2p
    return (res, ts4N, ntwk2)


def getwave_step_resp(ami_model):
    """Use a model's GetWave() function to extract its step response.

    Args:
        ami_model (): The AMI model to use.

    Returns:
        NumPy 1-D array: The model's step response.

    Raises:
        RuntimeError: When no step rise is detected.
    """
    # Delay the input edge slightly, in order to minimize high
    # frequency artifactual energy sometimes introduced near
    # the signal edges by frequency domain processing in some models.
    tmp = array([-0.5] * 128 + [0.5] * 896)  # Stick w/ 2^n, for freq. domain models' sake.
    tx_s, _ = ami_model.getWave(tmp)
    # Some models delay signal flow through GetWave() arbitrarily.
    tmp = array([0.5] * 1024)
    max_tries = 10
    n_tries = 0
    while max(tx_s) < 0 and n_tries < max_tries:  # Wait for step to rise, but not indefinitely.
        tx_s, _ = ami_model.getWave(tmp)
        n_tries += 1
    if n_tries == max_tries:
        raise RuntimeError("No step rise detected!")
    # Make one more call, just to ensure a sufficient "tail".
    tmp, _ = ami_model.getWave(tmp)
    tx_s = np.append(tx_s, tmp)
    return tx_s - tx_s[0]
