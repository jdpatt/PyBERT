import logging
from dataclasses import dataclass

import numpy as np
from numpy import (
    array,
    convolve,
    cumsum,
    diff,
    histogram,
    insert,
    mean,
    ones,
    real,
    reshape,
    resize,
    sqrt,
    where,
    zeros,
)
from numpy.fft import fft, ifft
from scipy.stats import norm

from pybert.utility import safe_log10

logger = logging.getLogger(__name__)


@dataclass
class Jitter:
    jitter: np.ndarray
    t_jitter: list
    isi: float
    dcd: float
    pj: float
    rj: float
    tie_ind: np.ndarray
    thresh: np.ndarray
    jitter_spectrum: np.ndarray
    tie_ind_spectrum: np.ndarray
    spectrum_freqs: list
    hist: np.ndarray
    hist_synth: list
    bin_centers: list


def calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh=6, num_bins=99, zero_mean=True):
    """Calculate the jitter in a set of actual zero crossings, given the ideal
    crossings and unit interval.

    Inputs:

      - ui               : The nominal unit interval.
      - nui              : The number of unit intervals spanned by the input signal.
      - pattern_len      : The number of unit intervals, before input symbol stream repeats.
      - ideal_xings      : The ideal zero crossing locations of the edges.
      - actual_xings     : The actual zero crossing locations of the edges.
      - rel_thresh       : (optional) The threshold for determining periodic jitter spectral components (sigma).
      - num_bins         : (optional) The number of bins to use, when forming histograms.
      - zero_mean        : (optional) Force the mean jitter to zero, when True.

    Outputs:

      - jitter   : The total jitter.
      - t_jitter : The times (taken from 'ideal_xings') corresponding to the returned jitter values.
      - isi      : The peak to peak jitter due to intersymbol interference.
      - dcd      : The peak to peak jitter due to duty cycle distortion.
      - pj       : The peak to peak jitter due to uncorrelated periodic sources.
      - rj       : The standard deviation of the jitter due to uncorrelated unbounded random sources.
      - tie_ind  : The data independent jitter.
      - thresh   : Threshold for determining periodic components.
      - jitter_spectrum  : The spectral magnitude of the total jitter.
      - tie_ind_spectrum : The spectral magnitude of the data independent jitter.
      - spectrum_freqs   : The frequencies corresponding to the spectrum components.
      - hist        : The histogram of the actual jitter.
      - hist_synth  : The histogram of the extrapolated jitter.
      - bin_centers : The bin center values for both histograms.

    Notes:
        1. The actual crossings should arrive pre-aligned to the ideal crossings.
        And both should start near zero time.
    """

    def my_hist(x):
        """Calculates the probability mass function (PMF) of the input vector,
        enforcing an output range of [-UI/2, +UI/2], sweeping everything in.

        [-UI, -UI/2] into the first bin, and everything in [UI/2, UI]
        into the last bin.
        """
        hist, bin_edges = histogram(
            x, [-ui] + [-ui / 2.0 + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui]
        )
        bin_centers = (
            [-ui / 2.0] + [mean([bin_edges[i + 1], bin_edges[i + 2]]) for i in range(len(bin_edges) - 3)] + [ui / 2.0]
        )

        return (array(list(map(float, hist))) / sum(hist), bin_centers)

    # Check inputs.
    if not ideal_xings.all():
        raise ValueError("calc_jitter(): zero length ideal crossings vector received!")
    if not actual_xings.all():
        raise ValueError("calc_jitter(): zero length actual crossings vector received!")

    num_patterns = nui // pattern_len
    assert num_patterns, f"Need at least one full pattern repetition! (pattern_len: {pattern_len}; nui: {nui})"
    xings_per_pattern = where(ideal_xings > (pattern_len * ui))[0][0]
    if xings_per_pattern % 2 or not xings_per_pattern:
        logger.error("xings_per_pattern:", xings_per_pattern)
        logger.error("min(ideal_xings):", min(ideal_xings))
        raise AssertionError("pybert_util.calc_jitter(): Odd number of (or, no) crossings per pattern detected!")

    # Assemble the TIE track.
    i = 0
    jitter = []
    t_jitter = []
    skip_next_ideal_xing = False
    for ideal_xing in ideal_xings:
        if skip_next_ideal_xing:
            t_jitter.append(ideal_xing)
            skip_next_ideal_xing = False
            continue
        # Confine our attention to those actual crossings occuring
        # within the interval [-UI/2, +UI/2] centered around the
        # ideal crossing.
        min_t = ideal_xing - ui / 2.0
        max_t = ideal_xing + ui / 2.0
        while i < len(actual_xings) and actual_xings[i] < min_t:
            i += 1
        if i == len(actual_xings):  # We've exhausted the list of actual crossings; we're done.
            break
        if actual_xings[i] > max_t:  # Means the xing we're looking for didn't occur, in the actual signal.
            jitter.append(3.0 * ui / 4.0)  # Pad the jitter w/ alternating +/- 3UI/4.
            jitter.append(-3.0 * ui / 4.0)  # (Will get pulled into [-UI/2, UI/2], later.
            skip_next_ideal_xing = True  # If we missed one, we missed two.
        else:  # Noise may produce several crossings.
            xings = []  # We find all those within the interval [-UI/2, +UI/2]
            j = i  # centered around the ideal crossing, and take the average.
            while j < len(actual_xings) and actual_xings[j] <= max_t:
                xings.append(actual_xings[j])
                j += 1
            tie = mean(xings) - ideal_xing
            jitter.append(tie)
        t_jitter.append(ideal_xing)
    jitter = array(jitter)

    logger.debug("mean(jitter):%d", mean(jitter))
    logger.debug("len(jitter):%d", len(jitter))

    if zero_mean:
        jitter -= mean(jitter)

    # Do the jitter decomposition.
    # - Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    tie_risings = jitter.take(list(range(0, len(jitter), 2)))
    tie_fallings = jitter.take(list(range(1, len(jitter), 2)))
    tie_risings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)
    tie_fallings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)
    tie_risings = reshape(tie_risings, (num_patterns, xings_per_pattern // 2))
    tie_fallings = reshape(tie_fallings, (num_patterns, xings_per_pattern // 2))

    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    try:
        tie_risings_ave = tie_risings.mean(axis=0)
        tie_fallings_ave = tie_fallings.mean(axis=0)
        isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    except:
        logger.error("xings_per_pattern:", xings_per_pattern)
        logger.error("len(ideal_xings):", len(ideal_xings))
        raise
    isi = min(isi, ui)  # Cap the ISI at the unit interval.
    dcd = abs(mean(tie_risings_ave) - mean(tie_fallings_ave))

    # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
    tie_ave = sum(list(zip(tie_risings_ave, tie_fallings_ave)), ())
    tie_ave = resize(tie_ave, len(jitter))
    tie_ind = jitter - tie_ave

    # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
    # -- Calculate the total jitter spectrum, for display purposes only.
    # --- Make vector uniformly sampled in time, via zero padding where necessary.
    # --- (It's necessary to keep track of those elements in the resultant vector, which aren't paddings; hence, 'valid_ix'.)
    x, valid_ix = make_uniform(t_jitter, jitter, ui, nui)
    y = fft(x)
    jitter_spectrum = abs(y[: len(y) // 2]) / sqrt(len(jitter))  # Normalized, in order to make power correct.
    f0 = 1.0 / (ui * nui)
    spectrum_freqs = [i * f0 for i in range(len(y) // 2)]

    # -- Use the data independent jitter spectrum for our calculations.
    tie_ind_uniform, valid_ix = make_uniform(t_jitter, tie_ind, ui, nui)

    # --- Normalized, in order to make power correct, since we grab Rj from the freq. domain.
    # --- (I'm using the length of the vector before zero padding, because zero padding doesn't add energy.)
    # --- (This has the effect of making our final Rj estimate more conservative.)
    y = fft(tie_ind_uniform) / sqrt(len(tie_ind))
    y_mag = abs(y)
    y_mean = moving_average(y_mag, n=len(y_mag) // 10)
    y_var = moving_average((y_mag - y_mean) ** 2, n=len(y_mag) // 10)
    y_sigma = sqrt(y_var)
    thresh = y_mean + rel_thresh * y_sigma
    y_per = where(y_mag > thresh, y, zeros(len(y)))  # Periodic components are those lying above the threshold.
    y_rnd = where(y_mag > thresh, zeros(len(y)), y)  # Random components are those lying below.
    y_rnd = abs(y_rnd)
    rj = sqrt(mean((y_rnd - mean(y_rnd)) ** 2))
    tie_per = real(ifft(y_per)).take(valid_ix) * sqrt(len(tie_ind))  # Restoring shape of vector to its original,
    pj = tie_per.ptp()  # non-uniformly sampled state.

    # --- Save the spectrum, for display purposes.
    tie_ind_spectrum = y_mag[: len(y_mag) // 2]

    # - Reassemble the jitter, excluding the Rj.
    # -- Here, we see why it was necessary to keep track of the non-padded elements with 'valid_ix':
    # -- It was so that we could add the average and periodic components back together,
    # -- maintaining correct alignment between them.
    if len(tie_per) > len(tie_ave):
        tie_per = tie_per[: len(tie_ave)]
    if len(tie_per) < len(tie_ave):
        tie_ave = tie_ave[: len(tie_per)]
    jitter_synth = tie_ave + tie_per

    # - Calculate the histogram of original, for comparison.
    hist, bin_centers = my_hist(jitter)

    # - Calculate the histogram of everything, except Rj.
    hist_synth, bin_centers = my_hist(jitter_synth)

    # - Extrapolate the tails by convolving w/ complete Gaussian.
    rv = norm(loc=0.0, scale=rj)
    rj_pdf = rv.pdf(bin_centers)
    rj_pmf = rj_pdf / sum(rj_pdf)
    hist_synth = convolve(hist_synth, rj_pmf)
    tail_len = (len(bin_centers) - 1) // 2
    hist_synth = (
        [sum(hist_synth[: tail_len + 1])]
        + list(hist_synth[tail_len + 1 : len(hist_synth) - tail_len - 1])
        + [sum(hist_synth[len(hist_synth) - tail_len - 1 :])]
    )

    return Jitter(
        jitter,
        t_jitter,
        isi,
        dcd,
        pj,
        rj,
        tie_ind,
        thresh[: len(thresh) // 2],
        jitter_spectrum,
        tie_ind_spectrum,
        spectrum_freqs,
        hist,
        hist_synth,
        bin_centers,
    )


def make_uniform(t, jitter, ui, nbits):
    """Make the jitter vector uniformly sampled in time, by zero-filling where
    necessary.

    The trick, here, is creating a uniformly sampled input vector for the FFT operation,
    since the jitter samples are almost certainly not uniformly sampled.
    We do this by simply zero padding the missing samples.

    Inputs:

    - t      : The sample times for the 'jitter' vector.

    - jitter : The input jitter samples.

    - ui     : The nominal unit interval.

    - nbits  : The desired number of unit intervals, in the time domain.

    Output:

    - y      : The uniformly sampled, zero padded jitter vector.

    - y_ix   : The indices where y is valid (i.e. - not zero padded).
    """

    if len(t) < len(jitter):
        jitter = jitter[: len(t)]

    run_lengths = list(map(int, diff(t) / ui + 0.5))
    valid_ix = [0] + list(cumsum(run_lengths))
    valid_ix = [x for x in valid_ix if x < nbits]
    missing = where(array(run_lengths) > 1)[0]
    num_insertions = 0
    jitter = list(jitter)  # Because we use 'insert'.

    for i in missing:
        for _ in range(run_lengths[i] - 1):
            jitter.insert(i + 1 + num_insertions, 0.0)
            num_insertions += 1

    if len(jitter) < nbits:
        jitter.extend([0.0] * (nbits - len(jitter)))
    if len(jitter) > nbits:
        jitter = jitter[:nbits]

    return jitter, valid_ix


def moving_average(a, n=3):
    """Calculates a sliding average over the input vector.

    Args:
        a([float]): Input vector to be averaged.
        n(int): Width of averaging window, in vector samples. (Optional;
            default = 3.)

    Returns:
        [float]: the moving average of the input vector, leaving the input
            vector unchanged.
    """

    ret = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return insert(ret[n - 1 :], 0, ret[n - 1] * ones(n - 1)) / n


def jitter_html_table(channel, tx, ctle, dfe):
    try:
        isi_chnl = channel.isi * 1.0e12
        dcd_chnl = channel.dcd * 1.0e12
        pj_chnl = channel.pj * 1.0e12
        rj_chnl = channel.rj * 1.0e12
        isi_tx = tx.isi * 1.0e12
        dcd_tx = tx.dcd * 1.0e12
        pj_tx = tx.pj * 1.0e12
        rj_tx = tx.rj * 1.0e12
        isi_ctle = ctle.isi * 1.0e12
        dcd_ctle = ctle.dcd * 1.0e12
        pj_ctle = ctle.pj * 1.0e12
        rj_ctle = ctle.rj * 1.0e12
        isi_dfe = dfe.isi * 1.0e12
        dcd_dfe = dfe.dcd * 1.0e12
        pj_dfe = dfe.pj * 1.0e12
        rj_dfe = dfe.rj * 1.0e12

        isi_rej_tx = 1.0e20
        dcd_rej_tx = 1.0e20
        isi_rej_ctle = 1.0e20
        dcd_rej_ctle = 1.0e20
        pj_rej_ctle = 1.0e20
        rj_rej_ctle = 1.0e20
        isi_rej_dfe = 1.0e20
        dcd_rej_dfe = 1.0e20
        pj_rej_dfe = 1.0e20
        rj_rej_dfe = 1.0e20
        isi_rej_total = 1.0e20
        dcd_rej_total = 1.0e20
        pj_rej_total = 1.0e20
        rj_rej_total = 1.0e20

        if isi_tx:
            isi_rej_tx = isi_chnl / isi_tx
        if dcd_tx:
            dcd_rej_tx = dcd_chnl / dcd_tx
        if isi_ctle:
            isi_rej_ctle = isi_tx / isi_ctle
        if dcd_ctle:
            dcd_rej_ctle = dcd_tx / dcd_ctle
        if pj_ctle:
            pj_rej_ctle = pj_tx / pj_ctle
        if rj_ctle:
            rj_rej_ctle = rj_tx / rj_ctle
        if isi_dfe:
            isi_rej_dfe = isi_ctle / isi_dfe
        if dcd_dfe:
            dcd_rej_dfe = dcd_ctle / dcd_dfe
        if pj_dfe:
            pj_rej_dfe = pj_ctle / pj_dfe
        if rj_dfe:
            rj_rej_dfe = rj_ctle / rj_dfe
        if isi_dfe:
            isi_rej_total = isi_chnl / isi_dfe
        if dcd_dfe:
            dcd_rej_total = dcd_chnl / dcd_dfe
        if pj_dfe:
            pj_rej_total = pj_tx / pj_dfe
        if rj_dfe:
            rj_rej_total = rj_tx / rj_dfe

        # Temporary, until I figure out DPI independence.
        info_str = "<style>\n"
        # info_str += ' table td {font-size: 36px;}\n'
        # info_str += ' table th {font-size: 38px;}\n'
        info_str += " table td {font-size: 12em;}\n"
        info_str += " table th {font-size: 14em;}\n"
        info_str += "</style>\n"
        # info_str += '<font size="+3">\n'
        # End Temp.

        info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"

        info_str += "<H2>Tx Preemphasis</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_chnl,
            isi_tx,
            10.0 * safe_log10(isi_rej_tx),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_chnl,
            dcd_tx,
            10.0 * safe_log10(dcd_rej_tx),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (
            pj_chnl,
            pj_tx,
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (
            rj_chnl,
            rj_tx,
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>CTLE (+ AMI DFE)</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_tx,
            isi_ctle,
            10.0 * safe_log10(isi_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_tx,
            dcd_ctle,
            10.0 * safe_log10(dcd_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_tx,
            pj_ctle,
            10.0 * safe_log10(pj_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_tx,
            rj_ctle,
            10.0 * safe_log10(rj_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>DFE</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_ctle,
            isi_dfe,
            10.0 * safe_log10(isi_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_ctle,
            dcd_dfe,
            10.0 * safe_log10(dcd_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_ctle,
            pj_dfe,
            10.0 * safe_log10(pj_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_ctle,
            rj_dfe,
            10.0 * safe_log10(rj_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>TOTAL</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_chnl,
            isi_dfe,
            10.0 * safe_log10(isi_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_chnl,
            dcd_dfe,
            10.0 * safe_log10(dcd_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_tx,
            pj_dfe,
            10.0 * safe_log10(pj_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_tx,
            rj_dfe,
            10.0 * safe_log10(rj_rej_total),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"
    except Exception as err:
        info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"
        info_str += "Sorry, the following error occurred:\n"
        info_str += str(err)

    return info_str
