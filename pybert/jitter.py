import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

# ? See https://numpy.org/devdocs/reference/typing.html on why ndnp.array gets typed as Any


log = logging.getLogger("pybert.jitter")


@dataclass
class JitterResults:
    """The object returned from calc_jitter.

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
    """

    total_jitter: Any
    jitter_times: list
    isi: float
    dcd: float
    pj: float
    rj: float
    tie_ind: Any
    thresh: Any
    jitter_spectrum: Any
    tie_ind_spectrum: Any
    spectrum_freqs: list
    hist: Any
    hist_synth: list
    bin_centers: list


def moving_average(a, n=3):
    """
    Calculates a sliding average over the input vector.

    Args:
        a([float]): Input vector to be averaged.
        n(int): Width of averaging window, in vector samples. (Optional;
            default = 3.)

    Returns:
        [float]: the moving average of the input vector, leaving the input
            vector unchanged.
    """

    ret = np.cumsum(a, dtype=float)
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

    Returns:
        [float]: Array of signal threshold crossing times.
    """

    if len(t) != len(x):
        raise ValueError(f"len(t) ({len(t)}) and len(x) ({len(x)}) need to be the same.")

    t = np.array(t)
    x = np.array(x)

    try:
        max_mag_x = max(abs(x))
    except:
        log.error("len(x): %d", len(x))
        raise
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
        assert min_delay < xings[-1], f"min_delay ({min_delay}) must be less than last crossing time ({xings[-1]})."
        while xings[i] < min_delay:
            i += 1

    log.debug("min_delay: %d", min_delay)
    log.debug("rising_first: %d", rising_first)
    log.debug("i: %d", i)
    log.debug("max_mag_x: %d", max_mag_x)
    log.debug("min_mag_x: %d", min_mag_x)
    log.debug("xings[0]: %d", xings[0])
    log.debug("xings[i]: %d", xings[i])

    try:
        if rising_first and diff_sign_x[xing_ix[i]] < 0.0:
            i += 1
    except:
        log.error("len(diff_sign_x): %d", len(diff_sign_x))
        log.error("len(xing_ix): %d", len(xing_ix))
        log.error("i: %d", i)
        raise

    return np.array(xings[i:])


def find_crossings(
    t,
    x,
    amplitude=1.0,
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

    Returns:
        [float]: The signal threshold crossing times.
    """

    assert mod_type >= 0 and mod_type <= 2, f"ERROR: utility.find_crossings(): Unknown modulation type: {mod_type}"

    xings = []
    if mod_type == 0:  # NRZ
        xings.append(
            find_crossing_times(t, x, min_delay=min_delay, rising_first=rising_first, min_init_dev=min_init_dev)
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
    elif mod_type == 2:  # PAM-4 (Enabling the +/-0.67 cases yields multiple ideal crossings at the same edge.)
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


def make_uniform(t, jitter, ui, nbits):
    """
    Make the jitter vector uniformly sampled in time, by zero-filling where necessary.

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

    run_lengths = list(map(int, np.diff(t) / ui + 0.5))
    valid_ix = [0] + list(np.cumsum(run_lengths))
    valid_ix = [x for x in valid_ix if x < nbits]
    missing = np.where(np.array(run_lengths) > 1)[0]
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


def calc_jitter(
    ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh=6, num_bins=99, zero_mean=True
) -> JitterResults:
    """
    Calculate the jitter in a set of actual zero crossings, given the ideal crossings and unit interval.

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

    """

    def my_hist(x):
        """
        Calculates the probability mass function (PMF) of the input vector,
        enforcing an output range of [-UI/2, +UI/2], sweeping everything in [-UI, -UI/2] into the first bin,
        and everything in [UI/2, UI] into the last bin.
        """
        hist, bin_edges = np.histogram(
            x, [-ui] + [-ui / 2.0 + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui]
        )
        bin_centers = (
            [-ui / 2.0]
            + [np.mean([bin_edges[i + 1], bin_edges[i + 2]]) for i in range(len(bin_edges) - 3)]
            + [ui / 2.0]
        )

        return (np.array(list(map(float, hist))) / sum(hist), bin_centers)

    # Check inputs.
    if not ideal_xings.all():
        raise ValueError("calc_jitter(): zero length ideal crossings vector received!")
    if not actual_xings.all():
        raise ValueError("calc_jitter(): zero length actual crossings vector received!")

    # Line up first ideal/actual crossings, and count/validate crossings per pattern.
    ideal_xings = np.array(ideal_xings) - (ideal_xings[0] - ui / 2.0)
    actual_xings = np.array(actual_xings) - (actual_xings[0] - ui / 2.0)
    xings_per_pattern = np.where(ideal_xings > (pattern_len * ui))[0][0]
    if xings_per_pattern % 2 or not xings_per_pattern:
        log.debug("xings_per_pattern: %d", xings_per_pattern)
        log.debug("len(ideal_xings): %d", len(ideal_xings))
        log.debug("min(ideal_xings): %d", min(ideal_xings))
        log.debug("max(ideal_xings): %d", max(ideal_xings))
        raise AssertionError("utility.calc_jitter(): Odd number of (or, no) crossings per pattern detected!")
    num_patterns = nui // pattern_len

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
        else:  # Noise may produce several crossings. We find all those
            xings = []  # within the interval [-UI/2, +UI/2] centered
            j = i  # around the ideal crossing, and take the average.
            while j < len(actual_xings) and actual_xings[j] <= max_t:
                xings.append(actual_xings[j])
                j += 1
            tie = np.mean(xings) - ideal_xing
            jitter.append(tie)
        t_jitter.append(ideal_xing)
    jitter = np.array(jitter)

    log.debug("mean(jitter): %d", np.mean(jitter))
    log.debug("len(jitter): %d", len(jitter))

    if zero_mean:
        jitter -= np.mean(jitter)

    # Do the jitter decomposition.
    # - Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    tie_risings = jitter.take(list(range(0, len(jitter), 2)))
    tie_fallings = jitter.take(list(range(1, len(jitter), 2)))
    tie_risings.resize(num_patterns * xings_per_pattern // 2)
    tie_fallings.resize(num_patterns * xings_per_pattern // 2)
    tie_risings = np.reshape(tie_risings, (num_patterns, xings_per_pattern // 2))
    tie_fallings = np.reshape(tie_fallings, (num_patterns, xings_per_pattern // 2))

    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    try:
        tie_risings_ave = tie_risings.mean(axis=0)
        tie_fallings_ave = tie_fallings.mean(axis=0)
        isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    except:
        log.error("xings_per_pattern: %d", xings_per_pattern)
        log.error("len(ideal_xings): %d", len(ideal_xings))
        raise
    isi = min(isi, ui)  # Cap the ISI at the unit interval.
    dcd = abs(np.mean(tie_risings_ave) - np.mean(tie_fallings_ave))

    # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
    tie_ave = sum(list(zip(tie_risings_ave, tie_fallings_ave)), ())
    tie_ave = np.resize(tie_ave, len(jitter))
    tie_ind = jitter - tie_ave

    # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
    # -- Calculate the total jitter spectrum, for display purposes only.
    # --- Make vector uniformly sampled in time, via zero padding np.where necessary.
    # --- (It's necessary to keep track of those elements in the resultant vector, which aren't paddings; hence, 'valid_ix'.)
    x, valid_ix = make_uniform(t_jitter, jitter, ui, nui)
    y = np.fft.fft(x)
    jitter_spectrum = abs(y[: len(y) // 2]) / np.sqrt(len(jitter))  # Normalized, in order to make power correct.
    f0 = 1.0 / (ui * nui)
    spectrum_freqs = [i * f0 for i in range(len(y) // 2)]

    # -- Use the data independent jitter spectrum for our calculations.
    tie_ind_uniform, valid_ix = make_uniform(t_jitter, tie_ind, ui, nui)

    # --- Normalized, in order to make power correct, since we grab Rj from the freq. domain.
    # --- (I'm using the length of the vector before zero padding, because zero padding doesn't add energy.)
    # --- (This has the effect of making our final Rj estimate more conservative.)
    y = np.fft.fft(tie_ind_uniform) / np.sqrt(len(tie_ind))
    y_mag = abs(y)
    y_mean = moving_average(y_mag, n=len(y_mag) // 10)
    y_var = moving_average((y_mag - y_mean) ** 2, n=len(y_mag) // 10)
    y_sigma = np.sqrt(y_var)
    thresh = y_mean + rel_thresh * y_sigma
    y_per = np.where(y_mag > thresh, y, np.zeros(len(y)))  # Periodic components are those lying above the threshold.
    y_rnd = np.where(y_mag > thresh, np.zeros(len(y)), y)  # Random components are those lying below.
    y_rnd = abs(y_rnd)
    rj = np.sqrt(np.mean((y_rnd - np.mean(y_rnd)) ** 2))
    tie_per = np.real(np.fft.ifft(y_per)).take(valid_ix) * np.sqrt(
        len(tie_ind)
    )  # Restoring shape of vector to its original,
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
    hist_synth = np.convolve(hist_synth, rj_pmf)
    tail_len = (len(bin_centers) - 1) // 2
    hist_synth = (
        [sum(hist_synth[: tail_len + 1])]
        + list(hist_synth[tail_len + 1 : len(hist_synth) - tail_len - 1])
        + [sum(hist_synth[len(hist_synth) - tail_len - 1 :])]
    )

    return JitterResults(
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
