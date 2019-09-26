"""Jitter calculation and container for the system."""
from dataclasses import dataclass
from logging import getLogger

import numpy as np
from numpy.fft import fft, ifft
from scipy.stats import norm

from pybert.utility import moving_average

# TODO: I'm not happy with the new jitter container.  Maybe a dict in a dict instead of a class?
#  Would update calc_jitter to return a dict instead of a class?
#  This would allow use of dict comprehensions when calculating the jitter info strings.


@dataclass
class Jitter:
    """docstring for Jitter"""

    jitter = 0.0
    t_jitter = 0.0
    isi = 0.0
    dcd = 0.0
    pj = 0.0
    rj = 0.0
    tie_ind = np.array([])
    thresh = np.array([])
    jitter_spectrum = np.array([])
    tie_ind_spectrum = np.array([])
    spectrum_freqs = np.array([])
    hist = np.array([])
    hist_synth = np.array([])
    bin_centers = np.array([])

    @classmethod
    def calc_jitter(
        cls,
        ui,
        nui,
        pattern_len,
        ideal_xings,
        actual_xings,
        rel_thresh=6,
        num_bins=99,
        zero_mean=True,
    ):
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
                x,
                [-ui] + [-ui / 2.0 + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui],
            )
            bin_centers = (
                [-ui / 2.0]
                + [
                    np.mean([bin_edges[i + 1], bin_edges[i + 2]])
                    for i in range(len(bin_edges) - 3)
                ]
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
            log = getLogger()
            log.error("xings_per_pattern: %d", xings_per_pattern)
            log.error("len(ideal_xings): %d", len(ideal_xings))
            log.error("min(ideal_xings): %d", min(ideal_xings))
            log.error("max(ideal_xings): %d", max(ideal_xings))
            raise AssertionError(
                "pybert_util.calc_jitter(): Odd number of (or, no) crossings per pattern detected!"
            )
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
            if (
                actual_xings[i] > max_t
            ):  # Means the xing we're looking for didn't occur, in the actual signal.
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
        except Exception as error:
            log = getLogger()
            log.error("xings_per_pattern: %d", xings_per_pattern)
            log.error("len(ideal_xings): %d", len(ideal_xings))
            raise error
        isi = min(isi, ui)  # Cap the ISI at the unit interval.
        dcd = abs(np.mean(tie_risings_ave) - np.mean(tie_fallings_ave))

        # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
        tie_ave = sum(list(zip(tie_risings_ave, tie_fallings_ave)), ())
        tie_ave = np.resize(tie_ave, len(jitter))
        tie_ind = jitter - tie_ave

        # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
        # -- Calculate the total jitter spectrum, for display purposes only.
        # --- Make vector uniformly sampled in time, via zero padding where necessary.
        # --- (It's necessary to keep track of those elements in the resultant vector, which aren't paddings; hence, 'valid_ix'.)
        x, valid_ix = make_uniform(t_jitter, jitter, ui, nui)
        y = fft(x)
        jitter_spectrum = abs(y[: len(y) // 2]) / np.sqrt(
            len(jitter)
        )  # Normalized, in order to make power correct.
        f0 = 1.0 / (ui * nui)
        spectrum_freqs = [i * f0 for i in range(len(y) // 2)]

        # -- Use the data independent jitter spectrum for our calculations.
        tie_ind_uniform, valid_ix = make_uniform(t_jitter, tie_ind, ui, nui)

        # --- Normalized, in order to make power correct, since we grab Rj from the freq. domain.
        # --- (I'm using the length of the vector before zero padding, because zero padding doesn't add energy.)
        # --- (This has the effect of making our final Rj estimate more conservative.)
        y = fft(tie_ind_uniform) / np.sqrt(len(tie_ind))
        y_mag = abs(y)
        y_mean = moving_average(y_mag, n=len(y_mag) // 10)
        y_var = moving_average((y_mag - y_mean) ** 2, n=len(y_mag) // 10)
        y_sigma = np.sqrt(y_var)
        thresh = y_mean + rel_thresh * y_sigma
        y_per = np.where(
            y_mag > thresh, y, np.zeros(len(y))
        )  # Periodic components are those lying above the threshold.
        y_rnd = np.where(
            y_mag > thresh, np.zeros(len(y)), y
        )  # Random components are those lying below.
        y_rnd = abs(y_rnd)
        rj = np.sqrt(np.mean((y_rnd - np.mean(y_rnd)) ** 2))
        tie_per = np.real(ifft(y_per)).take(valid_ix) * np.sqrt(
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

        return cls(
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
