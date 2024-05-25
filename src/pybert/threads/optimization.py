"""
PyBERT Linear Equalization Optimizer

Original author: David Banas <capn.freako@gmail.com>  
Original date: June 21, 2017

Copyright (c) 2017 David Banas; all rights reserved World wide.

TX, RX or co optimization are run in a separate thread to keep the gui responsive.
"""

import time

from numpy import arange, array, convolve, floor, ones
from numpy.fft import irfft
from scipy.interpolate import interp1d

from pybert.models.tx_tap import TxTapTuner
from pybert.threads.stoppable import StoppableThread
from pybert.utility import make_ctle, trim_impulse, calc_resps, pulse_center

gDebugOptimize = False


def mk_tx_weights(weightss: list[list[float]], enumerated_tuners: list[tuple[int, TxTapTuner]]) -> list[list[float]]:
    """
    Make all tap weight combinations possible from a list of Tx tap tuners.

    Args:
        weightss: The current list of tap weight combinations.
        enumerated_tuners: List of pairs, each containing:
            - the index of this tap in the list, and
            - this tap tuner.

    Return:
        tap_weights: List of all possible tap weight combinations.
    """
    if not enumerated_tuners:
        return weightss
    head, *tail = enumerated_tuners
    n, tuner = head
    if not tuner.enabled:
        return mk_tx_weights(weightss, tail) 
    weight_vals = arange(tuner.min_val, tuner.max_val + tuner.step, tuner.step)
    new_weightss = []
    for weights in weightss:
        for val in weight_vals:
            weights[n] = val
            new_weightss.append(weights.copy())
    return mk_tx_weights(new_weightss, tail) 


def coopt(pybert) -> tuple[list[float], float]:
    """
    Co-optimize the Tx/Rx linear equalization, assuming ideal bounded DFE.

    Args:
        pybert(PyBERT): The PyBERT instance on which to perform co-optimization.

    Returns:
        (tx_weights, ctle_peaking): The ideal Tx FFE / Rx CTLE settings.
    """

    w         = pybert.w
    t_irfft   = pybert.t_irfft
    min_mag   = pybert.min_mag_tune
    max_mag   = pybert.max_mag_tune
    step_mag  = pybert.step_mag_tune
    rx_bw     = pybert.rx_bw_tune * 1e9
    peak_freq = pybert.peak_freq_tune * 1e9
    dfe_taps  = pybert.dfe_tap_tuners
    tx_taps   = pybert.tx_tap_tuners
    min_len   = 20 * pybert.nspui
    max_len   = 100 * pybert.nspui

    h_chnl = pybert.calc_chnl_h()
    t = pybert.t
    ui = pybert.ui
    nspui = pybert.nspui
    f = pybert.f
    _, p_chnl, _ = calc_resps(t, h_chnl, ui, f)

    n_weights = len(tx_taps)
    tx_curs_pos = max(0, -tx_taps[0].pos)  # list position at which to insert main tap

    pybert.log("\n".join([
        "Optimizing linear EQ...",
        f"\tTime step: {t[1] * 1e12:5.1f} ps",
        f"\tUnit interval: {ui * 1e12:5.1f} ps",
        f"\tOversampling factor: {nspui}",
        f"\tNumber of Tx taps: {n_weights}",
        f"\tTx cursor tap position: {tx_curs_pos}",
        ""]))

    tx_weightss = mk_tx_weights([[0 for _ in range(n_weights)],], enumerate(pybert.tx_tap_tuners))
    for tx_weights in tx_weightss:
        tx_weights.insert(tx_curs_pos, 1 - sum(abs(array(tx_weights))))

    curs_ix = None
    fom_max = -1000
    for peak_mag in arange(min_mag, max_mag + step_mag, step_mag):
        _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w)
        _ctle_h = irfft(ctle_H)
        krnl = interp1d(t_irfft, _ctle_h, bounds_error=False, fill_value=0)
        ctle_h = krnl(t)
        ctle_h *= sum(_ctle_h) / sum(ctle_h)
        ctle_h, _ = trim_impulse(ctle_h, front_porch=False, min_len=min_len, max_len=max_len)
        p_ctle = convolve(p_chnl, ctle_h)[:len(p_chnl)]
        for tx_weights in tx_weightss:
            h_tx = array(sum([[tx_weight] + [0] * (nspui - 1) for tx_weight in tx_weights], []))  # sum = concatenate
            p_tx = convolve(p_ctle, h_tx)[:len(p_ctle)]
            curs_ix, _ = pulse_center(p_tx, nspui)
            curs_amp = p_tx[curs_ix]
            if pybert.use_dfe_tune:
                for k in range(pybert.n_taps_tune):
                    isi = p_tx[curs_ix + (k + 1) * nspui]
                    ideal_tap_weight = isi / curs_amp
                    actual_tap_weight = max(dfe_taps[k].min_val, min(dfe_taps[k].max_val, ideal_tap_weight))
                    p_tx[curs_ix + (k + 1) * nspui - nspui // 2:] -= actual_tap_weight * curs_amp
            n_pre_isi = int(curs_ix / nspui)
            isi_sum = sum(abs(p_tx[curs_ix - n_pre_isi * nspui::nspui])) - curs_amp
            fom = curs_amp / isi_sum
            if fom > fom_max:
                tx_weights_best = tx_weights.copy()
                del tx_weights_best[tx_curs_pos]
                peak_mag_best = peak_mag
                clocks = 1.1 * curs_amp * ones(len(p_tx))
                for ix in range(curs_ix - n_pre_isi * nspui, len(clocks), nspui):
                    clocks[ix] = 0
                pybert.plotdata.set_data("clocks_tune", clocks)
                pybert.plotdata.set_data("ctle_out_h_tune", p_tx)
                pybert.plotdata.set_data("t_ns_chnl", pybert.t_ns[:len(p_tx)])
                fom_max = fom
    return (tx_weights_best, peak_mag_best)


# pylint: disable=no-member
class TxOptThread(StoppableThread):
    """Used to run Tx tap weight optimization in its own thread, in order to
    preserve GUI responsiveness."""

    def run(self):
        """Run the Tx equalization optimization thread."""

        pybert = self.pybert

        if self.update_status:
            pybert.status = "Optimizing Tx..."

        max_iter = pybert.max_iter

        old_taps = []
        min_vals = []
        max_vals = []
        for tuner in pybert.tx_tap_tuners:
            if tuner.enabled:
                old_taps.append(tuner.value)
                min_vals.append(tuner.min_val)
                max_vals.append(tuner.max_val)

        cons = {"type": "ineq", "fun": lambda x: 0.7 - sum(abs(x))}

        bounds = list(zip(min_vals, max_vals))

        try:
            if gDebugOptimize:
                res = minimize(
                    self.do_opt_tx,
                    old_taps,
                    bounds=bounds,
                    constraints=cons,
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize(
                    self.do_opt_tx,
                    old_taps,
                    bounds=bounds,
                    constraints=cons,
                    options={"disp": False, "maxiter": max_iter},
                )

            if self.update_status:
                if res["success"]:
                    pybert.status = "Optimization succeeded."
                else:
                    pybert.status = f"Optimization failed: {res['message']}"

        except Exception as err:  # pylint: disable=broad-exception-caught
            pybert.status = err

    def do_opt_tx(self, taps):
        """Run the Tx Optimization."""
        time.sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        tuners = pybert.tx_tap_tuners
        taps = list(taps)
        for tuner in tuners:
            if tuner.enabled:
                tuner.value = taps.pop(0)
        return pybert.cost


class RxOptThread(StoppableThread):
    """Used to run Rx tap weight optimization in its own thread, in order to
    preserve GUI responsiveness."""

    def run(self):
        """Run the Rx equalization optimization thread."""

        pybert = self.pybert

        pybert.status = "Optimizing Rx..."
        max_iter = pybert.max_iter
        max_mag_tune = pybert.max_mag_tune

        try:
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = f"Optimization failed: {res['message']}"

        except Exception as err:  # pylint: disable=broad-exception-caught
            pybert.status = err

    def do_opt_rx(self, peak_mag):
        """Run the Rx Optimization."""
        time.sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        pybert.peak_mag_tune = peak_mag
        return pybert.cost


