"""Module to handle all of the equalization features of pybert."""
import logging
from functools import lru_cache
from time import sleep

import numpy as np
from numpy.fft import ifft
from pybert.config import DEBUG_OPTIMIZE
from pybert.sim.utility import CTLE_MODE, fir_numerator, make_ctle
from scipy.optimize import minimize, minimize_scalar


class Equalization:
    """Any equalization that the SerDes channel can make use of."""

    def __init__(self, config):
        super(Equalization, self).__init__()
        self.log = logging.getLogger("pybert.equalization")
        self.log.debug("Initializing Equalization")
        # EQ optimizer list of TxTapTuner objects.

        self.tx_taps = config.tx_taps
        self.tx_tap_tuners = config.tx_tap_tuners

        self.use_ctle_file = (
            config.use_ctle_file
        )  # For importing CTLE impulse/step response directly.
        self.ctle_file = (
            config.ctle_file
        )  # CTLE response file (when use_ctle_file = True). ["*.csv"]
        self.rx_bw = config.rx_bw  # CTLE bandwidth (GHz).
        self.peak_freq = config.peak_freq  # CTLE peaking frequency (GHz)
        self.peak_mag = config.peak_mag  # CTLE peaking magnitude (dB)
        self.ctle_offset = config.ctle_offset  # CTLE d.c. offset (dB)
        self.ctle_mode = config.ctle_mode  # CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
        self.rx_bw_tune = config.rx_bw  # EQ optimizer CTLE bandwidth (GHz).
        self.peak_freq_tune = config.peak_freq  # EQ optimizer CTLE peaking freq. (GHz).
        self.peak_mag_tune = config.peak_mag  # EQ optimizer CTLE peaking mag. (dB).
        self.ctle_offset_tune = config.ctle_offset_tune  # EQ optimizer CTLE d.c. offset (dB).
        self.ctle_mode_tune = config.ctle_mode_tune
        self.use_dfe_tune = config.use_dfe  # EQ optimizer DFE select (Bool).
        self.n_taps_tune = config.n_taps  # EQ optimizer # DFE taps.
        self.max_iter = config.max_iter  # EQ optimizer max. # of optimization iterations.

        # - DFE
        self.use_dfe = config.use_dfe
        self.sum_ideal = config.sum_ideal  # True = use an ideal
        self.decision_scaler = config.decision_scaler  # DFE slicer output voltage (V).
        self.gain = config.gain  # DFE error gain (unitless).
        self.n_ave = config.n_ave  # DFE # of averages to take, before making tap corrections.
        self.n_taps = config.n_taps  # DFE # of taps.
        self._old_n_taps = self.n_taps
        self.sum_bw = (
            config.sum_bw
        )  # DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

        # - CDR
        self.delta_t = config.delta_t  # proportional branch magnitude (ps).
        self.alpha = config.alpha  # integral branch magnitude (unitless).
        self.n_lock_ave = config.n_lock_ave  # # of averages to take in determining lock.
        self.rel_lock_tol = config.rel_lock_tol  # relative tolerance to use in determining lock.
        self.lock_sustain = config.lock_sustain  # hysteresis to use in determining lock.

        self.tx_h_tune = np.array([])
        self.ctle_h_tune = np.array([])
        self.ctle_out_h_tune = np.array([])
        self._ffe = np.array([])

    def handler(self, button_id):
        """Given which button is pressed, run one of the actions."""
        if button_id == -2:  # Reset EQ
            self.reset_equalization()
        elif button_id == -3:  # Save EQ
            self.save_equalization()
        elif button_id == -4:  # Start Tx Opt
            self.run_tx_optimization()
        elif button_id == -5:  # Start Rx Opt
            self.run_rx_optimization()
        elif button_id == -6:  # Start Co Opt
            self.run_co_optimization()

    def reset_equalization(self):
        """Reset the equalization to the last set values."""
        self.log.debug("Reset EQ")
        for index, tuner in enumerate(self.tx_tap_tuners):
            tuner.value = self.tx_taps[index].value
            tuner.enabled = self.tx_taps[index].enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_mode_tune = self.ctle_mode
        self.ctle_offset_tune = self.ctle_offset
        self.use_dfe_tune = self.use_dfe
        self.n_taps_tune = self.n_taps

    def save_equalization(self):
        """Save the current set of equalization values."""
        self.log.debug("Save EQ")
        for index, tuner in enumerate(self.tx_taps):
            tuner.value = self.tx_tap_tuners[index].value
            tuner.enabled = self.tx_tap_tuners[index].enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_mode = self.ctle_mode_tune
        self.ctle_offset = self.ctle_offset_tune
        self.use_dfe = self.use_dfe_tune
        self.n_taps = self.n_taps_tune

    def run_tx_optimization(self, update_status=True):
        """Kick off the tx optimization."""
        self.log.debug("Run Tx Opt")
        if any([self.tx_tap_tuners[i].enabled for i in range(len(self.tx_tap_tuners))]):
            # At least one tuner should be enabled.
            if update_status:
                self.status = "Optimizing Tx..."

            max_iter = self.max_iter

            old_taps = []
            min_vals = []
            max_vals = []
            for tuner in self.tx_tap_tuners:
                if tuner.enabled:
                    old_taps.append(tuner.value)
                    min_vals.append(tuner.min_val)
                    max_vals.append(tuner.max_val)

            cons = {"type": "ineq", "fun": lambda x: 0.7 - sum(abs(x))}

            bounds = list(zip(min_vals, max_vals))

            try:
                if DEBUG_OPTIMIZE:
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

                if update_status:
                    if res["success"]:
                        self.status = "Optimization succeeded."
                    else:
                        self.status = "Optimization failed: {}".format(res["message"])

            except Exception as err:
                self.status = err

    def do_opt_tx(self, taps):
        """Run the Tx Optimization."""
        tuners = self.tx_tap_tuners
        taps = list(taps)
        for tuner in tuners:
            if tuner.enabled:
                tuner.value = taps.pop(0)
        return self.cost

    def run_rx_optimization(self):
        """Kick off the rx optimization."""
        self.log.debug("Run Rx Opt")
        if not self.ctle_mode_tune == "Off":
            self.status = "Optimizing Rx..."
            max_iter = self.max_iter

            try:
                if DEBUG_OPTIMIZE:
                    res = minimize_scalar(
                        self.do_opt_rx,
                        bounds=(0, MAX_CTLE_PEAK),
                        method="Bounded",
                        options={"disp": True, "maxiter": max_iter},
                    )
                else:
                    res = minimize_scalar(
                        self.do_opt_rx,
                        bounds=(0, MAX_CTLE_PEAK),
                        method="Bounded",
                        options={"disp": False, "maxiter": max_iter},
                    )

                if res["success"]:
                    self.status = "Optimization succeeded."
                else:
                    self.status = "Optimization failed: {}".format(res["message"])

            except Exception as err:
                self.status = err

    def do_opt_rx(self, peak_mag):
        """Run the Rx Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        self.peak_mag_tune = peak_mag
        return self.cost

    def run_co_optimization(self):
        """Kick off the co-optimization between Tx and Rx."""
        self.log.debug("Run Co Opt")
        self.status = "Co-optimizing..."
        max_iter = self.max_iter

        try:
            if DEBUG_OPTIMIZE:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, MAX_CTLE_PEAK),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, MAX_CTLE_PEAK),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                self.status = "Optimization succeeded."
            else:
                self.status = "Optimization failed: {}".format(res["message"])

        except Exception as err:
            self.status = err

    def do_coopt(self, peak_mag):
        """Run the Tx and Rx Co-Optimization."""
        self.peak_mag_tune = peak_mag
        if any([self.tx_tap_tuners[i].enabled for i in range(len(self.tx_tap_tuners))]):
            while self.tx_opt_thread and self.tx_opt_thread.is_alive():
                sleep(0.001)
            self._do_opt_tx(update_status=False)
            while self.tx_opt_thread and self.tx_opt_thread.is_alive():
                sleep(0.001)
        return self.cost

    def toggle_dfe(self, new_value):
        """ Turn on/off the DFE."""
        if not new_value:
            for tuner in self.tx_taps:
                tuner.enabled = True
        else:
            for tuner in self.tx_taps:
                tuner.enabled = False

    def toggle_tuned_dfe(self, new_value):
        """ Turn on/off the tuned DFE."""
        if not new_value:
            for tuner in self.tx_tap_tuners:
                tuner.enabled = True
        else:
            for tuner in self.tx_tap_tuners:
                tuner.enabled = False

    @property
    @lru_cache(maxsize=None)
    def ffe(self):
        """
        Generate the Tx pre-emphasis FIR numerator.
        """
        return fir_numerator(self.tx_taps)

    @lru_cache(maxsize=None)
    def get_tx_h_tune(self, nspui):
        taps = fir_numerator(self.tx_tap_tuners)
        h = sum([[x] + list(np.zeros(nspui - 1)) for x in taps], [])
        return h

    @lru_cache(maxsize=None)
    def get_ctle_h_tune(self, w, len_h):
        rx_bw = self.rx_bw_tune * 1.0e9
        peak_freq = self.peak_freq_tune * 1.0e9
        peak_mag = self.peak_mag_tune
        offset = self.ctle_offset_tune
        mode = self.ctle_mode_tune

        _, H = make_ctle(rx_bw, peak_freq, peak_mag, w, mode, offset)
        h = np.real(ifft(H))[:len_h]
        h *= abs(H[0]) / sum(h)
        return h

    @lru_cache(maxsize=None)
    def get_ctle_out_h_tune(self, chnl_h):
        tx_h = self.tx_h_tune
        ctle_h = self.ctle_h_tune
        tx_out_h = np.convolve(tx_h, chnl_h)
        h = np.convolve(ctle_h, tx_out_h)
        return h
