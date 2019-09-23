"""Module to handle all of the equalization features of pybert."""
from threading import Event, Thread
from time import sleep

from pybert.defaults import (
    ALPHA,
    BANDWIDTH,
    CTLE_OFFSET,
    DEBUG_OPTIMIZE,
    DECISION_SCALER,
    DELTA_T,
    DFE_BW,
    DFE_IDEAL,
    DFE_NUM_AVG,
    GAIN,
    LOCK_SUSTAIN,
    MAX_CTLE_PEAK,
    NUM_LOCK_AVG,
    NUM_TAPS,
    PEAK_FREQ,
    PEAK_MAG,
    REL_LOCK_TOL,
    USE_DFE,
)
from scipy.optimize import minimize, minimize_scalar
from traits.api import (
    Array,
    Bool,
    Enum,
    File,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Property,
    String,
    cached_property,
)


class StoppableThread(Thread):
    """
    Thread class with a stop() method.

    The thread itself has to check regularly for the stopped() condition.

    All PyBERT thread classes are subclasses of this class.
    """

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = Event()

    def stop(self):
        """Called by thread invoker, when thread should be stopped prematurely."""
        self._stop_event.set()

    def stopped(self):
        """Should be called by thread (i.e. - subclass) periodically and, if this function
        returns True, thread should clean itself up and quit ASAP.
        """
        return self._stop_event.is_set()


class TxOptThread(StoppableThread):
    """Used to run Tx tap weight optimization in its own thread,
    in order to preserve GUI responsiveness.
    """

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

            if self.update_status:
                if res["success"]:
                    pybert.status = "Optimization succeeded."
                else:
                    pybert.status = "Optimization failed: {}".format(res["message"])

        except Exception as err:
            pybert.status = err

    def do_opt_tx(self, taps):
        """Run the Tx Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

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
    """Used to run Rx tap weight optimization in its own thread,
    in order to preserve GUI responsiveness.
    """

    def run(self):
        """Run the Rx equalization optimization thread."""

        pybert = self.pybert

        pybert.status = "Optimizing Rx..."
        max_iter = pybert.max_iter

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
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = "Optimization failed: {}".format(res["message"])

        except Exception as err:
            pybert.status = err

    def do_opt_rx(self, peak_mag):
        """Run the Rx Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        pybert.peak_mag_tune = peak_mag
        return pybert.cost


class CoOptThread(StoppableThread):
    """Used to run co-optimization in its own thread, in order to preserve GUI responsiveness."""

    def run(self):
        """Run the Tx/Rx equalization co-optimization thread."""

        pybert = self.pybert

        pybert.status = "Co-optimizing..."
        max_iter = pybert.max_iter

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
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = "Optimization failed: {}".format(res["message"])

        except Exception as err:
            pybert.status = err

    def do_coopt(self, peak_mag):
        """Run the Tx and Rx Co-Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        pybert.peak_mag_tune = peak_mag
        if any([pybert.tx_tap_tuners[i].enabled for i in range(len(pybert.tx_tap_tuners))]):
            while pybert.tx_opt_thread and pybert.tx_opt_thread.isAlive():
                sleep(0.001)
            pybert._do_opt_tx(update_status=False)
            while pybert.tx_opt_thread and pybert.tx_opt_thread.isAlive():
                sleep(0.001)
        return pybert.cost


class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name = String("(noname)")
    enabled = Bool(False)
    min_val = Float(0.0)
    max_val = Float(0.0)
    value = Float(0.0)
    steps = Int(0)  # Non-zero means we want to sweep it.

    def __init__(
        self, name="(noname)", enabled=False, min_val=0.0, max_val=0.0, value=0.0, steps=0
    ):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(TxTapTuner, self).__init__()

        self.name = name
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.steps = steps


class Equalization(HasTraits):
    """docstring for Equalization"""

    def __init__(self):
        super(Equalization, self).__init__()
        self.tx_taps = List(
            [
                TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
                TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
                TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
                TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
            ]
        )  #: List of TxTapTuner objects.
        self.tx_tap_tuners = List(
            [
                TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
                TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
                TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
                TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
            ]
        )  #: EQ optimizer list of TxTapTuner objects.
        self.use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
        self.ctle_file = File(
            "", entries=5, filter=["*.csv"]
        )  #: CTLE response file (when use_ctle_file = True).
        self.rx_bw = Float(BANDWIDTH)  #: CTLE bandwidth (GHz).
        self.peak_freq = Float(PEAK_FREQ)  #: CTLE peaking frequency (GHz)
        self.peak_mag = Float(PEAK_MAG)  #: CTLE peaking magnitude (dB)
        self.ctle_offset = Float(CTLE_OFFSET)  #: CTLE d.c. offset (dB)
        self.ctle_mode = Enum(
            "Off", "Passive", "AGC", "Manual"
        )  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').

        self.rx_bw_tune = Float(BANDWIDTH)  #: EQ optimizer CTLE bandwidth (GHz).
        self.peak_freq_tune = Float(PEAK_FREQ)  #: EQ optimizer CTLE peaking freq. (GHz).
        self.peak_mag_tune = Float(PEAK_MAG)  #: EQ optimizer CTLE peaking mag. (dB).
        self.ctle_offset_tune = Float(CTLE_OFFSET)  #: EQ optimizer CTLE d.c. offset (dB).
        self.ctle_mode_tune = Enum(
            "Off", "Passive", "AGC", "Manual"
        )  #: EQ optimizer CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
        self.use_dfe_tune = Bool(USE_DFE)  #: EQ optimizer DFE select (Bool).
        self.n_taps_tune = Int(NUM_TAPS)  #: EQ optimizer # DFE taps.
        self.max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.
        self.tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
        self.rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
        self.coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.

        # - DFE
        self.use_dfe = Bool(USE_DFE)  #: True = use a DFE (Bool).
        self.sum_ideal = Bool(
            DFE_IDEAL
        )  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
        self.decision_scaler = Float(DECISION_SCALER)  #: DFE slicer output voltage (V).
        self.gain = Float(GAIN)  #: DFE error gain (unitless).
        self.n_ave = Float(
            DFE_NUM_AVG
        )  #: DFE # of averages to take, before making tap corrections.
        self.n_taps = Int(NUM_TAPS)  #: DFE # of taps.
        self._old_n_taps = self.n_taps
        self.sum_bw = Float(
            DFE_BW
        )  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

        # - CDR
        self.delta_t = Float(DELTA_T)  #: CDR proportional branch magnitude (ps).
        self.alpha = Float(ALPHA)  #: CDR integral branch magnitude (unitless).
        self.n_lock_ave = Int(NUM_LOCK_AVG)  #: CDR # of averages to take in determining lock.
        self.rel_lock_tol = Float(
            REL_LOCK_TOL
        )  #: CDR relative tolerance to use in determining lock.
        self.lock_sustain = Int(LOCK_SUSTAIN)  #: CDR hysteresis to use in determining lock.

        self.tx_h_tune = Property(Array, depends_on=["tx_tap_tuners.value", "nspui"])
        self.ctle_h_tune = Property(
            Array,
            depends_on=[
                "peak_freq_tune",
                "peak_mag_tune",
                "rx_bw_tune",
                "w",
                "len_h",
                "ctle_mode_tune",
                "ctle_offset_tune",
                "use_dfe_tune",
                "n_taps_tune",
            ],
        )
        self.ctle_out_h_tune = Property(Array, depends_on=["tx_h_tune", "ctle_h_tune", "chnl_h"])

    def reset_equalization(self):
        """Reset the equalization to the last set values."""
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

    def run_tx_optimization(self):
        """Kick off the tx optimization thread if its not already running."""
        if (
            self.tx_opt_thread
            and self.tx_opt_thread.isAlive()
            or not any([self.tx_tap_tuners[i].enabled for i in range(len(self.tx_tap_tuners))])
        ):
            pass
        else:
            self._do_opt_tx()

    def _do_opt_tx(self, update_status=True):
        self.tx_opt_thread = TxOptThread()
        self.tx_opt_thread.pybert = self
        self.tx_opt_thread.update_status = update_status
        self.tx_opt_thread.start()

    def run_rx_optimization(self):
        """Kick off the rx optimization thread if its not already running."""
        if self.rx_opt_thread and self.rx_opt_thread.isAlive() or self.ctle_mode_tune == "Off":
            pass
        else:
            self.rx_opt_thread = RxOptThread()
            self.rx_opt_thread.pybert = self
            self.rx_opt_thread.start()

    def run_co_optimization(self):
        """Kick off the co-optimization between Tx and Rx thread if its not already running."""
        if self.coopt_thread and self.coopt_thread.isAlive():
            pass
        else:
            self.coopt_thread = CoOptThread()
            self.coopt_thread.pybert = self
            self.coopt_thread.start()

    def abort_optimization(self):
        """Halt all optimization threads."""
        if self.coopt_thread and self.coopt_thread.isAlive():
            self.coopt_thread.stop()
            self.coopt_thread.join(10)
        if self.tx_opt_thread and self.tx_opt_thread.isAlive():
            self.tx_opt_thread.stop()
            self.tx_opt_thread.join(10)
        if self.rx_opt_thread and self.rx_opt_thread.isAlive():
            self.rx_opt_thread.stop()
            self.rx_opt_thread.join(10)

    def toggle_dfe(self, new_value):
        """ Turn on/off the DFE."""
        if not new_value:
            for i in range(1, 4):
                self.tx_taps[i].enabled = True
        else:
            for i in range(1, 4):
                self.tx_taps[i].enabled = False

    def toggle_tunded_dfe(self, new_value):
        """ Turn on/off the tuned DFE."""
        if not new_value:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = True
        else:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = False

    @cached_property
    def _get_ffe(self):
        """
        Generate the Tx pre-emphasis FIR numerator.
        """

        return get_tap_fir_numerator(self.tx_taps)


def get_tap_fir_numerator(tap_tuners):
    """docstring"""
    taps = []
    for tuner in tap_tuners:
        if tuner.enabled:
            taps.append(tuner.value)
        else:
            taps.append(0.0)
    taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

    return taps
