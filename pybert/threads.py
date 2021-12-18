from threading import Event, Thread
from time import sleep

from scipy.optimize import minimize, minimize_scalar

from pybert.control import my_run_sweeps

gDebugOptimize = False
gMaxCTLEPeak = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)

class RunSimThread(Thread):
    """Used to run the simulation in its own thread, in order to preserve GUI responsiveness."""

    def run(self):
        """Run the simulation(s)."""
        my_run_sweeps(self.the_pybert)

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
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, gMaxCTLEPeak),
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
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, gMaxCTLEPeak),
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
