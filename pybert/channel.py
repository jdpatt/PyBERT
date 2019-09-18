from logging import getLogger

from pybert.buffer import Transmitter, Receiver


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


# - Channel Control
#     - parameters for Howard Johnson's "Metallic Transmission Model"
#     - (See "High Speed Signal Propagation", Sec. 3.1.)
#     - ToDo: These are the values for 24 guage twisted copper pair; need to add other options.
gRdc = 0.1876  # Ohms/m
gw0 = 10.0e6  # 10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
gR0 = 1.452  # skin-effect resistance (Ohms/m)
gTheta0 = 0.02  # loss tangent
gZ0 = 100.0  # characteristic impedance in LC region (Ohms)
gv0 = 0.67  # relative propagation velocity (c)
gl_ch = 1.0  # cable length (m)
gRn = (
    0.001
)  # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)

class Channel(object):
    """docstring for Channel"""
    def __init__(self):
        super(Channel, self).__init__()
        self.log = logging.getLogger("pybert.channel")
        self.log.debug("Creating Channel Object")
        self.use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
        self.padded = Bool(False)  #: Zero pad imported Touchstone data? (Default = False)
        self.windowed = Bool(False)  #: Apply windowing to the Touchstone data? (Default = False)
        self.f_step = Float(10)  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
        self.ch_file = File(
            "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
        )  #: Channel file name.
        self.impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
        self.Rdc = Float(gRdc)  #: Channel d.c. resistance (Ohms/m).
        self.w0 = Float(gw0)  #: Channel transition frequency (rads./s).
        self.R0 = Float(gR0)  #: Channel skin effect resistance (Ohms/m).
        self.Theta0 = Float(gTheta0)  #: Channel loss tangent (unitless).
        self.Z0 = Float(gZ0)  #: Channel characteristic impedance, in LC region (Ohms).
        self.v0 = Float(gv0)  #: Channel relative propagation velocity (c).
        self.l_ch = Float(gl_ch)  #: Channel length (m).

        self.tx = Transmitter()
        self.rx = Receiver()
        tx_tap_tuners = List(
            [
                TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
                TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
                TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
                TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
            ]
        )  #: EQ optimizer list of TxTapTuner objects.
        rx_bw_tune = Float(gBW)  #: EQ optimizer CTLE bandwidth (GHz).
        peak_freq_tune = Float(gPeakFreq)  #: EQ optimizer CTLE peaking freq. (GHz).
        peak_mag_tune = Float(gPeakMag)  #: EQ optimizer CTLE peaking mag. (dB).
        ctle_offset_tune = Float(gCTLEOffset)  #: EQ optimizer CTLE d.c. offset (dB).
        ctle_mode_tune = Enum(
            "Off", "Passive", "AGC", "Manual"
        )  #: EQ optimizer CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
        use_dfe_tune = Bool(gUseDfe)  #: EQ optimizer DFE select (Bool).
        n_taps_tune = Int(gNtaps)  #: EQ optimizer # DFE taps.
        max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.
        tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
        rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
        coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.