from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable, Optional

from pybert.optimizer.optimization import OptThread

if TYPE_CHECKING:
    from pybert.pybert import PyBERT

import logging

from pybert.constants import gPeakFreq, gPeakMag
from pybert.models.tx_tap import TxTapTuner

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self):
        self.opt_thread: Optional[OptThread] = None  #: EQ optimization thread.
        self._optimization_callbacks: list[Callable[[dict], None]] = []
        self._optimization_loop_callbacks: list[Callable[[dict], None]] = []
        self._optimization_future: Optional[Future] = None

        # - EQ Tune
        self.tx_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Pre-tap3", pos=-3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
            TxTapTuner(name="Pre-tap2", pos=-2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Pre-tap1", pos=-1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap1", pos=1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap2", pos=2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Post-tap3", pos=3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
        ]  #: EQ optimizer list of TxTapTuner objects.
        self.rx_bw_tune: float = 12.0  #: EQ optimizer CTLE bandwidth (GHz).
        self.peak_freq_tune: float = gPeakFreq  #: EQ optimizer CTLE peaking freq. (GHz).
        self.peak_mag_tune: float = gPeakMag  #: EQ optimizer CTLE peaking mag. (dB).
        self.min_mag_tune: float = 2  #: EQ optimizer CTLE peaking mag. min. (dB).
        self.max_mag_tune: float = 12  #: EQ optimizer CTLE peaking mag. max. (dB).
        self.step_mag_tune: float = 1  #: EQ optimizer CTLE peaking mag. step (dB).
        self.ctle_enable_tune: bool = True  #: EQ optimizer CTLE enable
        self.dfe_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Tap1", enabled=True, min_val=0.1, max_val=0.4, value=0.1),
            TxTapTuner(name="Tap2", enabled=True, min_val=-0.15, max_val=0.15, value=0.0),
            TxTapTuner(name="Tap3", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap4", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap5", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap6", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap7", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap8", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap9", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap10", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap11", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap12", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap13", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap14", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap15", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap16", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap17", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap18", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap19", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap20", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
        ]  #: EQ optimizer list of DFE tap tuner objects.

    def add_optimization_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called when optimization completes.

        Args:
            callback: Function that takes optimization result dict as argument
        """
        self._optimization_callbacks.append(callback)

    def add_optimization_loop_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called during optimization loop.

        Args:
            callback: Function that takes optimization loop result dict as argument
        """
        self._optimization_loop_callbacks.append(callback)

    def _notify_optimization_complete(self, result: dict) -> None:
        """Notify all optimization callbacks with result."""
        # Resolve the future if it exists (for blocking operations)
        if self._optimization_future and not self._optimization_future.done():
            self._optimization_future.set_result(result)
            self._optimization_future = None

        # Call all callbacks
        for callback in self._optimization_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in optimization callback: {e}")

    def _notify_optimization_loop_complete(self, result: dict) -> None:
        """Notify all optimization loop callbacks with result."""
        for callback in self._optimization_loop_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in optimization loop callback: {e}")

    def calculate_optimization_trials(self):
        """Calculate the number of trials for the optimization."""
        n_trials = int((self.max_mag_tune - self.min_mag_tune) / self.step_mag_tune)
        for tuner in self.tx_tap_tuners:
            n_trials *= int((tuner.max_val - tuner.min_val) / tuner.step)
        return n_trials

    def is_running(self):
        """Check if the optimization is running."""
        return self.opt_thread and self.opt_thread.is_alive()

    def optimize(self, pybert: "PyBERT", block: bool = False, timeout: int = 180):
        """Start the optimization process using the tuner values.

        Args:
            block: If True, wait for the optimization to complete and return results.
            timeout: Maximum time to wait for optimization completion in seconds.

        Returns:
            If block is True, returns the optimization results. Otherwise returns None.
        """
        if self.is_running():
            if block and self._optimization_future:
                return self._optimization_future.result(timeout=timeout)
            return None
        else:
            logger.info("Starting optimization.")

            # Create a future for blocking operations
            if block:
                self._optimization_future = Future()

            self.opt_thread = OptThread()
            self.opt_thread.pybert = pybert
            self.opt_thread.start()

            if block and self._optimization_future:
                return self._optimization_future.result(timeout=timeout)
        return None

    def stop(self):
        """Stop the running optimization."""
        logger.info("Stopping optimization.")
        if self.opt_thread and self.opt_thread.is_alive():
            self.opt_thread.stop()
            self.opt_thread.join(10)

    def apply(self, pybert: "PyBERT"):
        """Apply the optimization to the current configuration."""
        logger.info("Applying optimization.")
        for i, tap in enumerate(self.tx_tap_tuners):
            setattr(pybert.tx_taps[i], "value", tap.value)
            setattr(pybert.tx_taps[i], "enabled", tap.enabled)
        setattr(pybert, "rx_bw", self.rx_bw_tune)
        setattr(pybert, "peak_freq", self.peak_freq_tune)
        setattr(pybert, "peak_mag", self.peak_mag_tune)
        setattr(pybert, "ctle_enable", self.ctle_enable_tune)

    def reset(self):
        """Reset the optimization back to what the current configuration is."""
        logger.info("Resetting optimization.")
        # TODO: Implement reset
