import logging
from typing import Optional

import numpy as np
from numpy.fft import irfft
from traits.api import (
    Array,
    Bool,
    Button,
    Enum,
    File,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Property,
    cached_property,
)

from pybert.gui.optimizer import OPTIMIZER_VIEW
from pybert.models.tx_tap import TxTapTuner
from pybert.threads.optimization import CoOptThread, RxOptThread, TxOptThread
from pybert.utility import make_ctle

logger = logging.getLogger(__name__)

gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)

gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 1.7  # CTLE peaking magnitude (dB)

gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseDfe = True  # Include DFE when running simulation.
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)
gNtaps = 5
gMaxCTLEPeak = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)


class Optimizer(HasTraits):
    tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
    rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
    coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.

    tx_tap_tuners = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )

    #: EQ optimizer list of TxTapTuner objects.
    rx_bw_tune = Float(gBW)  #: EQ optimizer CTLE bandwidth (GHz).
    peak_freq_tune = Float(gPeakFreq)  #: EQ optimizer CTLE peaking freq. (GHz).
    peak_mag_tune = Float(gPeakMag)  #: EQ optimizer CTLE peaking mag. (dB).
    max_mag_tune = Float(20)  #: EQ optimizer CTLE peaking mag. (dB).
    ctle_offset_tune = Float(gCTLEOffset)  #: EQ optimizer CTLE d.c. offset (dB).
    ctle_mode_tune = Enum("Off", "Passive", "AGC", "Manual")  #: EQ optimizer CTLE mode
    use_dfe_tune = Bool(gUseDfe)  #: EQ optimizer DFE select (Bool).
    n_taps_tune = Int(gNtaps)  #: EQ optimizer # DFE taps.
    max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.

    # Custom buttons, which we'll use in particular tabs.
    # (Globally applicable buttons, such as "Run" and "Ok", are handled more simply, in the View.)
    btn_rst_eq = Button(label="ResetEq")
    btn_save_eq = Button(label="SaveEq")
    btn_opt_tx = Button(label="OptTx")
    btn_opt_rx = Button(label="OptRx")
    btn_coopt = Button(label="CoOpt")
    btn_abort = Button(label="Abort")

    def default_traits_view(self):
        return OPTIMIZER_VIEW

    # Independent variable setting intercepts
    # (Primarily, for debugging.)
    def _set_ctle_peak_mag_tune(self, val):
        if val > gMaxCTLEPeak or val < 0.0:
            raise RuntimeError("CTLE peak magnitude out of range!")
        self.peak_mag_tune = val

    def _btn_rst_eq_fired(self):
        """Reset the equalization."""
        for i in range(4):
            self.tx_tap_tuners[i].value = self.tx.taps[i].value
            self.tx_tap_tuners[i].enabled = self.tx.taps[i].enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_mode_tune = self.ctle_mode
        self.ctle_offset_tune = self.ctle_offset
        self.use_dfe_tune = self.use_dfe
        self.n_taps_tune = self.n_taps

    def _btn_save_eq_fired(self):
        """Save the equalization."""
        for i in range(4):
            self.tx.taps[i].value = self.tx_tap_tuners[i].value
            self.tx.taps[i].enabled = self.tx_tap_tuners[i].enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_mode = self.ctle_mode_tune
        self.ctle_offset = self.ctle_offset_tune
        self.use_dfe = self.use_dfe_tune
        self.n_taps = self.n_taps_tune

    def _btn_opt_tx_fired(self):
        if (
            self.tx_opt_thread
            and self.tx_opt_thread.is_alive()
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

    def _btn_opt_rx_fired(self):
        if self.rx_opt_thread and self.rx_opt_thread.is_alive() or self.ctle_mode_tune == "Off":
            pass
        else:
            self.rx_opt_thread = RxOptThread()
            self.rx_opt_thread.pybert = self
            self.rx_opt_thread.start()

    def _btn_coopt_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            pass
        else:
            self.coopt_thread = CoOptThread()
            self.coopt_thread.pybert = self
            self.coopt_thread.start()

    def _btn_abort_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            self.coopt_thread.stop()
            self.coopt_thread.join(10)
        if self.tx_opt_thread and self.tx_opt_thread.is_alive():
            self.tx_opt_thread.stop()
            self.tx_opt_thread.join(10)
        if self.rx_opt_thread and self.rx_opt_thread.is_alive():
            self.rx_opt_thread.stop()
            self.rx_opt_thread.join(10)

    def _use_dfe_tune_changed(self, new_value):
        if not new_value:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = True
        else:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = False
