from traits.api import Bool, Enum, File, Float, Int

from pybert.gui.reciever import RX_VIEW
from pybert.models.buffer import Buffer

gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)

gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseDfe = True  # Include DFE when running simulation.
gDfeIdeal = True  # DFE ideal summing node selector
gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 1.7  # CTLE peaking magnitude (dB)
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)


# - DFE
gDecisionScaler = 0.5
gNtaps = 5
gGain = 0.5
gNave = 100
gDfeBW = 12.0  # DFE summing node bandwidth (GHz)
# - CDR
gDeltaT = 0.1  # (ps)
gAlpha = 0.01
gNLockAve = 500  # number of UI used to average CDR locked status.
gRelLockTol = 0.1  # relative lock tolerance of CDR.
gLockSustain = 500


class Receiver(Buffer):
    resistance = Float(100)  # differential input resistance(Ohm)
    capacitance = Float(0.5)  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
    coupling_capacitance = Float(1.0)  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
    use_dfe = Bool(False)

    # - CTLE
    use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
    ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
    bandwidth = Float(gBW)  #: CTLE bandwidth (GHz).
    peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
    peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
    ctle_offset = Float(gCTLEOffset)  #: CTLE d.c. offset (dB)
    ctle_mode = Enum("Off", "Passive", "AGC", "Manual")  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    ctle_mode = "Passive"

    # - DFE
    use_dfe = Bool(gUseDfe)  #: True = use a DFE (Bool).
    sum_ideal = Bool(gDfeIdeal)  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
    decision_scaler = Float(gDecisionScaler)  #: DFE slicer output voltage (V).
    gain = Float(gGain)  #: DFE error gain (unitless).
    n_ave = Float(gNave)  #: DFE # of averages to take, before making tap corrections.
    n_taps = Int(gNtaps)  #: DFE # of taps.
    old_n_taps = n_taps
    sum_bw = Float(gDfeBW)  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

    # - CDR
    delta_t = Float(gDeltaT)  #: CDR proportional branch magnitude (ps).
    alpha = Float(gAlpha)  #: CDR integral branch magnitude (unitless).
    n_lock_ave = Int(gNLockAve)  #: CDR # of averages to take in determining lock.
    rel_lock_tol = Float(gRelLockTol)  #: CDR relative tolerance to use in determining lock.
    lock_sustain = Int(gLockSustain)  #: CDR hysteresis to use in determining lock.

    def default_traits_view(self):
        return RX_VIEW

    def _use_ami_changed(self, new_value):
        if new_value:
            self.use_dfe = False
