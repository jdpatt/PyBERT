from traits.api import (
    Bool,
    Enum,
    File,
    Float,
    Int,
    Range,
)

from pybert.buffer import Buffer

# - Rx
gRin = 100  # differential input resistance
gCin = 0.50  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gCac = 1.0  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 10.0  # CTLE peaking magnitude (dB)
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)


# - CDR
gDeltaT = 0.1  # (ps)
gAlpha = 0.01
gNLockAve = 500  # number of UI used to average CDR locked status.
gRelLockTol = 0.1  # relative lock tolerance of CDR.
gLockSustain = 500

class Receiver(Buffer):
    """docstring for Receiver"""

    def __init__(self):
        super(Receiver, self).__init__()
        self.rin = Float(gRin)  #: Rx input impedance (Ohm)
        self.cin = Range(low=0.001, value=gCin)  #: Rx parasitic input capacitance (pF)
        self.cac = Float(gCac)  #: Rx a.c. coupling capacitance (uF)
        self.use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
        self.ctle_file = File(
            "", entries=5, filter=["*.csv"]
        )  #: CTLE response file (when use_ctle_file = True).
        self.rx_bw = Float(gBW)  #: CTLE bandwidth (GHz).
        self.peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
        self.peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
        self.ctle_offset = Float(gCTLEOffset)  #: CTLE d.c. offset (dB)
        self.ctle_mode = Enum(
            "Off", "Passive", "AGC", "Manual"
        )  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').

        # - CDR
        self.delta_t = Float(gDeltaT)  #: CDR proportional branch magnitude (ps).
        self.alpha = Float(gAlpha)  #: CDR integral branch magnitude (unitless).
        self.n_lock_ave = Int(gNLockAve)  #: CDR # of averages to take in determining lock.
        self.rel_lock_tol = Float(gRelLockTol)  #: CDR relative tolerance to use in determining lock.
        self.lock_sustain = Int(gLockSustain)  #: CDR hysteresis to use in determining lock.
