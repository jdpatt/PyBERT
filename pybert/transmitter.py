from traits.api import (
    Bool,
    Float,
    Int,
    List,
    Range,
    HasTraits,
    String,
)

from pybert.buffer import Buffer

# - Tx
gVod = 1.0  # output drive strength (Vp)
gRs = 100  # differential source impedance (Ohms)
gCout = 0.50  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)
gRn = (
    0.001
)  # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)

# - DFE
gUseDfe = True  # Include DFE when running simulation.
gDfeIdeal = True  # DFE ideal summing node selector
gDecisionScaler = 0.5
gNtaps = 5
gGain = 0.5
gNave = 100
gDfeBW = 12.0  # DFE summing node bandwidth (GHz)

class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    def __init__(
        self,
        name: str = "(noname)",
        enabled: bool = False,
        min_val: float = 0.0,
        max_val: float = 0.0,
        value: float = 0.0,
        steps: int = 0,
    ):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(TxTapTuner, self).__init__()
        self.name = String(name)
        self.enabled = Bool(enabled)
        self.min_val = Float(min_val)
        self.max_val = Float(max_val)
        self.value = Float(value)
        self.steps = Int(steps)


class Transmitter(Buffer):
    """docstring for Transmitter"""

    def __init__(self):
        super(Transmitter, self).__init__()
        self.vod = Float(gVod)  #: Tx differential output voltage (V)
        self.rs = Float(gRs)  #: Tx source impedance (Ohms)
        self.cout = Range(low=0.001, value=gCout)  #: Tx parasitic output capacitance (pF)
        self.pn_mag = Float(gPnMag)  #: Periodic noise magnitude (V).
        self.pn_freq = Float(gPnFreq)  #: Periodic noise frequency (MHz).
        self.rn = Float(gRn)  #: Standard deviation of Gaussian random noise (V).
        self.tx_taps = List(
            [
                TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
                TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
                TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
                TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
            ]
        )  #: List of TxTapTuner objects.
        self.rel_power = Float(1.0)  #: Tx power dissipation (W).

        # - DFE
        self.sum_ideal = Bool(
            gDfeIdeal
        )  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
        self.decision_scaler = Float(gDecisionScaler)  #: DFE slicer output voltage (V).
        self.gain = Float(gGain)  #: DFE error gain (unitless).
        self.n_ave = Float(gNave)  #: DFE # of averages to take, before making tap corrections.
        self.n_taps = Int(gNtaps)  #: DFE # of taps.
        self._old_n_taps = n_taps
        self.sum_bw = Float(gDfeBW)  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).
