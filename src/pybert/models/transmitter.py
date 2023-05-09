from traits.api import Float, List, Range

from pybert.gui.transmitter import TX_VIEW
from pybert.models.buffer import Buffer
from pybert.models.tx_tap import TxTapTuner


class Transmitter(Buffer):
    impedance = Float(100)  # differential source impedance (Ohms)
    capacitance = Range(
        low=0.001, high=1000, value=0.50
    )  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)

    taps = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=-0.066),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )  #: List of TxTapTuner objects.

    def default_traits_view(self):
        return TX_VIEW
