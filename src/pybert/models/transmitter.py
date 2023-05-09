from traits.api import Float, List, Range

from pybert.gui.transmitter import TX_VIEW
from pybert.models.buffer import Buffer, add_ondie_s
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

    def add_characteristics_to_channel(self, channel):
        """Optionally, add the driver characteristics to the channel."""
        if self.use_ibis and self.use_ondie_sparameters:
            merged_channel, _, _ = add_ondie_s(channel, self.ami_config.get_touchstone_file(), is_rx=False)
            return merged_channel
        else:
            return channel  # Un-modified.

    def get_impedance(self) -> float:
        """Return the driver impedance.

        If an ibis is set and enabled, will return the value from it otherwise the native value.
        """
        if self.use_ibis:
            return self.ibis_model.zout * 2
        else:
            return self.impedance

    def get_capacitance(self) -> float:
        """Return the driver capacitance.

        If an ibis is set and enabled, will return the value from it otherwise the native value.
        """
        if self.use_ibis:
            return self.ibis_model.ccomp[0] / 2  # They're in series.
        else:
            return self.capacitance * 1.0e-12
