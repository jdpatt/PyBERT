import logging
from dataclasses import dataclass

import numpy as np
import skrf as rf
from traits.api import Bool, File, Float, HasTraits

from pybert.gui.channel import CHANNEL_VIEW
from pybert.utility import calc_gamma, import_channel

logger = logging.getLogger(__name__)


@dataclass
class NativeChannelModel:
    """THe default channel model uses Howard Johnson's "Metallic Transmission Model" for 24 gauge twisted copper pair.

    See "High Speed Signal Propagation", Sec. 3.1.
    """

    dc_resistance: float = 0.1876  # Ohms/m
    w0: float = 10.0e6  # transition frequency.  10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
    skin_resistance: float = 1.452  # skin-effect resistance (Ohms/m)
    loss_tangent: float = 0.02  # loss tangent (unitless)
    Z0: float = 100.0  # characteristic impedance in LC region (Ohms)
    v0: float = 0.67  # relative propagation velocity (c)
    length: float = 1.0  # cable length (m)


class Channel(HasTraits):
    """The representation of the physical traces, cable or channel between the buffers.

    By default, uses the native channel model but when `use_ch_file` is set, imports the model from a file.
    """

    use_ch_file = Bool(False)  #: Import channel description from file?
    filepath = File("", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"])
    f_step = Float(10)  # MHz

    dc_resistance = Float(NativeChannelModel.dc_resistance)
    w0 = Float(NativeChannelModel.w0)
    skin_resistance = Float(NativeChannelModel.skin_resistance)
    loss_tangent = Float(NativeChannelModel.loss_tangent)
    Z0 = Float(NativeChannelModel.Z0)
    v0 = Float(NativeChannelModel.v0)
    length = Float(NativeChannelModel.length)

    def default_traits_view(self):
        return CHANNEL_VIEW

    def network_from_file(self, ts, fs):
        """Create a 2 port network from an external file that should be a s-parameter or similar file."""
        if self.filepath:
            try:
                return import_channel(self.filepath, ts, fs)
            except Exception:
                raise RuntimeError("Unable to import channel file.  Verify format and data.")
        raise ValueError("Filepath is not set or is empty.")

    def network_from_native_model(self, f, w):
        """Create a 2 port network from the builtin channel model."""
        len_f = len(f)

        # - Calculate propagation constant, characteristic impedance, and transfer function.
        gamma, Zc = calc_gamma(
            self.skin_resistance, self.w0, self.dc_resistance, self.Z0, self.v0 * 3.0e8, self.loss_tangent, w
        )
        H = np.exp(-self.length * gamma)

        # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
        tmp = np.array(list(zip(zip(np.zeros(len_f), H), zip(H, np.zeros(len_f)))))
        ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)

        return ch_s2p_pre
