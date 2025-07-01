import logging
from typing import TypedDict

import numpy as np
import skrf as rf

from pybert.utility.channel import calc_gamma
from pybert.utility.sparam import import_channel, import_channel_cascade

logger = logging.getLogger(__name__)


class ChannelElement(TypedDict):
    file: str
    renumber: bool  #: Automatically fix "1=>3/2=>4" port numbering? (Default = False)


class Channel:
    """The channel model that allows for a native channel or for the user to cascade multiple s-parameters."""

    def __init__(
        self,
        elements: list[ChannelElement] = [],
        use_ch_file: bool = False,
        f_step: float = 10,
        f_max: float = 40,
        impulse_length: float = 0.0,
        Rdc: float = 0.1876,
        w0: float = 10e6,
        R0: float = 1.452,
        Theta0: float = 0.02,
        Z0: float = 100,
        v0: float = 0.67,
        l_ch: float = 0.5,
        use_window: bool = False,
    ):
        self.elements: list[ChannelElement] = elements  #: Channel elements.
        self.use_ch_file: bool = use_ch_file  #: Import channel description from file? (Default = False)
        self.f_step: float = f_step  #: Frequency step to use when constructing H(f) (MHz). (Default = 10 MHz)
        self.f_max: float = f_max  #: Frequency maximum to use when constructing H(f) (GHz). (Default = 40 GHz)
        self.impulse_length: float = impulse_length  #: Impulse response length. (Determined automatically, when 0.)
        self.Rdc: float = Rdc  #: Channel d.c. resistance (Ohms/m).
        self.w0: float = w0  #: Channel transition frequency (rads./s).
        self.R0: float = R0  #: Channel skin effect resistance (Ohms/m).
        self.Theta0: float = Theta0  #: Channel loss tangent (unitless).
        self.Z0: float = Z0  #: Channel characteristic impedance, in LC region (Ohms).
        self.v0: float = v0  #: Channel relative propagation velocity (c).
        self.l_ch: float = l_ch  #: Channel length (m).
        self.use_window: bool = (
            use_window  #: Apply raised cosine to frequency response before FFT()-ing? (Default = False)
        )

    def to_dict(self) -> dict:
        return {
            "elements": self.elements,
            "use_ch_file": self.use_ch_file,
            "f_step": self.f_step,
            "f_max": self.f_max,
            "impulse_length": self.impulse_length,
            "Rdc": self.Rdc,
            "w0": self.w0,
            "R0": self.R0,
            "Theta0": self.Theta0,
            "Z0": self.Z0,
            "v0": self.v0,
            "l_ch": self.l_ch,
            "use_window": self.use_window,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Channel":
        return cls(
            elements=data.get("elements", []),
            use_ch_file=data.get("use_ch_file", False),
            f_step=data.get("f_step", 10),
            f_max=data.get("f_max", 40),
            impulse_length=data.get("impulse_length", 0.0),
            Rdc=data.get("Rdc", 0.1876),
            w0=data.get("w0", 10e6),
            R0=data.get("R0", 1.452),
            Theta0=data.get("Theta0", 0.02),
            Z0=data.get("Z0", 100),
            v0=data.get("v0", 0.67),
            l_ch=data.get("l_ch", 0.5),
            use_window=data.get("use_window", False),
        )

    def form_file_based_channel_response(self, ts: float, f: np.ndarray) -> tuple[np.ndarray, rf.Network]:
        """Import channel response from file(s)."""
        if not self.elements:
            raise ValueError("No channel elements defined")

        if len(self.elements) == 1:
            # Single element - use existing logic for backward compatibility
            ch_s2p_pre = import_channel(self.elements[0]["file"], ts, f, renumber=self.elements[0]["renumber"])
            logger.debug(str(ch_s2p_pre))
        else:
            # Multiple elements - cascade them
            ch_s2p_pre = import_channel_cascade(self.elements, ts, f)
            logger.debug(f"Cascaded channel network: {ch_s2p_pre}")

        H = ch_s2p_pre.s21.s.flatten()
        return H, ch_s2p_pre

    def form_native_channel_response(
        self, f: np.ndarray, w: np.ndarray, len_f: int, Rs: float
    ) -> tuple[np.ndarray, rf.Network]:
        """Construct PyBERT default channel model (i.e. - Howard Johnson's UTP model)."""
        l_ch = self.l_ch
        v0 = self.v0 * 3.0e8
        R0 = self.R0
        w0 = self.w0
        Rdc = self.Rdc
        Z0 = self.Z0
        Theta0 = self.Theta0
        # - Calculate propagation constant, characteristic impedance, and transfer function.
        gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
        self.Zc = Zc
        H = np.exp(-l_ch * gamma)  # pylint: disable=invalid-unary-operand-type
        self.H = H
        # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
        tmp = np.array(list(zip(zip(np.zeros(len_f), H), zip(H, np.zeros(len_f)))))
        ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)
        # - And, finally, renormalize to driver impedance.
        ch_s2p_pre.renormalize(Rs)
        return H, ch_s2p_pre

    def form_channel_response(
        self, ts: float, f: np.ndarray, w: np.ndarray, len_f: int, Rs: float
    ) -> tuple[np.ndarray, rf.Network]:
        """Form the channel response."""
        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.use_ch_file:
            H, ch_s2p_pre = self.form_file_based_channel_response(ts, f)
        else:
            H, ch_s2p_pre = self.form_native_channel_response(f, w, len_f, Rs)
        try:
            ch_s2p_pre.name = "ch_s2p_pre"
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception(f"ch_s2p_pre: {ch_s2p_pre}")
            raise
        return H, ch_s2p_pre
