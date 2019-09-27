"""A serDes channel consists of a driver, a channel and a receiver."""
from functools import lru_cache
from logging import getLogger

from pybert.materials import MATERIALS


class Channel:
    """docstring for Channel"""

    def __init__(self):
        super(Channel, self).__init__()
        self.log = getLogger("pybert.channel")
        self.log.debug("Initializing Channel")
        self.use_ch_file: bool = False  #: Import channel description from file? (Default = False)
        self.padded: bool = False  #: Zero pad imported Touchstone data? (Default = False)
        self.windowed: bool = False  #: Apply windowing to the Touchstone data? (Default = False)
        self.f_step = 10.0  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
        self.ch_file = (
            None
        )  #: Channel file name. "*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"
        self.impulse_length = 0.0  #: Impulse response length. (Determined automatically, when 0.)
        self.material = MATERIALS["UTP_24Gauge"]

    def change_material(self, new_material):
        """Update the material properties of the channel."""
        if new_material not in MATERIALS:
            raise ValueError("Not a valid material choice.")
        self.material = MATERIALS[new_material]
