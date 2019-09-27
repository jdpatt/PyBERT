"""A serDes channel consists of a driver, a channel and a receiver."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pybert.materials import MATERIALS, Materials


@dataclass
class Channel:
    """docstring for Channel"""

    use_ch_file: bool = False  # Import channel description from file?
    padded: bool = False  # Zero pad imported Touchstone data?
    windowed: bool = False  # Apply windowing to the Touchstone data?
    f_step: float = 10.0  # Frequency step to use when constructing H(f). (MHz)
    filename: Union[
        Path, None
    ] = None  # "*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"
    impulse_length: float = 0.0  # Impulse response length. (Determined automatically, when 0.)
    material: Materials = MATERIALS["UTP_24Gauge"]

    chnl_dly: float = 0.0

    def change_material(self, new_material):
        """Update the material properties of the channel."""
        if new_material not in MATERIALS:
            raise ValueError("Not a valid material choice.")
        self.material = MATERIALS[new_material]
