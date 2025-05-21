
from enum import Enum


class ModulationType(Enum):
    NRZ = "NRZ"
    DUO = "Duobinary"
    PAM4 = "PAM4"

class BitPattern(Enum):
    PRBS7 = [7, 6]
    PRBS9 = [9, 5]
    PRBS11 = [11, 9]
    PRBS13 = [13, 12, 2, 1]
    PRBS15 = [15, 14]
    PRBS20 = [20, 3]
    PRBS23 = [23, 18]
    PRBS31 = [31, 28]
