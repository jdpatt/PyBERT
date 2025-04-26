"Tx FFE tap weight tuner, used by the optimizer."

from dataclasses import dataclass


@dataclass
class TxTapTuner:
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name: str = "(noname)"
    pos: int = 0  # negative = pre-cursor; positive = post-cursor
    enabled: bool = False
    min_val: float = -0.1
    max_val: float = 0.1
    step: float = 0.01
    value: float = 0.0
    steps: int = 0  # Non-zero means we want to sweep it.
