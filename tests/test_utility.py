import time

import numpy as np
import pybert.utility as utility
import pytest
from pybert.equalization import TxTapTuner
from pybert.utility import StoppableThread


class Test_Utility:
    def test_safe_log10(self):
        assert utility.safe_log10(0) == -20.0
        assert utility.safe_log10(10) == 1
        np.testing.assert_array_equal(
            utility.safe_log10(np.array([0.0, 1, 10, 0])), np.array([-20.0, 0.0, 1.0, -20.0])
        )

    def test_moving_average(self):
        sample = np.arange(20)
        np.testing.assert_array_equal(
            utility.moving_average(sample),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                ]
            ),
        )

    def test_fir_numerator(self):
        tuner = [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=True, min_val=-0.3, max_val=0.3, value=2.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
        # [Pre-tap, Fir, Post-tap1, Post-tap2, Post-tap3]
        assert utility.fir_numerator(tuner) == [0.0, -1.0, 0.0, 2.0, 0.0]

    def test_lfsr_bits(self):
        lfsr = utility.lfsr_bits([2], 1)
        # Compare the first 32 bits of the Linear-Feedback Shift Register
        assert [next(lfsr) for _ in range(32)] == [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
