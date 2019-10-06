import numpy as np
import pybert.view.static as static


class Test_Static:
    def test_calc_reject(self):
        assert static.calc_reject(1 * 1.0e12, 1 * 1.0e12) == 0.0
        assert static.calc_reject(10 * 1.0e12, 1 * 1.0e12) == 10.0

    def test_safe_log10(self):
        assert static.safe_log10(0) == -20.0
        assert static.safe_log10(10) == 1
        np.testing.assert_array_equal(
            static.safe_log10(np.array([0.0, 1, 10, 0])), np.array([-20.0, 0.0, 1.0, -20.0])
        )
