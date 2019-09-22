import pybert.static as static


def test_calc_reject():
    assert static.calc_reject(1 * 1.0e12, 1 * 1.0e12) == 0.0
    assert static.calc_reject(10 * 1.0e12, 1 * 1.0e12) == 10.0
