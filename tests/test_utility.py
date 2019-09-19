import pybert.utility as utility


def test_safe_log10():
    assert utility.safe_log10(0) == -20.0
