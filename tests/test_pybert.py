"""Test the core of pybert."""
import pytest

from pybert.pybert import PyBERT


@pytest.fixture(scope="module")
def pybert():
    return PyBERT(run_simulation=False, gui=False)


@pytest.mark.parametrize("mod_type, results", ((0, 1e-09), (1, 1e-09), (2, 2e-09)))
def test_get_unit_interval(pybert, mod_type, results):
    pybert.bit_rate = 1.0  # Gbps
    pybert.mod_type = list([mod_type])
    assert pybert._get_ui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 8000), (1, 8000), (2, 4000)))
def test_get_number_of_unit_intervals(pybert, mod_type, results):
    pybert.nbits = 8000
    pybert.mod_type = list([mod_type])
    assert pybert._get_nui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 32), (1, 32), (2, 64)))
def test_get_number_of_sample_unit_intervals(pybert, mod_type, results):
    pybert.nspb = 32
    pybert.mod_type = list([mod_type])
    assert pybert._get_nspui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 1600), (1, 1600), (2, 800)))
def test_get_number_of_eye_unit_intervals(pybert, mod_type, results):
    pybert.mod_type = list([mod_type])
    assert pybert._get_eye_uis() == results
