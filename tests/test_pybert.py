"""Test the core of pybert."""
from unittest.mock import patch

import pytest

from pybert import pybert
from pyibisami.ami.configurator import AMIParamConfigurator

# pylint: disable=redefined-outer-name,protected-access


@pytest.fixture(scope="module")
def app():
    """Return a pybert object without a GUI."""
    return pybert.PyBERT(run_simulation=False, gui=False)


@pytest.mark.parametrize("mod_type, results", ((0, 1e-09), (1, 1e-09), (2, 2e-09)))
def test_get_unit_interval(app, mod_type, results):
    """Give a modulation type, make sure the unit interval in ns is correct."""
    app.bit_rate = 1.0  # Gbps
    app.mod_type = list([mod_type])
    assert app._get_ui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 8000), (1, 8000), (2, 4000)))
def test_get_number_of_unit_intervals(app, mod_type, results):
    """Give a modulation type and number of bits, calculate number of unit intervals."""
    app.nbits = 8000
    app.mod_type = list([mod_type])
    assert app._get_nui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 32), (1, 32), (2, 64)))
def test_get_number_of_sample_unit_intervals(app, mod_type, results):
    """Give a modulation type and number of sample per bit, calculate number of sample ui."""
    app.nspb = 32
    app.mod_type = list([mod_type])
    assert app._get_nspui() == results


@pytest.mark.parametrize("mod_type, results", ((0, 1600), (1, 1600), (2, 800)))
def test_get_number_of_eye_unit_intervals(app, mod_type, results):
    """Give a modulation type, calculate number of unit intervals used for the eye diagrams."""
    app.mod_type = list([mod_type])
    assert app._get_eye_uis() == results


def test_enable_using_dfe_should_enable_tx_taps(app):
    """Verify that when the `use_dfe` checkbox is cleared, it disables all taps and the inverse."""
    app._use_dfe_changed(False)
    for tuner in app.tx_taps:
        assert not tuner.enabled
    app._use_dfe_changed(True)
    for tuner in app.tx_taps:
        assert tuner.enabled


@patch.object(pybert, "my_run_simulation", autospec=True)
def test_simulation_sweeping(mock_run_sim, app):
    """No step values are set, so pybert would only run one scenario * `sweep_aves` simulations."""
    # pylint: disable=unused-argument
    app.sweep_sim = True
    app.sweep_aves = 5
    app.run_simulations()
    assert app.num_sweeps == 5


def test_set_ibis_model_without_ami(ibis_test_file, app):
    """When the IBIS file is updated in the GUI, read it and set flags accordlying.

    Since this ibis model does not have an algorithmic model section it won't try to load the ami
    and dll files.
    """
    assert not app.tx_ibis_valid
    app._tx_ibis_file_changed(ibis_test_file)
    assert app.tx_ibis_valid
    assert not app.tx_dll_valid
    assert not app.tx_ami_valid
    assert app.tx_dll_file == ""
    assert app.tx_ami_file == ""
    assert "example_tx" in app._tx_ibis.models


def test_set_ami_file(ami_test_file, app):
    """Change the ami file directly, which should validate the file."""
    app._tx_ami_file_changed(ami_test_file)
    assert isinstance(app._tx_cfg, AMIParamConfigurator)
    assert app.tx_has_getwave
    assert not app.rx_has_ts4
