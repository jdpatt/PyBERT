"""Unit test coverage to make sure that the pybert can correctly save and load files."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from pybert import __version__
from pybert.configuration import Configuration
from pybert.models.stimulus import BitPattern, ModulationType
from pybert.pybert import PyBERT
from pybert.results import Results


@pytest.mark.parametrize("filepath_converter", [str, Path])
@pytest.mark.usefixtures("dut")
def test_config_save_as_yaml(dut: PyBERT, filepath_converter, tmp_path: Path):
    """Make sure that pybert can correctly generate a yaml file that can get reloaded."""
    save_file = tmp_path.joinpath("config.yaml")
    dut.save_configuration(filepath_converter(save_file))

    assert save_file.exists()  # File was created.

    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        assert user_config.version == __version__


@pytest.mark.usefixtures("dut")
def test_config_save_as_invalid(dut: PyBERT, tmp_path: Path, caplog):
    """When given an unsupported file suffix, no file should be generated and an message logged."""
    save_file = tmp_path.joinpath("config.json")
    dut.save_configuration(save_file)

    assert not save_file.exists()  # File should not have been created.
    assert "This filetype is not currently supported." in caplog.text


@pytest.mark.parametrize("filepath_converter", [str, Path])
@pytest.mark.usefixtures("dut")
def test_config_load_from_yaml(dut: PyBERT, filepath_converter, tmp_path: Path):
    """Make sure that pybert can correctly load a yaml file."""
    save_file = tmp_path.joinpath("config.yaml")
    dut.save_configuration(save_file)

    # Modify the saved yaml file.
    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        # Change a lot of settings throughout the different tabs of the application.
        user_config.eye_bits = 1234  # Normally 8000
        user_config.bit_rate = 20  # Normally 10
        user_config.mod_type = "PAM4"  # Normally [0]
        user_config.pattern = "PRBS23"  # Normally PRBS-7
        user_config.Rdc = 2  # Normally 0.1876
        user_config.rin = 85  # Normally 100
        user_config.n_taps = 2  # Normally 5
        user_config.delta_t = 0.01  # Normally 0.1
        user_config.thresh = 5  # Normally 6
    with open(save_file, "w", encoding="UTF-8") as saved_config_file:
        yaml.dump(user_config, saved_config_file)

    dut.load_configuration(filepath_converter(save_file))

    # For everything saved in configuration, make sure they match.
    # All items should exist in both, so fail if one isn't found.
    for name in user_config.__dict__.keys():
        if name == "mod_type":
            assert ModulationType(getattr(user_config, name)) == getattr(dut, name)
        elif name == "pattern":
            assert BitPattern[getattr(user_config, name)] == getattr(dut, name)
        else:
            # These are handled differently so skip them.
            if name not in ["tx_taps", "tx_tap_tuners", "dfe_tap_tuners", "version", "date_created"]:
                # Test the values
                assert getattr(user_config, name) == getattr(dut, name)


@pytest.mark.usefixtures("dut")
def test_config_load_from_invalid(dut: PyBERT, tmp_path: Path, caplog):
    """When given an unsupported file suffix, no file should be read and an message logged."""
    save_file = tmp_path.joinpath("config.json")
    save_file.touch()
    dut.load_configuration(save_file)

    assert "This filetype is not currently supported." in caplog.text


@pytest.mark.usefixtures("dut")
def test_config_reset_to_default(dut: PyBERT, caplog):
    """When given an unsupported file suffix, no file should be read and an message logged."""
    dut.bit_rate = 100.0
    assert dut.bit_rate == 100.0
    dut.reset_configuration()
    assert dut.bit_rate == 10.0


@pytest.mark.usefixtures("dut")
def test_results_load_from_pickle(dut: PyBERT, tmp_path: Path):
    """Make sure that pybert can correctly load a pickle file."""
    save_file = tmp_path.joinpath("config.pybert_data")
    dut.save_results(save_file)

    results = dut.load_results(save_file)
    np.testing.assert_allclose(dut.t_ns, results.results["t_ns"])


@pytest.mark.usefixtures("dut")
def test_results_save_as_pickle(dut: PyBERT, tmp_path: Path):
    """Make sure that pybert can correctly generate a waveform pickle file that can get reloaded."""
    save_file = tmp_path.joinpath("results.pybert_data")
    dut.save_results(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "rb") as saved_results_file:
        results: Results = pickle.load(saved_results_file)
        assert results.version == __version__
