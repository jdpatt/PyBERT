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


@pytest.mark.usefixtures("dut")
def test_results_load_from_pickle(dut: PyBERT, tmp_path: Path):
    """Make sure that pybert can correctly load a pickle file."""
    dut.simulate(block=True)
    save_file = tmp_path.joinpath("config.pybert_data")
    dut.save_results(save_file)

    results = dut.load_results(save_file)
    np.testing.assert_allclose(dut.t_ns, results.data["t_ns"])


@pytest.mark.usefixtures("dut")
def test_results_save_as_pickle(dut: PyBERT, tmp_path: Path):
    """Make sure that pybert can correctly generate a waveform pickle file that can get reloaded."""
    dut.simulate(block=True)
    save_file = tmp_path.joinpath("results.pybert_data")
    dut.save_results(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "rb") as saved_results_file:
        results: Results = pickle.load(saved_results_file)
        assert results.version == __version__
