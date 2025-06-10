"""Shared fixtures across the pybert testing infrastructure."""

import logging
import time

import pytest

from pybert.pybert import PyBERT

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="module")
def dut():
    """Return an initialized pybert object that has already run the initial simulation."""
    dut = PyBERT()
    dut.simulate(wait_for_completion=True)
    yield dut


@pytest.fixture(scope="module")
def dut_imp_len():
    """Return an initialized pybert object with manually controlled channel impulse response length."""
    dut = PyBERT(run_simulation=False)
    dut.impulse_length = 10  # (ns)
    dut.simulate()
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_init():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in statistical mode.
    """
    dut = PyBERT(run_simulation=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.simulate()
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_getwave():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode.
    """
    dut = PyBERT(run_simulation=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.rx_use_getwave = True
    dut.simulate()
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_getwave_clocked():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode and making use of clock times.
    """
    dut = PyBERT(run_simulation=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.rx_use_getwave = True
    dut.rx_use_clocks = True
    dut.simulate()
    yield dut
