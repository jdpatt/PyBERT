"""Shared fixtures across the pybert testing infrastructure."""

import logging
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pybert.pybert import PyBERT

RX_IBIS_EXAMPLE = Path(__file__).parent / "ibisami" / "example_rx.ibs"


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests without interfering with PyBERT's logging setup."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.WARNING)


@pytest.fixture(scope="function")
def dut_for_gui_tests():
    """Return an initialized pybert object without running a simulation.

    This gets reset for every function test so we can test individual
    gui elements without affecting the other tests.
    """
    dut = PyBERT(run_simulation=False)
    yield dut


@pytest.fixture(scope="class")
def dut():
    """Return an initialized pybert object that has already run the initial simulation."""
    dut = PyBERT(run_simulation=False)
    dut.simulate(block=True)
    yield dut


@pytest.fixture(scope="module")
def dut_imp_len():
    """Return an initialized pybert object with manually controlled channel impulse response length."""
    dut = PyBERT(run_simulation=False)
    dut.impulse_length = 10  # (ns)
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_init():
    """Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in statistical mode."""
    dut = PyBERT(run_simulation=False)

    dut.rx.load_ibis_file(RX_IBIS_EXAMPLE)

    # Ensure these are set to True to use the IBIS-AMI model.
    dut.rx.use_ibis = True
    dut.rx.use_ami = True
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_getwave():
    """Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode."""
    dut = PyBERT(run_simulation=False)
    dut.rx.load_ibis_file(RX_IBIS_EXAMPLE)

    # Ensure these are set to True to use the IBIS-AMI model with getwave.
    dut.rx.use_ibis = True
    dut.rx.use_ami = True
    dut.rx.use_getwave = True
    yield dut


@pytest.fixture(scope="module")
def ibisami_rx_getwave_clocked():
    """Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode and making use of clock times."""
    dut = PyBERT(run_simulation=False)
    dut.rx.load_ibis_file(RX_IBIS_EXAMPLE)

    # Ensure these are set to True to use the IBIS-AMI model with getwave and clocks.
    dut.rx.use_ibis = True
    dut.rx.use_ami = True
    dut.rx.use_getwave = True
    dut.rx.use_clocks = True
    yield dut


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for GUI tests without this QtWidgets will not work."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
