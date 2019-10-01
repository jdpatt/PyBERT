from unittest.mock import patch

import numpy as np
import pytest
from pybert.simulation import Simulation


@pytest.fixture(scope="function")
def sim():
    """Use all the default values for the test suite."""
    return Simulation()


@pytest.mark.usefixtures("sim")
class Test_Simulation:
    def test_simulation_initialized(self, sim):
        assert sim.status == "Ready."

    def test_update_status(self, sim):
        sim.status = "Test Status"
        assert sim.status == "Test Status"

    def test_generate_time_series(self, sim):
        sim.bit_rate = 0.000_000_001
        sim.nspb = 1  # One samples per UI
        sim.nbits = 4  # Only generate 4 bits instead of 8000
        np.testing.assert_array_equal(sim.t, [0.00e00, 1.0, 2.0, 3.0])

    def test_get_time_in_nanoseconds(self, sim):
        sim.bit_rate = 0.000_000_001
        sim.nspb = 1  # One samples per UI
        sim.nbits = 4  # Only generate 4 bits instead of 8000
        np.testing.assert_array_equal(sim.t_ns, [0.00e9, 1.0e9, 2.0e9, 3.0e9])

    def test_generate_frequency_vector(self, sim):
        sim.bit_rate = 0.000_000_001
        sim.nspb = 1  # One samples per UI
        sim.nbits = 4  # Only generate 4 bits instead of 8000
        np.testing.assert_array_equal(sim.f, [0.0, 0.25, 0.5, -0.25])

    def test_get_freq_vector_in_rads(self, sim):
        sim.bit_rate = 0.000_000_001
        sim.nspb = 1  # One samples per UI
        sim.nbits = 4  # Only generate 4 bits instead of 8000
        np.testing.assert_array_equal(
            sim.w, [0, 1.5707963267948966, 3.141592653589793, -1.5707963267948966]
        )

    def test_unit_interval(self, sim):
        """For 10Gbps, we should get a 1e-10 UI."""
        assert sim.ui == 0.000_000_000_1

    def test_num_of_unit_intervals(self, sim):
        """How many bits should we simulate."""
        assert sim.nui == 8000

    def test_num_of_samples_per_ui(self, sim):
        assert sim.nspui == 32

    def test_unit_interval_for_eye_diagrams(self, sim):
        assert sim.eye_uis == 1600

    @patch("pybert.simulation.randint")
    def test_generate_bit_stream(self, mock_randint, sim):
        mock_randint.return_value = 8  # Have the same seed across test runs.
        sim.nbits = 10
        #  First four bits are the prequel alignment bits.
        np.testing.assert_array_equal(sim.bits, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    @patch("pybert.simulation.randint")
    def test_generate_symbol_stream(self, mock_randint, sim):
        mock_randint.return_value = 8  # Have the same seed across test runs.
        sim.nbits = 10
        np.testing.assert_array_equal(
            sim.symbols, [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
        )
