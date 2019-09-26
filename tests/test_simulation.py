import pytest

from pybert.simulation import Simulation


@pytest.fixture(scope="module")
def sim():
    return Simulation()


@pytest.mark.usefixtures("sim")
class Test_Simulation:
    def test_simulation_initialized(self, sim):
        assert sim.status == "Ready."
