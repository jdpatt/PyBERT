import numpy as np
import pytest

from pybert.bert import SimulationPerf
from pybert.pybert import PyBERT


@pytest.mark.parametrize(
    "dut_fixture",
    [
        "dut",  # Basic Native simulation
        "dut_imp_len",  # Native simulation with controlled impulse length
        "ibisami_rx_init",  # Rx IBIS-AMI statistical mode
        "ibisami_rx_getwave",  # Rx IBIS-AMI bit-by-bit mode
        "ibisami_rx_getwave_clocked",  # Rx IBIS-AMI bit-by-bit with clocks
    ],
    ids=[
        "Native simulation",
        "Native simulation with controlled impulse length",
        "Rx IBIS-AMI statistical mode",
        "Rx IBIS-AMI bit-by-bit mode",
        "Rx IBIS-AMI bit-by-bit with clocks",
    ],
)
class TestAcrossAllSimulationWorkflows:
    """Basic tests of a properly initialized PyBERT.

    Tests are organized into groups:
    - Performance tests (test_performance_calculations)
    - Signal quality tests (test_ber, test_dly)
    - Jitter tests (test_isi, test_dcd, test_pj, test_rj)
    - Adaptation tests (test_lock, test_adapt)
    """

    @pytest.fixture(autouse=True)
    def _setup(self, request, dut_fixture):
        """Setup fixture to get the correct DUT based on the parametrized fixture name."""
        self.dut = request.getfixturevalue(dut_fixture)

    def test_initialization(self):
        """Test that the DUT is initialized correctly."""
        assert self.dut is not None
        assert self.dut.last_results is not None  # Simulation data is available

    def test_performance_calculations(self):
        """Test simulation performance."""
        perf: SimulationPerf = self.dut.last_results["performance"]
        assert perf.total > (1e6 / 60), "Performance dropped below 1 Msmpls/min.!"

    def test_ber(self):
        """Test simulation bit errors."""
        assert not self.dut.bit_errs, "Bit errors detected!"

    def test_dly(self):
        """Test channel delay."""
        assert self.dut.chnl_dly > 1e-9 and self.dut.chnl_dly < 10e-9, "Channel delay is out of range!"

    def test_isi(self):
        """Test ISI portion of jitter."""
        assert self.dut.dfe_jitter.isi < 50e-12, "ISI is too high!"

    def test_dcd(self):
        """Test DCD portion of jitter."""
        assert self.dut.dfe_jitter.dcd < 20e-12, "DCD is too high!"

    def test_pj(self):
        """Test periodic portion of jitter."""
        assert self.dut.dfe_jitter.pj < 20e-12, "Periodic jitter is too high!"

    def test_rj(self):
        """Test random portion of jitter."""
        assert self.dut.dfe_jitter.rj < 20e-12, "Random jitter is too high!"

    def test_lock(self):
        """Test CDR lock, by ensuring that last 20% of locked indication vector
        is all True."""
        _lockeds = self.dut.lockeds
        assert all(_lockeds[4 * len(_lockeds) // 5 :]), "CDR lock is unstable!"

    def test_adapt(self):
        """Test DFE lock, by ensuring that last 20% of all coefficient vectors
        are stable to within +/-20% of their mean."""
        _weights = self.dut.adaptation  # rows = step; cols = tap
        _ws = np.array(list(zip(*_weights[4 * len(_weights) // 5 :])))  # zip(*x) = unzip(x)
        _means = list(map(lambda xs: sum(xs) / len(xs), _ws))
        assert all(
            list(map(lambda pr: pr[1] == 0 or all(abs(pr[0] - pr[1]) / pr[1] < 0.2), zip(_ws, _means)))
        ), f"DFE adaptation is unstable! {max(_ws[-1])} {min(_ws[-1])}"
