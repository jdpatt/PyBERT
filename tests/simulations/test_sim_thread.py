from pybert.bert import SimulationThread
from pybert.pybert import PyBERT


def test_simulation_can_abort(caplog):
    """Test that spawning a simulation thread can be aborted.

    The timeout on join does not kill the thread but if reached will stop
    blocking.  This just guards against pytest infinitely hanging up because
    of some event.  The simulation should abort within a second or two.
    """
    app = PyBERT(run_simulation=False)

    sim = SimulationThread()
    sim.pybert = app
    sim.start()  # Start the thread
    sim.stop()  # Abort the thread
    sim.join(60)  # Join and wait until it ends or 10 seconds passed.

    assert "Simulation aborted by User." in caplog.text
