from pybert.bert import SimulationThread
from pybert.pybert import PyBERT


def test_simulation_can_abort():
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

    # Get all messages from the queue and check for "Aborted" string
    while not app.result_queue.empty():
        result = app.result_queue.get()
        if result.get("type") == "message":
            if "Simulation aborted by User." in result.get("message", ""):
                assert True
                return
    assert False
