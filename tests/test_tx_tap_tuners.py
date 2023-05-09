from pybert.pybert import TxTapTuner


def test_sweep_value_when_disabled():
    """When disabled, we should always just get a zero."""
    tuner = TxTapTuner()
    assert tuner.sweep_values() == [0.0]


def test_sweep_value_when_enabled_but_step_is_zero():
    """When enabled but step is zero, we should just get the current value."""
    tuner = TxTapTuner()
    tuner.value = 1.2
    tuner.enabled = True
    assert tuner.sweep_values() == [1.2]


def test_sweep_value_when_enabled_and_has_step():
    """When enabled and step is set, we should just get a list of min to max with a step between."""
    tuner = TxTapTuner()
    tuner.value = 1.0
    tuner.min_val = 1.0
    tuner.max_val = 3.0
    tuner.steps = 1
    tuner.enabled = True
    assert tuner.sweep_values() == [1.0, 2.0, 3.0]
