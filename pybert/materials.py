from dataclasses import dataclass


@dataclass
class Materials:
    """Container to hold the properties of a transmission line."""

    dc_resistance_per_meter = 0.0  # Ohms/m
    w_transition_freq = 0.0
    skin_effect_resistance = 0.0  # Ohms/m
    loss_tangent = 0.0
    characteristic_impedance = 0.0  # Ohms
    rel_velocity = 0.0  # relative propagation velocity (c)
    channel_length = 0.0  # cable length (m)
    # The random noise is applied at end of channel, so as to appear white to Rx.
    random_noise = 0.0  # standard deviation of Gaussian random noise (V)


@dataclass
class TwistedCopperPair24Gauge(Materials):
    """Parameters for Howard Johnson's "Metallic Transmission Model"
    (See "High Speed Signal Propagation", Sec. 3.1.)
    """

    dc_resistance_per_meter = 0.1876
    w_transition_freq = (
        10.0e6
    )  # 10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
    skin_effect_resistance = 1.452
    loss_tangent = 0.02
    characteristic_impedance = 100.0
    rel_velocity = 0.67
    channel_length = 1.0
    random_noise = 0.001
