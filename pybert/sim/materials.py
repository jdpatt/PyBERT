from dataclasses import dataclass


@dataclass
class Materials:
    """Every Material should have these properties of a transmission line.

    If you don't provide them, they default to the same properties as "TwistedCopperPair24Gauge"
    """

    dc_resistance_per_meter = 0.1876  # Ohms/m
    w_transition_freq = 10.0e6  # rads./s
    skin_effect_resistance = 1.452  # Ohms/m
    loss_tangent = 0.02  # unitless
    characteristic_impedance = 100.0  # Channel characteristic impedance, in LC region (Ohms).
    rel_velocity = 0.67  # relative propagation velocity (c)
    channel_length = 1.0  # cable length (m)


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


MATERIALS = {"UTP_24Gauge": TwistedCopperPair24Gauge()}
