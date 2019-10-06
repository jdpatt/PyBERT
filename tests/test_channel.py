import pytest
from pybert.sim.channel import Channel


@pytest.fixture(scope="module")
def channel():
    return Channel()


@pytest.mark.usefixtures("channel")
class Test_Channel:
    def test_invalid_material(self, channel):
        with pytest.raises(ValueError):
            channel.change_material("BLAH")

    def test_valid_material(self, channel):
        channel.change_material("UTP_24Gauge")
        assert channel.material.random_noise == 0.001
