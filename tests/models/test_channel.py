from pybert.models.channel import Channel


class TestChannelSerialization:
    """Test channel serialization and deserialization methods."""

    def test_to_dict_from_dict(self):
        """Test Channel to_dict and from_dict methods."""
        # Create a channel with custom values
        channel = Channel(elements=[{"file": "test.s2p", "renumber": True}])

        # Test to_dict
        channel_dict = channel.to_dict()
        expected = {
            "elements": [{"file": "test.s2p", "renumber": True}],
            "use_ch_file": False,
            "f_step": 10,
            "f_max": 40,
            "impulse_length": 0.0,
            "Rdc": 0.1876,
            "w0": 10e6,
            "R0": 1.452,
            "Theta0": 0.02,
            "Z0": 100,
            "v0": 0.67,
            "l_ch": 0.5,
            "use_window": False,
        }
        assert channel_dict == expected

        # Test from_dict
        new_channel = Channel.from_dict(channel_dict)
        assert new_channel.elements == [{"file": "test.s2p", "renumber": True}]
        assert new_channel.use_ch_file is False
        assert new_channel.f_step == 10
        assert new_channel.f_max == 40
        assert new_channel.impulse_length == 0.0
        assert new_channel.Rdc == 0.1876
        assert new_channel.w0 == 10e6
        assert new_channel.R0 == 1.452

        # Test with partial data (should use defaults)
        partial_dict = {"Rdc": 75.0}
        partial_channel = Channel.from_dict(partial_dict)
        assert partial_channel.Rdc == 75.0
        assert partial_channel.w0 == 10e6
        assert partial_channel.R0 == 1.452
        assert partial_channel.Theta0 == 0.02
        assert partial_channel.Z0 == 100
        assert partial_channel.v0 == 0.67
        assert partial_channel.l_ch == 0.5
        assert partial_channel.use_window is False
