import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from pybert import __version__
from pybert.configuration import Configuration
from pybert.models.buffer import Buffer, IbisAmiBuffer, Receiver, Transmitter
from pybert.models.stimulus import BitPattern, ModulationType
from pybert.pybert import PyBERT
from pybert.results import Results


class TestBufferSerialization:
    """Test buffer serialization and deserialization methods."""

    def test_buffer_to_dict_from_dict(self):
        """Test Buffer to_dict and from_dict methods."""
        # Create a buffer with custom values
        buffer = Buffer(impedance=50.0, capacitance=1.0, inductance=0.1)

        # Test to_dict
        buffer_dict = buffer.to_dict()
        expected = {
            "impedance": 50.0,
            "capacitance": 1.0,
            "inductance": 0.1,
        }
        assert buffer_dict == expected

        # Test from_dict
        new_buffer = Buffer.from_dict(buffer_dict)
        assert new_buffer.impedance == 50.0
        assert new_buffer.capacitance == 1.0
        assert new_buffer.inductance == 0.1

        # Test with partial data (should use defaults)
        partial_dict = {"impedance": 75.0}
        partial_buffer = Buffer.from_dict(partial_dict)
        assert partial_buffer.impedance == 75.0
        assert partial_buffer.capacitance == 0.0  # default
        assert partial_buffer.inductance == 0.0  # default

    def test_ibis_ami_buffer_to_dict_from_dict(self):
        """Test IbisAmiBuffer to_dict and from_dict methods."""
        # Create buffer with IBIS/AMI settings
        buffer = IbisAmiBuffer(
            ibis_file="test.ibs",
            use_ami=True,
            use_ts4=True,
            use_getwave=True,
            use_ibis=True,
        )

        # Test to_dict
        buffer_dict = buffer.to_dict()
        expected_keys = {"ibis_file", "use_ami", "use_ts4", "use_getwave", "use_ibis", "ami_file", "dll_file"}
        assert set(buffer_dict.keys()) == expected_keys
        assert buffer_dict["ibis_file"] == "test.ibs"
        assert buffer_dict["use_ami"] is True
        assert buffer_dict["use_ts4"] is True
        assert buffer_dict["use_getwave"] is True
        assert buffer_dict["use_ibis"] is True

    def test_transmitter_to_dict_from_dict(self):
        """Test Transmitter to_dict and from_dict methods."""
        # Create transmitter with custom values
        tx = Transmitter(
            impedance=75,
            capacitance=0.8,
            inductance=0.05,
            ibis_file="tx.ibs",
            use_ami=True,
        )

        # Test to_dict
        tx_dict = tx.to_dict()
        assert tx_dict["impedance"] == 75
        assert tx_dict["capacitance"] == 0.8
        assert tx_dict["inductance"] == 0.05
        assert tx_dict["ibis_file"] == "tx.ibs"
        assert tx_dict["use_ami"] is True

        # Test from_dict
        new_tx = Transmitter.from_dict(tx_dict)
        assert new_tx.impedance == 75
        assert new_tx.capacitance == 0.8
        assert new_tx.inductance == 0.05
        assert new_tx.ibis_file == "tx.ibs"
        assert new_tx.use_ami is True

    def test_receiver_to_dict_from_dict(self):
        """Test Receiver to_dict and from_dict methods."""
        # Create receiver with custom values
        rx = Receiver(
            impedance=85,
            capacitance=0.6,
            inductance=0.02,
            ac_coupling=2.0,
            use_clocks=True,
            ibis_file="rx.ibs",
            use_ami=True,
        )

        # Test to_dict
        rx_dict = rx.to_dict()
        assert rx_dict["impedance"] == 85
        assert rx_dict["capacitance"] == 0.6
        assert rx_dict["inductance"] == 0.02
        assert rx_dict["ac_coupling"] == 2.0
        assert rx_dict["use_clocks"] is True
        assert rx_dict["ibis_file"] == "rx.ibs"
        assert rx_dict["use_ami"] is True

        # Test from_dict
        new_rx = Receiver.from_dict(rx_dict)
        assert new_rx.impedance == 85
        assert new_rx.capacitance == 0.6
        assert new_rx.inductance == 0.02
        assert new_rx.ac_coupling == 2.0
        assert new_rx.use_clocks is True
        assert new_rx.ibis_file == "rx.ibs"
        assert new_rx.use_ami is True

    def test_buffer_inheritance_structure(self):
        """Test that Transmitter and Receiver properly inherit from both Buffer and IbisAmiBuffer."""
        tx = Transmitter()
        rx = Receiver()

        # Check that they have both Buffer and IbisAmiBuffer attributes
        assert hasattr(tx, "impedance")  # From Buffer
        assert hasattr(tx, "capacitance")  # From Buffer
        assert hasattr(tx, "inductance")  # From Buffer
        assert hasattr(tx, "ibis_file")  # From IbisAmiBuffer
        assert hasattr(tx, "use_ami")  # From IbisAmiBuffer

        assert hasattr(rx, "impedance")  # From Buffer
        assert hasattr(rx, "capacitance")  # From Buffer
        assert hasattr(rx, "inductance")  # From Buffer
        assert hasattr(rx, "ac_coupling")  # Receiver-specific
        assert hasattr(rx, "use_clocks")  # Receiver-specific
        assert hasattr(rx, "ibis_file")  # From IbisAmiBuffer
        assert hasattr(rx, "use_ami")  # From IbisAmiBuffer


class TestConfigurationBufferHandling:
    """Test configuration system's handling of buffer serialization."""

    @pytest.mark.usefixtures("dut")
    def test_configuration_saves_buffer_structure(self, dut: PyBERT, tmp_path: Path):
        """Test that configuration properly saves the new buffer structure."""
        # Modify buffer settings
        dut.tx.impedance = 75
        dut.tx.capacitance = 0.8
        dut.tx.use_ami = True
        dut.rx.impedance = 85
        dut.rx.capacitance = 0.6
        dut.rx.ac_coupling = 2.0
        dut.rx.use_clocks = True

        # Save configuration
        save_file = tmp_path.joinpath("config.yaml")
        dut.save_configuration(save_file)

        # Load and verify
        with open(save_file, "r", encoding="UTF-8") as f:
            config_data: Configuration = yaml.load(f, Loader=yaml.Loader)

        # Check that transmitter and receiver are saved as dictionaries
        assert hasattr(config_data, "transmitter")
        assert hasattr(config_data, "receiver")
        assert isinstance(config_data.transmitter, dict)
        assert isinstance(config_data.receiver, dict)

        # Verify transmitter settings
        assert config_data.transmitter["impedance"] == 75
        assert config_data.transmitter["capacitance"] == 0.8
        assert config_data.transmitter["use_ami"] is True

        # Verify receiver settings
        assert config_data.receiver["impedance"] == 85
        assert config_data.receiver["capacitance"] == 0.6
        assert config_data.receiver["ac_coupling"] == 2.0
        assert config_data.receiver["use_clocks"] is True

    @pytest.mark.usefixtures("dut")
    def test_configuration_loads_buffer_structure(self, dut: PyBERT, tmp_path: Path):
        """Test that configuration properly loads the new buffer structure."""
        # Save current configuration
        save_file = tmp_path.joinpath("config.yaml")
        dut.save_configuration(save_file)

        # Modify the saved configuration
        with open(save_file, "r", encoding="UTF-8") as f:
            config_data: Configuration = yaml.load(f, Loader=yaml.Loader)

        # Modify buffer settings
        config_data.transmitter["impedance"] = 75
        config_data.transmitter["capacitance"] = 0.8
        config_data.transmitter["use_ami"] = True
        config_data.receiver["impedance"] = 85
        config_data.receiver["capacitance"] = 0.6
        config_data.receiver["ac_coupling"] = 2.0
        config_data.receiver["use_clocks"] = True

        # Save modified configuration
        with open(save_file, "w", encoding="UTF-8") as f:
            yaml.dump(config_data, f)

        # Load configuration
        dut.load_configuration(save_file)

        # Verify that buffer settings were loaded correctly
        assert dut.tx.impedance == 75
        assert dut.tx.capacitance == 0.8
        assert dut.tx.use_ami is True
        assert dut.rx.impedance == 85
        assert dut.rx.capacitance == 0.6
        assert dut.rx.ac_coupling == 2.0
        assert dut.rx.use_clocks is True

    @pytest.mark.usefixtures("dut")
    def test_configuration_buffer_roundtrip(self, dut: PyBERT, tmp_path: Path):
        """Test that buffer settings survive a save/load roundtrip."""
        # Set custom buffer settings
        dut.tx.impedance = 75
        dut.tx.capacitance = 0.8
        dut.tx.use_ami = True
        dut.rx.impedance = 85
        dut.rx.capacitance = 0.6
        dut.rx.ac_coupling = 2.0
        dut.rx.use_clocks = True

        # Save and reload
        save_file = tmp_path.joinpath("config.yaml")
        dut.save_configuration(save_file)

        # Create new PyBERT instance and load configuration
        new_dut = PyBERT()
        new_dut.load_configuration(save_file)

        # Verify all settings survived
        assert new_dut.tx.impedance == 75
        assert new_dut.tx.capacitance == 0.8
        assert new_dut.tx.use_ami is True
        assert new_dut.rx.impedance == 85
        assert new_dut.rx.capacitance == 0.6
        assert new_dut.rx.ac_coupling == 2.0
        assert new_dut.rx.use_clocks is True

    def test_configuration_create_default_config(self):
        """Test that create_default_config includes proper buffer structure."""
        config = Configuration.create_default_config()

        # Check that transmitter and receiver are present
        assert hasattr(config, "transmitter")
        assert hasattr(config, "receiver")
        assert isinstance(config.transmitter, dict)
        assert isinstance(config.receiver, dict)

        # Check default transmitter values
        assert config.transmitter["impedance"] == 100
        assert config.transmitter["capacitance"] == 0.5
        assert config.transmitter["use_ami"] is False

        # Check default receiver values
        assert config.receiver["impedance"] == 100
        assert config.receiver["capacitance"] == 0.5
        assert config.receiver["ac_coupling"] == 1.0
        assert config.receiver["use_clocks"] is False
