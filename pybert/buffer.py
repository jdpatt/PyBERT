"""Common features between the transmitter and receiver."""
from logging import getLogger
from pathlib import Path

from pyibisami.ami_model import AMIModel, AMIModelInitializer
from pyibisami.ami_parse import AMIParamConfigurator

from pybert.defaults import (
    AC_CAPACITANCE,
    INPUT_CAPACITANCE,
    INPUT_IMPEDANCE,
    OUTPUT_CAPACITANCE,
    OUTPUT_DRIVE_STRENGTH,
    OUTPUT_IMPEDANCE,
    PN_FREQ,
    PN_MAG,
)


class Buffer:
    """Object to hold items common between the Tx and Rx buffers."""

    def __init__(self):
        super(Buffer, self).__init__()
        self.log = getLogger("pybert.buffer")
        self.use_ami: bool = False
        self.use_getwave: bool = False
        self.has_getwave: bool = False
        self.ami_file = None
        self.ami_valid: bool = False
        self.dll_file = None
        self.dll_valid: bool = False
        self.model = None
        self.configurator = None

    def ami_file_changed(self, new_file: Path):
        """Read and configure the new ami model from the user."""
        try:
            if new_file.suffix not in [".ami"]:
                raise ValueError("File is not an ami file.")
            self.ami_valid = False
            with open(new_file) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log.info("Parsing AMI file, '%s'...\n%s", new_file, pcfg.ami_parsing_errors)
            self.has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self.configurator = pcfg.open_gui
            self.ami_valid = True
        except Exception as err:
            self.log.error(err)
            FailedAMILoad("Failed to open and/or parse AMI file!\n")

    def dll_file_changed(self, new_file: Path):
        """Read and set the new DLL file from the users."""
        try:
            if new_file.suffix not in ["dll", "so"]:
                raise ValueError("File is not a DLL or SO file.")
            self.dll_valid = False
            model = AMIModel(str(new_file))
            self.model = model
            self.dll_valid = True
        except Exception as err:
            self.log.error(err)
            raise FailledDLLLoad("Failed to open DLL/SO file!\n")

    def open_config_gui(self):
        """Open the AMI configuration GUI."""
        if self.configurator:
            self.configurator()


class FailedAMILoad(Exception):
    pass


class FailledDLLLoad(Exception):
    pass


class Transmitter(Buffer):
    """docstring for Transmitter"""

    def __init__(self, random_noise):
        super(Transmitter, self).__init__()
        self.log.debug("Initializing Tx")
        self.vod = OUTPUT_DRIVE_STRENGTH  #: Tx differential output voltage (V)
        self.output_impedance = OUTPUT_IMPEDANCE  #: Tx source impedance (Ohms)
        self.output_capacitance = OUTPUT_CAPACITANCE  #: Tx parasitic output capacitance (pF)
        self.pn_mag = PN_MAG  #: Periodic noise magnitude (V).
        self.pn_freq = PN_FREQ  #: Periodic noise frequency (MHz).
        self.random_noise = random_noise  #: Standard deviation of Gaussian random noise (V).
        self.rel_power = 1.0  #: Tx power dissipation (W).

    def initialize_model(self, sample_interval, channel_response, bit_time):
        """Within the PyBERT computational environment, we use normalized impulse responses,
        which have units of (V/ts), where 'ts' is the sample interval. However, IBIS-AMI models expect
        units of (V/s). So, we have to scale accordingly, as we transit the boundary between these two worlds.

        Get the model invoked and initialized, except for 'channel_response', which
        we need to do several different ways, in order to gather all the data we need.
        """
        tx_param_dict = self.configurator.input_ami_params
        tx_model_init = AMIModelInitializer(tx_param_dict)
        tx_model_init.sample_interval = sample_interval  # Must be set, before 'channel_response'!
        tx_model_init.channel_response = channel_response
        tx_model_init.bit_time = bit_time
        tx_model = AMIModel(self.dll_file)
        tx_model.initialize(tx_model_init)
        self.log.info("Tx IBIS-AMI model initialization results:")
        self.log.info("Input parameters: %s", tx_model.ami_params_in)
        self.log.info("Output parameters: %s", tx_model.ami_params_out)
        self.log.info("Message: %s", tx_model.msg)
        if not self.configurator.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
            raise TypeError("Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!")
        if not self.use_getwave:
            raise ValueError(
                "You have elected not to use GetWave for a model, which does not \
                        return an impulse response! Aborting... Please, select 'Use GetWave'"
            )
        return tx_model


class Receiver(Buffer):
    """docstring for Receiver"""

    def __init__(self):
        super(Receiver, self).__init__()
        self.log.debug("Initializing Rx")
        self.input_impedance = INPUT_IMPEDANCE  #: Rx input impedance (Ohm)
        self.input_capacitance = INPUT_CAPACITANCE  #: Rx parasitic input capacitance (pF)
        self.cac = AC_CAPACITANCE  #: Rx a.c. coupling capacitance (uF)

    def initialize_model(self, sample_interval, channel_response, bit_time):
        """Get the model invoked and initialized, except for 'channel_response', which
        we need to do several different ways, in order to gather all the data we need."""
        rx_param_dict = self.configurator.input_ami_params
        rx_model_init = AMIModelInitializer(rx_param_dict)
        rx_model_init.sample_interval = sample_interval  # Must be set, before 'channel_response'!
        rx_model_init.channel_response = channel_response
        rx_model_init.bit_time = bit_time
        rx_model = AMIModel(self.dll_file)
        rx_model.initialize(rx_model_init)
        self.log.info("Rx IBIS-AMI model initialization results:")
        self.log.info("Input parameters: %s", rx_model.ami_params_in)
        self.log.info("Output parameters: %s", rx_model.ami_params_out)
        self.log.info("Message: %s", rx_model.msg)
        if not self.configurator.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
            raise TypeError("Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!")
        if not self.use_getwave:
            raise ValueError(
                "You have elected not to use GetWave for a model, which does not \
                        return an impulse response! Aborting... Please, select 'Use GetWave'"
            )
        return rx_model
