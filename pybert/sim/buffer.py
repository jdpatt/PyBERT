"""Common features between the transmitter and receiver."""
from logging import getLogger
from pathlib import Path

from pyibisami.ami_model import AMIModel, AMIModelInitializer
from pyibisami.ami_parse import AMIParamConfigurator


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

    def initialize_model(self, sample_interval, channel_response, bit_time):
        """Within the PyBERT computational environment, we use normalized impulse responses,
        which have units of (V/ts), where 'ts' is the sample interval. However, IBIS-AMI models expect
        units of (V/s). So, we have to scale accordingly, as we transit the boundary between these two worlds.

        Get the model invoked and initialized, except for 'channel_response', which
        we need to do several different ways, in order to gather all the data we need.
        """
        param_dict = self.configurator.input_ami_params
        model_init = AMIModelInitializer(param_dict)
        model_init.sample_interval = sample_interval  # Must be set, before 'channel_response'!
        model_init.channel_response = channel_response
        model_init.bit_time = bit_time
        model = AMIModel(self.dll_file)
        model.initialize(model_init)
        self.log.info("Input parameters: %s", model.ami_params_in)
        self.log.info("Output parameters: %s", model.ami_params_out)
        self.log.info("Message: %s", model.msg)
        if not self.configurator.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
            raise TypeError("Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!")
        if not self.use_getwave:
            raise ValueError(
                "You have elected not to use GetWave for a model, which does not \
                        return an impulse response! Aborting... Please, select 'Use GetWave'"
            )
        return model

    def open_config_gui(self):
        """Open the AMI configuration GUI."""
        if self.configurator:
            self.configurator()


class FailedAMILoad(Exception):
    """Raised when something goes wrong with the AMI file.

    The original exception is logged but not raised so that the TypeError and ValueError
    can be caught independently by pybert.
    """


class FailledDLLLoad(Exception):
    """Raised when something goes wrong with the DLL file.

    The original exception is logged but not raised so that the TypeError and ValueError
    can be caught independently by pybert.
    """


class Transmitter(Buffer):
    """Transmitter in the Channel"""

    def __init__(self, config):
        super(Transmitter, self).__init__()
        self.log.debug("Initializing Tx")
        self.vod = config.vod
        self.output_impedance = config.output_impedance
        self.output_capacitance = config.output_capacitance
        self.pn_mag = config.pn_mag
        self.pn_freq = config.pn_freq
        self.random_noise = config.random_noise
        self.rel_power = 1.0

        self.use_ami = config.tx_use_ami
        self.use_getwave = config.tx_use_getwave
        self.ami_file = config.tx_ami_file
        self.dll_file = config.tx_dll_file


class Receiver(Buffer):
    """Receiver in the Channel"""

    def __init__(self, config):
        super(Receiver, self).__init__()
        self.log.debug("Initializing Rx")
        self.input_impedance = config.input_impedance
        self.input_capacitance = config.input_capacitance
        self.cac = config.ac_capacitance

        self.use_ami = config.rx_use_ami
        self.use_getwave = config.rx_use_getwave
        self.ami_file = config.rx_ami_file
        self.dll_file = config.rx_dll_file
