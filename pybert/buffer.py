"""Common features between the transmitter and receiver."""
from logging import getLogger

from pybert.defaults import (
    AC_CAPACITANCE,
    INPUT_CAPACITANCE,
    INPUT_IMPEDANCE,
    OUTPUT_CAPACITANCE,
    OUTPUT_DRIVE_STRENGTH,
    OUTPUT_IMPEDANCE,
    PN_FREQ,
    PN_MAG,
    RANDOM_NOISE,
)
from pybert.view import popup_alert
from pyibisami.ami_model import AMIModel, AMIModelInitializer
from pyibisami.ami_parse import AMIParamConfigurator
from traits.api import Bool, File, Float, HasTraits, Range


class Buffer(HasTraits):
    """Object to hold items common between the Tx and Rx buffers."""

    def __init__(self):
        super(Buffer, self).__init__()
        self.log = getLogger("pybert.buffer")
        self.log.debug("Creating Buffer Object")
        self.use_ami: bool = Bool(False)
        self.use_getwave: bool = Bool(False)
        self.has_getwave: bool = Bool(False)
        self.ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
        self.ami_valid: bool = Bool(False)
        self.dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
        self.dll_valid: bool = Bool(False)
        self.model = None
        self.configurator = None

    def ami_file_changed(self, new_file):
        """Read and configure the new ami model from the user."""
        try:
            self.ami_valid = False
            with open(new_file) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log.info("Parsing AMI file, '%s'...\n%s", new_file, pcfg.ami_parsing_errors)
            self.has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self.configurator = pcfg.open_gui
            self.ami_valid = True
        except Exception as err:
            popup_alert("Failed to open and/or parse AMI file!\n", err)

    def dll_file_changed(self, new_file):
        """Read and set the new DLL file from the users."""
        try:
            self.dll_valid = False
            model = AMIModel(str(new_file))
            self.model = model
            self.dll_valid = True
        except Exception as err:
            popup_alert("Failed to open DLL/SO file!\n{}", err)

    def open_config_gui(self):
        """Open the AMI configuration GUI."""
        if self.configurator:
            self.configurator()


class Transmitter(Buffer):
    """docstring for Transmitter"""

    def __init__(self):
        super(Transmitter, self).__init__()
        self.log.debug("Creating Tx")
        self.vod = Float(OUTPUT_DRIVE_STRENGTH)  #: Tx differential output voltage (V)
        self.output_impedance = Float(OUTPUT_IMPEDANCE)  #: Tx source impedance (Ohms)
        self.output_capacitance = Range(
            low=0.001, value=OUTPUT_CAPACITANCE
        )  #: Tx parasitic output capacitance (pF)
        self.pn_mag = Float(PN_MAG)  #: Periodic noise magnitude (V).
        self.pn_freq = Float(PN_FREQ)  #: Periodic noise frequency (MHz).
        self.random_noise = Float(
            RANDOM_NOISE
        )  #: Standard deviation of Gaussian random noise (V).
        self.rel_power = Float(1.0)  #: Tx power dissipation (W).

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
            raise TypeError()
        if not self.use_getwave:
            raise ValueError()
        return tx_model


class Receiver(Buffer):
    """docstring for Receiver"""

    def __init__(self):
        super(Receiver, self).__init__()
        self.log.debug("Creating Rx")
        self.input_impedance = Float(INPUT_IMPEDANCE)  #: Rx input impedance (Ohm)
        self.input_capacitance = Range(
            low=0.001, value=INPUT_CAPACITANCE
        )  #: Rx parasitic input capacitance (pF)
        self.cac = Float(AC_CAPACITANCE)  #: Rx a.c. coupling capacitance (uF)

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
            raise TypeError()
        if not self.use_getwave:
            raise ValueError()
        return rx_model
