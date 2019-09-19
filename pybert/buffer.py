import logging

from traits.api import (
    Bool,
    File,
)

from pyibisami.ami_parse import AMIParamConfigurator
from pyibisami.ami_model import AMIModel
from pybert.view import popup_error


class Buffer(object):
    """Object to hold items common between the Tx and Rx buffers."""

    def __init__(self):
        super(Buffer, self).__init__()
        self.log = logging.getLogger("pybert.buffer")
        self.log.debug("Creating Buffer Object")
        self.use_ami: bool = Bool(False)
        self.use_getwave: bool = Bool(False)
        self.has_getwave: bool = Bool(False)
        self.ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
        self.ami_valid: bool = Bool(False)
        self.dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
        self.dll_valid: bool = Bool(False)
        self.model = None
        self.open_gui = None

    def ami_file_changed(self, new_file):
        """Read and configure the new ami model from the user."""
        try:
            self.ami_valid = False
            with open(new_file) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log.info(
                "Parsing AMI file, '{}'...\n{}".format(new_file, pcfg.ami_parsing_errors)
            )
            self.has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self.open_gui = pcfg.open_gui
            self.ami_valid = True
        except Exception as err:
            popup_error("Failed to open and/or parse AMI file!\n{}".format(err))

    def dll_file_changed(self, new_file):
        """Read and set the new DLL file from the users."""
        try:
            self.dll_valid = False
            model = AMIModel(str(new_file))
            self.model = model
            self.dll_valid = True
        except Exception as err:
            popup_error("Failed to open DLL/SO file!\n{}".format(err))

    def open_config_gui(self):
        """Open the AMI configuration GUI."""
        if self.open_gui:
            self.open_gui()
