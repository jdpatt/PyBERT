import logging
from typing import Optional

from traits.api import Bool, Button, File, HasTraits

from pyibisami import AMIModel, AMIParamConfigurator, IBISModel

logger = logging.getLogger(__name__)


class Buffer(HasTraits):
    ibis_filepath = File("", entries=5, filter=["*.ibs"])
    dll_filepath = File("", entries=5, filter=["*.dll", "*.so"])
    ami_filepath = File("", entries=5, filter=["*.ami"])

    use_getwave = Bool(False)
    use_ami = Bool(False)
    use_ibis = Bool(False)
    use_ondie_sparameters = Bool(False)

    btn_select_model = Button(label="Select")
    btn_view_model = Button(label="View")
    btn_ami_config = Button(label="Configure AMI")

    def __init__(self):
        super().__init__()

        self.ibis_model: Optional[IBISModel] = None
        self.ami_config: Optional[AMIParamConfigurator] = None
        self.ami_model: Optional[AMIModel] = None

    def _ibis_filepath_changed(self, new_filepath):
        """If the IBIS file changes, parse in and read in the new one."""
        logger.info(f"Parsing IBIS file: {new_filepath}")
        try:
            ibis = IBISModel.from_ibis_file(new_filepath)
            self.use_ibis = True
            self.ibis_model = ibis
            self.ibis_filepath = ibis.filepath
        except Exception as err:
            logger.exception(f"Failed to open and/or parse IBIS file!\n{err}")
            # error_popup("Failed to open and/or parse IBIS file. See Console for more Information.")
            self.use_ibis = False
            self.ibis_filepath = ""
            self.ibis_model = None

    def _ami_filepath_changed(self, ami_filepath):
        logger.info(f"Parsing AMI file, {ami_filepath}")
        try:
            configurator = AMIParamConfigurator(ami_filepath)
            if not configurator.init_returns_impulse():
                self.use_getwave = True
            self.ami_config = configurator
        except Exception as err:
            logger.exception(f"Failed to open and/or parse AMI file!\n{err}")
            # error_popup("Failed to open and/or parse AMI file. See Console for more Information.")
            self.ami_config = None

    def _dll_filepath_changed(self, new_value):
        try:
            ami = AMIModel(new_value)
            self.ami_model = ami
        except Exception as err:
            self.ami_model = None
            logger.exception(f"Failed to open DLL/SO file!\n{err}")
            # error_popup("Failed to open DLL/SO file. See Console for more Information.")

    def _btn_select_model_fired(self):
        self.ibis_model.open_model_selector()

    def _btn_view_model_fired(self):
        self.ibis_model.open_gui()

    def _btn_ami_config_fired(self):
        self.ami_config.open_gui()
