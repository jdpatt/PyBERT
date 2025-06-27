import logging
from dataclasses import dataclass
from pathlib import Path

from pyibisami import AMIModel, AMIParamConfigurator, IBISModel

logger = logging.getLogger("pybert.buffer")


class Buffer:
    """Base buffer class.

    This class contains the basic parameters for a native buffer.
    """

    def __init__(self, impedance: float = 0, capacitance: float = 0, inductance: float = 0):
        self.impedance = impedance
        self.capacitance = capacitance
        self.inductance = inductance

    def to_dict(self) -> dict:
        """Convert buffer to dictionary for serialization."""
        return {
            "impedance": self.impedance,
            "capacitance": self.capacitance,
            "inductance": self.inductance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Buffer":
        """Create buffer from dictionary."""
        return cls(
            impedance=data.get("impedance", 0),
            capacitance=data.get("capacitance", 0),
            inductance=data.get("inductance", 0),
        )


class IbisAmiBuffer:
    """IBIS-AMI buffer class."""

    def __init__(
        self,
        ibis_file: str | Path | None = None,
        use_ami: bool = False,
        use_ts4: bool = False,
        use_getwave: bool = False,
        use_ibis: bool = False,
        config_data: dict | None = None,
    ):
        self.ibis_file = ibis_file
        self.use_ami = use_ami
        self.has_ts4 = False
        self.use_ts4 = use_ts4
        self.use_getwave = use_getwave
        self.has_getwave = False
        self.use_ibis = use_ibis
        self.ibis: IBISModel | None = None
        self.ami: AMIParamConfigurator | None = None
        self.model: AMIModel | None = None
        # Load IBIS/AMI if provided
        if self.ibis_file:
            self.load_ibis_file(self.ibis_file)
            if self.is_ibis_loaded() and self.ibis is not None and self.ibis.has_algorithmic_model:
                ami_file = self.ibis.ami_file
                dll_file = self.ibis.dll_file
                if isinstance(ami_file, (str, Path)) and str(ami_file):
                    self.load_ami_configurator(str(ami_file))
                if isinstance(dll_file, (str, Path)) and str(dll_file):
                    self.load_dll_model(str(dll_file))

    def is_ibis_loaded(self) -> bool:
        """Check if IBIS model is loaded and valid."""
        return self.ibis is not None

    def is_ami_loaded(self) -> bool:
        """Check if AMI model is loaded and valid."""
        return self.ami is not None

    def is_dll_loaded(self) -> bool:
        """Check if DLL model is loaded and valid."""
        return self.model is not None

    def get_ibis_file_path(self) -> str | Path | None:
        """Get the current IBIS file path."""
        return self.ibis_file

    def get_ami_file_path(self) -> str | Path | None:
        """Get the current AMI file path."""
        return self.ibis.ami_file if self.ibis is not None else None

    def get_dll_file_path(self) -> str | Path | None:
        """Get the current DLL file path."""
        return self.ibis.dll_file if self.ibis is not None else None

    def load_ibis_file(
        self,
        filepath: Path | str,
        current_component: str | None = None,
        current_pin: str | None = None,
        current_model: str | None = None,
        auto_load_ami: bool = True,
    ) -> bool:
        """Load a new IBIS file and return True if successful, False otherwise."""
        if not Path(filepath).exists() or not Path(filepath).is_file():
            logger.error("Invalid IBIS file path: %s", filepath)
            return False

        try:
            logger.info("Parsing IBIS file: %s", str(filepath))
            # TODO: Some models are very large, we should lazy load them.
            is_tx = isinstance(self, Transmitter)
            ibis = IBISModel.from_file(filepath, is_tx=is_tx)
            if current_component and current_pin and current_model:
                ibis.current_component = current_component  # This updates the pin dictionary
                ibis.current_pin = current_pin  # This updates the model dictionary
                ibis.current_model = current_model  # Finally set the model.

            self.ibis = ibis
            self.ibis_file = filepath
            self.use_ibis = True

            logger.info("Loaded new IBIS file. Switching to IBIS mode.")
            if auto_load_ami and self.ibis.has_algorithmic_model:
                self.load_ami_configurator(self.ibis.ami_file)
                self.load_dll_model(self.ibis.dll_file)
            return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse IBIS file!\n{err}"
            logger.exception(error_message)
            return False

    def load_ami_configurator(self, ami_file: Path | str, is_tx: bool = True) -> bool:
        """Load the AMI file and return True if successful, False otherwise."""
        if not Path(ami_file).exists() or not Path(ami_file).is_file():
            logger.error("Invalid AMI file path: %s", ami_file)
            return False

        try:
            logger.info("Parsing AMI file, '%s'...", str(ami_file))
            pcfg = AMIParamConfigurator.from_file(ami_file)
            if pcfg is not None:
                self.ami = pcfg
                self.ami_file = ami_file
                self.has_ts4 = pcfg.ts4file is not None
                self.use_ami = True
                logger.info("Loaded new AMI file, switching to AMI equalization mode.")
            else:
                logger.warning("Failed to load AMI file, using native equalization.")
                self.use_ami = False
            if pcfg.ami_parsing_errors:
                logger.warning(f"Non-fatal parsing errors:\n{pcfg.ami_parsing_errors}")
            else:
                logger.info("Success.")
            self.has_getwave = pcfg.getwave_exists()
            init_returns_impulse = pcfg.returns_impulse()
            if not init_returns_impulse:
                self.use_getwave = True
            if pcfg.ts4file():
                self.has_ts4 = True
            else:
                self.has_ts4 = False
            self.use_ami = True
            return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            logger.exception(error_message)
            return False

    def load_dll_model(self, dll_file: Path | str) -> bool:
        """Load the DLL/SO file and return True if successful, False otherwise."""
        if not Path(dll_file).exists() or not Path(dll_file).is_file():
            logger.error("Invalid DLL file path: %s", dll_file)
            return False

        try:
            model = AMIModel(dll_file)
            self.model = model
            self.dll_file = dll_file
            return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to open DLL/SO file! %s", err)
            return False

    def to_dict(self) -> dict:
        """Convert IBIS-AMI buffer to dictionary for serialization."""
        ibis_config = {}
        if self.ibis:
            ibis_config = {
                "current_component": self.ibis._current_component,
                "current_pin": self.ibis._current_pin,
                "current_model": self.ibis._current_model,
            }

        return {
            "ibis_file": str(self.ibis_file) if self.ibis_file else None,
            "use_ami": self.use_ami,
            "use_ts4": self.use_ts4,
            "use_getwave": self.use_getwave,
            "use_ibis": self.use_ibis,
            "ami_file": str(self.ami_file) if hasattr(self, "ami_file") and self.ami_file else None,
            "dll_file": str(self.dll_file) if hasattr(self, "dll_file") and self.dll_file else None,
            **ibis_config,
        }


class Transmitter(Buffer, IbisAmiBuffer):
    """Transmitter class (inherits Buffer, IbisAmiBuffer)."""

    def __init__(
        self,
        impedance: int = 100,
        capacitance: float = 0.5,
        inductance: float = 0,
        ibis_file: str | Path | None = None,
        use_ami: bool = False,
        use_ts4: bool = False,
        use_getwave: bool = False,
        use_ibis: bool = False,
        config_data: dict | None = None,
    ):
        Buffer.__init__(self, impedance, capacitance, inductance)
        IbisAmiBuffer.__init__(
            self,
            ibis_file=ibis_file,
            use_ami=use_ami,
            use_ts4=use_ts4,
            use_getwave=use_getwave,
            use_ibis=use_ibis,
            config_data=config_data,
        )

    def to_dict(self) -> dict:
        base_dict = Buffer.to_dict(self)
        ibis_dict = IbisAmiBuffer.to_dict(self)
        return {**base_dict, **ibis_dict, "impedance": self.impedance, "capacitance": self.capacitance}

    @classmethod
    def from_dict(cls, data: dict) -> "Transmitter":
        return cls(
            impedance=data.get("impedance", 100),
            capacitance=data.get("capacitance", 0.5),
            inductance=data.get("inductance", 0),
            ibis_file=data.get("ibis_file"),
            use_ami=data.get("use_ami", False),
            use_ts4=data.get("use_ts4", False),
            use_getwave=data.get("use_getwave", False),
            use_ibis=data.get("use_ibis", False),
            config_data=data,
        )


class Receiver(Buffer, IbisAmiBuffer):
    """Receiver class (inherits Buffer, IbisAmiBuffer)."""

    def __init__(
        self,
        impedance: int = 100,
        capacitance: float = 0.5,
        inductance: float = 0,
        ac_coupling: float = 1.0,
        use_clocks: bool = False,
        ibis_file: str | Path | None = None,
        use_ami: bool = False,
        use_ts4: bool = False,
        use_getwave: bool = False,
        use_ibis: bool = False,
        config_data: dict | None = None,
    ):
        Buffer.__init__(self, impedance, capacitance, inductance)
        IbisAmiBuffer.__init__(
            self,
            ibis_file=ibis_file,
            use_ami=use_ami,
            use_ts4=use_ts4,
            use_getwave=use_getwave,
            use_ibis=use_ibis,
            config_data=config_data,
        )
        self.ac_coupling = ac_coupling
        self.use_clocks = use_clocks

    def to_dict(self) -> dict:
        base_dict = Buffer.to_dict(self)
        ibis_dict = IbisAmiBuffer.to_dict(self)
        return {
            **base_dict,
            **ibis_dict,
            "impedance": self.impedance,
            "capacitance": self.capacitance,
            "ac_coupling": self.ac_coupling,
            "use_clocks": self.use_clocks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Receiver":
        return cls(
            impedance=data.get("impedance", 100),
            capacitance=data.get("capacitance", 0.5),
            inductance=data.get("inductance", 0),
            ac_coupling=data.get("ac_coupling", 1.0),
            use_clocks=data.get("use_clocks", False),
            ibis_file=data.get("ibis_file"),
            use_ami=data.get("use_ami", False),
            use_ts4=data.get("use_ts4", False),
            use_getwave=data.get("use_getwave", False),
            use_ibis=data.get("use_ibis", False),
            config_data=data,
        )
