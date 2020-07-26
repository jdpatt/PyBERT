"""The logging and debug functionality for powertree."""
import logging
from logging import Logger, LogRecord
from pathlib import Path
import platform

from pybert import __version__ as VERSION


def setup_logger(pybert, directory: Path = Path(), debug: bool = False) -> Logger:
    """Create a console and file handler with the level set to Debug."""
    log = logging.getLogger("pybert")

    # Setup a Console Logger
    console_handler = logging.StreamHandler()
    ch_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(ch_format)
    console_handler.setLevel(logging.ERROR)
    log.addHandler(console_handler)

    # Setup a File Logger
    file_path = directory.joinpath("pybert.log")
    file_handler = logging.FileHandler(file_path, mode="w", delay=True)
    fh_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(fh_format)
    file_handler.setLevel(logging.DEBUG)
    log.addHandler(file_handler)

    # Setup the Traits Console in the GUI
    gui_handler = GUILogHandler(pybert)
    log.addHandler(gui_handler)

    if debug:
        log.setLevel(logging.DEBUG)
        log.info("Debug Mode Enabled.")
    else:
        log.setLevel(logging.INFO)

    log.info(f"Log file created at: {file_path.resolve()}")
    log.info(f"System: {platform.system()} {platform.release()}")
    log.info(f"Python Version: {platform.python_version()}")
    log.info(f"PyBERT Version: {VERSION}")
    return log


class GUILogHandler(logging.Handler):
    """Create a custom logging handler that appends each record to the Console Log in the GUI."""

    def __init__(self, pybert) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.setLevel(logging.DEBUG)  # Default Level to show

        self.pybert = pybert

    def emit(self, record: LogRecord) -> None:
        """Append the record to the console in the gui."""
        msg = self.format(record)
        self.pybert.console_log += msg + "\n"
