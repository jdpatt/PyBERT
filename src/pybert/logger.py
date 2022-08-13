"""The logging and debug functionality of pybert."""
import json
import logging
import time
from logging import Logger
from pathlib import Path

from traitsui.message import message


def setup_logging(verbose=False) -> Logger:
    """Create a console and file handler with the level set to Debug."""
    log = logging.getLogger()

    # Setup a Console Logger
    console_handler = logging.StreamHandler()
    ch_format = logging.Formatter("%(levelname)s - %(message)s")

    console_handler.setFormatter(ch_format)
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.ERROR)

    # Setup a File Logger
    log_file = Path.cwd().joinpath("pybert.log")
    file_handler = logging.FileHandler(log_file, mode="w", delay=True)
    fh_format = StructuredLogger()
    file_handler.setFormatter(fh_format)
    file_handler.setLevel(logging.DEBUG)

    # Remove pyibisami's NullHandler
    logging.getLogger("pyibisami").removeHandler(logging.NullHandler())

    log.addHandler(console_handler)
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)

    log.info("Log file created at: %s", log_file)

    return log


class ConsoleTextLogHandler(logging.Handler):
    """A Log Handler that uses pybert's log console."""

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -  %(message)s"))
        self.setLevel(logging.INFO)

    def emit(self, record, **kwargs):
        """Emit a new record to be displayed in the GUI.

        When running headless, this Handler doesn't do anything. You can use any of the logging
        features and they will show up in the GUI. *Note*, if debug is disabled in the GUI only
        messages INFO and higher will be shown.
        See the [documentation](https://docs.python.org/3/library/logging.html#logging-levels) for
        more infomation.

        If you want a pop-up warning, you need to add alert to the LogRecord.  You can do this by
        adding `extra={"alert":True})` to the function call.  If you want to include the exception
        traceback, you can also include `exc_info=True`.

        Examples:
        --------
        ```python
        # Simple logging
        logging.info("This will show up everywhere.")

        # Prompt the user only if debug is enabled.
        self._log.debug("%s reached uncharted waters.", username, extra={"alert":True}))

        # Include the exception traceback of the last exception raised.
        raise InvalidFileExtension("Pybert does not support this file type.")
        self._log.error("Failed to load configuration.\n", exc_info=True, extra={"alert":True})
        ```
        """
        # pylint: disable=unused-argument
        msg = self.format(record)
        self.application.console_log += f"{msg}\n"
        show_user_alert = False

        if "alert" in record.__dict__:  # alert gets added  when using extra={"alert":True}
            show_user_alert = record.alert
        if self.application.has_gui and show_user_alert:
            message(msg, f"PyBERT Alert: {record.levelname}")


class StructuredLogger(logging.Formatter):
    """Log formatter that will change the logging.LogRecord into a json format."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert the record into a json string."""
        payload = self.record_to_dict(record)
        if hasattr(record, "exc_info") and record.exc_info:
            payload["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(payload)

    @staticmethod
    def record_to_dict(record: logging.LogRecord) -> dict:
        """Convert the record into whatever dictionary format you want to capture."""
        if isinstance(record.args, dict):
            user_payload = record.args
        else:
            user_payload = {"args": repr(record.args)}

        return {
            "message": record.getMessage(),
            "payload": user_payload,
            "meta": {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "name": record.name,
                "pid": record.process,
                "file_url": record.pathname,
                "line": record.lineno,
            },
            "version": "0.0.1",  # This is a version for the log schema and not the program,
        }
