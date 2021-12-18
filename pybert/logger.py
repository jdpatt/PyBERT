"""The logging and debug functionality of pybert."""
import logging
from logging import Logger
from pathlib import Path
import json
import time

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

    # Remove pyami's NullHandler
    logging.getLogger("pyami").removeHandler(logging.NullHandler())

    log.addHandler(console_handler)
    log.addHandler(file_handler)
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    log.info(f"Log file created at: {log_file}")

    return log

class ConsoleTextLogHandler(logging.Handler):
    """A Log Handler that uses pybert's log console."""

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s -  %(message)s"
            )
        )
        self.setLevel(logging.INFO)

    def emit(self, record, **kwargs):
        """Emit a new record with the level before the message."""
        msg = self.format(record)
        self.application.console_log += f"{msg}\n"

        show_user_alert = False
        if isinstance(record.args, dict): # if no kwargs are passed, record.args is an empty tuple.
            show_user_alert = record.args.get("alert", False)
        if self.application.has_gui and show_user_alert:
            message(msg, "PyBERT Alert")

class StructuredLogger(logging.Formatter):
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
