import logging
import platform

from pyibisami import __version__ as PyAMI_VERSION  # type: ignore
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QStatusBar, QTextEdit

from pybert import __version__ as VERSION


class StructuredLogger(logging.Formatter):
    """A formatter that formats log messages into a structured format."""

    def format(self, record):
        """Format the log record into a JSON structure.

        Args:
            record: The log record to format

        Returns:
            str: JSON formatted log string
        """
        # Create base log structure
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any custom attributes from record
        for key, value in record.__dict__.items():
            if key not in ["timestamp", "level", "name", "message", "exc_info"] and not key.startswith("_"):
                log_data[key] = value

        # Convert to JSON string
        import json

        return json.dumps(log_data)


class QTextEditHandler(QObject, logging.Handler):
    """A logging handler that emits color-coded messages to something that can render html like a QTextEdit.

    This class only emits a signal that the GUI can register a slot to receive the messages. This is done to avoid
    threading issues.
    """

    new_record = Signal(object)

    COLORS = {
        logging.DEBUG: "#808080",  # Gray
        logging.INFO: "#000000",  # Black
        logging.WARNING: "#FFA500",  # Orange
        logging.ERROR: "#FF0000",  # Red
        logging.CRITICAL: "#8B0000",  # Dark Red
    }

    def __init__(self, parent=None, level: int = logging.INFO):
        """Initialize the handler.

        Args:
            parent: The parent object which can always be None.
            level: The logging level for this handler
        """
        QObject.__init__(self, parent)
        logging.Handler.__init__(self, level)
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))

    def emit(self, record):
        """Emit a color-coded log record to the text edit.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            color = self.COLORS.get(record.levelno, "#000000")
            html_msg = f'<span style="color: {color};">{msg}</span>'
            self.new_record.emit(html_msg)
        except Exception:
            self.handleError(record)


class QStatusBarHandler(QObject, logging.Handler):
    """A logging handler that emits messages to a QStatusBar.

    The level or any other record attributes are not used and dropped so only the message is shown.
    """

    new_record = Signal(object)

    def __init__(self, parent=None, level: int = logging.INFO):
        """Initialize the handler.

        Args:
            parent: The parent object which can always be None.
            level: The logging level for this handler
        """
        QObject.__init__(self, parent)
        logging.Handler.__init__(self, level)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        """Emit a log record to the status bar.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            self.new_record.emit(msg)
        except Exception:
            self.handleError(record)


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with a specific name and level.

    Args:
        level: The logging level for the console handler

    Returns:
        logging.Logger: The configured logger
    """
    # Get the pybert logger
    logger = logging.getLogger("pybert")

    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)

    # Create a file handler that logs all information using the structured formatter
    fh = logging.FileHandler("pybert.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(StructuredLogger())
    logger.addHandler(fh)

    return logger


def log_user_system_information(logger: logging.Logger):
    """Log the system information."""
    logger.info(f"Python Version: {platform.python_version()}, System: {platform.system()} {platform.release()}")
    logger.info(f"PyBERT Version: {VERSION}, PyAMI Version: {PyAMI_VERSION}")
