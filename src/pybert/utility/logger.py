import logging
from typing import Optional
from PySide6.QtWidgets import QStatusBar, QTextEdit
from PySide6.QtCore import Qt

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
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage()
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add any custom attributes from record
        for key, value in record.__dict__.items():
            if key not in ['timestamp', 'level', 'name', 'message', 'exc_info'] and not key.startswith('_'):
                log_data[key] = value

        # Convert to JSON string
        import json
        return json.dumps(log_data)

class QTextEditHandler(logging.Handler):
    """A logging handler that emits color-coded messages to a QTextEdit."""

    COLORS = {
        logging.DEBUG: '#808080',      # Gray
        logging.INFO: '#000000',       # Black
        logging.WARNING: '#FFA500',    # Orange
        logging.ERROR: '#FF0000',      # Red
        logging.CRITICAL: '#8B0000',   # Dark Red
    }

    def __init__(self, text_edit: QTextEdit, level: int = logging.INFO):
        """Initialize the handler.

        Args:
            text_edit: The QTextEdit to emit messages to
            level: The logging level for this handler
        """
        super().__init__(level)
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))

    def emit(self, record):
        """Emit a color-coded log record to the text edit.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            color = self.COLORS.get(record.levelno, '#000000')
            html_msg = f'<span style="color: {color};">{msg}</span>'
            self.text_edit.append(html_msg)
            # Scroll to bottom to show latest messages
            self.text_edit.verticalScrollBar().setValue(
                self.text_edit.verticalScrollBar().maximum()
            )
        except Exception:
            self.handleError(record)

class QStatusBarHandler(logging.Handler):
    """A logging handler that emits messages to a QStatusBar."""

    def __init__(self, status_bar: QStatusBar, level: int = logging.INFO):
        """Initialize the handler.

        Args:
            status_bar: The QStatusBar to emit messages to
            level: The logging level for this handler
        """
        super().__init__(level)
        self.status_bar = status_bar
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        """Emit a log record to the status bar.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            self.status_bar.showMessage(msg, timeout=2000)  # milli-seconds, default is 0.
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

    # Set the logger level to DEBUG to allow all messages
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs all information using the structured formatter
    fh = logging.FileHandler('pybert.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(StructuredLogger())
    logger.addHandler(fh)

    return logger
