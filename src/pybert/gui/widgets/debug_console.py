"""Debug console widget for PyBERT application.

This module implements a debug console widget that displays log
messages.
"""

import logging

from PySide6.QtWidgets import QDockWidget, QTextEdit

from pybert.utility.logger import QTextEditHandler, log_user_system_information


class DebugConsoleWidget(QDockWidget):
    """Debug console widget that displays log messages."""

    def __init__(self, parent=None):
        """Initialize the debug console widget.

        Args:
            parent: Optional parent widget
        """
        super().__init__("Debug Console", parent)

        # Create text edit widget
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self.text_edit.setStyleSheet("QTextEdit { font-family: monospace; }")

        # Set as dock widget's content
        self.setWidget(self.text_edit)

        # Create and add log handler using the existing QTextEditHandler
        self.log_handler = QTextEditHandler(self.text_edit, level=logging.DEBUG)
        logger = logging.getLogger("pybert")
        logger.addHandler(self.log_handler)

        # Show initial message
        logger.debug("Debug console initialized")
        log_user_system_information(logger)

    def append_message(self, msg: str):
        """Append a message to the console.

        Args:
            msg: Message to append
        """
        self.text_edit.append(msg)
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class DebugConsoleHandler(logging.Handler):
    """Logging handler that writes messages to the debug console widget."""

    def __init__(self, widget: DebugConsoleWidget):
        """Initialize the handler.

        Args:
            widget: Debug console widget to write messages to
        """
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record):
        """Emit a log record.

        Args:
            record: Log record to emit
        """
        msg = self.format(record)
        self.widget.append_message(msg)
