import logging

from PySide2.QtGui import QColor
from PySide2.QtWidgets import QTextEdit

TEXT_COLOR = {
    "WARNING": QColor("black"),
    "INFO": QColor("black"),
    "DEBUG": QColor("red"),
    "CRITICAL": QColor("red"),
    "ERROR": QColor("red"),
}


class QTextEditLogger(logging.Handler):
    """Create a custom logginer handler that appends each record to the TextEdit Widget."""

    def __init__(self, parent):
        super().__init__()
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        """Append the record to the Widget.  Color according to 'TEXT_COLOR'."""
        msg = self.format(record)
        level = record.levelname
        if level in TEXT_COLOR:
            self.widget.setTextColor(TEXT_COLOR[level])
            self.widget.append(msg)
