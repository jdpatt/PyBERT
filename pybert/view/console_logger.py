import logging

from PySide2.QtGui import QColor, QSyntaxHighlighter
from PySide2.QtWidgets import QPlainTextEdit

TEXT_COLOR = {
    "WARNING": "black",
    "INFO": "black",
    "DEBUG": "navy",
    "CRITICAL": "red",
    "ERROR": "red",
}


class QTextEditLogger(logging.Handler):
    """Create a custom logginer handler that appends each record to the TextEdit Widget."""

    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(100)
        self.widget.setCenterOnScroll(True)

    def emit(self, record):
        """Append the record to the Widget.  Color according to 'TEXT_COLOR'."""
        msg = self.format(record)
        level = record.levelname
        if level in TEXT_COLOR:
            self.widget.appendHtml(f'<p style="color:{TEXT_COLOR[level]};">{msg}</p>')
        else:
            self.widget.appendPlainText(msg)
        self.widget.ensureCursorVisible()
