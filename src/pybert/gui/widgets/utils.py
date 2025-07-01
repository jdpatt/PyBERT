from contextlib import contextmanager

from PySide6.QtWidgets import (
    QWidget,
)


@contextmanager
def block_signals(widget: QWidget):
    """Context manager to temporarily block all signals from a widget and its children."""
    try:
        # Block signals on the parent widget itself
        widget.blockSignals(True)
        # Block signals on all children
        for child in widget.findChildren(QWidget):
            child.blockSignals(True)
        yield
    finally:
        # Unblock signals on the parent widget
        widget.blockSignals(False)
        # Unblock signals on all children
        for child in widget.findChildren(QWidget):
            child.blockSignals(False)
