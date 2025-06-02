from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


def alert_dialog(title: str, message: str):
    """Display an alert dialog with the given title and message."""
    return QMessageBox.critical(None, title, message) == QMessageBox.Ok


def info_dialog(title: str, message: str):
    """Display an info dialog with the given title and message."""
    return QMessageBox.information(None, title, message) == QMessageBox.Ok


def warning_dialog(title: str, message: str):
    """Display a warning dialog with the given title and message."""
    return QMessageBox.warning(None, title, message) == QMessageBox.Ok


def confirm_dialog(title: str, message: str):
    """Display a confirmation dialog with the given title and message."""
    return QMessageBox.question(None, title, message) == QMessageBox.Ok


def select_file_dialog(parent: QWidget, caption: str, file_filter: str) -> Optional[str]:
    """Open a file dialog and return the selected file path or None."""
    filename, _ = QFileDialog.getOpenFileName(parent, caption, "", file_filter)
    return filename if filename else None
