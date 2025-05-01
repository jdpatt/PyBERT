from PySide6 import QMessageBox


def alert(title: str, message: str):
    """
    Display an alert dialog with the given title and message.
    """
    msg = QMessageBox.critical(None, title, message)
    return msg.exec() == QMessageBox.Ok


def warning(title: str, message: str):
    """
    Display a warning dialog with the given title and message.
    """
    msg = QMessageBox.warning(None, title, message)
    return msg.exec() == QMessageBox.Ok


def confirm(title: str, message: str):
    """
    Display a confirmation dialog with the given title and message.
    """
    msg = QMessageBox.question(None, title, message)
    return msg.exec() == QMessageBox.Ok
