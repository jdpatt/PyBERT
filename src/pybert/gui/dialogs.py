from PySide6.QtWidgets import QMessageBox, QWidget

# Global flag to disable dialogs during testing
_DISABLE_DIALOGS = False


def set_dialog_enabled(enabled: bool) -> None:
    """Enable or disable dialog display. Useful for testing.

    Args:
        enabled: If True, dialogs will be shown. If False, dialogs will be suppressed
                and messages will be printed to console instead.
    """
    global _DISABLE_DIALOGS
    _DISABLE_DIALOGS = not enabled


def alert_dialog(parent: QWidget, title: str, message: str):
    """Display an alert dialog with the given title and message."""
    if _DISABLE_DIALOGS:
        print(f"ALERT: {title} - {message}")
        return True
    return QMessageBox.critical(parent, title, message) == QMessageBox.StandardButton.Ok


def info_dialog(parent: QWidget, title: str, message: str):
    """Display an info dialog with the given title and message."""
    if _DISABLE_DIALOGS:
        print(f"INFO: {title} - {message}")
        return True
    return QMessageBox.information(parent, title, message) == QMessageBox.StandardButton.Ok


def warning_dialog(parent: QWidget, title: str, message: str):
    """Display a warning dialog with the given title and message."""
    if _DISABLE_DIALOGS:
        print(f"WARNING: {title} - {message}")
        return True
    return QMessageBox.warning(parent, title, message) == QMessageBox.StandardButton.Ok


def confirm_dialog(parent: QWidget, title: str, message: str):
    """Display a confirmation dialog with the given title and message."""
    if _DISABLE_DIALOGS:
        print(f"CONFIRM: {title} - {message}")
        return True
    return QMessageBox.question(parent, title, message) == QMessageBox.StandardButton.Ok
