from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton


class StatusIndicator(QLabel):
    """A reusable status indicator widget that shows validation/status states with appropriate styling.

    States:
        - not_loaded: Gray, neutral state
        - valid: Green, success state
        - invalid: Red, error state
        - warning: Yellow, warning state
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Get a temporary button to get the height of the button.
        button = QPushButton()
        button_height = button.sizeHint().height()

        self.setStyleSheet(
            f"""
            QLabel {{
                padding: 0px 8px;
                border-radius: 3px;
                font-weight: bold;
                min-height: {button_height}px;
                max-height: {button_height}px;
            }}
            QLabel[status="valid"] {{
                background-color: #d4edda;
                color: #155724;
            }}
            QLabel[status="invalid"] {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            QLabel[status="warning"] {{
                background-color: #fff3cd;
                color: #856404;
            }}
            QLabel[status="not_loaded"] {{
                background-color: #e2e3e5;
                color: #383d41;
            }}
        """
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status("not_loaded")

    def set_status(self, status: Literal["valid", "invalid", "warning", "not_loaded"]):
        """Update the status and appearance of the indicator."""
        status_text = {"valid": "Valid", "invalid": "Invalid", "warning": "Warning", "not_loaded": "Not Loaded"}
        self.setText(status_text[status])
        self.setProperty("status", status)
        # Force style update
        self.style().unpolish(self)
        self.style().polish(self)
