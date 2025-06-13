"""Manager class for IBIS and AMI configuration widgets.

This class manages the IBIS and AMI configuration widgets, handling their state
and interactions with the PyBERT model. It provides a clean interface for both
transmitter and receiver widgets to manage their IBIS and AMI configurations.
"""

import logging
from typing import Optional, Tuple

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget

from pybert.gui.widgets.ibis_ami_config import IbisAmiConfigWidget
from pybert.gui.widgets.ibis_config import IbisConfigWidget
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.ibis_ami_manager")


class IbisAmiManager(QObject):
    """Manager class for IBIS and AMI configuration widgets."""

    ibis_changed = Signal()
    ami_changed = Signal()

    def __init__(
        self,
        pybert: PyBERT,
        is_tx: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the IBIS-AMI manager.

        Args:
            pybert: PyBERT instance
            is_tx: True if this is for transmitter, False if for receiver
            parent: Parent widget
        """
        super().__init__(parent)  # Initialize QObject
        self.pybert = pybert
        self.is_tx = is_tx
        self.direction = "tx" if is_tx else "rx"
        self.parent = parent

        self.ibis_model = None
        self.ami_model = None
        self.dll_model = None

        # Create the widgets with their respective parents
        self.ibis_widget = IbisConfigWidget(pybert=pybert, is_tx=is_tx, parent=parent)
        self.ami_widget = IbisAmiConfigWidget(pybert=pybert, is_tx=is_tx, parent=parent)

        # Connect signals
        self.ibis_widget.file_picker.file_selected.connect(self._handle_ibis_file_selected)
        self.ibis_widget.view_btn.clicked.connect(self._handle_ibis_view_btn_clicked)
        self.ami_widget.configure_btn.clicked.connect(self._handle_ami_configure_btn_clicked)

    def update_widget_from_model(self) -> None:
        """Update the widgets from the model."""
        # TODO: This currently doesn't handle parsing the IBIS and AMI files if loaded from a configuration file.
        self.ibis_widget.update_widget_from_model()
        self.ami_widget.update_widget_from_model()

    def get_ibis_widget(self) -> IbisConfigWidget:
        """Get the IBIS and AMI configuration widgets.

        Returns:
            Tuple containing the IBIS and AMI configuration widgets
        """
        return self.ibis_widget

    def get_ami_widget(self) -> IbisAmiConfigWidget:
        """Get the AMI configuration widget.

        Returns:
            AMI configuration widget
        """
        return self.ami_widget

    def _handle_ibis_file_selected(self, file_path: str | None) -> None:
        """The user has selected a new IBIS file, load it and update the view."""
        if file_path is None:
            self.ibis_widget.set_status("not_loaded")
        self.ibis_model = self.pybert.load_ibis_file(file_path, is_tx=self.is_tx)
        if self.ibis_model is not None:
            self.ibis_widget.set_status("valid")
            setattr(self.pybert, f"{self.direction}_ibis_valid", True)
            setattr(self.pybert, f"{self.direction}_ibis", self.ibis_model)
            setattr(self.pybert, f"{self.direction}_use_ibis", True)
            setattr(self.pybert, f"{self.direction}_ibis_file", file_path)
            self.ibis_widget.view_btn.setEnabled(True)
            self._handle_new_ami_model_selected()
            self.ibis_changed.emit()
        else:
            self.ibis_widget.set_status("invalid")
            setattr(self.pybert, f"{self.direction}_ibis_valid", False)
            setattr(self.pybert, f"{self.direction}_ibis", None)
            setattr(self.pybert, f"{self.direction}_use_ibis", False)
            setattr(self.pybert, f"{self.direction}_ibis_file", "")
            self.ibis_widget.view_btn.setEnabled(False)
            self.ami_widget.reset()
            self.ibis_changed.emit()

    def _handle_ibis_view_btn_clicked(self) -> None:
        """The user has clicked the view button, show the IBIS model."""
        if self.ibis_model is not None:
            current_model = self.ibis_model.current_model
            self.ibis_model.gui()
            if self.ibis_model.current_model != current_model:
                self._handle_new_ami_model_selected()

    def _handle_ami_configure_btn_clicked(self) -> None:
        """The user has clicked the configure button, show the AMI configurator."""
        if self.ami_model is not None:
            self.ami_model.gui()

    def _handle_new_ami_model_selected(self) -> None:
        """The user has selected a new AMI model, load it and update the view."""
        self.ami_model = self.pybert.load_ami_configurator(self.ibis_model.ami_file, is_tx=self.is_tx)
        if self.ami_model is not None:
            self.ami_widget.set_status("valid")
            setattr(self.pybert, f"{self.direction}_ami", self.ami_model)
            setattr(self.pybert, f"{self.direction}_ami_valid", True)
            setattr(self.pybert, f"{self.direction}_use_ami", True)
            setattr(self.pybert, f"{self.direction}_ami_file", self.ibis_model.ami_file)
            setattr(self.pybert, f"{self.direction}_dll_file", self.ibis_model.dll_file)
            setattr(self.pybert, f"{self.direction}_has_getwave", self.ami_model.getwave_exists())
            setattr(self.pybert, f"{self.direction}_has_ts4", self.ami_model.ts4file() is not None)
            self.ami_widget.configure_btn.setEnabled(True)
            self.ami_widget.set_filepaths(self.ibis_model.ami_file, self.ibis_model.dll_file)
            self.ami_changed.emit()
        else:
            self.ami_widget.set_status("invalid")
            setattr(self.pybert, f"{self.direction}_ami", None)
            setattr(self.pybert, f"{self.direction}_ami_valid", False)
            setattr(self.pybert, f"{self.direction}_use_ami", False)
            setattr(self.pybert, f"{self.direction}_has_getwave", False)
            setattr(self.pybert, f"{self.direction}_has_ts4", False)
            self.ami_widget.configure_btn.setEnabled(False)
            self.ami_widget.set_filepaths(None, None)
            self.ami_changed.emit()
