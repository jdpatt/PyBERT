"""Manager class for IBIS and AMI configuration widgets.

This class manages the IBIS and AMI configuration widgets, handling their state
and interactions with the PyBERT model. It provides a clean interface for both
transmitter and receiver widgets to manage their IBIS and AMI configurations.
"""

import logging
from enum import IntEnum
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QStackedWidget, QWidget

from pybert.gui.widgets.ibis_ami_config import IbisAmiConfigWidget
from pybert.gui.widgets.ibis_config import IbisConfigWidget
from pybert.models.buffer import Receiver, Transmitter
from pybert.pybert import PyBERT

logger = logging.getLogger("pybert.ibis_ami_manager")


class StackedWidgetView(IntEnum):
    """Enum for the views of the stacked widgets."""

    NATIVE = 0
    IBIS = 1


class IbisAmiWidgetsManager(QObject):
    """Manager class for IBIS and AMI configuration widgets."""

    def __init__(
        self,
        obj_accessor: Callable[[], Transmitter | Receiver],
        ibis_widget: IbisConfigWidget,
        ami_widget: IbisAmiConfigWidget,
        ibis_stacked_widget: QStackedWidget,
        ami_stacked_widget: QStackedWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the IBIS-AMI manager.

        Args:
            obj_accessor: Function that returns the current Transmitter or Receiver instance
            ibis_widget: IBIS configuration widget
            ami_widget: AMI configuration widget
            ibis_stacked_widget: Stacked widget for IBIS configuration
            ami_stacked_widget: Stacked widget for AMI configuration
            parent: Parent widget
        """
        super().__init__(parent)
        self._obj_accessor = obj_accessor
        self.ibis_widget = ibis_widget
        self.ami_widget = ami_widget
        self.ibis_stacked_widget = ibis_stacked_widget
        self.ami_stacked_widget = ami_stacked_widget
        self.connect_signals()

    @property
    def obj(self) -> Transmitter | Receiver:
        """Get the current object instance."""
        return self._obj_accessor()

    def connect_signals(self) -> None:
        """Connect signals to the widgets."""
        self.ibis_widget.file_picker.file_selected.connect(self._handle_ibis_file_selected)
        self.ibis_widget.configure_btn.clicked.connect(self._handle_ibis_view_btn_clicked)
        self.ami_widget.configure_btn.clicked.connect(self._handle_ami_configure_btn_clicked)

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
            self.ami_widget.set_status("not_loaded")
            return

        current_obj = self.obj  # Get current object
        # Load IBIS file, but don't auto-load AMI so we can handle it ourselves.
        success = current_obj.load_ibis_file(file_path)
        if success:
            self.ibis_widget.set_status("valid")
            self.ibis_stacked_widget.setCurrentIndex(StackedWidgetView.IBIS)  # Switch to the IBIS mode
            self._handle_new_model_selected()
        else:
            self.ibis_widget.set_status("invalid")
            self.ami_widget.set_status("not_loaded")

    def _handle_ibis_view_btn_clicked(self) -> None:
        """The user has clicked the view button, show the IBIS model.

        Grab the current model so we can track if it changes. The IBIS gui will make direct
        changes to the IbisModel class.
        """
        current_obj = self.obj  # Get current object
        if not current_obj.is_ibis_loaded() or current_obj.ibis is None:
            logger.warning("Cannot view IBIS model: no IBIS model loaded")
            self.reset()  # The view button shouldn't be enabled if no model is loaded
            return

        try:
            current_model = current_obj.ibis.current_model
            current_obj.ibis.gui()
            if (
                current_obj.ibis.current_model != current_model
            ):  # TODO: We should track the path as well, multiple models can use the same ami files.
                self._handle_new_model_selected()
        except Exception as e:
            logger.error(f"Failed to show IBIS GUI: {e}")

    def _handle_ami_configure_btn_clicked(self) -> None:
        """The user has clicked the configure button, show the AMI configurator.

        The AMI gui will make direct changes to the AmiConfigurator class.
        """
        current_obj = self.obj  # Get current object
        if not current_obj.is_ami_loaded() or current_obj.ami is None:
            logger.warning("Cannot configure AMI: no AMI model loaded")
            self.ami_widget.set_status("not_loaded")  # The configure button shouldn't be enabled if no model is loaded
            return

        try:
            current_obj.ami.gui()
        except Exception as e:
            logger.error(f"Failed to show AMI GUI: {e}")

    def _handle_new_model_selected(self) -> None:
        """Using the IBIS gui, the user has selected a new pin model which means we need to update the AMI model."""
        current_obj = self.obj  # Get current object
        if not current_obj.is_ibis_loaded():
            logger.warning("Cannot load AMI model: no IBIS model loaded")
            self.reset()
            return

        # Get file paths from IBIS model
        ami_file = current_obj.get_ami_file_path()
        dll_file = current_obj.get_dll_file_path()

        # Load AMI and DLL models using model methods
        if ami_file and dll_file:
            ami_success = current_obj.load_ami_configurator(ami_file)  # type: ignore
            dll_success = current_obj.load_dll_model(dll_file)  # type: ignore
        else:
            ami_success = False
            dll_success = False

        if ami_success and dll_success:
            self.ami_widget.set_status("valid")
            self.ami_stacked_widget.setCurrentIndex(StackedWidgetView.IBIS)
        else:
            self.ami_widget.set_status("invalid")

    def _handle_ami_changed(self) -> None:
        """The AMI model has changed, update the widgets."""
        self.ami_widget.update_widget_from_model()

    def reset(self) -> None:
        """Reset the widgets."""
        self.ibis_widget.set_status("not_loaded")
        self.ami_widget.set_status("not_loaded")
        self.ibis_stacked_widget.setCurrentIndex(StackedWidgetView.NATIVE)
        self.ami_stacked_widget.setCurrentIndex(StackedWidgetView.NATIVE)

    def update_widget_from_model(self) -> None:
        """Update the widgets from the model.

        We assume all of the loading and parsing was already done in the configuration loading, so we need to just
        update the view.
        """
        self.ibis_widget.update_widget_from_model()
        self.ami_widget.update_widget_from_model()
