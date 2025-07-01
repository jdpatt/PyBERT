"""Channel configuration widget for PyBERT GUI.

This widget contains controls for channel parameters including file-
based and native (Howard Johnson) channel models.
"""

from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from pybert.gui.widgets.file_picker import FilePickerWidget
from pybert.gui.widgets.utils import block_signals
from pybert.models.channel import ChannelElement

if TYPE_CHECKING:
    from pybert.pybert import PyBERT

SPARAMETER_FILE_FILTER = "S-Parameter Files (*.s4p *.s2p);;Touchstone Files (*.s4p *.s2p);;All Files (*.*)"
MAX_CHANNEL_FILES = 5


class ChannelItemWidget(QWidget):
    """A widget representing a single file group in the File based channel model."""

    text_changed = Signal()

    def __init__(self, order=1, remove_callback=None, parent=None, file_changed_callback=None):
        super().__init__(parent)
        self.file_changed_callback = file_changed_callback

        # Create main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)  # Reduced margins for list widget items
        layout.setSpacing(8)  # Consistent spacing between elements

        # Order label
        self.order_label = QLabel(str(order))
        self.order_label.setMinimumWidth(20)  # Fixed width for order number
        layout.addWidget(self.order_label)

        # File selection using FilePickerWidget
        self.channel_file = FilePickerWidget(
            label_text="",  # Empty label since we have the order number
            file_filter=SPARAMETER_FILE_FILTER,
            parent=self,
        )
        self.channel_file.file_selected.connect(lambda _: self.text_changed.emit())
        self.channel_file.file_selected.connect(
            lambda _: self.file_changed_callback() if self.file_changed_callback else None
        )
        self.channel_file.file_edit.setPlaceholderText("Select channel file...")
        layout.addWidget(self.channel_file, stretch=1)  # Allow file path to expand

        # Port order
        layout.addWidget(QLabel("Renumber:"))
        self.renumber = QCheckBox()
        self.renumber.setChecked(False)
        self.renumber.stateChanged.connect(lambda _: self.text_changed.emit())
        layout.addWidget(self.renumber)

        # Remove button
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedWidth(24)
        self.remove_btn.setToolTip("Remove this channel file")
        if remove_callback:
            self.remove_btn.clicked.connect(remove_callback)
        layout.addWidget(self.remove_btn)

    def set_order(self, order):
        self.order_label.setText(str(order))

    def set_remove_enabled(self, enabled):
        self.remove_btn.setEnabled(enabled)

    def get_filename(self):
        """Get the filename from the channel file line edit."""
        return self.channel_file.text()

    def get_renumber(self):
        """Get the renumber from the renumber check box."""
        return self.renumber.isChecked()

    def is_empty(self):
        """Check if the file group has no file selected."""
        return not bool(self.channel_file.text().strip())


class ChannelFileListWidget(QListWidget):
    """A specialized list widget for managing channel file groups."""

    def __init__(self, parent=None, items_changed_callback=None):
        super().__init__(parent)
        self.items_changed_callback = items_changed_callback
        self._loading = False  # Flag to prevent callbacks during loading
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setAlternatingRowColors(True)
        self.model().rowsMoved.connect(self._update_file_groups_state)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def items_from_list(self, file_elements):
        """Load items from a list of file element dictionaries."""
        # Prevent callbacks during loading
        self._loading = True

        # Properly clean up existing widgets before clearing
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if widget:
                widget.deleteLater()

        self.clear()

        for file_info in file_elements:
            group = ChannelItemWidget(
                order=self.count() + 1,
                remove_callback=lambda: self._update_file_groups_state(),
                parent=self,
                file_changed_callback=lambda: self._update_file_groups_state(),
            )
            group.channel_file.set_filepath(file_info["file"])
            group.renumber.setChecked(file_info["renumber"])
            item = QListWidgetItem()
            self.addItem(item)
            self.setItemWidget(item, group)

        # Always ensure there's at least one group for user convenience
        if self.count() == 0:
            self._add_file_group()

        self._update_file_groups_state()

        # Re-enable callbacks and trigger one final update
        self._loading = False
        if self.items_changed_callback:
            self.items_changed_callback()

    def get_items_as_list(self):
        """Get the list of files and their parameters from the file groups.

        Returns:
            list: List of dictionaries containing {"file": filename, "port": port_order} for each file group
        """
        files = []
        for i in range(self.count()):
            item = self.item(i)
            widget = cast(ChannelItemWidget, self.itemWidget(item))
            if widget and widget.get_filename():
                files.append(ChannelElement(file=widget.get_filename(), renumber=widget.get_renumber()))
        return files

    def _update_file_groups_state(self):
        """Update all file group related state in one place."""
        count = self.count()
        has_empty = False

        # Update orders and check for empty groups
        for i in range(count):
            item = self.item(i)
            widget = cast(ChannelItemWidget, self.itemWidget(item))
            if widget:
                widget.set_order(i + 1)
                widget.set_remove_enabled(count > 1)
                if widget.is_empty():
                    has_empty = True

        # Update add button state if we have a reference to it
        if hasattr(self, "add_btn"):
            can_add = (count < MAX_CHANNEL_FILES) and not has_empty
            self.add_btn.setEnabled(can_add)

        # Notify parent of changes only when not loading
        if self.items_changed_callback and not self._loading:
            self.items_changed_callback()

    def _add_file_group(self):
        """Add a new file group if validation passes and no empty group exists."""
        # Prevent adding if an empty group already exists
        for i in range(self.count()):
            item = self.item(i)
            widget = cast(ChannelItemWidget, self.itemWidget(item))
            if widget and widget.is_empty():
                return

        if self.count() >= MAX_CHANNEL_FILES:
            return

        # Check if any existing groups are empty
        if not self._validate_existing_groups():
            return

        item = QListWidgetItem()

        def remove():
            if self.count() > 1:
                row = self.row(item)
                self.takeItem(row)
                self._update_file_groups_state()

        widget = ChannelItemWidget(
            parent=self,
            order=self.count() + 1,
            remove_callback=remove,
            file_changed_callback=self._update_file_groups_state,
        )
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        self._update_file_groups_state()

    def _validate_existing_groups(self):
        """Check if any existing file groups are empty."""
        for i in range(self.count()):
            item = self.item(i)
            widget = cast(ChannelItemWidget, self.itemWidget(item))
            if widget and widget.is_empty():
                return False
        return True

    def set_add_button(self, add_btn):
        """Set a reference to the add button for state management."""
        self.add_btn = add_btn


class ChannelConfigWidget(QGroupBox):
    """Widget for configuring channel parameters."""

    def __init__(self, pybert: "PyBERT", parent=None):
        """Initialize the channel configuration widget.

        Args:
            pybert: PyBERT model instance
            parent: Parent widget
        """
        super().__init__("Channel", parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Mode selection radio buttons ---
        mode_layout = QHBoxLayout()
        self.native_radio = QRadioButton("Native")
        self.file_radio = QRadioButton("From File(s)")
        self.native_radio.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.native_radio, 0)
        self.mode_group.addButton(self.file_radio, 1)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- Stacked layout for channel config groups ---
        self.stacked_layout = QStackedLayout()

        # From File group (now supports multiple file groups)
        self.file_group = QWidget(self)
        file_layout = QVBoxLayout(self.file_group)
        self.file_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        # Add button at the top, right-aligned
        add_btn_layout = QHBoxLayout()
        add_btn_layout.addStretch()
        add_btn = QPushButton("Add Channel File")
        self.add_btn = add_btn  # Store reference for enabling/disabling
        add_btn_layout.addWidget(add_btn)
        file_layout.addLayout(add_btn_layout)

        # Use the new ChannelFileListWidget
        self.file_list = ChannelFileListWidget(self, items_changed_callback=self._on_file_list_changed)
        add_btn.clicked.connect(self.file_list._add_file_group)
        self.file_list.set_add_button(self.add_btn)
        file_layout.addWidget(self.file_list, stretch=1)
        file_layout.addStretch()

        # Native model group
        self.native_group = QWidget(self)
        native_form = QFormLayout()
        self.native_group.setLayout(native_form)

        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 10.0)
        self.length.setSuffix(" m")
        native_form.addRow(QLabel("Length"), self.length)

        self.loss_tan = QDoubleSpinBox()
        self.loss_tan.setRange(0.0, 1.0)
        self.loss_tan.setDecimals(3)
        native_form.addRow(
            QLabel(
                "Loss Tangent",
            ),
            self.loss_tan,
        )

        self.z0 = QDoubleSpinBox()
        self.z0.setRange(0.0, 200.0)
        self.z0.setSuffix(" Ohms")
        native_form.addRow(QLabel("Characteristic Impedance"), self.z0)

        self.v0 = QDoubleSpinBox()
        self.v0.setRange(0.0, 1.0)
        self.v0.setSuffix(" c")
        native_form.addRow(QLabel("Relative Velocity"), self.v0)

        self.rdc = QDoubleSpinBox()
        self.rdc.setRange(0.0, 100.0)
        self.rdc.setSuffix(" Ohms")
        native_form.addRow(QLabel("DC Resistance"), self.rdc)

        self.w0 = QDoubleSpinBox()
        self.w0.setRange(0.0, 1e12)
        self.w0.setSuffix(" rads/s")
        native_form.addRow(QLabel("Transition Frequency"), self.w0)

        self.r0 = QDoubleSpinBox()
        self.r0.setRange(0.0, 100.0)
        self.r0.setSuffix(" Ohms")
        native_form.addRow(QLabel("Skin Effect Resistance"), self.r0)

        # Add both groups to stacked layout (after both are constructed)
        self.stacked_layout.addWidget(self.native_group)
        self.stacked_layout.addWidget(self.file_group)
        layout.addLayout(self.stacked_layout, stretch=1)

        # Misc group
        misc_group = QGroupBox("Misc.")
        misc_layout = QHBoxLayout()
        misc_group.setLayout(misc_layout)

        self.use_window = QCheckBox("Apply window")
        self.use_window.setToolTip("Apply raised cosine window to frequency response before FFT()'ing.")
        misc_layout.addWidget(self.use_window)

        layout.addWidget(misc_group)

        # Add stretch to push everything to the top
        layout.addStretch()

        self.update_widget_from_model()
        self.connect_signals()

    def update_widget_from_model(self) -> None:
        """Update all widget values from the PyBERT model."""
        with block_signals(self):
            # Update mode
            self.native_radio.setChecked(not self.pybert.channel.use_ch_file)
            self.file_radio.setChecked(self.pybert.channel.use_ch_file)

            # Update native parameters
            self.length.setValue(self.pybert.channel.l_ch)
            self.loss_tan.setValue(self.pybert.channel.Theta0)
            self.z0.setValue(self.pybert.channel.Z0)
            self.v0.setValue(self.pybert.channel.v0)
            self.rdc.setValue(self.pybert.channel.Rdc)
            self.w0.setValue(self.pybert.channel.w0)
            self.r0.setValue(self.pybert.channel.R0)
            self.use_window.setChecked(self.pybert.channel.use_window)

            # Update file groups
            self.file_list.items_from_list(self.pybert.channel.elements)

        # Update stacked layout after signals are unblocked
        self.stacked_layout.setCurrentIndex(1 if self.pybert.channel.use_ch_file else 0)

    def connect_signals(self) -> None:
        """Connect widget signals to PyBERT instance."""
        self.mode_group.buttonReleased.connect(self._handle_channel_mode_change)

        # Connect native parameters
        self.length.valueChanged.connect(lambda val: setattr(self.pybert.channel, "l_ch", val))
        self.loss_tan.valueChanged.connect(lambda val: setattr(self.pybert.channel, "Theta0", val))
        self.z0.valueChanged.connect(lambda val: setattr(self.pybert.channel, "Z0", val))
        self.v0.valueChanged.connect(lambda val: setattr(self.pybert.channel, "v0", val))
        self.rdc.valueChanged.connect(lambda val: setattr(self.pybert.channel, "Rdc", val))
        self.w0.valueChanged.connect(lambda val: setattr(self.pybert.channel, "w0", val))
        self.r0.valueChanged.connect(lambda val: setattr(self.pybert.channel, "R0", val))

        # Connect use_window
        self.use_window.toggled.connect(lambda val: setattr(self.pybert.channel, "use_window", val))

    def _handle_channel_mode_change(self) -> None:
        """Handle mode change from radio buttons."""
        # Update stacked layout
        self.stacked_layout.setCurrentIndex(1 if self.file_radio.isChecked() else 0)
        # Update PyBERT model
        setattr(self.pybert.channel, "use_ch_file", self.file_radio.isChecked())

    def _on_file_list_changed(self) -> None:
        """Callback when file list items change."""
        new_elements = self.get_channel_elements()
        # Only update if the value has actually changed
        if new_elements != self.pybert.channel.elements:
            setattr(self.pybert.channel, "elements", new_elements)

    def get_channel_elements(self):
        """Get the list of files and their parameters from the file groups.

        Returns:
            list: List of dictionaries containing {"file": filename, "port": port_order} for each file group
        """
        return self.file_list.get_items_as_list()
