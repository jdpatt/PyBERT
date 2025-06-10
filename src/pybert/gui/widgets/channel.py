"""Channel configuration widget for PyBERT GUI.

This widget contains controls for channel parameters including file-
based and native (Howard Johnson) channel models.
"""

from PySide6.QtCore import Signal
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

MAX_CHANNEL_FILES = 5
SPARAMETER_FILE_FILTER = "S-parameters (*.s*p);;CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
from pybert.gui.widgets.utils import FilePickerWidget, block_signals
from pybert.pybert import PyBERT


class FileGroupWidget(QWidget):
    """A widget representing a single file group (file selection, port options, remove button, and order number)."""

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
        layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.addItems(["Even", "Odd"])
        self.port_combo.setCurrentText("Odd")
        self.port_combo.currentIndexChanged.connect(lambda _: self.text_changed.emit())
        self.port_combo.setFixedWidth(80)  # Fixed width for port combo
        layout.addWidget(self.port_combo)

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

    def get_port_order(self):
        """Get the port order from the port combo box."""
        return self.port_combo.currentText()

    def is_empty(self):
        """Check if the file group has no file selected."""
        return not bool(self.channel_file.text().strip())


class ChannelConfigWidget(QGroupBox):
    """Widget for configuring channel parameters."""

    def __init__(self, pybert: PyBERT | None = None, parent=None):
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
        self.mode_group.addButton(self.native_radio)
        self.mode_group.addButton(self.file_radio)
        mode_layout.addWidget(self.native_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- Stacked layout for channel config groups ---
        self.stacked_layout = QStackedLayout()

        # From File group (now supports multiple file groups)
        self.file_group = QWidget(self)
        file_layout = QVBoxLayout(self.file_group)
        self.file_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # Add button at the top, right-aligned
        add_btn_layout = QHBoxLayout()
        add_btn_layout.addStretch()
        add_btn = QPushButton("Add Channel File")
        add_btn.clicked.connect(self._add_file_group)
        self.add_btn = add_btn  # Store reference for enabling/disabling
        add_btn_layout.addWidget(add_btn)
        file_layout.addLayout(add_btn_layout)
        self.file_list = QListWidget(self)
        self.file_list.setDragDropMode(QListWidget.InternalMove)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setStyleSheet(
            """
            QListWidget::item:alternate { background: #f0f0f0; }
            QListWidget::item { background: #ffffff; }
            """
        )
        self.file_list.model().rowsMoved.connect(self._update_file_groups_state)
        self.file_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
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

        if pybert is not None:
            self.update_from_model()
            self.connect_signals(pybert)

    def update_from_model(self) -> None:
        """Update all widget values from the PyBERT model.

        Args:
            pybert: PyBERT model instance to update from
        """
        if self.pybert is None:
            return

        with block_signals(self):
            # Update mode
            self.native_radio.setChecked(self.pybert.use_ch_file == False)
            self.file_radio.setChecked(self.pybert.use_ch_file == True)

            # Update native parameters
            self.length.setValue(self.pybert.l_ch)
            self.loss_tan.setValue(self.pybert.Theta0)
            self.z0.setValue(self.pybert.Z0)
            self.v0.setValue(self.pybert.v0)
            self.rdc.setValue(self.pybert.Rdc)
            self.w0.setValue(self.pybert.w0)
            self.r0.setValue(self.pybert.R0)

            # Update file groups
            self.file_list.clear()
            for file_info in self.pybert.channel_elements:
                group = FileGroupWidget(
                    order=len(self.file_list) + 1,
                    remove_callback=lambda: self._update_file_groups_state(),
                    parent=self,
                    file_changed_callback=lambda: self._update_file_groups_state(),
                )
                group.channel_file.setText(file_info["file"])
                group.port_combo.setCurrentText(file_info["port"])
                item = QListWidgetItem()
                self.file_list.addItem(item)
                self.file_list.setItemWidget(item, group)

            self._update_file_groups_state()
            if self.file_list.count() == 0:
                self._add_file_group()
        self._update_mode()

    def connect_signals(self, pybert: "PyBERT") -> None:
        """Connect widget signals to PyBERT instance."""
        self.native_radio.toggled.connect(self._update_mode)
        self.file_radio.toggled.connect(self._update_mode)

        # Connect mode selection
        self.mode_group.buttonReleased.connect(
            lambda val: setattr(pybert, "use_ch_file", self.native_radio.isChecked() == False)
        )

        # Connect native parameters
        self.length.valueChanged.connect(lambda val: setattr(pybert, "channel_length", val))
        self.loss_tan.valueChanged.connect(lambda val: setattr(pybert, "channel_loss_tan", val))
        self.z0.valueChanged.connect(lambda val: setattr(pybert, "channel_z0", val))
        self.v0.valueChanged.connect(lambda val: setattr(pybert, "channel_v0", val))
        self.rdc.valueChanged.connect(lambda val: setattr(pybert, "channel_rdc", val))
        self.w0.valueChanged.connect(lambda val: setattr(pybert, "channel_w0", val))
        self.r0.valueChanged.connect(lambda val: setattr(pybert, "channel_r0", val))

        # Connect use_window
        self.use_window.toggled.connect(lambda val: setattr(pybert, "use_window", val))

    def get_channel_elements(self):
        """Get the list of files and their parameters from the file groups.

        Returns:
            list: List of tuples containing (filename, port_order) for each file group
        """
        files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            widget: FileGroupWidget = self.file_list.itemWidget(item)
            if widget and widget.get_filename():
                files.append((widget.get_filename(), widget.get_port_order()))
        return files

    def _update_mode(self):
        """Show only the selected group (Native or From File) using stacked layout."""
        if self.native_radio.isChecked():
            self.stacked_layout.setCurrentWidget(self.native_group)
        else:
            self.stacked_layout.setCurrentWidget(self.file_group)

    def _update_file_groups_state(self):
        """Update all file group related state in one place."""
        count = self.file_list.count()
        has_empty = False

        # Update orders and check for empty groups
        for i in range(count):
            item = self.file_list.item(i)
            widget = self.file_list.itemWidget(item)
            if widget:
                widget.set_order(i + 1)
                widget.set_remove_enabled(count > 1)
                if widget.is_empty():
                    has_empty = True

        # Update add button state
        can_add = (count < MAX_CHANNEL_FILES) and not has_empty
        self.add_btn.setEnabled(can_add)

        # Update PyBERT if available
        if self.pybert:
            setattr(self.pybert, "elements", self.get_channel_elements())

    def _add_file_group(self):
        """Add a new file group if validation passes."""
        if self.file_list.count() >= MAX_CHANNEL_FILES:
            return

        # Check if any existing groups are empty
        if not self._validate_existing_groups():
            return

        item = QListWidgetItem()

        def remove():
            if self.file_list.count() > 1:
                row = self.file_list.row(item)
                self.file_list.takeItem(row)
                self._update_file_groups_state()

        widget = FileGroupWidget(
            parent=self,
            order=self.file_list.count() + 1,
            remove_callback=remove,
            file_changed_callback=self._update_file_groups_state,
        )
        widget.text_changed.connect(lambda: setattr(self.pybert, "elements", self.get_channel_elements()))
        item.setSizeHint(widget.sizeHint())
        self.file_list.addItem(item)
        self.file_list.setItemWidget(item, widget)
        self._update_file_groups_state()

    def _validate_existing_groups(self):
        """Check if any existing file groups are empty."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            widget = self.file_list.itemWidget(item)
            if widget and widget.is_empty():
                return False
        return True
