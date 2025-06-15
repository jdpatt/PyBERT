from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybert.pybert import PyBERT
from pybert.utility.jitter import JitterAnalysis
from pybert.utility.math import safe_log10


class ThickBottomBorderDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, group_size=4):
        super().__init__(parent)
        self.group_size = group_size

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        # Draw thick bottom border for every 4th row
        if (index.row() + 1) % self.group_size == 0:
            pen = painter.pen()
            pen.setWidth(4)
            painter.setPen(pen)
            rect = option.rect
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())


@dataclass
class JitterComponent:
    """Represents a single jitter component (ISI, DCD, Pj, Rj) with input and output values."""

    name: str
    input_val: float  # in ps
    output_val: float  # in ps

    @property
    def rejection_ratio(self) -> Optional[float]:
        """Calculate rejection ratio in dB if possible."""
        if not self.output_val:
            return None
        ratio = self.input_val / self.output_val
        if ratio != ratio or ratio in (float("inf"), float("-inf")):  # Check for NaN or inf
            return None
        return 10.0 * safe_log10(ratio)


@dataclass
class JitterStage:
    """Represents a stage in the signal chain with its jitter components."""

    name: str
    input_jitter: JitterAnalysis
    output_jitter: JitterAnalysis

    def get_components(self) -> list[JitterComponent]:
        """Get all jitter components for this stage."""
        return [
            JitterComponent("ISI", self.input_jitter.isi * 1e12, self.output_jitter.isi * 1e12),
            JitterComponent("DCD", self.input_jitter.dcd * 1e12, self.output_jitter.dcd * 1e12),
            JitterComponent("Pj", self.input_jitter.pj * 1e12, self.output_jitter.pj * 1e12),
            JitterComponent("Rj", self.input_jitter.rj * 1e12, self.output_jitter.rj * 1e12),
        ]


class JitterInfoTable(QTableWidget):
    """Table widget displaying jitter analysis information."""

    def __init__(self, pybert: PyBERT | None = None, parent=None):
        super().__init__(parent)
        self.pybert = pybert
        self.setup_ui()

    def setup_ui(self):
        """Initialize the table UI."""
        self.setAlternatingRowColors(True)
        self.setStyleSheet(
            """
            QTableWidget {
                alternate-background-color: #f0f0f0;
                background-color: white;
            }
        """
        )
        self.setColumnCount(5)
        self.setRowCount(16)
        self.setHorizontalHeaderLabels(["Location", "Component", "Input (ps)", "Output (ps)", "Rejection (dB)"])

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        self.locations = ["Tx Preemphasis", "CTLE (+ AMI DFE)", "DFE", "Total"]
        self.components = ["ISI", "DCD", "Pj", "Rj"]
        self.setItemDelegate(ThickBottomBorderDelegate(self, group_size=len(self.locations)))

        for row, (location, component) in enumerate(product(self.locations, self.components)):
            location_cell = QTableWidgetItem(location)
            component_cell = QTableWidgetItem(component)
            input_cell = QTableWidgetItem("0.0")
            output_cell = QTableWidgetItem("0.0")
            rejection_cell = QTableWidgetItem("0.0")

            # Set cells in table
            cells = [location_cell, component_cell, input_cell, output_cell, rejection_cell]
            for col, cell in enumerate(cells):
                # Add thicker bottom border if last component in a location group
                if component == self.components[-1]:
                    cell.setData(Qt.UserRole, "border-bottom: 2px solid black;")
                self.setItem(row, col, cell)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        selection = self.selectedRanges()
        if not selection:
            return
        s = ""
        for r in selection:
            for row in range(r.topRow(), r.bottomRow() + 1):
                row_data = []
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    item = self.item(row, col)
                    row_data.append(item.text() if item else "")
                s += "\t".join(row_data) + "\n"
        QGuiApplication.clipboard().setText(s)

    def update_rejection(self):
        """Update the jitter rejection table with current values."""
        if not self.pybert or not all(
            [self.pybert.chnl_jitter, self.pybert.tx_jitter, self.pybert.ctle_jitter, self.pybert.dfe_jitter]
        ):
            return

        # Define the stages in the signal chain
        stages = [
            JitterStage("Tx Preemphasis", self.pybert.chnl_jitter, self.pybert.tx_jitter),
            JitterStage("CTLE (+ AMI DFE)", self.pybert.tx_jitter, self.pybert.ctle_jitter),
            JitterStage("DFE", self.pybert.ctle_jitter, self.pybert.dfe_jitter),
            JitterStage("Total", self.pybert.chnl_jitter, self.pybert.dfe_jitter),
        ]

        # Update table with values
        row = 0
        for stage in stages:
            for component in stage.get_components():
                # Input value
                self.item(row, 2).setText(f"{component.input_val:6.3f}")
                # Output value
                self.item(row, 3).setText(f"{component.output_val:6.3f}")
                # Rejection ratio
                rejection = component.rejection_ratio
                self.item(row, 4).setText(f"{rejection:4.1f}" if rejection is not None else "n/a")
                row += 1
