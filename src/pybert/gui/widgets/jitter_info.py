from itertools import product

from PySide6.QtCore import Qt
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
from pybert.utility.debug import setattr
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


class JitterInfoTable(QTableWidget):
    def __init__(self, pybert: PyBERT| None = None, parent=None):
        super().__init__(parent)
        self.pybert = pybert

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.jitter_info_table = QTableWidget()
        self.jitter_info_table.setAlternatingRowColors(True)
        self.jitter_info_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
                background-color: white;
            }
        """)
        self.jitter_info_table.setColumnCount(5)
        self.jitter_info_table.setRowCount(16)
        self.jitter_info_table.setHorizontalHeaderLabels(
            ["Location", "Component", "Input (ps)", "Output (ps)", "Rejection (dB)"]
        )

        header = self.jitter_info_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        self.locations = ["Tx Preemphasis", "CTLE (+ AMI DFE)", "DFE", "Total"]
        self.components = ["ISI", "DCD", "Pj", "Rj"]
        self.jitter_info_table.setItemDelegate(ThickBottomBorderDelegate(self.jitter_info_table, group_size=len(self.locations)))

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
                self.jitter_info_table.setItem(row, col, cell)

        self.layout.addWidget(self.jitter_info_table)


    def update_rejection(self):
        # Gather jitter values from pybert (in ps)
        isi_chnl = self.pybert.isi_chnl * 1.0e12
        dcd_chnl = self.pybert.dcd_chnl * 1.0e12
        pj_chnl = self.pybert.pj_chnl * 1.0e12
        rj_chnl = self.pybert.rj_chnl * 1.0e12
        isi_tx = self.pybert.isi_tx * 1.0e12
        dcd_tx = self.pybert.dcd_tx * 1.0e12
        pj_tx = self.pybert.pj_tx * 1.0e12
        rj_tx = self.pybert.rj_tx * 1.0e12
        isi_ctle = self.pybert.isi_ctle * 1.0e12
        dcd_ctle = self.pybert.dcd_ctle * 1.0e12
        pj_ctle = self.pybert.pj_ctle * 1.0e12
        rj_ctle = self.pybert.rj_ctle * 1.0e12
        isi_dfe = self.pybert.isi_dfe * 1.0e12
        dcd_dfe = self.pybert.dcd_dfe * 1.0e12
        pj_dfe = self.pybert.pj_dfe * 1.0e12
        rj_dfe = self.pybert.rj_dfe * 1.0e12

        # Calculate rejection ratios
        def rej(in_val, out_val):
            if out_val:
                return in_val / out_val
            return float('nan')

        # Table order: locations x components
        # locations = ["Tx Preemphasis", "CTLE (+ AMI DFE)", "DFE", "Total"]
        # components = ["ISI", "DCD", "Pj", "Rj"]
        values = [
            # Tx Preemphasis
            (isi_chnl, isi_tx, rej(isi_chnl, isi_tx)),
            (dcd_chnl, dcd_tx, rej(dcd_chnl, dcd_tx)),
            (pj_chnl, pj_tx, float('nan')),
            (rj_chnl, rj_tx, float('nan')),
            # CTLE (+ AMI DFE)
            (isi_tx, isi_ctle, rej(isi_tx, isi_ctle)),
            (dcd_tx, dcd_ctle, rej(dcd_tx, dcd_ctle)),
            (pj_tx, pj_ctle, rej(pj_tx, pj_ctle)),
            (rj_tx, rj_ctle, rej(rj_tx, rj_ctle)),
            # DFE
            (isi_ctle, isi_dfe, rej(isi_ctle, isi_dfe)),
            (dcd_ctle, dcd_dfe, rej(dcd_ctle, dcd_dfe)),
            (pj_ctle, pj_dfe, rej(pj_ctle, pj_dfe)),
            (rj_ctle, rj_dfe, rej(rj_ctle, rj_dfe)),
            # Total
            (isi_chnl, isi_dfe, rej(isi_chnl, isi_dfe)),
            (dcd_chnl, dcd_dfe, rej(dcd_chnl, dcd_dfe)),
            (pj_tx, pj_dfe, rej(pj_tx, pj_dfe)),
            (rj_tx, rj_dfe, rej(rj_tx, rj_dfe)),
        ]

        for row, (input_val, output_val, rejection) in enumerate(values):
            # Input (ps)
            self.jitter_info_table.item(row, 2).setText(f"{input_val:6.3f}")
            # Output (ps)
            self.jitter_info_table.item(row, 3).setText(f"{output_val:6.3f}")
            # Rejection (dB)
            if not (rejection is None or rejection != rejection or rejection == float('inf') or rejection == float('-inf')):
                self.jitter_info_table.item(row, 4).setText(f"{10.0 * safe_log10(rejection):4.1f}")
            else:
                self.jitter_info_table.item(row, 4).setText("n/a")
