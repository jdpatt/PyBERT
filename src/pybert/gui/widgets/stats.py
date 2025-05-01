from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StatisticsWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.jitter_info_table = QTableWidget()
        self.jitter_info_table.setColumnCount(5)
        self.jitter_info_table.setHorizontalHeaderLabels(
            ["Location", "Component", "Input (ps)", "Output (ps)", "Rejection (dB)"]
        )
        self.layout.addWidget(self.jitter_info_table)

        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(2)
        self.performance_table.setHorizontalHeaderLabels(["Parameter", "Value"])

        self.performance_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.performance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.add_default_rows(["Channel", "Tx Preemphasis", "CTLE", "DFE", "Jitter", "Plotting", "Total"])

        self.layout.addWidget(self.performance_table)

    def _create_jitter_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        return widget

    def update_performance(self, performance_data: dict):
        """Update the performance table with the given data."""
        self.performance_table.setRowCount(len(performance_data))
        for row, (key, value) in enumerate(performance_data.items()):
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            self.performance_table.setItem(row, 0, key_item)
            self.performance_table.setItem(row, 1, value_item)

    def add_default_rows(self, rows: list[str]):
        """Add default rows to the performance table for the simulation components."""
        self.performance_table.setRowCount(len(rows))  # Set the number of rows first
        for row_idx, row_name in enumerate(rows):
            key_item = QTableWidgetItem(row_name)
            value_item = QTableWidgetItem("0.0")
            self.performance_table.setItem(row_idx, 0, key_item)
            self.performance_table.setItem(row_idx, 1, value_item)
