from typing import List, Optional

from PySide6.QtWidgets import QTableWidget


def setup_table(table: QTableWidget, headers: List[str], editable_cols: Optional[List[int]] = None):
    """Set up a QTableWidget with headers and editable columns."""
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    for col in range(len(headers)):
        for row in range(table.rowCount()):
            item = table.item(row, col)
            if item and (editable_cols is None or col not in editable_cols):
                item.setFlags(item.flags() & ~table.Qt.ItemIsEditable)
