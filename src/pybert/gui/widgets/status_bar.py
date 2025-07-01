import logging
from typing import Optional

from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

from pybert.pybert import PyBERT
from pybert.results import Results
from pybert.utility.logger import QStatusBarHandler

logger = logging.getLogger(__name__)


class StatusBar(QStatusBar):
    """Status bar for PyBERT GUI."""

    def __init__(self, pybert: PyBERT, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.pybert = pybert

        # Create permanent status widgets with styling
        style = "padding: 0px 10px; margin: 2px; border-left: 1px solid #cccccc;"

        self.perf_label = QLabel("Perf: 0.0 Msmpls/min")
        self.perf_label.setStyleSheet(style)

        self.delay_label = QLabel("Channel Delay: 0.0 ns")
        self.delay_label.setStyleSheet(style)

        self.errors_label = QLabel("Bit Errors: 0")
        self.errors_label.setStyleSheet(style)

        self.power_label = QLabel("Tx Power: 0.0 mW")
        self.power_label.setStyleSheet(style)

        # Jitter metrics
        self.isi_label = QLabel("ISI: 0.0 ps")
        self.isi_label.setStyleSheet(style)

        self.dcd_label = QLabel("DCD: 0.0 ps")
        self.dcd_label.setStyleSheet(style)

        self.pj_label = QLabel("Pj: 0.0 ps")
        self.pj_label.setStyleSheet(style)

        self.rj_label = QLabel("Rj: 0.0 ps")
        self.rj_label.setStyleSheet(style)

        # Add permanent widgets to status bar (right-aligned)
        self.addPermanentWidget(self.perf_label)
        self.addPermanentWidget(self.delay_label)
        self.addPermanentWidget(self.errors_label)
        self.addPermanentWidget(self.power_label)
        self.addPermanentWidget(self.isi_label)
        self.addPermanentWidget(self.dcd_label)
        self.addPermanentWidget(self.pj_label)
        self.addPermanentWidget(self.rj_label)

        # Add the status bar handler for logging
        status_bar_handler = QStatusBarHandler()
        logger.addHandler(status_bar_handler)
        status_bar_handler.new_record.connect(self.showMessage)

    def update(self, results: Results):
        """Update the status bar with performance metrics and jitter values.

        Args:
            results: Simulation results
        """

        self.perf_label.setText(f"Perf: {results.performance.total * 6e-05:6.3f} Msmpls/min")
        self.delay_label.setText(f"Channel Delay: {self.pybert.chnl_dly * 1e9:5.3f} ns")
        self.errors_label.setText(f"Bit Errors: {int(results.bit_errs)}")
        self.power_label.setText(f"Tx Power: {self.pybert.rel_power * 1e3:3.0f} mW")
        self.isi_label.setText(f"ISI: {self.pybert.dfe_jitter.isi * 1.0e12:6.1f} ps")
        self.dcd_label.setText(f"DCD: {self.pybert.dfe_jitter.dcd * 1.0e12:6.1f} ps")
        self.pj_label.setText(
            f"Pj: {self.pybert.dfe_jitter.pj * 1.0e12:6.1f} ({self.pybert.dfe_jitter.pjDD * 1.0e12:6.1f}) ps"
        )
        self.rj_label.setText(
            f"Rj: {self.pybert.dfe_jitter.rj * 1.0e12:6.1f} ({self.pybert.dfe_jitter.rjDD * 1.0e12:6.1f}) ps"
        )
