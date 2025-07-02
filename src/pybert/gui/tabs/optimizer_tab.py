"""Optimization tab for PyBERT GUI.

This tab contains controls for configuring and tuning transmitter and
receiver equalization.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from pybert.gui.widgets import (
    RxOptimizationCTLEWidget,
    RxOptimizationDFEWidget,
    TxOptimizationWidget,
)
from pybert.pybert import PyBERT


class OptimizerTab(QWidget):
    """Tab for configuring and tuning equalization."""

    def __init__(self, pybert: PyBERT, parent=None):
        """Initialize the equalization tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create horizontal splitter for controls and plot
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Left side - Controls
        controls = QWidget(self)
        controls_layout = QHBoxLayout()
        controls.setLayout(controls_layout)

        # Add Tx and Rx equalization widgets
        self.tx_optimization = TxOptimizationWidget(pybert=self.pybert, parent=self)
        self.rx_ctle = RxOptimizationCTLEWidget(pybert=self.pybert, parent=self)
        self.rx_dfe = RxOptimizationDFEWidget(pybert=self.pybert, parent=self)

        controls_layout.addWidget(self.tx_optimization, stretch=1)
        controls_layout.addWidget(self.rx_ctle, stretch=1)
        controls_layout.addWidget(self.rx_dfe, stretch=1)

        splitter.addWidget(controls)

        # Right side - Plot
        plot_widget = pg.PlotWidget(
            parent=self,
            title="Channel + Tx Preemphasis + CTLE (+ AMI DFE) + Ideal DFE",
        )
        plot_widget.showGrid(x=False, y=False)
        plot_widget.setLabel("left", "Pulse Response", units="V")
        plot_widget.setLabel("bottom", "Time", units="ns")

        # Add legend
        plot_widget.addLegend()

        # Create plot curves
        self.clocks_curve = plot_widget.plot(name="Clocks", pen=pg.mkPen("b", width=2))
        self.eq_curve = plot_widget.plot(name="Equalized Pulse Response", pen=pg.mkPen("r", width=2))
        self.pulse_curve = plot_widget.plot(name="Channel Pulse Response", pen=pg.mkPen("b", width=2))
        self.cursor_curve = plot_widget.plot(
            name="Main Cursor", pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine)
        )

        splitter.addWidget(plot_widget)

        # Set initial splitter sizes (40% controls, 60% plot)
        splitter.setSizes([400, 600])

        self.connect_signals()

    def update_plot(self, result: dict):
        """Update the plot with new data.

        Args:
            result: Result from optimization
        """
        self.clocks_curve.setData(result.get("t_ns_opt"), result.get("clocks_tune"))
        self.eq_curve.setData(result.get("t_ns_opt"), result.get("ctle_out_h_tune"))
        self.cursor_curve.setData(result.get("curs_ix"), result.get("curs_amp"))
        self.pulse_curve.setData(result.get("t_ns_opt"), result.get("p_chnl"))

    def update_channel_response(self, result: dict):
        """Update the channel response plot."""
        self.pulse_curve.setData(result.get("t_ns_opt"), result.get("p_chnl"))

    def connect_signals(self):
        """Connect signals to PyBERT instance."""
        self.tx_optimization.connect_signals(self.pybert)
        self.rx_ctle.connect_signals(self.pybert)
        self.rx_dfe.connect_signals(self.pybert)

    def update_results(self, opt_result: dict):
        """Update the tap values and CTLE boost with the new optimization result."""
        self.tx_optimization.set_tap_values(opt_result["tx_weights"])
        self.rx_ctle.set_ctle_boost(opt_result["rx_peaking"])
        self.rx_dfe.set_tap_values(opt_result["dfe_weights"])

    def clear_results(self):
        """Clear the results."""
        self.clocks_curve.clear()
        self.eq_curve.clear()
        self.pulse_curve.clear()
        self.cursor_curve.clear()
