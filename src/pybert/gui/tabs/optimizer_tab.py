"""Optimization tab for PyBERT GUI.

This tab contains controls for configuring and tuning transmitter and receiver equalization.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from pybert.gui.widgets import (
    RxOptimizationCTLEWidget,
    RxOptimizationDFEWidget,
    TxOptimizationWidget,
)


class OptimizerTab(QWidget):
    """Tab for configuring and tuning equalization."""

    def __init__(self, parent=None):
        """Initialize the equalization tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create horizontal splitter for controls and plot
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Left side - Controls
        controls = QWidget()
        controls_layout = QHBoxLayout()
        controls.setLayout(controls_layout)

        # Add Tx and Rx equalization widgets
        self.tx_optimization = TxOptimizationWidget()
        self.rx_ctle = RxOptimizationCTLEWidget()
        self.rx_dfe = RxOptimizationDFEWidget()

        controls_layout.addWidget(self.tx_optimization, stretch=1)
        controls_layout.addWidget(self.rx_ctle, stretch=1)
        controls_layout.addWidget(self.rx_dfe, stretch=1)

        splitter.addWidget(controls)

        # Right side - Plot
        plot_widget = pg.PlotWidget(
            title="Equalization Results",
        )
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel("left", "Amplitude")
        plot_widget.setLabel("bottom", "Time", units="s")

        # Add legend
        plot_widget.addLegend()

        # Create plot curves
        self.pulse_curve = plot_widget.plot(name="Pulse Response", pen=pg.mkPen("b", width=2))
        self.eq_curve = plot_widget.plot(name="Equalized Response", pen=pg.mkPen("r", width=2))
        self.target_curve = plot_widget.plot(name="Target Response", pen=pg.mkPen("g", width=2, style=Qt.DashLine))

        splitter.addWidget(plot_widget)

        # Set initial splitter sizes (40% controls, 60% plot)
        splitter.setSizes([400, 600])

    def update_plot(self, time_points, pulse_data, eq_data, target_data=None):
        """Update the plot with new data.

        Args:
            time_points: Array of time points
            pulse_data: Array of pulse response data
            eq_data: Array of equalized response data
            target_data: Optional array of target response data
        """
        self.pulse_curve.setData(time_points, pulse_data)
        self.eq_curve.setData(time_points, eq_data)

        if target_data is not None:
            self.target_curve.setData(time_points, target_data)
            self.target_curve.show()
        else:
            self.target_curve.hide()
