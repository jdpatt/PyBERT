"""Optimizer tab for PyBERT GUI.

This tab provides controls for optimizing equalization settings and visualizing the results.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

import pyqtgraph as pg


class OptimizerTab(QWidget):
    """Tab for optimizing equalization settings."""

    def __init__(self, parent=None):
        """Initialize the optimizer tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Add zoom instructions
        zoom_label = QLabel(
            "To zoom: Click in the plot, hit `z` (Cursor will change to crosshair), "
            "and click/drag to select region of interest. Hit <ESC> to exit zoom."
        )
        layout.addWidget(zoom_label)

        # Create plot widget
        plot_widget = pg.PlotWidget(title="Channel + Tx Preemphasis + CTLE (+ AMI DFE) + Ideal DFE")
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel("left", "Pulse Response (V)")
        plot_widget.setLabel("bottom", "Time (ns)")

        # Add legend
        plot_widget.addLegend()

        # Create plot curves
        self.clocks_curve = plot_widget.plot(name="Clocks", pen=pg.mkPen("gray", width=2))
        self.eq_curve = plot_widget.plot(name="Equalized Pulse Response", pen=pg.mkPen("blue", width=2))
        self.channel_curve = plot_widget.plot(name="Channel Pulse Response", pen=pg.mkPen("magenta", width=2))
        self.cursor_curve = plot_widget.plot(name="Main Cursor", pen=pg.mkPen("red", width=2))

        layout.addWidget(plot_widget)

    def update_plot(self, t_ns_opt, clocks_tune, ctle_out_h_tune, p_chnl, curs_ix, curs_amp):
        """Update the plot with new data.

        Args:
            t_ns_opt: Time points in ns
            clocks_tune: Clock signal data
            ctle_out_h_tune: Equalized pulse response data
            p_chnl: Channel pulse response data
            curs_ix: Cursor x positions
            curs_amp: Cursor y positions
        """
        self.clocks_curve.setData(t_ns_opt, clocks_tune)
        self.eq_curve.setData(t_ns_opt, ctle_out_h_tune)
        self.channel_curve.setData(t_ns_opt, p_chnl)
        self.cursor_curve.setData(curs_ix, curs_amp)
