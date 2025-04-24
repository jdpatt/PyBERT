"""Responses tab for PyBERT GUI.

This tab shows various signal responses including impulse, step, pulse and frequency responses.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt

import pyqtgraph as pg


class ResponsesTab(QWidget):
    """Tab for displaying various signal responses."""

    def __init__(self, parent=None):
        """Initialize the responses tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for different response types
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Create tabs for each response type
        self.impulse_tab = self._create_response_tab("Impulses")
        self.step_tab = self._create_response_tab("Steps")
        self.pulse_tab = self._create_response_tab("Pulses")
        self.freq_tab = self._create_response_tab("Freq. Resp.")

        tab_widget.addTab(self.impulse_tab, "Impulses")
        tab_widget.addTab(self.step_tab, "Steps")
        tab_widget.addTab(self.pulse_tab, "Pulses")
        tab_widget.addTab(self.freq_tab, "Freq. Resp.")

    def _create_response_tab(self, title):
        """Create a tab with a 2x2 grid of plots.

        Args:
            title: Title for the plots

        Returns:
            QWidget: Widget containing the plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of plots
        plots = []
        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.addLegend()

            # Configure axes
            plot.getAxis("left").setLabel("Amplitude")
            plot.getAxis("bottom").setLabel("Time", units="ns")

            # Add to list for later access
            plots.append(plot)

        # Link x-axes for synchronized zooming
        for plot in plots[1:]:
            plot.setXLink(plots[0])

        # Store plots as attributes
        self.plots = plots

        return widget

    def update_impulse_plots(self, t_ns, chnl_h, tx_out_h, ctle_out_h, dfe_out_h):
        """Update impulse response plots.

        Args:
            t_ns: Time points
            chnl_h: Channel impulse response
            tx_out_h: Tx output impulse response
            ctle_out_h: CTLE output impulse response
            dfe_out_h: DFE output impulse response
        """
        self.plots[0].plot(t_ns, chnl_h, pen="b", name="Channel", clear=True)
        self.plots[1].plot(t_ns, tx_out_h, pen="r", name="+ Tx", clear=True)
        self.plots[2].plot(t_ns, ctle_out_h, pen="r", name="+ CTLE", clear=True)
        self.plots[3].plot(t_ns, dfe_out_h, pen="r", name="+ DFE", clear=True)

    def update_step_plots(self, t_ns, chnl_s, tx_s, tx_out_s, ctle_s, ctle_out_s, dfe_s, dfe_out_s):
        """Update step response plots.

        Args:
            t_ns: Time points
            chnl_s: Channel step response
            tx_s: Tx step response
            tx_out_s: Tx output step response
            ctle_s: CTLE step response
            ctle_out_s: CTLE output step response
            dfe_s: DFE step response
            dfe_out_s: DFE output step response
        """
        self.plots[0].plot(t_ns, chnl_s, pen="b", name="Channel", clear=True)

        self.plots[1].plot(t_ns, tx_s, pen="b", name="Incremental", clear=True)
        self.plots[1].plot(t_ns, tx_out_s, pen="r", name="Cumulative")

        self.plots[2].plot(t_ns, ctle_s, pen="b", name="Incremental", clear=True)
        self.plots[2].plot(t_ns, ctle_out_s, pen="r", name="Cumulative")

        self.plots[3].plot(t_ns, dfe_s, pen="b", name="Incremental", clear=True)
        self.plots[3].plot(t_ns, dfe_out_s, pen="r", name="Cumulative")

    def update_pulse_plots(self, t_ns, chnl_p, tx_out_p, ctle_out_p, dfe_out_p):
        """Update pulse response plots.

        Args:
            t_ns: Time points
            chnl_p: Channel pulse response
            tx_out_p: Tx output pulse response
            ctle_out_p: CTLE output pulse response
            dfe_out_p: DFE output pulse response
        """
        self.plots[0].plot(t_ns, chnl_p, pen="b", name="Channel", clear=True)
        self.plots[1].plot(t_ns, tx_out_p, pen="r", name="+ Tx", clear=True)
        self.plots[2].plot(t_ns, ctle_out_p, pen="r", name="+ CTLE", clear=True)
        self.plots[3].plot(t_ns, dfe_out_p, pen="r", name="+ DFE", clear=True)

    def update_freq_plots(
        self, f_GHz, chnl_H, chnl_H_raw, chnl_trimmed_H, tx_H, tx_out_H, ctle_H, ctle_out_H, dfe_H, dfe_out_H
    ):
        """Update frequency response plots.

        Args:
            f_GHz: Frequency points
            chnl_H: Channel frequency response
            chnl_H_raw: Raw channel frequency response
            chnl_trimmed_H: Trimmed channel frequency response
            tx_H: Tx frequency response
            tx_out_H: Tx output frequency response
            ctle_H: CTLE frequency response
            ctle_out_H: CTLE output frequency response
            dfe_H: DFE frequency response
            dfe_out_H: DFE output frequency response
        """
        # Channel plot
        self.plots[0].plot(f_GHz, chnl_H_raw, pen="k", name="Perfect Term.", clear=True)
        self.plots[0].plot(f_GHz, chnl_H, pen="b", name="Actual Term.")
        self.plots[0].plot(f_GHz, chnl_trimmed_H, pen="r", name="Trimmed")
        self.plots[0].setLogMode(x=True, y=False)

        # Tx plot
        self.plots[1].plot(f_GHz, tx_H, pen="b", name="Incremental", clear=True)
        self.plots[1].plot(f_GHz, tx_out_H, pen="r", name="Cumulative")
        self.plots[1].setLogMode(x=True, y=False)

        # CTLE plot
        self.plots[2].plot(f_GHz, ctle_H, pen="b", name="Incremental", clear=True)
        self.plots[2].plot(f_GHz, ctle_out_H, pen="r", name="Cumulative")
        self.plots[2].setLogMode(x=True, y=False)

        # DFE plot
        self.plots[3].plot(f_GHz, dfe_H, pen="b", name="Incremental", clear=True)
        self.plots[3].plot(f_GHz, dfe_out_H, pen="r", name="Cumulative")
        self.plots[3].setLogMode(x=True, y=False)

        # Set frequency axis labels
        for plot in self.plots:
            plot.getAxis("bottom").setLabel("Frequency", units="GHz")
            plot.getAxis("left").setLabel("Response", units="dB")
