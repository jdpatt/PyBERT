"""Results tab for PyBERT GUI.

This tab shows simulation results including DFE adaptation, output waveforms, eye diagrams and bathtub curves.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt

import numpy as np
import pyqtgraph as pg


class ResultsTab(QWidget):
    """Tab for displaying simulation results."""

    def __init__(self, parent=None):
        """Initialize the results tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for different result types
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Create tabs for each result type
        self.dfe_tab = self._create_dfe_tab()
        self.outputs_tab = self._create_outputs_tab()
        self.eyes_tab = self._create_eyes_tab()
        self.bathtub_tab = self._create_bathtub_tab()

        tab_widget.addTab(self.dfe_tab, "DFE")
        tab_widget.addTab(self.outputs_tab, "Outputs")
        tab_widget.addTab(self.eyes_tab, "Eyes")
        tab_widget.addTab(self.bathtub_tab, "Bathtubs")

    def _create_dfe_tab(self):
        """Create the DFE adaptation tab.

        Returns:
            QWidget: Widget containing the DFE plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # CDR adaptation plot
        cdr_plot = plot_grid.addPlot(row=0, col=0)
        cdr_plot.showGrid(x=True, y=True)
        cdr_plot.setTitle("CDR Adaptation")
        cdr_plot.getAxis("left").setLabel("UI", units="ps")
        cdr_plot.getAxis("bottom").setLabel("Time", units="ns")
        self.cdr_curve = cdr_plot.plot(pen="b")

        # DFE adaptation plot
        dfe_plot = plot_grid.addPlot(row=0, col=1)
        dfe_plot.showGrid(x=True, y=True)
        dfe_plot.setTitle("DFE Adaptation")
        dfe_plot.getAxis("left").setLabel("Tap Weight")
        dfe_plot.getAxis("bottom").setLabel("Sample Number")
        dfe_plot.addLegend()

        # Create curves for DFE taps
        colors = ["m", "r", "orange", "y", "g", "c", "b", "purple", "brown", "k"]
        styles = [Qt.SolidLine, Qt.DashLine]
        self.dfe_curves = []
        for i in range(10):  # Support up to 10 taps
            pen = pg.mkPen(color=colors[i % len(colors)], style=styles[i // len(colors)])
            curve = dfe_plot.plot(pen=pen, name=f"Tap {i+1}")
            self.dfe_curves.append(curve)

        # Clock period histogram
        hist_plot = plot_grid.addPlot(row=1, col=0)
        hist_plot.showGrid(x=True, y=True)
        hist_plot.setTitle("CDR Clock Period Histogram")
        hist_plot.getAxis("left").setLabel("Bin Count")
        hist_plot.getAxis("bottom").setLabel("Clock Period", units="ps")
        self.hist_curve = hist_plot.plot(pen="b")

        # Clock period spectrum
        spec_plot = plot_grid.addPlot(row=1, col=1)
        spec_plot.showGrid(x=True, y=True)
        spec_plot.setTitle("CDR Clock Period Spectrum")
        spec_plot.getAxis("left").setLabel("|H(f)|", units="dB norm.")
        spec_plot.getAxis("bottom").setLabel("Frequency", units="symbol rate")
        self.spec_curve = spec_plot.plot(pen="b")

        return widget

    def _create_outputs_tab(self):
        """Create the outputs tab.

        Returns:
            QWidget: Widget containing the output plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of plots
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.output_plots = []

        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("Output", units="V")
            plot.getAxis("bottom").setLabel("Time", units="ns")

            # Add curves
            if i == 0:  # Channel output includes ideal signal
                plot.plot(pen=pg.mkPen("lightgray"), name="Ideal")
            plot.plot(pen=pg.mkPen("b"), name="Output")

            # Link x-axes for synchronized zooming
            if i > 0:
                plot.setXLink(self.output_plots[0])

            self.output_plots.append(plot)

        return widget

    def _create_eyes_tab(self):
        """Create the eye diagrams tab.

        Returns:
            QWidget: Widget containing the eye diagrams
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of eye diagrams
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.eye_plots = []

        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("Signal Level", units="V")
            plot.getAxis("bottom").setLabel("Time", units="ps")

            # Create image item for eye diagram
            img = pg.ImageItem()
            plot.addItem(img)

            # Link x-axes for synchronized zooming
            if i > 0:
                plot.setXLink(plot_grid.getItem(0, 0))

            self.eye_plots.append(img)

        return widget

    def _create_bathtub_tab(self):
        """Create the bathtub curves tab.

        Returns:
            QWidget: Widget containing the bathtub plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of bathtub plots
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.bathtub_plots = []

        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("Log10(P(Transition occurs inside))")
            plot.getAxis("bottom").setLabel("Time", units="ps")

            # Set y-axis range
            plot.setYRange(-12, 0)
            plot.getAxis("left").setTickSpacing(3, 1)

            # Add curve
            curve = plot.plot(pen="b")
            self.bathtub_plots.append(curve)

            # Link x-axes for synchronized zooming
            if i > 0:
                plot.setXLink(plot_grid.getItem(0, 0))

        return widget

    def update_dfe_plots(self, t_ns, ui_ests, tap_weights, clk_per_hist_bins, clk_per_hist_vals, clk_freqs, clk_spec):
        """Update DFE adaptation plots.

        Args:
            t_ns: Time points
            ui_ests: UI estimates for CDR
            tap_weights: List of tap weight arrays
            clk_per_hist_bins: Clock period histogram bins
            clk_per_hist_vals: Clock period histogram values
            clk_freqs: Clock frequency points
            clk_spec: Clock spectrum values
        """
        self.cdr_curve.setData(t_ns, ui_ests)

        for i, weights in enumerate(tap_weights):
            if i < len(self.dfe_curves):
                self.dfe_curves[i].setData(np.arange(len(weights)), weights)

        self.hist_curve.setData(clk_per_hist_bins, clk_per_hist_vals)
        self.spec_curve.setData(clk_freqs, clk_spec)

    def update_output_plots(self, t_ns, ideal_signal, chnl_out, rx_in, ctle_out, dfe_out):
        """Update output waveform plots.

        Args:
            t_ns: Time points
            ideal_signal: Ideal input signal
            chnl_out: Channel output
            rx_in: Rx input (after Tx)
            ctle_out: CTLE output
            dfe_out: DFE output
        """
        signals = [(ideal_signal, chnl_out), (None, rx_in), (None, ctle_out), (None, dfe_out)]

        for plot, (ideal, signal) in zip(self.output_plots, signals):
            plot.clear()
            if ideal is not None:
                plot.plot(t_ns, ideal, pen=pg.mkPen("lightgray"), name="Ideal")
            plot.plot(t_ns, signal, pen=pg.mkPen("b"), name="Output")

    def update_eye_plots(self, eye_data, ui_ps, v_range):
        """Update eye diagram plots.

        Args:
            eye_data: List of 2D arrays containing eye diagram data
            ui_ps: UI period in ps
            v_range: Voltage range tuple (min, max)
        """
        for img, data in zip(self.eye_plots, eye_data):
            img.setImage(data)
            img.setRect(pg.QtCore.QRectF(0, v_range[0], ui_ps, v_range[1] - v_range[0]))

    def update_bathtub_plots(self, jitter_bins, bathtub_data):
        """Update bathtub curve plots.

        Args:
            jitter_bins: Time points for bathtub curves
            bathtub_data: List of bathtub curve data arrays
        """
        for curve, data in zip(self.bathtub_plots, bathtub_data):
            curve.setData(jitter_bins, data)
