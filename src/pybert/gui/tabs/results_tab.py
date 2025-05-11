"""Results tab for PyBERT GUI.

This tab shows simulation results including DFE adaptation, output waveforms, eye diagrams and bathtub curves.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from pybert.gui.widgets.stats import StatisticsWidget
from pybert.utility.math import make_bathtub, safe_log10
from pybert.utility.sigproc import calc_eye

from scipy.interpolate import interp1d


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

        # --- Responses tabs ---
        self.impulse_tab = self._create_response_tab("Impulses")
        self.step_tab = self._create_response_tab("Steps")
        self.pulse_tab = self._create_response_tab("Pulses")
        self.freq_tab = self._create_response_tab("Frequency Response")

        tab_widget.addTab(self.impulse_tab, "Impulses")
        tab_widget.addTab(self.step_tab, "Steps")
        tab_widget.addTab(self.pulse_tab, "Pulses")
        tab_widget.addTab(self.freq_tab, "Frequency Response")

        # --- Original Results tabs ---
        self.dfe_tab = self._create_dfe_tab()
        self.outputs_tab = self._create_outputs_tab()
        self.eyes_tab = self._create_eyes_tab()
        self.bathtub_tab = self._create_bathtub_tab()

        tab_widget.addTab(self.dfe_tab, "DFE")
        tab_widget.addTab(self.outputs_tab, "Outputs")
        tab_widget.addTab(self.eyes_tab, "Eyes")
        tab_widget.addTab(self.bathtub_tab, "Bathtubs")

        # --- Jitter tabs ---
        self.jitter_dist_tab = self._create_jitter_dist_tab()
        self.jitter_spec_tab = self._create_jitter_spec_tab()

        tab_widget.addTab(self.jitter_dist_tab, "Jitter Dist.")
        tab_widget.addTab(self.jitter_spec_tab, "Jitter Spec.")

        # --- Performance tab ---
        self.statistics_tab = StatisticsWidget()
        tab_widget.addTab(self.statistics_tab, "Statistics")

    def connect_signals(self, pybert):
        """Connect signals to PyBERT instance."""
        pass


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

    # --- ResponsesTab logic ---
    def _create_response_tab(self, title):
        """Create a tab with a 2x2 grid of plots."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        plots = []
        for i in range(4):
            row = i // 2
            col = i % 2
            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.addLegend()
            plot.getAxis("left").setLabel("Amplitude")
            # Set appropriate x-axis label based on plot type
            if title == "Frequency Response":
                plot.getAxis("bottom").setLabel("Frequency", units="GHz")
            else:
                plot.getAxis("bottom").setLabel("Time", units="ns")
            plots.append(plot)
        for plot in plots[1:]:
            plot.setXLink(plots[0])
        # Store plots for update methods
        if title == "Impulses":
            self.impulse_plots = plots
        elif title == "Steps":
            self.step_plots = plots
        elif title == "Pulses":
            self.pulse_plots = plots
        elif title == "Frequency Response":
            self.freq_plots = plots
        return widget

    def update_impulse_plots(self, t_ns, chnl_h, tx_out_h, ctle_out_h, dfe_out_h):
        self.impulse_plots[0].plot(t_ns, chnl_h, pen="b", name="Channel", clear=True)
        self.impulse_plots[1].plot(t_ns, tx_out_h, pen="r", name="+ Tx", clear=True)
        self.impulse_plots[2].plot(t_ns, ctle_out_h, pen="r", name="+ CTLE", clear=True)
        self.impulse_plots[3].plot(t_ns, dfe_out_h, pen="r", name="+ DFE", clear=True)

    def update_step_plots(self, t_ns, chnl_s, tx_s, tx_out_s, ctle_s, ctle_out_s, dfe_s, dfe_out_s):
        self.step_plots[0].plot(t_ns, chnl_s, pen="b", name="Channel", clear=True)
        self.step_plots[1].plot(t_ns, tx_s, pen="b", name="Incremental", clear=True)
        self.step_plots[1].plot(t_ns, tx_out_s, pen="r", name="Cumulative")
        self.step_plots[2].plot(t_ns, ctle_s, pen="b", name="Incremental", clear=True)
        self.step_plots[2].plot(t_ns, ctle_out_s, pen="r", name="Cumulative")
        self.step_plots[3].plot(t_ns, dfe_s, pen="b", name="Incremental", clear=True)
        self.step_plots[3].plot(t_ns, dfe_out_s, pen="r", name="Cumulative")

    def update_pulse_plots(self, t_ns, chnl_p, tx_out_p, ctle_out_p, dfe_out_p):
        self.pulse_plots[0].plot(t_ns, chnl_p, pen="b", name="Channel", clear=True)
        self.pulse_plots[1].plot(t_ns, tx_out_p, pen="r", name="+ Tx", clear=True)
        self.pulse_plots[2].plot(t_ns, ctle_out_p, pen="r", name="+ CTLE", clear=True)
        self.pulse_plots[3].plot(t_ns, dfe_out_p, pen="r", name="+ DFE", clear=True)

    def update_freq_plots(
        self, f_GHz, chnl_H, chnl_H_raw, chnl_trimmed_H, tx_H, tx_out_H, ctle_H, ctle_out_H, dfe_H, dfe_out_H
    ):
        self.freq_plots[0].plot(f_GHz, chnl_H_raw, pen="k", name="Perfect Term.", clear=True)
        self.freq_plots[0].plot(f_GHz, chnl_H, pen="b", name="Actual Term.")
        self.freq_plots[0].plot(f_GHz, chnl_trimmed_H, pen="r", name="Trimmed")
        self.freq_plots[0].setLogMode(x=True, y=False)
        self.freq_plots[1].plot(f_GHz, tx_H, pen="b", name="Incremental", clear=True)
        self.freq_plots[1].plot(f_GHz, tx_out_H, pen="r", name="Cumulative")
        self.freq_plots[1].setLogMode(x=True, y=False)
        self.freq_plots[2].plot(f_GHz, ctle_H, pen="b", name="Incremental", clear=True)
        self.freq_plots[2].plot(f_GHz, ctle_out_H, pen="r", name="Cumulative")
        self.freq_plots[2].setLogMode(x=True, y=False)
        self.freq_plots[3].plot(f_GHz, dfe_H, pen="b", name="Incremental", clear=True)
        self.freq_plots[3].plot(f_GHz, dfe_out_H, pen="r", name="Cumulative")
        self.freq_plots[3].setLogMode(x=True, y=False)
        for plot in self.freq_plots:
            plot.getAxis("bottom").setLabel("Frequency", units="GHz")
            plot.getAxis("left").setLabel("Response", units="dB")

    # --- JitterTab logic ---
    def _create_jitter_dist_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.jitter_dist_plots = []
        for i in range(4):
            row = i // 2
            col = i % 2
            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("PDF")
            plot.getAxis("bottom").setLabel("Time", units="ps")
            plot.addLegend()
            total_curve = plot.plot(pen=pg.mkPen("b"), name="Total")
            di_curve = plot.plot(pen=pg.mkPen("r"), name="Data-Ind.")
            self.jitter_dist_plots.append((total_curve, di_curve))
        return widget

    def _create_jitter_spec_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.jitter_spec_plots = []
        for i in range(4):
            row = i // 2
            col = i % 2
            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("|FFT(TIE)|", units="dBui")
            plot.getAxis("bottom").setLabel("Frequency", units="MHz")
            plot.addLegend()
            total_curve = plot.plot(pen=pg.mkPen("b"), name="Total")
            di_curve = plot.plot(pen=pg.mkPen("r"), name="Data Independent")
            thresh_curve = plot.plot(pen=pg.mkPen("m"), name="Pj Threshold")
            self.jitter_spec_plots.append((total_curve, di_curve, thresh_curve))
            plot.setMouseEnabled(x=True, y=True)
        return widget

    def update_jitter_dist_plots(self, jitter_bins, jitter_data, jitter_ext_data):
        for (total_curve, di_curve), total, di in zip(self.jitter_dist_plots, jitter_data, jitter_ext_data):
            total_curve.setData(jitter_bins, total)
            di_curve.setData(jitter_bins, di)

    def update_jitter_spec_plots(self, f_MHz, jitter_spectrum, jitter_ind_spectrum, thresh):
        for (total_curve, di_curve, thresh_curve), total, di, th in zip(
            self.jitter_spec_plots, jitter_spectrum, jitter_ind_spectrum, thresh
        ):
            total_curve.setData(f_MHz, total)
            di_curve.setData(f_MHz, di)
            thresh_curve.setData(f_MHz, th)

# Plot updating

def update_results(self):
    """Updates all plot data used by GUI.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
    """

    # Copy globals into local namespace.
    ui = self.ui
    samps_per_ui = self.nspui
    eye_uis = self.eye_uis
    num_ui = self.nui
    clock_times = self.clock_times
    ui_ests = self.ui_ests
    f = self.f
    t = self.t
    t_ns = self.t_ns
    t_ns_chnl = self.t_ns_chnl
    t_irfft = self.t_irfft

    ignore_until = (num_ui - eye_uis) * ui
    ignore_samps = (num_ui - eye_uis) * samps_per_ui

    # Misc.
    f_GHz = f / 1.0e9
    len_f_GHz = len(f_GHz)
    len_t = len(t_ns)
    self.plotdata.set_data("f_GHz", f_GHz[1:])
    self.plotdata.set_data("t_ns_chnl", t_ns_chnl)
    self.plotdata.set_data("t_ns_irfft", t_irfft * 1e9)
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        t_ns_plot = np.linspace(0, t_ns[-1], 1000)
    else:
        t_ns_plot = t_ns
    self.plotdata.set_data("t_ns", t_ns_plot)

    # DFE.
    tap_weights = np.transpose(np.array(self.adaptation))
    if len(tap_weights):
        for k, tap_weight in enumerate(tap_weights):
            self.plotdata.set_data(f"tap{k + 1}_weights", tap_weight)
        self.plotdata.set_data("tap_weight_index", list(range(len(tap_weight))))
    else:
        for k in range(len(self.dfe_tap_tuners)):
            self.plotdata.set_data(f"tap{k + 1}_weights", zeros(10))
        self.plotdata.set_data("tap_weight_index", list(range(10)))

    (bin_counts, bin_edges) = np.histogram(ui_ests, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    clock_spec = np.fft.rfft(ui_ests)
    _f0 = 1 / (t[1] * len(t))
    spec_freqs = [_f0 * k for k in range(len(t) // 2 + 1)]
    self.plotdata.set_data("clk_per_hist_bins", bin_centers)
    self.plotdata.set_data("clk_per_hist_vals", bin_counts)
    self.plotdata.set_data("clk_spec", safe_log10(abs(clock_spec[1:]) / abs(clock_spec[1])))  # Omit the d.c. value and normalize to fundamental magnitude.
    self.plotdata.set_data("clk_freqs", np.array(spec_freqs[1:]) * ui)
    self.plotdata.set_data("dfe_out", self.dfe_out)
    self.plotdata.set_data("clocks", self.clocks)
    self.plotdata.set_data("lockeds", self.lockeds)
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        krnl = interp1d(t_ns, self.ui_ests)
        ui_ests_plot = krnl(t_ns_plot)
    else:
        ui_ests_plot = self.ui_ests
    self.plotdata.set_data("ui_ests", ui_ests_plot)

    # Impulse responses
    self.plotdata.set_data("chnl_h", self.chnl_h)
    self.plotdata.set_data("tx_h", self.tx_h)
    self.plotdata.set_data("tx_out_h", self.tx_out_h)
    self.plotdata.set_data("ctle_h", self.ctle_h)
    self.plotdata.set_data("ctle_out_h", self.ctle_out_h)
    self.plotdata.set_data("dfe_h", self.dfe_h)
    self.plotdata.set_data("dfe_out_h", self.dfe_out_h)

    # Step responses
    self.plotdata.set_data("chnl_s", self.chnl_s)
    self.plotdata.set_data("tx_s", self.tx_s)
    self.plotdata.set_data("tx_out_s", self.tx_out_s)
    self.plotdata.set_data("ctle_s", self.ctle_s)
    self.plotdata.set_data("ctle_out_s", self.ctle_out_s)
    self.plotdata.set_data("dfe_s", self.dfe_s)
    self.plotdata.set_data("dfe_out_s", self.dfe_out_s)

    # Pulse responses
    self.plotdata.set_data("chnl_p", self.chnl_p)
    self.plotdata.set_data("tx_out_p", self.tx_out_p)
    self.plotdata.set_data("ctle_out_p", self.ctle_out_p)
    self.plotdata.set_data("dfe_out_p", self.dfe_out_p)

    # Frequency responses
    self.plotdata.set_data("chnl_H_raw", 20.0 * safe_log10(abs(self.chnl_H_raw[1:len_f_GHz])))
    self.plotdata.set_data("chnl_H", 20.0 * safe_log10(abs(self.chnl_H[1:len_f_GHz])))
    self.plotdata.set_data("chnl_trimmed_H", 20.0 * safe_log10(abs(self.chnl_trimmed_H[1:len_f_GHz])))
    self.plotdata.set_data("tx_H", 20.0 * safe_log10(abs(self.tx_H[1:])))
    self.plotdata.set_data("tx_out_H", 20.0 * safe_log10(abs(self.tx_out_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_H", 20.0 * safe_log10(abs(self.ctle_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_out_H", 20.0 * safe_log10(abs(self.ctle_out_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_H", 20.0 * safe_log10(abs(self.dfe_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_out_H", 20.0 * safe_log10(abs(self.dfe_out_H[1:len_f_GHz])))

    # Outputs
    ideal_signal = self.ideal_signal[:len_t]
    chnl_out = self.chnl_out[:len_t]
    rx_in = self.rx_in[:len_t]
    ctle_out = self.ctle_out[:len_t]
    dfe_out = self.dfe_out[:len_t]
    lockeds = self.lockeds[:len_t]
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        krnl = interp1d(t_ns, ideal_signal)
        ideal_signal_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, chnl_out)
        chnl_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, rx_in)
        rx_in_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, ctle_out)
        ctle_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, dfe_out)
        dfe_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, lockeds)
        lockeds_plot = krnl(t_ns_plot)
    else:
        ideal_signal_plot = ideal_signal
        chnl_out_plot = chnl_out
        rx_in_plot = rx_in
        ctle_out_plot = ctle_out
        dfe_out_plot = dfe_out
        lockeds_plot = lockeds
    self.plotdata.set_data("ideal_signal", ideal_signal_plot)
    self.plotdata.set_data("chnl_out", chnl_out_plot)
    self.plotdata.set_data("rx_in", rx_in_plot)
    self.plotdata.set_data("ctle_out", ctle_out_plot)
    self.plotdata.set_data("dfe_out", dfe_out_plot)
    self.plotdata.set_data("dbg_out", lockeds_plot)

    # Jitter distributions
    jitter_chnl = self.jitter_chnl  # These are used again in bathtub curve generation, below.
    jitter_tx   = self.jitter_tx
    jitter_ctle = self.jitter_ctle
    jitter_dfe  = self.jitter_dfe
    jitter_bins = self.jitter_bins
    self.plotdata.set_data("jitter_bins", np.array(self.jitter_bins)  * 1e12)
    self.plotdata.set_data("jitter_chnl",     jitter_chnl          * 1e-12)  # PDF (/ps)
    self.plotdata.set_data("jitter_ext_chnl", self.jitter_ext_chnl * 1e-12)
    self.plotdata.set_data("jitter_tx",       jitter_tx            * 1e-12)
    self.plotdata.set_data("jitter_ext_tx",   self.jitter_ext_tx   * 1e-12)
    self.plotdata.set_data("jitter_ctle",     jitter_ctle          * 1e-12)
    self.plotdata.set_data("jitter_ext_ctle", self.jitter_ext_ctle * 1e-12)
    self.plotdata.set_data("jitter_dfe",      jitter_dfe           * 1e-12)
    self.plotdata.set_data("jitter_ext_dfe",  self.jitter_ext_dfe  * 1e-12)

    # Jitter spectrums
    log10_ui = safe_log10(ui)
    self.plotdata.set_data("f_MHz", self.f_MHz[1:])
    self.plotdata.set_data("f_MHz_dfe", self.f_MHz_dfe[1:])
    self.plotdata.set_data("jitter_spectrum_chnl", 10.0 * (safe_log10(self.jitter_spectrum_chnl[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_chnl", 10.0 * (safe_log10(self.jitter_ind_spectrum_chnl[1:]) - log10_ui))
    self.plotdata.set_data("thresh_chnl", 10.0 * (safe_log10(self.thresh_chnl[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_tx", 10.0 * (safe_log10(self.jitter_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_tx", 10.0 * (safe_log10(self.jitter_ind_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("thresh_tx", 10.0 * (safe_log10(self.thresh_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_ctle", 10.0 * (safe_log10(self.jitter_spectrum_ctle[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_ctle", 10.0 * (safe_log10(self.jitter_ind_spectrum_ctle[1:]) - log10_ui))
    self.plotdata.set_data("thresh_ctle", 10.0 * (safe_log10(self.thresh_ctle[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_dfe", 10.0 * (safe_log10(self.jitter_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_dfe", 10.0 * (safe_log10(self.jitter_ind_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("thresh_dfe", 10.0 * (safe_log10(self.thresh_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])

    # Bathtubs
    bathtub_chnl = make_bathtub(
        jitter_bins, jitter_chnl, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_chnl, mu_r=self.mu_pos_chnl, mu_l=self.mu_neg_chnl, extrap=True)
    bathtub_tx = make_bathtub(
        jitter_bins, jitter_tx,   min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_tx, mu_r=self.mu_pos_tx, mu_l=self.mu_neg_tx, extrap=True)
    bathtub_ctle = make_bathtub(
        jitter_bins, jitter_ctle, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_ctle, mu_r=self.mu_pos_ctle, mu_l=self.mu_neg_ctle, extrap=True)
    bathtub_dfe = make_bathtub(
        jitter_bins, jitter_dfe,  min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_dfe, mu_r=self.mu_pos_dfe, mu_l=self.mu_neg_dfe, extrap=True)
    self.plotdata.set_data("bathtub_chnl", safe_log10(bathtub_chnl))
    self.plotdata.set_data("bathtub_tx",   safe_log10(bathtub_tx))
    self.plotdata.set_data("bathtub_ctle", safe_log10(bathtub_ctle))
    self.plotdata.set_data("bathtub_dfe",  safe_log10(bathtub_dfe))

    # Eyes
    width = 2 * samps_per_ui
    xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)
    height = 1000
    tiny_noise = np.random.normal(scale=1e-3, size=len(chnl_out[ignore_samps:]))  # to make channel eye easier to view.
    chnl_out_noisy = self.chnl_out[ignore_samps:] + tiny_noise
    y_max = 1.1 * max(abs(np.array(chnl_out_noisy)))
    eye_chnl = calc_eye(ui, samps_per_ui, height, chnl_out_noisy, y_max)
    y_max = 1.1 * max(abs(np.array(self.rx_in[ignore_samps:])))
    eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(np.array(self.ctle_out[ignore_samps:])))
    eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(np.array(self.dfe_out[ignore_samps:])))
    i = 0
    len_clock_times = len(clock_times)
    while i < len_clock_times and clock_times[i] < ignore_until:
        i += 1
    if i >= len(clock_times):
        logger.error("ERROR: Insufficient coverage in 'clock_times' vector.")
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out[ignore_samps:], y_max)
    else:
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out[ignore_samps:], y_max, np.array(clock_times[i:]) - ignore_until)
    self.plotdata.set_data("eye_index", xs)
    self.plotdata.set_data("eye_chnl", eye_chnl)
    self.plotdata.set_data("eye_tx", eye_tx)
    self.plotdata.set_data("eye_ctle", eye_ctle)
    self.plotdata.set_data("eye_dfe", eye_dfe)


def update_eyes(self):
    """Update the heat plots representing the eye diagrams.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
    """

    ui = self.ui
    samps_per_ui = self.nspui

    width = 2 * samps_per_ui
    height = 100
    xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)

    y_max = 1.1 * max(abs(np.array(self.chnl_out)))
    ys = np.linspace(-y_max, y_max, height)
    self.plots_eye.components[0].components[0].index.set_data(xs, ys)
    self.plots_eye.components[0].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[0].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[0].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[0].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[0].invalidate_draw()

    y_max = 1.1 * max(abs(np.array(self.rx_in)))
    ys = np.linspace(-y_max, y_max, height)
    self.plots_eye.components[1].components[0].index.set_data(xs, ys)
    self.plots_eye.components[1].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[1].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[1].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[1].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[1].invalidate_draw()

    y_max = 1.1 * max(abs(np.array(self.dfe_out)))
    ys = np.linspace(-y_max, y_max, height)
    self.plots_eye.components[3].components[0].index.set_data(xs, ys)
    self.plots_eye.components[3].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[3].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[3].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[3].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[3].invalidate_draw()

    self.plots_eye.components[2].components[0].index.set_data(xs, ys)
    self.plots_eye.components[2].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[2].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[2].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[2].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[2].invalidate_draw()

    self.plots_eye.request_redraw()
