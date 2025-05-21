"""Results tab for PyBERT GUI.

This tab shows simulation results including DFE adaptation, output
waveforms, eye diagrams and bathtub curves.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget
from scipy.interpolate import interp1d

from pybert.gui.widgets.jitter_info import JitterInfoTable
from pybert.models.bert import MIN_BATHTUB_VAL
from pybert.pybert import PyBERT
from pybert.utility.math import make_bathtub, safe_log10
from pybert.utility.sigproc import calc_eye

pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')
# TODO: Either limit auto sizing or limit the range of the plot(s)
class ResultsTab(QWidget):
    """Tab for displaying simulation results."""

    def __init__(self, pybert: PyBERT | None = None, parent=None):
        """Initialize the results tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.pybert = pybert

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for different result types
        tab_widget = QTabWidget(self)
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
        self.jitter_info = JitterInfoTable(self.pybert, parent=self)
        tab_widget.addTab(self.jitter_info, "Jitter Info")

    def connect_signals(self, pybert):
        """Connect signals to PyBERT instance."""
        pybert.sim_complete.connect(self.update_results)


    def _create_dfe_tab(self):
        """Create the DFE adaptation tab.

        Returns:
            QWidget: Widget containing the DFE plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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

        try:
            num_taps = len(self.pybert.tx_taps)
        except:
            num_taps = 10

        for i in range(num_taps):  # Support up to 10 taps
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
        spec_plot.setXRange(0, 0.5, padding=0)
        spec_plot.setYRange(-10, 2, padding=0)
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
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
            plot.setMouseEnabled(False, False)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("Signal Level", units="V")
            plot.getAxis("bottom").setLabel("Time", units="ps")

            # Create image item for eye diagram
            img = pg.ImageItem()
            plot.addItem(img)
            self.eye_plots.append(img)

        return widget

    def _create_bathtub_tab(self):
        """Create the bathtub curves tab.

        Returns:
            QWidget: Widget containing the bathtub plots
        """
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
            plot.setMouseEnabled(False, False)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("Log10(P(Transition occurs inside))")
            plot.getAxis("bottom").setLabel("Time", units="ps")

            # Set y-axis range
            plot.setYRange(-12, 0, padding=0)
            plot.getAxis("left").setTickSpacing(3, 1)

            # Add curve
            curve = plot.plot(pen="b")
            self.bathtub_plots.append(curve)


        return widget

    def _create_response_tab(self, title):
        """Create a tab with a 2x2 grid of plots."""
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
                plot.setMouseEnabled(False, False)  # Disable zooming for freq response
                plot.setYRange(-40, 10, padding=0)  # Set min/max y values for freq response
            else:
                plot.getAxis("bottom").setLabel("Time", units="ns")
            plots.append(plot)

        # Only link x-axes for non-frequency response plots
        if title != "Frequency Response":
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

    def _create_jitter_dist_tab(self):
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
            plot.setMouseEnabled(False, False)  # Disable zooming for jitter dist plots
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("PDF")
            plot.getAxis("bottom").setLabel("Time", units="ps")
            plot.addLegend()
            total_curve = plot.plot(pen=pg.mkPen("b"), name="Total")
            di_curve = plot.plot(pen=pg.mkPen("r"), name="Data-Ind.")
            self.jitter_dist_plots.append((total_curve, di_curve))
        return widget

    def _create_jitter_spec_tab(self):
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        plot_grid = pg.GraphicsLayoutWidget(parent=self)
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
        colormap = get_custom_colormap()
        lut = colormap.getLookupTable(0.0, 1.0, 256)
        for img, data in zip(self.eye_plots, eye_data):
            img.setImage(np.rot90(data), lut=lut)
            img.setRect(pg.QtCore.QRectF(0, v_range[0], ui_ps, v_range[1] - v_range[0]))

    def update_bathtub_plots(self, jitter_bins, bathtub_data):
        """Update bathtub curve plots.

        Args:
            jitter_bins: Time points for bathtub curves
            bathtub_data: List of bathtub curve data arrays
        """
        for curve, data in zip(self.bathtub_plots, bathtub_data):
            curve.setData(jitter_bins, data)

    def update_impulse_plots(self, t_ns, chnl_h, tx_out_h, ctle_out_h, dfe_out_h):
        self.impulse_plots[0].plot(t_ns, chnl_h, pen="b", name="Channel", clear=True)
        self.impulse_plots[1].plot(t_ns, tx_out_h, pen="r", name="+ Tx", clear=True)
        self.impulse_plots[2].plot(t_ns, ctle_out_h, pen="r", name="+ CTLE", clear=True)
        self.impulse_plots[3].plot(t_ns, dfe_out_h, pen="r", name="+ DFE", clear=True)

    def update_step_plots(self, t_ns, chnl_s, tx_s, tx_out_s, ctle_s, ctle_out_s, dfe_s, dfe_out_s):
        self.step_plots[0].plot(t_ns, chnl_s, pen="b", name="Channel", clear=True)
        self.step_plots[1].plot(t_ns, tx_s, pen="b", name="Incremental", clear=True)
        self.step_plots[1].plot(t_ns, tx_out_s, pen="r", name="Cumulative")
        self.step_plots[2].plot(t_ns, ctle_s[:len(t_ns)], pen="b", name="Incremental", clear=True)
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
            plot.setYRange(-40, 10, padding=0)
            plot.getAxis("bottom").setLabel("Frequency", units="GHz")
            plot.getAxis("left").setLabel("Response", units="dB")

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

    def update_results(self, results, perf):
        """Update all plots using the current PyBERT simulation results."""
        pb = self.pybert
        if pb is None:
            return

        self.jitter_info.update_rejection()

        t_ns = pb.t_ns
        t_ns_chnl = pb.t_ns_chnl

        # --- DFE plots ---
        ui_ests = pb.ui_ests
        try:
            tap_weights = np.transpose(np.array(pb.adaptation))
        except Exception:
            tap_weights = []
        (bin_counts, bin_edges) = np.histogram(ui_ests, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        clock_spec = np.fft.rfft(ui_ests)
        t = pb.t
        ui = pb.ui
        _f0 = 1 / (t[1] * len(t)) if len(t) > 1 else 0
        spec_freqs = np.array([_f0 * k for k in range(len(t) // 2 + 1)])
        clk_per_hist_bins = bin_centers
        clk_per_hist_vals = bin_counts
        clk_spec = safe_log10(np.abs(clock_spec[1:]) / np.abs(clock_spec[1])) if len(clock_spec) > 1 else np.zeros(1)
        clk_freqs = spec_freqs[1:] * ui if len(spec_freqs) > 1 else np.zeros(1)
        self.update_dfe_plots(
            t_ns,
            ui_ests,
            tap_weights,
            clk_per_hist_bins,
            clk_per_hist_vals,
            clk_freqs,
            clk_spec,
        )

        # --- Output plots ---
        len_t = len(t_ns)
        ideal_signal = pb.ideal_signal[:len_t]
        chnl_out = pb.chnl_out[:len_t]
        rx_in = pb.rx_in[:len_t]
        ctle_out = pb.ctle_out[:len_t]
        dfe_out = pb.dfe_out[:len_t]
        self.update_output_plots(
            t_ns,
            ideal_signal,
            chnl_out,
            rx_in,
            ctle_out,
            dfe_out,
        )

        # --- Eye plots ---
        samps_per_ui = pb.nspui
        eye_uis = pb.eye_uis
        num_ui = pb.nui
        clock_times = pb.clock_times
        ignore_until = (num_ui - eye_uis) * pb.ui
        ignore_samps = (num_ui - eye_uis) * samps_per_ui
        width = 2 * samps_per_ui
        height = 1000
        tiny_noise = np.random.normal(scale=1e-3, size=len(chnl_out[ignore_samps:]))
        chnl_out_noisy = pb.chnl_out[ignore_samps:] + tiny_noise
        y_max = 1.1 * max(abs(np.array(chnl_out_noisy)))
        eye_chnl = calc_eye(pb.ui, samps_per_ui, height, chnl_out_noisy, y_max)
        y_max = 1.1 * max(abs(np.array(pb.rx_in[ignore_samps:])))
        eye_tx = calc_eye(pb.ui, samps_per_ui, height, pb.rx_in[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(np.array(pb.ctle_out[ignore_samps:])))
        eye_ctle = calc_eye(pb.ui, samps_per_ui, height, pb.ctle_out[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(np.array(pb.dfe_out[ignore_samps:])))
        i = 0
        len_clock_times = len(clock_times)
        while i < len_clock_times and clock_times[i] < ignore_until:
            i += 1
        if i >= len(clock_times):
                # logger.error("ERROR: Insufficient coverage in 'clock_times' vector.")
                eye_dfe = calc_eye(pb.ui, samps_per_ui, height, pb.dfe_out[ignore_samps:], y_max)
        else:
                eye_dfe = calc_eye(pb.ui, samps_per_ui, height, pb.dfe_out[ignore_samps:], y_max, np.array(clock_times[i:]) - ignore_until)
        eye_data = [eye_chnl, eye_tx, eye_ctle, eye_dfe]
        ui_ps = pb.ui * 1e12
        v_range = (
            min(np.min(eye_chnl), np.min(eye_tx), np.min(eye_ctle), np.min(eye_dfe)),
            max(np.max(eye_chnl), np.max(eye_tx), np.max(eye_ctle), np.max(eye_dfe)),
        )
        self.update_eye_plots(eye_data, ui_ps, v_range)

        # --- Bathtub plots ---
        jitter_bins = pb.jitter_bins
        bathtub_chnl = make_bathtub(
            jitter_bins, pb.jitter_chnl, min_val=0.1 * MIN_BATHTUB_VAL,
            rj=pb.rjDD_chnl, mu_r=pb.mu_pos_chnl, mu_l=pb.mu_neg_chnl, extrap=True)
        bathtub_tx = make_bathtub(
            jitter_bins, pb.jitter_tx, min_val=0.1 * MIN_BATHTUB_VAL,
            rj=pb.rjDD_tx, mu_r=pb.mu_pos_tx, mu_l=pb.mu_neg_tx, extrap=True)
        bathtub_ctle = make_bathtub(
            jitter_bins, pb.jitter_ctle, min_val=0.1 * MIN_BATHTUB_VAL,
            rj=pb.rjDD_ctle, mu_r=pb.mu_pos_ctle, mu_l=pb.mu_neg_ctle, extrap=True)
        bathtub_dfe = make_bathtub(
            jitter_bins, pb.jitter_dfe, min_val=0.1 * MIN_BATHTUB_VAL,
            rj=pb.rjDD_dfe, mu_r=pb.mu_pos_dfe, mu_l=pb.mu_neg_dfe, extrap=True)
        bathtub_data = [safe_log10(bathtub_chnl), safe_log10(bathtub_tx), safe_log10(bathtub_ctle), safe_log10(bathtub_dfe)]
        self.update_bathtub_plots(np.array(jitter_bins) * 1e12, bathtub_data)

        # --- Impulse plots ---
        self.update_impulse_plots(t_ns_chnl, pb.chnl_h, pb.tx_out_h, pb.ctle_out_h, pb.dfe_out_h)

        # --- Step plots ---
        self.update_step_plots(t_ns_chnl, pb.chnl_s, pb.tx_s, pb.tx_out_s, pb.ctle_s, pb.ctle_out_s, pb.dfe_s, pb.dfe_out_s)

        # --- Pulse plots ---
        self.update_pulse_plots(t_ns_chnl, pb.chnl_p, pb.tx_out_p, pb.ctle_out_p, pb.dfe_out_p)

        # --- Frequency plots ---
        f_GHz = pb.f / 1.0e9
        len_f_GHz = len(f_GHz)
        self.update_freq_plots(
            f_GHz[1:],
            20.0 * safe_log10(np.abs(pb.chnl_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.chnl_H_raw[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.chnl_trimmed_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.tx_H[1:])),
            20.0 * safe_log10(np.abs(pb.tx_out_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.ctle_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.ctle_out_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.dfe_H[1:len_f_GHz])),
            20.0 * safe_log10(np.abs(pb.dfe_out_H[1:len_f_GHz])),
        )

        # --- Jitter distribution plots ---
        jitter_data = [pb.jitter_chnl * 1e-12, pb.jitter_tx * 1e-12, pb.jitter_ctle * 1e-12, pb.jitter_dfe * 1e-12]
        jitter_ext_data = [pb.jitter_ext_chnl * 1e-12, pb.jitter_ext_tx * 1e-12, pb.jitter_ext_ctle * 1e-12, pb.jitter_ext_dfe * 1e-12]
        self.update_jitter_dist_plots(jitter_bins, jitter_data, jitter_ext_data)

        # --- Jitter spectrum plots ---
        log10_ui = safe_log10(pb.ui)
        f_MHz = pb.f_MHz[1:]
        self.update_jitter_spec_plots(
            f_MHz,
            [10.0 * (safe_log10(pb.jitter_spectrum_chnl[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_spectrum_tx[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_spectrum_ctle[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_spectrum_dfe[1:]) - log10_ui)],
            [10.0 * (safe_log10(pb.jitter_ind_spectrum_chnl[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_ind_spectrum_tx[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_ind_spectrum_ctle[1:]) - log10_ui),
             10.0 * (safe_log10(pb.jitter_ind_spectrum_dfe[1:]) - log10_ui)],
            [10.0 * (safe_log10(pb.thresh_chnl[1:]) - log10_ui),
             10.0 * (safe_log10(pb.thresh_tx[1:]) - log10_ui),
             10.0 * (safe_log10(pb.thresh_ctle[1:]) - log10_ui),
             10.0 * (safe_log10(pb.thresh_dfe[1:]) - log10_ui)],
        )

    def clear_waveforms(self):
        """Clear all waveform plots."""
        pass # TODO: Implement this


def get_custom_colormap():
    seg_map = {
        "red": [
            (0.00, 0.00, 0.00),  # black
            (0.00001, 0.00, 0.00),  # blue
            (0.15, 0.00, 0.00),  # cyan
            (0.30, 0.00, 0.00),  # green
            (0.45, 1.00, 1.00),  # yellow
            (0.60, 1.00, 1.00),  # orange
            (0.75, 1.00, 1.00),  # red
            (0.90, 1.00, 1.00),  # pink
            (1.00, 1.00, 1.00),  # white
        ],
        "green": [
            (0.00, 0.00, 0.00),  # black
            (0.00001, 0.00, 0.00),  # blue
            (0.15, 0.50, 0.50),  # cyan
            (0.30, 0.50, 0.50),  # green
            (0.45, 1.00, 1.00),  # yellow
            (0.60, 0.50, 0.50),  # orange
            (0.75, 0.00, 0.00),  # red
            (0.90, 0.50, 0.50),  # pink
            (1.00, 1.00, 1.00),  # white
        ],
        "blue": [
            (0.00, 0.00, 0.00),  # black
            (1e-18, 0.50, 0.50),  # blue
            (0.15, 0.50, 0.50),  # cyan
            (0.30, 0.00, 0.00),  # green
            (0.45, 0.00, 0.00),  # yellow
            (0.60, 0.00, 0.00),  # orange
            (0.75, 0.00, 0.00),  # red
            (0.90, 0.50, 0.50),  # pink
            (1.00, 1.00, 1.00),  # white
        ]
    }
    positions = [x[0] for x in seg_map["red"]]
    colors = [
        [int(255 * seg_map["red"][i][1]), int(255 * seg_map["green"][i][1]), int(255 * seg_map["blue"][i][1])]
        for i in range(len(positions))
    ]
    return pg.ColorMap(positions, colors)
