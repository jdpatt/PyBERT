"""Results tab for PyBERT GUI.

This tab shows simulation results including DFE adaptation, output
waveforms, eye diagrams and bathtub curves.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget
from scipy.interpolate import interp1d

from pybert.bert import MIN_BATHTUB_VAL
from pybert.gui.widgets.jitter_info import JitterInfoTable
from pybert.pybert import PyBERT
from pybert.utility.math import make_bathtub, safe_log10
from pybert.utility.sigproc import calc_eye

pg.setConfigOption("background", "w")
# TODO: Either limit auto sizing or limit the range of the plot(s) pyqtgraph defaults to a padding of non-zero values.
pg.ViewBox.suggestPadding = lambda *_: 0.001  # Normally this is 0.02 but we want to reduce the padding.
# pg.setConfigOption('foreground', 'k')


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

        for tab_name, creator_func in [
            ("Impulses", self._create_response_tab),
            ("Steps", self._create_response_tab),
            ("Pulses", self._create_response_tab),
            ("Frequency", self._create_response_tab),
            ("DFE", self._create_dfe_tab),
            ("Outputs", self._create_outputs_tab),
            ("Eyes", self._create_eyes_tab),
            ("Bathtubs", self._create_bathtub_tab),
            ("Jitter Dist.", self._create_jitter_dist_tab),
            ("Jitter Spec.", self._create_jitter_spec_tab),
            ("Jitter Info", self._create_jitter_info_tab),
        ]:
            tab = creator_func(tab_name)
            tab_widget.addTab(tab, tab_name)
            if tab_name == "Eyes":
                tab_widget.setCurrentWidget(tab)

        if pybert:
            self.connect_signals(pybert)

    def connect_signals(self, pybert) -> None:
        """Connect the simulation complete signal to the update_results method.

        This is triggered by the PyBERT instance's when a simulation is complete and the GUI needs to be updated.
        """
        pybert.sim_complete.connect(self.update_results)

    def update_results(self, results, perf):
        """Update all plots using the current PyBERT simulation results."""
        pb = self.pybert
        if pb is None:
            return

        self.jitter_info_table.update_rejection()

        # --- DFE plots ---
        self.update_dfe_plots(
            pb.t_ns,
            results["ui_ests"],
            results["tap_weights"],
            results["clk_per_hist_bins"],
            results["clk_per_hist_vals"],
            results["clk_freqs"],
            results["clk_spec"],
        )

        # --- Output plots ---
        len_t = len(pb.t_ns)
        ideal_signal = pb.ideal_signal[:len_t]
        chnl_out = pb.chnl_out[:len_t]
        rx_in = pb.rx_in[:len_t]
        ctle_out = pb.ctle_out[:len_t]
        dfe_out = pb.dfe_out[:len_t]
        self.update_output_plots(
            pb.t_ns,
            ideal_signal,
            chnl_out,
            rx_in,
            ctle_out,
            dfe_out,
        )

        # --- Eye plots ---
        self.update_eye_plots(results["eye_xs"], results["eye_data"], results["y_max_values"])

        # --- Bathtub plots ---
        self.update_bathtub_plots(results["jitter_bins"], results["bathtub_data"])

        # --- Impulse plots ---
        self.update_impulse_plots(pb.t_ns_chnl, pb.chnl_h, pb.tx_out_h, pb.ctle_out_h, pb.dfe_out_h)

        # --- Step plots ---
        self.update_step_plots(
            pb.t_ns_chnl, pb.chnl_s, pb.tx_s, pb.tx_out_s, pb.ctle_s, pb.ctle_out_s, pb.dfe_s, pb.dfe_out_s
        )

        # --- Pulse plots ---
        self.update_pulse_plots(pb.t_ns_chnl, pb.chnl_p, pb.tx_out_p, pb.ctle_out_p, pb.dfe_out_p)

        # --- Frequency plots ---
        freq_responses = results["freq_responses"]
        self.update_freq_plots(
            results["f_GHz"][1:],
            freq_responses["chnl_H"],
            freq_responses["chnl_H_raw"],
            freq_responses["chnl_trimmed_H"],
            freq_responses["tx_H"],
            freq_responses["tx_out_H"],
            freq_responses["ctle_H"],
            freq_responses["ctle_out_H"],
            freq_responses["dfe_H"],
            freq_responses["dfe_out_H"],
        )

        # --- Jitter distribution plots ---
        self.update_jitter_dist_plots(results["jitter_bins"], results["jitter_data"], results["jitter_ext_data"])

        # --- Jitter spectrum plots ---
        self.update_jitter_spec_plots(
            results["f_MHz"],
            results["jitter_spectrum"],
            results["jitter_ind_spectrum"],
            results["jitter_thresh"],
        )

    # -- Plot Creation Methods ------------------------------------------------------------------------------------

    def _create_jitter_info_tab(self, _: str):
        """Create the jitter info tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

        Returns:
            QWidget: Widget containing the jitter info table
        """
        self.jitter_info_table = JitterInfoTable(self.pybert, parent=self)
        return self.jitter_info_table

    def _create_dfe_tab(self, _: str):
        """Create the DFE adaptation tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

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

    def _create_outputs_tab(self, _: str):
        """Create the outputs tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

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

    def _create_eyes_tab(self, _: str):
        """Create the eye diagrams tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

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

    def _create_bathtub_tab(self, _: str):
        """Create the bathtub curves tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

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

    def _create_response_tab(self, response_name: str):
        """Create a tab with a 2x2 grid of plots.

        Args:
            response_name: The type of response to create.

        Returns:
            QWidget: Widget containing the response plots
        """
        widget = QWidget(self)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        plot_grid = pg.GraphicsLayoutWidget(parent=self)
        layout.addWidget(plot_grid)

        plots = []
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        for i, title in enumerate(titles):
            row = i // 2
            col = i % 2
            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(title)
            if response_name == "Frequency":
                plot.addLegend(offset=(-1, 1))  # Upper Right
                plot.getAxis("left").setLabel("Frequency Response", units="dB")
                plot.getAxis("bottom").setLabel("Frequency", units="GHz")
                plot.setMouseEnabled(False, False)  # Disable zooming for freq response
                plot.setYRange(-40, 10, padding=0)  # Set min/max y values for freq response
            elif response_name == "Impulses":
                plot.addLegend(offset=(-1, 1))  # Upper Right
                plot.getAxis("left").setLabel("Impulse Response", units="V/sample")
                plot.getAxis("bottom").setLabel("Time", units="ns")
            elif response_name == "Steps":
                plot.addLegend(offset=(-1, -1))  # Lower Right
                plot.getAxis("left").setLabel("Step Response", units="V")
                plot.getAxis("bottom").setLabel("Time", units="ns")
            elif response_name == "Pulses":
                plot.addLegend(offset=(-1, 1))  # Upper Right
                plot.getAxis("left").setLabel("Pulse Response", units="V")
                plot.getAxis("bottom").setLabel("Time", units="ns")
            plots.append(plot)

        # Only link x-axes for non-frequency response plots
        if response_name != "Frequency":
            for plot in plots[1:]:
                plot.setXLink(plots[0])

        # Store plots for update methods
        if response_name == "Impulses":
            self.impulse_plots = plots
        elif response_name == "Steps":
            self.step_plots = plots
        elif response_name == "Pulses":
            self.pulse_plots = plots
        elif response_name == "Frequency":
            self.freq_plots = plots

        return widget

    def _create_jitter_dist_tab(self, _: str):
        """Create the jitter distribution tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

        Returns:
            QWidget: Widget containing the jitter distribution plots
        """
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

    def _create_jitter_spec_tab(self, _: str):
        """Create the jitter spectrum tab.

        Args:
            _: The name of the tab, which is ignored but passed to allow us to loop over creation.

        Returns:
            QWidget: Widget containing the jitter spectrum plots
        """
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
            plot.addLegend(offset=(-1, -1))  # Lower Right
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

    # -- Plot Update Methods ------------------------------------------------------------------------------------

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

    def update_eye_plots(self, xs, eye_data, y_max_values):
        """Update eye diagram plots.

        Args:
            xs: Time points for x-axis
            eye_data: List of 2D arrays containing eye diagram data
            y_max_values: List of y_max values for each eye diagram
        """
        colormap = get_custom_colormap()
        lut = colormap.getLookupTable(0.0, 1.0, 256)

        for img, data, y_max in zip(self.eye_plots, eye_data, y_max_values):
            # Set the image with proper voltage mapping
            img.setImage(np.rot90(data), lut=lut)
            img.setRect(pg.QtCore.QRectF(xs[0], -y_max, xs[-1] - xs[0], 2 * y_max))

    def update_bathtub_plots(self, jitter_bins, bathtub_data):
        """Update bathtub curve plots.

        Args:
            jitter_bins: Time points for bathtub curves
            bathtub_data: List of bathtub curve data arrays
        """
        for curve, data in zip(self.bathtub_plots, bathtub_data):
            curve.setData(jitter_bins, data)

    def update_impulse_plots(self, t_ns, chnl_h, tx_out_h, ctle_out_h, dfe_out_h):
        """Update impulse response plots.

        Args:
            t_ns: Time points
            chnl_h: Channel impulse response
            tx_out_h: Tx output impulse response
            ctle_out_h: CTLE output impulse response
            dfe_out_h: DFE output impulse response
        """
        self.impulse_plots[0].plot(t_ns, chnl_h, pen="b", name="Incremental", clear=True)
        self.impulse_plots[1].plot(t_ns, tx_out_h, pen="r", name="Cumulative", clear=True)
        self.impulse_plots[2].plot(t_ns, ctle_out_h, pen="r", name="Cumulative", clear=True)
        self.impulse_plots[3].plot(t_ns, dfe_out_h, pen="r", name="Cumulative", clear=True)

    def update_step_plots(self, t_ns, chnl_s, tx_s, tx_out_s, ctle_s, ctle_out_s, dfe_s, dfe_out_s):
        """Update step response plots.

        Args:
            t_ns: Time points
            chnl_s: Channel step response
            tx_out_h: Tx output step response
            ctle_out_h: CTLE output step response
            dfe_out_h: DFE output step response
        """
        self.step_plots[0].plot(t_ns, chnl_s, pen="b", name="Channel", clear=True)
        self.step_plots[1].plot(t_ns, tx_s, pen="b", name="Incremental", clear=True)
        self.step_plots[1].plot(t_ns, tx_out_s, pen="r", name="Cumulative")
        self.step_plots[2].plot(t_ns, ctle_s[: len(t_ns)], pen="b", name="Incremental", clear=True)
        self.step_plots[2].plot(t_ns, ctle_out_s, pen="r", name="Cumulative")
        self.step_plots[3].plot(t_ns, dfe_s, pen="b", name="Incremental", clear=True)
        self.step_plots[3].plot(t_ns, dfe_out_s, pen="r", name="Cumulative")

    def update_pulse_plots(self, t_ns, chnl_p, tx_out_p, ctle_out_p, dfe_out_p):
        """Update pulse response plots.

        Args:
            t_ns: Time points
            chnl_p: Channel pulse response
            tx_out_p: Tx output pulse response
            ctle_out_p: CTLE output pulse response
            dfe_out_p: DFE output pulse response
        """
        self.pulse_plots[0].plot(t_ns, chnl_p, pen="b", name="Incremental", clear=True)
        self.pulse_plots[1].plot(t_ns, tx_out_p, pen="r", name="Cumulative", clear=True)
        self.pulse_plots[2].plot(t_ns, ctle_out_p, pen="r", name="Cumulative", clear=True)
        self.pulse_plots[3].plot(t_ns, dfe_out_p, pen="r", name="Cumulative", clear=True)

    def update_freq_plots(
        self, f_GHz, chnl_H, chnl_H_raw, chnl_trimmed_H, tx_H, tx_out_H, ctle_H, ctle_out_H, dfe_H, dfe_out_H
    ):
        """Update frequency response plots.

        Args:
            f_GHz: Frequency points
            chnl_H: Channel frequency response
            chnl_H_raw: Channel frequency response (raw)
            chnl_trimmed_H: Channel frequency response (trimmed)
            tx_H: Tx frequency response
            tx_out_H: Tx output frequency response
            ctle_H: CTLE frequency response
            ctle_out_H: CTLE output frequency response
            dfe_H: DFE frequency response
            dfe_out_H: DFE output frequency response
        """
        self.freq_plots[0].plot(f_GHz, chnl_H_raw, pen="k", name="Perfect Term.", clear=True)
        self.freq_plots[0].plot(f_GHz, chnl_H, pen="b", name="Actual Term.")
        self.freq_plots[0].plot(f_GHz, chnl_trimmed_H, pen="r", name="Trimmed Impulse")
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

    def update_jitter_dist_plots(self, jitter_bins, jitter_data, jitter_ext_data):
        """Update jitter distribution plots.

        Args:
            jitter_bins: Time points
            jitter_data: Jitter data
            jitter_ext_data: Jitter extended data
        """
        for (total_curve, di_curve), total, di in zip(self.jitter_dist_plots, jitter_data, jitter_ext_data):
            total_curve.setData(jitter_bins, total)
            di_curve.setData(jitter_bins, di)

    def update_jitter_spec_plots(self, f_MHz, jitter_spectrum, jitter_ind_spectrum, thresh):
        """Update jitter spectrum plots.
        Args:
            f_MHz: Frequency points
            jitter_spectrum: Jitter spectrum
            jitter_ind_spectrum: Jitter independent spectrum
            thresh: Threshold
        """
        for (total_curve, di_curve, thresh_curve), total, di, th in zip(
            self.jitter_spec_plots, jitter_spectrum, jitter_ind_spectrum, thresh
        ):
            total_curve.setData(f_MHz, total)
            di_curve.setData(f_MHz, di)
            thresh_curve.setData(f_MHz, th)


def get_custom_colormap() -> pg.ColorMap:
    """Get a custom colormap for the eye plots."""
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
        ],
    }
    positions = [x[0] for x in seg_map["red"]]
    colors = [
        [int(255 * seg_map["red"][i][1]), int(255 * seg_map["green"][i][1]), int(255 * seg_map["blue"][i][1])]
        for i in range(len(positions))
    ]
    return pg.ColorMap(positions, colors)
