"""
Plot definitions for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 21, 2015 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""
# pylint: skip-file

from logging import getLogger

import numpy as np
from pybert.defaults import MIN_BATHTUB_VAL
from pybert.sim.utility import calc_eye


class Plots:
    """A lot of this class goes away with pyqtgraph but any other items needed
    to generate the plots live here.
    """

    def __init__(self):
        super(Plots, self).__init__()
        self.log = getLogger("pybert.plots")
        self.log.debug("Initializing Plot Object")

        # Plots (plot containers, actually)
        # self.data = ArrayPlotData()
        # self.plots_h = Instance(GridPlotContainer)
        # self.plots_s = Instance(GridPlotContainer)
        # self.plots_p = Instance(GridPlotContainer)
        # self.plots_H = Instance(GridPlotContainer)
        # self.plots_dfe = Instance(GridPlotContainer)
        # self.eyes = Instance(GridPlotContainer)
        # self.plots_jitter_dist = Instance(GridPlotContainer)
        # self.plots_jitter_spec = Instance(GridPlotContainer)
        # self.plots_bathtub = Instance(GridPlotContainer)

        self._dfe_plot = None
        self.plot_h_tune = None
        self.plots_out = None
        self.clr_map = None

    def update_data(self, name: str, data):
        """Update the plot with new data."""
        self.data.set_data(name, data)

    def init_plots(self, n_dfe_taps):
        """ Create the plots used by the PyBERT GUI."""

        data = self.data

        # - DFE tab
        plot2 = Plot(data, padding_left=75)
        plot2.plot(("t_ns", "ui_ests"), type="line", color="blue")
        plot2.title = "CDR Adaptation"
        plot2.index_axis.title = "Time (ns)"
        plot2.value_axis.title = "UI (ps)"

        plot9 = Plot(
            data,
            auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
            padding_left=75,
        )
        for i in range(n_dfe_taps):
            plot9.plot(
                ("tap_weight_index", "tap%d_weights" % (i + 1)),
                type="line",
                color="auto",
                name="tap%d" % (i + 1),
            )
        plot9.title = "DFE Adaptation"
        plot9.tools.append(
            PanTool(plot9, constrain=True, constrain_key=None, constrain_direction="x")
        )
        zoom9 = ZoomTool(plot9, tool_mode="range", axis="index", always_on=False)
        plot9.overlays.append(zoom9)
        plot9.legend.visible = True
        plot9.legend.align = "ul"

        plot_clk_per_hist = Plot(data, padding_left=75)
        plot_clk_per_hist.plot(
            ("clk_per_hist_bins", "clk_per_hist_vals"), type="line", color="blue"
        )
        plot_clk_per_hist.title = "CDR Clock Period Histogram"
        plot_clk_per_hist.index_axis.title = "Clock Period (ps)"
        plot_clk_per_hist.value_axis.title = "Bin Count"

        plot_clk_per_spec = Plot(data, padding_left=75)
        plot_clk_per_spec.plot(("clk_freqs", "clk_spec"), type="line", color="blue")
        plot_clk_per_spec.title = "CDR Clock Period Spectrum"
        plot_clk_per_spec.index_axis.title = "Frequency (bit rate)"
        plot_clk_per_spec.value_axis.title = "|H(f)| (dB mean)"
        plot_clk_per_spec.value_range.low_setting = -10
        zoom_clk_per_spec = ZoomTool(
            plot_clk_per_spec, tool_mode="range", axis="index", always_on=False
        )
        plot_clk_per_spec.overlays.append(zoom_clk_per_spec)

        container_dfe = GridPlotContainer(shape=(2, 2))
        container_dfe.add(plot2)
        container_dfe.add(plot9)
        container_dfe.add(plot_clk_per_hist)
        container_dfe.add(plot_clk_per_spec)
        self.plots_dfe = container_dfe
        self._dfe_plot = plot9

        # - EQ Tune tab
        plot_h_tune = Plot(data, padding_left=75)
        plot_h_tune.plot(("t_ns_chnl", "ctle_out_h_tune"), type="line", color="blue")
        plot_h_tune.plot(("t_ns_chnl", "clocks_tune"), type="line", color="gray")
        plot_h_tune.title = "Channel + Tx Preemphasis + CTLE + Ideal DFE"
        plot_h_tune.index_axis.title = "Time (ns)"
        plot_h_tune.y_axis.title = "Post-CTLE Pulse Response (V)"
        zoom_tune = ZoomTool(plot_h_tune, tool_mode="range", axis="index", always_on=False)
        plot_h_tune.overlays.append(zoom_tune)
        self.plot_h_tune = plot_h_tune

        # - Impulse Responses tab
        plot_h_chnl = Plot(data, padding_left=75)
        plot_h_chnl.plot(("t_ns_chnl", "chnl_h"), type="line", color="blue", name="Incremental")
        plot_h_chnl.title = "Channel"
        plot_h_chnl.index_axis.title = "Time (ns)"
        plot_h_chnl.y_axis.title = "Impulse Response (V/ns)"
        plot_h_chnl.legend.visible = True
        plot_h_chnl.legend.align = "ur"
        zoom_h = ZoomTool(plot_h_chnl, tool_mode="range", axis="index", always_on=False)
        plot_h_chnl.overlays.append(zoom_h)

        plot_h_tx = Plot(data, padding_left=75)
        plot_h_tx.plot(("t_ns_chnl", "tx_out_h"), type="line", color="red", name="Cumulative")
        plot_h_tx.title = "Channel + Tx Preemphasis"
        plot_h_tx.index_axis.title = "Time (ns)"
        plot_h_tx.y_axis.title = "Impulse Response (V/ns)"
        plot_h_tx.legend.visible = True
        plot_h_tx.legend.align = "ur"
        plot_h_tx.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

        plot_h_ctle = Plot(data, padding_left=75)
        plot_h_ctle.plot(("t_ns_chnl", "ctle_out_h"), type="line", color="red", name="Cumulative")
        plot_h_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_h_ctle.index_axis.title = "Time (ns)"
        plot_h_ctle.y_axis.title = "Impulse Response (V/ns)"
        plot_h_ctle.legend.visible = True
        plot_h_ctle.legend.align = "ur"
        plot_h_ctle.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

        plot_h_dfe = Plot(data, padding_left=75)
        plot_h_dfe.plot(("t_ns_chnl", "dfe_out_h"), type="line", color="red", name="Cumulative")
        plot_h_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_h_dfe.index_axis.title = "Time (ns)"
        plot_h_dfe.y_axis.title = "Impulse Response (V/ns)"
        plot_h_dfe.legend.visible = True
        plot_h_dfe.legend.align = "ur"
        plot_h_dfe.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

        container_h = GridPlotContainer(shape=(2, 2))
        container_h.add(plot_h_chnl)
        container_h.add(plot_h_tx)
        container_h.add(plot_h_ctle)
        container_h.add(plot_h_dfe)
        self.plots_h = container_h

        # - Step Responses tab
        plot_s_chnl = Plot(data, padding_left=75)
        plot_s_chnl.plot(("t_ns_chnl", "chnl_s"), type="line", color="blue", name="Incremental")
        plot_s_chnl.title = "Channel"
        plot_s_chnl.index_axis.title = "Time (ns)"
        plot_s_chnl.y_axis.title = "Step Response (V)"
        plot_s_chnl.legend.visible = True
        plot_s_chnl.legend.align = "lr"
        zoom_s = ZoomTool(plot_s_chnl, tool_mode="range", axis="index", always_on=False)
        plot_s_chnl.overlays.append(zoom_s)

        plot_s_tx = Plot(data, padding_left=75)
        plot_s_tx.plot(("t_ns_chnl", "tx_s"), type="line", color="blue", name="Incremental")
        plot_s_tx.plot(("t_ns_chnl", "tx_out_s"), type="line", color="red", name="Cumulative")
        plot_s_tx.title = "Channel + Tx Preemphasis"
        plot_s_tx.index_axis.title = "Time (ns)"
        plot_s_tx.y_axis.title = "Step Response (V)"
        plot_s_tx.legend.visible = True
        plot_s_tx.legend.align = "lr"
        plot_s_tx.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

        plot_s_ctle = Plot(data, padding_left=75)
        plot_s_ctle.plot(("t_ns_chnl", "ctle_s"), type="line", color="blue", name="Incremental")
        plot_s_ctle.plot(("t_ns_chnl", "ctle_out_s"), type="line", color="red", name="Cumulative")
        plot_s_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_s_ctle.index_axis.title = "Time (ns)"
        plot_s_ctle.y_axis.title = "Step Response (V)"
        plot_s_ctle.legend.visible = True
        plot_s_ctle.legend.align = "lr"
        plot_s_ctle.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

        plot_s_dfe = Plot(data, padding_left=75)
        plot_s_dfe.plot(("t_ns_chnl", "dfe_s"), type="line", color="blue", name="Incremental")
        plot_s_dfe.plot(("t_ns_chnl", "dfe_out_s"), type="line", color="red", name="Cumulative")
        plot_s_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_s_dfe.index_axis.title = "Time (ns)"
        plot_s_dfe.y_axis.title = "Step Response (V)"
        plot_s_dfe.legend.visible = True
        plot_s_dfe.legend.align = "lr"
        plot_s_dfe.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

        container_s = GridPlotContainer(shape=(2, 2))
        container_s.add(plot_s_chnl)
        container_s.add(plot_s_tx)
        container_s.add(plot_s_ctle)
        container_s.add(plot_s_dfe)
        self.plots_s = container_s

        # - Pulse Responses tab
        plot_p_chnl = Plot(data, padding_left=75)
        plot_p_chnl.plot(("t_ns_chnl", "chnl_p"), type="line", color="blue", name="Incremental")
        plot_p_chnl.title = "Channel"
        plot_p_chnl.index_axis.title = "Time (ns)"
        plot_p_chnl.y_axis.title = "Pulse Response (V)"
        plot_p_chnl.legend.visible = True
        plot_p_chnl.legend.align = "ur"
        zoom_p = ZoomTool(plot_p_chnl, tool_mode="range", axis="index", always_on=False)
        plot_p_chnl.overlays.append(zoom_p)

        plot_p_tx = Plot(data, padding_left=75)
        plot_p_tx.plot(("t_ns_chnl", "tx_out_p"), type="line", color="red", name="Cumulative")
        plot_p_tx.title = "Channel + Tx Preemphasis"
        plot_p_tx.index_axis.title = "Time (ns)"
        plot_p_tx.y_axis.title = "Pulse Response (V)"
        plot_p_tx.legend.visible = True
        plot_p_tx.legend.align = "ur"
        plot_p_tx.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

        plot_p_ctle = Plot(data, padding_left=75)
        plot_p_ctle.plot(("t_ns_chnl", "ctle_out_p"), type="line", color="red", name="Cumulative")
        plot_p_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_p_ctle.index_axis.title = "Time (ns)"
        plot_p_ctle.y_axis.title = "Pulse Response (V)"
        plot_p_ctle.legend.visible = True
        plot_p_ctle.legend.align = "ur"
        plot_p_ctle.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

        plot_p_dfe = Plot(data, padding_left=75)
        plot_p_dfe.plot(("t_ns_chnl", "dfe_out_p"), type="line", color="red", name="Cumulative")
        plot_p_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_p_dfe.index_axis.title = "Time (ns)"
        plot_p_dfe.y_axis.title = "Pulse Response (V)"
        plot_p_dfe.legend.visible = True
        plot_p_dfe.legend.align = "ur"
        plot_p_dfe.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

        container_p = GridPlotContainer(shape=(2, 2))
        container_p.add(plot_p_chnl)
        container_p.add(plot_p_tx)
        container_p.add(plot_p_ctle)
        container_p.add(plot_p_dfe)
        self.plots_p = container_p

        # - Frequency Responses tab
        plot_H_chnl = Plot(data, padding_left=75)
        plot_H_chnl.plot(
            ("f_GHz", "chnl_H"),
            type="line",
            color="blue",
            name="Original Impulse",
            index_scale="log",
        )
        plot_H_chnl.plot(
            ("f_GHz", "chnl_trimmed_H"),
            type="line",
            color="red",
            name="Trimmed Impulse",
            index_scale="log",
        )
        plot_H_chnl.title = "Channel"
        plot_H_chnl.index_axis.title = "Frequency (GHz)"
        plot_H_chnl.y_axis.title = "Frequency Response (dB)"
        plot_H_chnl.index_range.low_setting = 0.01
        plot_H_chnl.index_range.high_setting = 40.0
        plot_H_chnl.legend.visible = True
        plot_H_chnl.legend.align = "ll"

        plot_H_tx = Plot(data, padding_left=75)
        plot_H_tx.plot(
            ("f_GHz", "tx_H"), type="line", color="blue", name="Incremental", index_scale="log"
        )
        plot_H_tx.plot(
            ("f_GHz", "tx_out_H"), type="line", color="red", name="Cumulative", index_scale="log"
        )
        plot_H_tx.title = "Channel + Tx Preemphasis"
        plot_H_tx.index_axis.title = "Frequency (GHz)"
        plot_H_tx.y_axis.title = "Frequency Response (dB)"
        plot_H_tx.index_range.low_setting = 0.01
        plot_H_tx.index_range.high_setting = 40.0
        plot_H_tx.legend.visible = True
        plot_H_tx.legend.align = "ll"

        plot_H_ctle = Plot(data, padding_left=75)
        plot_H_ctle.plot(
            ("f_GHz", "ctle_H"), type="line", color="blue", name="Incremental", index_scale="log"
        )
        plot_H_ctle.plot(
            ("f_GHz", "ctle_out_H"), type="line", color="red", name="Cumulative", index_scale="log"
        )
        plot_H_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_H_ctle.index_axis.title = "Frequency (GHz)"
        plot_H_ctle.y_axis.title = "Frequency Response (dB)"
        plot_H_ctle.index_range.low_setting = 0.01
        plot_H_ctle.index_range.high_setting = 40.0
        plot_H_ctle.value_range.low_setting = -40.0
        plot_H_ctle.legend.visible = True
        plot_H_ctle.legend.align = "ll"

        plot_H_chnl.value_range = plot_H_ctle.value_range
        plot_H_tx.value_range = plot_H_ctle.value_range

        plot_H_dfe = Plot(data, padding_left=75)
        plot_H_dfe.plot(
            ("f_GHz", "dfe_H"), type="line", color="blue", name="Incremental", index_scale="log"
        )
        plot_H_dfe.plot(
            ("f_GHz", "dfe_out_H"), type="line", color="red", name="Cumulative", index_scale="log"
        )
        plot_H_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_H_dfe.index_axis.title = "Frequency (GHz)"
        plot_H_dfe.y_axis.title = "Frequency Response (dB)"
        plot_H_dfe.index_range.low_setting = 0.01
        plot_H_dfe.index_range.high_setting = 40.0
        plot_H_dfe.value_range = plot_H_ctle.value_range
        plot_H_dfe.legend.visible = True
        plot_H_dfe.legend.align = "ll"

        container_H = GridPlotContainer(shape=(2, 2))
        container_H.add(plot_H_chnl)
        container_H.add(plot_H_tx)
        container_H.add(plot_H_ctle)
        container_H.add(plot_H_dfe)
        self.plots_H = container_H

        # - Outputs tab
        plot_out_chnl = Plot(data, padding_left=75)
        plot_out_chnl.plot(("t_ns", "ideal_signal"), type="line", color="lightgrey")
        plot_out_chnl.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot_out_chnl.title = "Channel"
        plot_out_chnl.index_axis.title = "Time (ns)"
        plot_out_chnl.y_axis.title = "Output (V)"
        plot_out_chnl.tools.append(
            PanTool(plot_out_chnl, constrain=True, constrain_key=None, constrain_direction="x")
        )
        zoom_out_chnl = ZoomTool(plot_out_chnl, tool_mode="range", axis="index", always_on=False)
        plot_out_chnl.overlays.append(zoom_out_chnl)

        plot_out_tx = Plot(data, padding_left=75)
        plot_out_tx.plot(("t_ns", "tx_out"), type="line", color="blue")
        plot_out_tx.title = "Channel + Tx Preemphasis (Noise added here.)"
        plot_out_tx.index_axis.title = "Time (ns)"
        plot_out_tx.y_axis.title = "Output (V)"
        plot_out_tx.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

        plot_out_ctle = Plot(data, padding_left=75)
        plot_out_ctle.plot(("t_ns", "ctle_out"), type="line", color="blue")
        plot_out_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_out_ctle.index_axis.title = "Time (ns)"
        plot_out_ctle.y_axis.title = "Output (V)"
        plot_out_ctle.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

        plot_out_dfe = Plot(data, padding_left=75)
        plot_out_dfe.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot_out_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_out_dfe.index_axis.title = "Time (ns)"
        plot_out_dfe.y_axis.title = "Output (V)"
        plot_out_dfe.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

        container_out = GridPlotContainer(shape=(2, 2))
        container_out.add(plot_out_chnl)
        container_out.add(plot_out_tx)
        container_out.add(plot_out_ctle)
        container_out.add(plot_out_dfe)
        self.plots_out = container_out

        # - Eye Diagrams tab
        seg_map = dict(
            red=[
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
            green=[
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
            blue=[
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
        )
        clr_map = ColorMapper.from_segment_map(seg_map)
        self.clr_map = clr_map

        plot_eye_chnl = Plot(data, padding_left=75)
        plot_eye_chnl.img_plot("eye_chnl", colormap=clr_map)
        plot_eye_chnl.y_direction = "normal"
        plot_eye_chnl.components[0].y_direction = "normal"
        plot_eye_chnl.title = "Channel"
        plot_eye_chnl.x_axis.title = "Time (ps)"
        plot_eye_chnl.x_axis.orientation = "bottom"
        plot_eye_chnl.y_axis.title = "Signal Level (V)"
        plot_eye_chnl.x_grid.visible = True
        plot_eye_chnl.y_grid.visible = True
        plot_eye_chnl.x_grid.line_color = "gray"
        plot_eye_chnl.y_grid.line_color = "gray"

        plot_eye_tx = Plot(data, padding_left=75)
        plot_eye_tx.img_plot("eye_tx", colormap=clr_map)
        plot_eye_tx.y_direction = "normal"
        plot_eye_tx.components[0].y_direction = "normal"
        plot_eye_tx.title = "Channel + Tx Preemphasis (Noise added here.)"
        plot_eye_tx.x_axis.title = "Time (ps)"
        plot_eye_tx.x_axis.orientation = "bottom"
        plot_eye_tx.y_axis.title = "Signal Level (V)"
        plot_eye_tx.x_grid.visible = True
        plot_eye_tx.y_grid.visible = True
        plot_eye_tx.x_grid.line_color = "gray"
        plot_eye_tx.y_grid.line_color = "gray"

        plot_eye_ctle = Plot(data, padding_left=75)
        plot_eye_ctle.img_plot("eye_ctle", colormap=clr_map)
        plot_eye_ctle.y_direction = "normal"
        plot_eye_ctle.components[0].y_direction = "normal"
        plot_eye_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_eye_ctle.x_axis.title = "Time (ps)"
        plot_eye_ctle.x_axis.orientation = "bottom"
        plot_eye_ctle.y_axis.title = "Signal Level (V)"
        plot_eye_ctle.x_grid.visible = True
        plot_eye_ctle.y_grid.visible = True
        plot_eye_ctle.x_grid.line_color = "gray"
        plot_eye_ctle.y_grid.line_color = "gray"

        plot_eye_dfe = Plot(data, padding_left=75)
        plot_eye_dfe.img_plot("eye_dfe", colormap=clr_map)
        plot_eye_dfe.y_direction = "normal"
        plot_eye_dfe.components[0].y_direction = "normal"
        plot_eye_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_eye_dfe.x_axis.title = "Time (ps)"
        plot_eye_dfe.x_axis.orientation = "bottom"
        plot_eye_dfe.y_axis.title = "Signal Level (V)"
        plot_eye_dfe.x_grid.visible = True
        plot_eye_dfe.y_grid.visible = True
        plot_eye_dfe.x_grid.line_color = "gray"
        plot_eye_dfe.y_grid.line_color = "gray"

        container_eye = GridPlotContainer(shape=(2, 2))
        container_eye.add(plot_eye_chnl)
        container_eye.add(plot_eye_tx)
        container_eye.add(plot_eye_ctle)
        container_eye.add(plot_eye_dfe)
        self.eyes = container_eye

        # - Jitter Distributions tab
        plot_jitter_dist_chnl = Plot(data, padding_left=75)
        plot_jitter_dist_chnl.plot(
            ("jitter_bins", "jitter_chnl"), type="line", color="blue", name="Measured"
        )
        plot_jitter_dist_chnl.plot(
            ("jitter_bins", "jitter_ext_chnl"), type="line", color="red", name="Extrapolated"
        )
        plot_jitter_dist_chnl.title = "Channel"
        plot_jitter_dist_chnl.index_axis.title = "Time (ps)"
        plot_jitter_dist_chnl.value_axis.title = "Count"
        plot_jitter_dist_chnl.legend.visible = True
        plot_jitter_dist_chnl.legend.align = "ur"

        plot_jitter_dist_tx = Plot(data, padding_left=75)
        plot_jitter_dist_tx.plot(
            ("jitter_bins", "jitter_tx"), type="line", color="blue", name="Measured"
        )
        plot_jitter_dist_tx.plot(
            ("jitter_bins", "jitter_ext_tx"), type="line", color="red", name="Extrapolated"
        )
        plot_jitter_dist_tx.title = "Channel + Tx Preemphasis (Noise added here.)"
        plot_jitter_dist_tx.index_axis.title = "Time (ps)"
        plot_jitter_dist_tx.value_axis.title = "Count"
        plot_jitter_dist_tx.legend.visible = True
        plot_jitter_dist_tx.legend.align = "ur"

        plot_jitter_dist_ctle = Plot(data, padding_left=75)
        plot_jitter_dist_ctle.plot(
            ("jitter_bins", "jitter_ctle"), type="line", color="blue", name="Measured"
        )
        plot_jitter_dist_ctle.plot(
            ("jitter_bins", "jitter_ext_ctle"), type="line", color="red", name="Extrapolated"
        )
        plot_jitter_dist_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_jitter_dist_ctle.index_axis.title = "Time (ps)"
        plot_jitter_dist_ctle.value_axis.title = "Count"
        plot_jitter_dist_ctle.legend.visible = True
        plot_jitter_dist_ctle.legend.align = "ur"

        plot_jitter_dist_dfe = Plot(data, padding_left=75)
        plot_jitter_dist_dfe.plot(
            ("jitter_bins", "jitter_dfe"), type="line", color="blue", name="Measured"
        )
        plot_jitter_dist_dfe.plot(
            ("jitter_bins", "jitter_ext_dfe"), type="line", color="red", name="Extrapolated"
        )
        plot_jitter_dist_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_jitter_dist_dfe.index_axis.title = "Time (ps)"
        plot_jitter_dist_dfe.value_axis.title = "Count"
        plot_jitter_dist_dfe.legend.visible = True
        plot_jitter_dist_dfe.legend.align = "ur"

        container_jitter_dist = GridPlotContainer(shape=(2, 2))
        container_jitter_dist.add(plot_jitter_dist_chnl)
        container_jitter_dist.add(plot_jitter_dist_tx)
        container_jitter_dist.add(plot_jitter_dist_ctle)
        container_jitter_dist.add(plot_jitter_dist_dfe)
        self.plots_jitter_dist = container_jitter_dist

        # - Jitter Spectrums tab
        plot_jitter_spec_chnl = Plot(data)
        plot_jitter_spec_chnl.plot(
            ("f_MHz", "jitter_spectrum_chnl"), type="line", color="blue", name="Total"
        )
        plot_jitter_spec_chnl.plot(
            ("f_MHz", "jitter_ind_spectrum_chnl"),
            type="line",
            color="red",
            name="Data Independent",
        )
        plot_jitter_spec_chnl.plot(
            ("f_MHz", "thresh_chnl"), type="line", color="magenta", name="Pj Threshold"
        )
        plot_jitter_spec_chnl.title = "Channel"
        plot_jitter_spec_chnl.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_chnl.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_chnl.tools.append(
            PanTool(
                plot_jitter_spec_chnl, constrain=True, constrain_key=None, constrain_direction="x"
            )
        )
        zoom_jitter_spec_chnl = ZoomTool(
            plot_jitter_spec_chnl, tool_mode="range", axis="index", always_on=False
        )
        plot_jitter_spec_chnl.overlays.append(zoom_jitter_spec_chnl)
        plot_jitter_spec_chnl.legend.visible = True
        plot_jitter_spec_chnl.legend.align = "lr"

        plot_jitter_spec_tx = Plot(data)
        plot_jitter_spec_tx.plot(
            ("f_MHz", "jitter_spectrum_tx"), type="line", color="blue", name="Total"
        )
        plot_jitter_spec_tx.plot(
            ("f_MHz", "jitter_ind_spectrum_tx"), type="line", color="red", name="Data Independent"
        )
        plot_jitter_spec_tx.plot(
            ("f_MHz", "thresh_tx"), type="line", color="magenta", name="Pj Threshold"
        )
        plot_jitter_spec_tx.title = "Channel + Tx Preemphasis (Noise added here.)"
        plot_jitter_spec_tx.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_tx.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_tx.value_range.low_setting = -40.0
        plot_jitter_spec_tx.index_range = (
            plot_jitter_spec_chnl.index_range
        )  # Zoom x-axes in tandem.
        plot_jitter_spec_tx.legend.visible = True
        plot_jitter_spec_tx.legend.align = "lr"

        plot_jitter_spec_chnl.value_range = plot_jitter_spec_tx.value_range

        plot_jitter_spec_ctle = Plot(data)
        plot_jitter_spec_ctle.plot(
            ("f_MHz", "jitter_spectrum_ctle"), type="line", color="blue", name="Total"
        )
        plot_jitter_spec_ctle.plot(
            ("f_MHz", "jitter_ind_spectrum_ctle"),
            type="line",
            color="red",
            name="Data Independent",
        )
        plot_jitter_spec_ctle.plot(
            ("f_MHz", "thresh_ctle"), type="line", color="magenta", name="Pj Threshold"
        )
        plot_jitter_spec_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_jitter_spec_ctle.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_ctle.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_ctle.index_range = (
            plot_jitter_spec_chnl.index_range
        )  # Zoom x-axes in tandem.
        plot_jitter_spec_ctle.legend.visible = True
        plot_jitter_spec_ctle.legend.align = "lr"
        plot_jitter_spec_ctle.value_range = plot_jitter_spec_tx.value_range

        plot_jitter_spec_dfe = Plot(data)
        plot_jitter_spec_dfe.plot(
            ("f_MHz_dfe", "jitter_spectrum_dfe"), type="line", color="blue", name="Total"
        )
        plot_jitter_spec_dfe.plot(
            ("f_MHz_dfe", "jitter_ind_spectrum_dfe"),
            type="line",
            color="red",
            name="Data Independent",
        )
        plot_jitter_spec_dfe.plot(
            ("f_MHz_dfe", "thresh_dfe"), type="line", color="magenta", name="Pj Threshold"
        )
        plot_jitter_spec_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_jitter_spec_dfe.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_dfe.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_dfe.index_range = (
            plot_jitter_spec_chnl.index_range
        )  # Zoom x-axes in tandem.
        plot_jitter_spec_dfe.legend.visible = True
        plot_jitter_spec_dfe.legend.align = "lr"
        plot_jitter_spec_dfe.value_range = plot_jitter_spec_tx.value_range

        container_jitter_spec = GridPlotContainer(shape=(2, 2))
        container_jitter_spec.add(plot_jitter_spec_chnl)
        container_jitter_spec.add(plot_jitter_spec_tx)
        container_jitter_spec.add(plot_jitter_spec_ctle)
        container_jitter_spec.add(plot_jitter_spec_dfe)
        self.plots_jitter_spec = container_jitter_spec

        # - Bathtub Curves tab
        plot_bathtub_chnl = Plot(data)
        plot_bathtub_chnl.plot(("jitter_bins", "bathtub_chnl"), type="line", color="blue")
        plot_bathtub_chnl.value_range.high_setting = 0
        plot_bathtub_chnl.value_range.low_setting = -18
        plot_bathtub_chnl.value_axis.tick_interval = 3
        plot_bathtub_chnl.title = "Channel"
        plot_bathtub_chnl.index_axis.title = "Time (ps)"
        plot_bathtub_chnl.value_axis.title = "Log10(P(Transition occurs inside.))"

        plot_bathtub_tx = Plot(data)
        plot_bathtub_tx.plot(("jitter_bins", "bathtub_tx"), type="line", color="blue")
        plot_bathtub_tx.value_range.high_setting = 0
        plot_bathtub_tx.value_range.low_setting = -18
        plot_bathtub_tx.value_axis.tick_interval = 3
        plot_bathtub_tx.title = "Channel + Tx Preemphasis (Noise added here.)"
        plot_bathtub_tx.index_axis.title = "Time (ps)"
        plot_bathtub_tx.value_axis.title = "Log10(P(Transition occurs inside.))"

        plot_bathtub_ctle = Plot(data)
        plot_bathtub_ctle.plot(("jitter_bins", "bathtub_ctle"), type="line", color="blue")
        plot_bathtub_ctle.value_range.high_setting = 0
        plot_bathtub_ctle.value_range.low_setting = -18
        plot_bathtub_ctle.value_axis.tick_interval = 3
        plot_bathtub_ctle.title = "Channel + Tx Preemphasis + CTLE"
        plot_bathtub_ctle.index_axis.title = "Time (ps)"
        plot_bathtub_ctle.value_axis.title = "Log10(P(Transition occurs inside.))"

        plot_bathtub_dfe = Plot(data)
        plot_bathtub_dfe.plot(("jitter_bins", "bathtub_dfe"), type="line", color="blue")
        plot_bathtub_dfe.value_range.high_setting = 0
        plot_bathtub_dfe.value_range.low_setting = -18
        plot_bathtub_dfe.value_axis.tick_interval = 3
        plot_bathtub_dfe.title = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_bathtub_dfe.index_axis.title = "Time (ps)"
        plot_bathtub_dfe.value_axis.title = "Log10(P(Transition occurs inside.))"

        container_bathtub = GridPlotContainer(shape=(2, 2))
        container_bathtub.add(plot_bathtub_chnl)
        container_bathtub.add(plot_bathtub_tx)
        container_bathtub.add(plot_bathtub_ctle)
        container_bathtub.add(plot_bathtub_dfe)
        self.plots_bathtub = container_bathtub

    def update_results(self):
        """
        Updates all plot data used by GUI.

        Args:
            self(PyBERT): Reference to an instance of the *PyBERT* class.

        """

        # Copy globals into local namespace.
        ui = self.ui
        samps_per_ui = self.nspui
        eye_uis = self.eye_uis
        num_ui = self.nui
        clock_times = self.clock_times
        f = self.f
        t = self.t
        t_ns = self.t_ns
        t_ns_chnl = self.channel.t_ns_chnl
        n_taps = self.eq.n_taps

        Ts = t[1]
        ignore_until = (num_ui - eye_uis) * ui
        ignore_samps = (num_ui - eye_uis) * samps_per_ui

        # Misc.
        f_GHz = f[: len(f) // 2] / 1.0e9
        len_f_GHz = len(f_GHz)
        self.update_data("f_GHz", f_GHz[1:])
        self.update_data("t_ns", t_ns)
        self.update_data("t_ns_chnl", t_ns_chnl)

        # DFE.
        tap_weights = np.transpose(np.array(self.adaptation))
        i = 1
        for tap_weight in tap_weights:
            self.update_data("tap%d_weights" % i, tap_weight)
            i += 1
        self.update_data("tap_weight_index", list(range(len(tap_weight))))
        if self.eq._old_n_taps != n_taps:
            new_plot = Plot(
                self.data,
                auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
                padding_left=75,
            )
            for i in range(self.eq.n_taps):
                new_plot.plot(
                    ("tap_weight_index", "tap%d_weights" % (i + 1)),
                    type="line",
                    color="auto",
                    name="tap%d" % (i + 1),
                )
            new_plot.title = "DFE Adaptation"
            new_plot.tools.append(
                PanTool(new_plot, constrain=True, constrain_key=None, constrain_direction="x")
            )
            zoom9 = ZoomTool(new_plot, tool_mode="range", axis="index", always_on=False)
            new_plot.overlays.append(zoom9)
            new_plot.legend.visible = True
            new_plot.legend.align = "ul"
            self.plots_dfe.remove(self._dfe_plot)
            self.plots_dfe.insert(1, new_plot)
            self._dfe_plot = new_plot
            self.eq._old_n_taps = n_taps

        clock_pers = np.diff(clock_times)
        start_t = t[np.where(self.lockeds)[0][0]]
        start_ix = np.where(clock_times > start_t)[0][0]
        (bin_counts, bin_edges) = np.histogram(clock_pers[start_ix:], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        clock_spec = fft(clock_pers[start_ix:])
        clock_spec = abs(clock_spec[: len(clock_spec) // 2])
        spec_freqs = np.arange(len(clock_spec)) / (
            2.0 * len(clock_spec)
        )  # In this case, fNyquist = half the bit rate.
        clock_spec /= clock_spec[1:].mean()  # Normalize the mean non-d.c. value to 0 dB.
        self.update_data("clk_per_hist_bins", bin_centers * 1.0e12)  # (ps)
        self.update_data("clk_per_hist_vals", bin_counts)
        self.update_data("clk_spec", 10.0 * np.log10(clock_spec[1:]))  # Omit the d.c. value.
        self.update_data("clk_freqs", spec_freqs[1:])
        self.update_data("dfe_out", self.dfe_out)
        self.update_data("ui_ests", self.ui_ests)
        self.update_data("clocks", self.clocks)
        self.update_data("lockeds", self.lockeds)

        # Impulse responses
        self.update_data(
            "chnl_h", self.channel.chnl_h * 1.0e-9 / Ts
        )  # Re-normalize to (V/ns), for plotting.
        self.update_data("tx_h", self.tx_h * 1.0e-9 / Ts)
        self.update_data("tx_out_h", self.tx_out_h * 1.0e-9 / Ts)
        self.update_data("ctle_h", self.ctle_h * 1.0e-9 / Ts)
        self.update_data("ctle_out_h", self.ctle_out_h * 1.0e-9 / Ts)
        self.update_data("dfe_h", self.dfe_h * 1.0e-9 / Ts)
        self.update_data("dfe_out_h", self.dfe_out_h * 1.0e-9 / Ts)

        # Step responses
        self.update_data("chnl_s", self.channel.chnl_s)
        self.update_data("tx_s", self.tx_s)
        self.update_data("tx_out_s", self.tx_out_s)
        self.update_data("ctle_s", self.ctle_s)
        self.update_data("ctle_out_s", self.ctle_out_s)
        self.update_data("dfe_s", self.dfe_s)
        self.update_data("dfe_out_s", self.dfe_out_s)

        # Pulse responses
        self.update_data("chnl_p", self.channel.chnl_p)
        self.update_data("tx_out_p", self.tx_out_p)
        self.update_data("ctle_out_p", self.ctle_out_p)
        self.update_data("dfe_out_p", self.dfe_out_p)

        # Outputs
        self.update_data("ideal_signal", self.ideal_signal)
        self.update_data("chnl_out", self.chnl_out)
        self.update_data("tx_out", self.rx_in)
        self.update_data("ctle_out", self.ctle_out)
        self.update_data("dfe_out", self.dfe_out)
        self.update_data("auto_corr", self.auto_corr)

        # Frequency responses
        self.update_data("chnl_H", 20.0 * np.log10(abs(self.channel.chnl_H[1:len_f_GHz])))
        self.update_data(
            "chnl_trimmed_H", 20.0 * np.log10(abs(self.channel.chnl_trimmed_H[1:len_f_GHz]))
        )
        self.update_data("tx_H", 20.0 * np.log10(abs(self.tx_H[1:len_f_GHz])))
        self.update_data("tx_out_H", 20.0 * np.log10(abs(self.tx_out_H[1:len_f_GHz])))
        self.update_data("ctle_H", 20.0 * np.log10(abs(self.ctle_H[1:len_f_GHz])))
        self.update_data("ctle_out_H", 20.0 * np.log10(abs(self.ctle_out_H[1:len_f_GHz])))
        self.update_data("dfe_H", 20.0 * np.log10(abs(self.dfe_H[1:len_f_GHz])))
        self.update_data("dfe_out_H", 20.0 * np.log10(abs(self.dfe_out_H[1:len_f_GHz])))

        self.update_data("jitter_bins", np.array(self.jitter["channel"].jitter_bins) * 1.0e12)
        self.update_data("jitter_chnl", self.jitter["channel"].hist)
        self.update_data("jitter_ext_chnl", self.jitter["channel"].hist_synth)
        self.update_data("jitter_tx", self.jitter["tx"].hist)
        self.update_data("jitter_ext_tx", self.jitter["tx"].hist_synth)
        self.update_data("jitter_ctle", self.jitter["ctle"].hist)
        self.update_data("jitter_ext_ctle", self.jitter["ctle"].hist_synth)
        self.update_data("jitter_dfe", self.jitter["dfe"].hist)
        self.update_data("jitter_ext_dfe", self.jitter["dfe"].hist_synth)

        # Jitter spectrums
        log10_ui = np.log10(ui)
        self.update_data("f_MHz", self.jitter["f_MHz"][1:])
        self.update_data("f_MHz_dfe", self.jitter["f_MHz_dfe"][1:])
        self.update_data(
            "jitter_spectrum_chnl",
            10.0 * (np.log10(self.jitter["channel"].jitter_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "jitter_ind_spectrum_chnl",
            10.0 * (np.log10(self.jitter["channel"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "thresh_chnl", 10.0 * (np.log10(self.jitter["channel"].thresh[1:]) - log10_ui)
        )
        self.update_data(
            "jitter_spectrum_tx",
            10.0 * (np.log10(self.jitter["tx"].jitter_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "jitter_ind_spectrum_tx",
            10.0 * (np.log10(self.jitter["tx"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.update_data("thresh_tx", 10.0 * (np.log10(self.jitter["tx"].thresh[1:]) - log10_ui))
        self.update_data(
            "jitter_spectrum_ctle",
            10.0 * (np.log10(self.jitter["ctle"].jitter_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "jitter_ind_spectrum_ctle",
            10.0 * (np.log10(self.jitter["ctle"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "thresh_ctle", 10.0 * (np.log10(self.jitter["ctle"].thresh[1:]) - log10_ui)
        )
        self.update_data(
            "jitter_spectrum_dfe",
            10.0 * (np.log10(self.jitter["dfe"].jitter_spectrum[1:]) - log10_ui),
        )
        self.update_data(
            "jitter_ind_spectrum_dfe",
            10.0 * (np.log10(self.jitter["dfe"].tie_ind_spectrum[1:]) - log10_ui),
        )
        self.update_data("thresh_dfe", 10.0 * (np.log10(self.jitter["dfe"].thresh[1:]) - log10_ui))
        self.update_data("jitter_rejection_ratio", self.jitter["rejection_ratio"][1:])

        # Bathtubs
        half_len = len(self.jitter["channel"].hist_synth) // 2
        #  - Channel
        bathtub_chnl = list(
            np.cumsum(self.jitter["channel"].hist_synth[-1 : -(half_len + 1) : -1])
        )
        bathtub_chnl.reverse()
        bathtub_chnl = np.array(
            bathtub_chnl + list(np.cumsum(self.jitter["channel"].hist_synth[: half_len + 1]))
        )
        bathtub_chnl = np.where(
            bathtub_chnl < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_chnl)),
            bathtub_chnl,
        )  # To avoid Chaco log scale plot wierdness.
        self.update_data("bathtub_chnl", np.log10(bathtub_chnl))
        #  - Tx
        bathtub_tx = list(np.cumsum(self.jitter["tx"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_tx.reverse()
        bathtub_tx = np.array(
            bathtub_tx + list(np.cumsum(self.jitter["tx"].hist_synth[: half_len + 1]))
        )
        bathtub_tx = np.where(
            bathtub_tx < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_tx)),
            bathtub_tx,
        )  # To avoid Chaco log scale plot wierdness.
        self.update_data("bathtub_tx", np.log10(bathtub_tx))
        #  - CTLE
        bathtub_ctle = list(np.cumsum(self.jitter["ctle"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_ctle.reverse()
        bathtub_ctle = np.array(
            bathtub_ctle + list(np.cumsum(self.jitter["ctle"].hist_synth[: half_len + 1]))
        )
        bathtub_ctle = np.where(
            bathtub_ctle < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_ctle)),
            bathtub_ctle,
        )  # To avoid Chaco log scale plot weirdness.
        self.update_data("bathtub_ctle", np.log10(bathtub_ctle))
        #  - DFE
        bathtub_dfe = list(np.cumsum(self.jitter["dfe"].hist_synth[-1 : -(half_len + 1) : -1]))
        bathtub_dfe.reverse()
        bathtub_dfe = np.array(
            bathtub_dfe + list(np.cumsum(self.jitter["dfe"].hist_synth[: half_len + 1]))
        )
        bathtub_dfe = np.where(
            bathtub_dfe < MIN_BATHTUB_VAL,
            0.1 * MIN_BATHTUB_VAL * np.ones(len(bathtub_dfe)),
            bathtub_dfe,
        )  # To avoid Chaco log scale plot weirdness.
        self.update_data("bathtub_dfe", np.log10(bathtub_dfe))

        # Eyes
        width = 2 * samps_per_ui
        xs = np.linspace(-ui * 1.0e12, ui * 1.0e12, width)
        height = 100
        y_max = 1.1 * max(abs(np.array(self.chnl_out)))
        eye_chnl = calc_eye(ui, samps_per_ui, height, self.chnl_out[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(np.array(self.rx_in)))
        eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
        y_max = 1.1 * max(abs(np.array(self.ctle_out)))
        eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max)
        i = 0
        while clock_times[i] <= ignore_until:
            i += 1
            assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
        y_max = 1.1 * max(abs(np.array(self.dfe_out)))
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out, y_max, clock_times[i:])
        self.update_data("eye_index", xs)
        self.update_data("eye_chnl", eye_chnl)
        self.update_data("eye_tx", eye_tx)
        self.update_data("eye_ctle", eye_ctle)
        self.update_data("eye_dfe", eye_dfe)

    def update_eyes(self):
        """
        Update the heat plots representing the eye diagrams.

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
        self.eyes.components[0].components[0].index.set_data(xs, ys)
        self.eyes.components[0].x_axis.mapper.range.low = xs[0]
        self.eyes.components[0].x_axis.mapper.range.high = xs[-1]
        self.eyes.components[0].y_axis.mapper.range.low = ys[0]
        self.eyes.components[0].y_axis.mapper.range.high = ys[-1]
        self.eyes.components[0].invalidate_draw()

        y_max = 1.1 * max(abs(np.array(self.rx_in)))
        ys = np.linspace(-y_max, y_max, height)
        self.eyes.components[1].components[0].index.set_data(xs, ys)
        self.eyes.components[1].x_axis.mapper.range.low = xs[0]
        self.eyes.components[1].x_axis.mapper.range.high = xs[-1]
        self.eyes.components[1].y_axis.mapper.range.low = ys[0]
        self.eyes.components[1].y_axis.mapper.range.high = ys[-1]
        self.eyes.components[1].invalidate_draw()

        y_max = 1.1 * max(abs(np.array(self.dfe_out)))
        ys = np.linspace(-y_max, y_max, height)
        self.eyes.components[3].components[0].index.set_data(xs, ys)
        self.eyes.components[3].x_axis.mapper.range.low = xs[0]
        self.eyes.components[3].x_axis.mapper.range.high = xs[-1]
        self.eyes.components[3].y_axis.mapper.range.low = ys[0]
        self.eyes.components[3].y_axis.mapper.range.high = ys[-1]
        self.eyes.components[3].invalidate_draw()

        self.eyes.components[2].components[0].index.set_data(xs, ys)
        self.eyes.components[2].x_axis.mapper.range.low = xs[0]
        self.eyes.components[2].x_axis.mapper.range.high = xs[-1]
        self.eyes.components[2].y_axis.mapper.range.low = ys[0]
        self.eyes.components[2].y_axis.mapper.range.high = ys[-1]
        self.eyes.components[2].invalidate_draw()

        self.eyes.request_redraw()
