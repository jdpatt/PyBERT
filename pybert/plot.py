"""
Plot definitions for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 21, 2015 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""
from typing import Tuple

from chaco.api import ArrayPlotData, ColorMapper, GridPlotContainer, Plot
from chaco.tools.api import PanTool, ZoomTool

PLOT_SPACING = 20

TITLE_CHANNEL = "Channel"
TITLE_TX_CHANNEL = "Channel + Tx Preemphasis"
TITLE_CTLE_CHANNEL = "Channel + Tx Preemphasis + CTLE (+ AMI DFE)"
TITLE_DFE_CHANNEL = "Channel + Tx Preemphasis + CTLE (+ AMI DFE) + PyBERT DFE"


def create_dfe_adaption_plot(plotdata: ArrayPlotData, n_taps: int):
    """DFE Adaption gets updated and replaced per simulation run so it is returned seperately."""
    plot_dfe_adapt = Plot(
        plotdata,
        auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
        padding_left=75,
    )
    for tap in range(n_taps):
        plot_dfe_adapt.plot(
            ("tap_weight_index", f"tap{tap + 1}_weights"),
            type="line",
            color="auto",
            name=f"tap{tap + 1}",
        )
    plot_dfe_adapt.title = "DFE Adaptation"
    plot_dfe_adapt.tools.append(PanTool(plot_dfe_adapt, constrain=True, constrain_key=None, constrain_direction="x"))
    zoom9 = ZoomTool(plot_dfe_adapt, tool_mode="range", axis="index", always_on=False)
    plot_dfe_adapt.overlays.append(zoom9)
    plot_dfe_adapt.legend.visible = True
    plot_dfe_adapt.legend.align = "ul"
    return plot_dfe_adapt


def init_dfe_tab_plots(plotdata: ArrayPlotData, n_dfe_taps: int) -> Tuple[GridPlotContainer, Plot]:
    """Create the plots found under Results/DFE."""
    plot_cdr_adapt = Plot(plotdata, padding_left=75)
    plot_cdr_adapt.plot(("t_ns", "ui_ests"), type="line", color="blue")
    plot_cdr_adapt.title = "CDR Adaptation"
    plot_cdr_adapt.index_axis.title = "Time (ns)"
    plot_cdr_adapt.value_axis.title = "UI (ps)"

    plot_dfe_adapt = create_dfe_adaption_plot(plotdata, n_dfe_taps)

    plot_clk_per_hist = Plot(plotdata, padding_left=75)
    plot_clk_per_hist.plot(("clk_per_hist_bins", "clk_per_hist_vals"), type="line", color="blue")
    plot_clk_per_hist.title = "CDR Clock Period Histogram"
    plot_clk_per_hist.index_axis.title = "Clock Period (ps)"
    plot_clk_per_hist.value_axis.title = "Bin Count"

    plot_clk_per_spec = Plot(plotdata, padding_left=75)
    plot_clk_per_spec.plot(("clk_freqs", "clk_spec"), type="line", color="blue")
    plot_clk_per_spec.title = "CDR Clock Period Spectrum"
    plot_clk_per_spec.index_axis.title = "Frequency (bit rate)"
    plot_clk_per_spec.value_axis.title = "|H(f)| (dB mean)"
    plot_clk_per_spec.value_range.low_setting = -10
    zoom_clk_per_spec = ZoomTool(plot_clk_per_spec, tool_mode="range", axis="index", always_on=False)
    plot_clk_per_spec.overlays.append(zoom_clk_per_spec)

    container_dfe = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_dfe.add(plot_cdr_adapt)
    container_dfe.add(plot_dfe_adapt)
    container_dfe.add(plot_clk_per_hist)
    container_dfe.add(plot_clk_per_spec)

    return container_dfe, plot_dfe_adapt


def init_eq_tune_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create the plot on the Optimizer tab."""
    plot_h_tune = Plot(plotdata, padding_bottom=75)
    plot_h_tune.plot(("t_ns_chnl", "ctle_out_h_tune"), type="line", color="blue")
    plot_h_tune.plot(("t_ns_chnl", "clocks_tune"), type="line", color="gray")
    plot_h_tune.title = "Channel + Tx Preemphasis + CTLE (+ AMI DFE) + Ideal DFE"
    plot_h_tune.index_axis.title = "Time (ns)"
    plot_h_tune.y_axis.title = "Pulse Response (V)"
    zoom_tune = ZoomTool(plot_h_tune, tool_mode="range", axis="index", always_on=False)
    plot_h_tune.overlays.append(zoom_tune)
    return plot_h_tune


def init_impulse_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of impulse reponse plots after each stage under Responses/Impulse.

    Setup a zoom that will zoom all four in tandem.
    """
    container_h = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    first_plot = None

    for title, data, color in (
        (TITLE_CHANNEL, "chnl_h", "blue"),
        (TITLE_TX_CHANNEL, "tx_out_h", "red"),
        (TITLE_CTLE_CHANNEL, "ctle_out_h", "red"),
        (TITLE_DFE_CHANNEL, "dfe_out_h", "red"),
    ):
        plot_impulse = Plot(plotdata, padding_left=75)
        plot_impulse.plot(("t_ns_chnl", data), type="line", color=color, name="Cumulative")
        plot_impulse.title = title
        plot_impulse.index_axis.title = "Time (ns)"
        plot_impulse.y_axis.title = "Impulse Response (V/ns)"
        plot_impulse.legend.visible = True
        plot_impulse.legend.align = "ur"
        if data == "chnl_h":  # Setup the zoom for all four.
            zoom_h = ZoomTool(plot_impulse, tool_mode="range", axis="index", always_on=False)
            plot_impulse.overlays.append(zoom_h)
            first_plot = plot_impulse
        else:  # Zoom x-axes in tandem.
            plot_impulse.index_range = first_plot.index_range
        container_h.add(plot_impulse)
    return container_h


def init_step_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of step response plots after each stage under Responses/Steps.

    Setup a zoom that will zoom all four in tandem.
    """
    container_s = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    first_plot = None

    for title, incremental, cumulative in (
        (TITLE_CHANNEL, "chnl_s", None),
        (TITLE_TX_CHANNEL, "tx_s", "tx_out_s"),
        (TITLE_CTLE_CHANNEL, "ctle_s", "ctle_out_s"),
        (TITLE_DFE_CHANNEL, "dfe_s", "dfe_out_s"),
    ):
        plot_step = Plot(plotdata, padding_left=75)
        plot_step.plot(("t_ns_chnl", incremental), type="line", color="blue", name="Incremental")
        if cumulative:
            plot_step.plot(("t_ns_chnl", cumulative), type="line", color="red", name="Cumulative")
        plot_step.title = title
        plot_step.index_axis.title = "Time (ns)"
        plot_step.y_axis.title = "Step Response (V)"
        plot_step.legend.visible = True
        plot_step.legend.align = "lr"
        if incremental == "chnl_s":  # Setup the zoom for all four.
            zoom_h = ZoomTool(plot_step, tool_mode="range", axis="index", always_on=False)
            plot_step.overlays.append(zoom_h)
            first_plot = plot_step
        else:  # Zoom x-axes in tandem.
            plot_step.index_range = first_plot.index_range
        container_s.add(plot_step)
    return container_s


def init_pulse_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of pulse response plots after each stage under Responses/Pulses.

    Setup a zoom that will zoom all four in tandem.
    """
    container_p = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))

    first_plot = None

    for title, data, color in (
        (TITLE_CHANNEL, "chnl_p", "blue"),
        (TITLE_TX_CHANNEL, "tx_out_p", "red"),
        (TITLE_CTLE_CHANNEL, "ctle_out_p", "red"),
        (TITLE_DFE_CHANNEL, "dfe_out_p", "red"),
    ):
        plot_pulse = Plot(plotdata, padding_left=75)
        plot_pulse.plot(("t_ns_chnl", data), type="line", color=color, name="Cumulative")
        plot_pulse.title = title
        plot_pulse.index_axis.title = "Time (ns)"
        plot_pulse.y_axis.title = "Pulse Response (V)"
        plot_pulse.legend.visible = True
        plot_pulse.legend.align = "ur"
        if data == "chnl_p":  # Setup the zoom for all four.
            zoom_h = ZoomTool(plot_pulse, tool_mode="range", axis="index", always_on=False)
            plot_pulse.overlays.append(zoom_h)
            first_plot = plot_pulse
        else:  # Zoom x-axes in tandem.
            plot_pulse.index_range = first_plot.index_range
        container_p.add(plot_pulse)
    return container_p


def init_frequency_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of freq. response plots after each stage under Responses/Freq. Resp."""
    plot_H_chnl = Plot(plotdata, padding_left=75)
    plot_H_chnl.plot(("f_GHz", "chnl_H"), type="line", color="blue", name="Original Impulse", index_scale="log")
    plot_H_chnl.plot(("f_GHz", "chnl_trimmed_H"), type="line", color="red", name="Trimmed Impulse", index_scale="log")
    plot_H_chnl.title = TITLE_CHANNEL
    plot_H_chnl.index_axis.title = "Frequency (GHz)"
    plot_H_chnl.y_axis.title = "Frequency Response (dB)"
    plot_H_chnl.index_range.low_setting = 0.01
    plot_H_chnl.index_range.high_setting = 40.0
    plot_H_chnl.legend.visible = True
    plot_H_chnl.legend.align = "ll"

    plot_H_tx = Plot(plotdata, padding_left=75)
    plot_H_tx.plot(("f_GHz", "tx_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_tx.plot(("f_GHz", "tx_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_tx.title = TITLE_TX_CHANNEL
    plot_H_tx.index_axis.title = "Frequency (GHz)"
    plot_H_tx.y_axis.title = "Frequency Response (dB)"
    plot_H_tx.index_range.low_setting = 0.01
    plot_H_tx.index_range.high_setting = 40.0
    plot_H_tx.legend.visible = True
    plot_H_tx.legend.align = "ll"

    plot_H_ctle = Plot(plotdata, padding_left=75)
    plot_H_ctle.plot(("f_GHz", "ctle_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_ctle.plot(("f_GHz", "ctle_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_ctle.title = TITLE_CTLE_CHANNEL
    plot_H_ctle.index_axis.title = "Frequency (GHz)"
    plot_H_ctle.y_axis.title = "Frequency Response (dB)"
    plot_H_ctle.index_range.low_setting = 0.01
    plot_H_ctle.index_range.high_setting = 40.0
    plot_H_ctle.value_range.low_setting = -40.0
    plot_H_ctle.legend.visible = True
    plot_H_ctle.legend.align = "ll"

    plot_H_chnl.value_range = plot_H_ctle.value_range
    plot_H_tx.value_range = plot_H_ctle.value_range

    plot_H_dfe = Plot(plotdata, padding_left=75)
    plot_H_dfe.plot(("f_GHz", "dfe_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_dfe.plot(("f_GHz", "dfe_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_dfe.title = TITLE_DFE_CHANNEL
    plot_H_dfe.index_axis.title = "Frequency (GHz)"
    plot_H_dfe.y_axis.title = "Frequency Response (dB)"
    plot_H_dfe.index_range.low_setting = 0.01
    plot_H_dfe.index_range.high_setting = 40.0
    plot_H_dfe.value_range = plot_H_ctle.value_range
    plot_H_dfe.legend.visible = True
    plot_H_dfe.legend.align = "ll"

    container_H = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_H.add(plot_H_chnl)
    container_H.add(plot_H_tx)
    container_H.add(plot_H_ctle)
    container_H.add(plot_H_dfe)
    return container_H


def init_output_tab_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of time domain plots after each stage under Responses/Outputs."""
    container_out = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    first_plot = None

    for (title, data) in (
        (TITLE_CHANNEL, "chnl_out"),
        (TITLE_TX_CHANNEL, "tx_out"),
        (TITLE_CTLE_CHANNEL, "ctle_out"),
        (TITLE_DFE_CHANNEL, "dfe_out"),
    ):
        plot_output = Plot(plotdata, padding_left=75)
        plot_output.plot(("t_ns", data), type="line", color="blue")
        plot_output.title = title
        plot_output.index_axis.title = "Time (ns)"
        plot_output.y_axis.title = "Output (V)"
        if data == "chnl_out":  # Setup the zoom for all four.
            zoom_h = ZoomTool(plot_output, tool_mode="range", axis="index", always_on=False)
            plot_output.overlays.append(zoom_h)
            first_plot = plot_output
        else:  # Zoom x-axes in tandem.
            plot_output.index_range = first_plot.index_range
        container_out.add(plot_output)
    return container_out


def init_eye_diagram_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of eye diagram heatmaps after each stage under Responses/Eyes."""
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

    container_eye = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))

    for (title, data,) in (
        (TITLE_CHANNEL, "eye_chnl"),
        (TITLE_TX_CHANNEL, "eye_tx"),
        (TITLE_CTLE_CHANNEL, "eye_ctle"),
        (TITLE_DFE_CHANNEL, "eye_dfe"),
    ):
        plot_eye_diagram = Plot(plotdata, padding_left=75)
        plot_eye_diagram.img_plot(data, colormap=clr_map)
        plot_eye_diagram.y_direction = "normal"
        plot_eye_diagram.components[0].y_direction = "normal"
        plot_eye_diagram.title = title
        plot_eye_diagram.x_axis.title = "Time (ps)"
        plot_eye_diagram.x_axis.orientation = "bottom"
        plot_eye_diagram.y_axis.title = "Signal Level (V)"
        plot_eye_diagram.x_grid.visible = True
        plot_eye_diagram.y_grid.visible = True
        plot_eye_diagram.x_grid.line_color = "gray"
        plot_eye_diagram.y_grid.line_color = "gray"
        container_eye.add(plot_eye_diagram)
    return container_eye


def init_jitter_dist_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of jitter distribution after each stage under Jitter/Jitter Dist."""
    container_jitter_dist = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))

    for title, measured, extrapolated, in (
        (TITLE_CHANNEL, "jitter_chnl", "jitter_ext_chnl"),
        (TITLE_TX_CHANNEL, "jitter_tx", "jitter_ext_tx"),
        (TITLE_CTLE_CHANNEL, "jitter_ctle", "jitter_ext_ctle"),
        (TITLE_DFE_CHANNEL, "jitter_dfe", "jitter_ext_dfe"),
    ):
        plot_jitter_dist = Plot(plotdata, padding_left=75)
        plot_jitter_dist.plot(("jitter_bins", measured), type="line", color="blue", name="Measured")
        plot_jitter_dist.plot(("jitter_bins", extrapolated), type="line", color="red", name="Extrapolated")
        plot_jitter_dist.title = title
        plot_jitter_dist.index_axis.title = "Time (ps)"
        plot_jitter_dist.value_axis.title = "Count"
        plot_jitter_dist.legend.visible = True
        plot_jitter_dist.legend.align = "ur"
        container_jitter_dist.add(plot_jitter_dist)
    return container_jitter_dist


def init_jitter_spec_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of jitter distribution after each stage under Jitter/Jitter Spec."""

    plot_jitter_spec_chnl = Plot(plotdata)
    plot_jitter_spec_chnl.plot(("f_MHz", "jitter_spectrum_chnl"), type="line", color="blue", name="Total")
    plot_jitter_spec_chnl.plot(
        ("f_MHz", "jitter_ind_spectrum_chnl"), type="line", color="red", name="Data Independent"
    )
    plot_jitter_spec_chnl.plot(("f_MHz", "thresh_chnl"), type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_chnl.title = TITLE_CHANNEL
    plot_jitter_spec_chnl.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_chnl.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_chnl.tools.append(
        PanTool(plot_jitter_spec_chnl, constrain=True, constrain_key=None, constrain_direction="x")
    )
    zoom_jitter_spec_chnl = ZoomTool(plot_jitter_spec_chnl, tool_mode="range", axis="index", always_on=False)
    plot_jitter_spec_chnl.overlays.append(zoom_jitter_spec_chnl)
    plot_jitter_spec_chnl.legend.visible = True
    plot_jitter_spec_chnl.legend.align = "lr"

    plot_jitter_spec_tx = Plot(plotdata)
    plot_jitter_spec_tx.plot(("f_MHz", "jitter_spectrum_tx"), type="line", color="blue", name="Total")
    plot_jitter_spec_tx.plot(("f_MHz", "jitter_ind_spectrum_tx"), type="line", color="red", name="Data Independent")
    plot_jitter_spec_tx.plot(("f_MHz", "thresh_tx"), type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_tx.title = TITLE_TX_CHANNEL
    plot_jitter_spec_tx.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_tx.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_tx.value_range.low_setting = -40.0
    plot_jitter_spec_tx.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_tx.legend.visible = True
    plot_jitter_spec_tx.legend.align = "lr"

    plot_jitter_spec_chnl.value_range = plot_jitter_spec_tx.value_range

    plot_jitter_spec_ctle = Plot(plotdata)
    plot_jitter_spec_ctle.plot(("f_MHz", "jitter_spectrum_ctle"), type="line", color="blue", name="Total")
    plot_jitter_spec_ctle.plot(
        ("f_MHz", "jitter_ind_spectrum_ctle"), type="line", color="red", name="Data Independent"
    )
    plot_jitter_spec_ctle.plot(("f_MHz", "thresh_ctle"), type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_ctle.title = TITLE_CTLE_CHANNEL
    plot_jitter_spec_ctle.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_ctle.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_ctle.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_ctle.legend.visible = True
    plot_jitter_spec_ctle.legend.align = "lr"
    plot_jitter_spec_ctle.value_range = plot_jitter_spec_tx.value_range

    plot_jitter_spec_dfe = Plot(plotdata)
    plot_jitter_spec_dfe.plot(("f_MHz_dfe", "jitter_spectrum_dfe"), type="line", color="blue", name="Total")
    plot_jitter_spec_dfe.plot(
        ("f_MHz_dfe", "jitter_ind_spectrum_dfe"), type="line", color="red", name="Data Independent"
    )
    plot_jitter_spec_dfe.plot(("f_MHz_dfe", "thresh_dfe"), type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_dfe.title = TITLE_DFE_CHANNEL
    plot_jitter_spec_dfe.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_dfe.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_dfe.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_dfe.legend.visible = True
    plot_jitter_spec_dfe.legend.align = "lr"
    plot_jitter_spec_dfe.value_range = plot_jitter_spec_tx.value_range

    container_jitter_spec = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_jitter_spec.add(plot_jitter_spec_chnl)
    container_jitter_spec.add(plot_jitter_spec_tx)
    container_jitter_spec.add(plot_jitter_spec_ctle)
    container_jitter_spec.add(plot_jitter_spec_dfe)
    return container_jitter_spec


def init_bathtub_plots(plotdata: ArrayPlotData) -> GridPlotContainer:
    """Create a 4x4 grid of bathtub curves after each stage under Responses/Bathtubs"""

    container_bathtub = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))

    for title, data, in (
        (TITLE_CHANNEL, "bathtub_chnl"),
        (TITLE_TX_CHANNEL, "bathtub_tx"),
        (TITLE_CTLE_CHANNEL, "bathtub_ctle"),
        (TITLE_DFE_CHANNEL, "bathtub_dfe"),
    ):
        plot_bathtub = Plot(plotdata)
        plot_bathtub.plot(("jitter_bins", data), type="line", color="blue")
        plot_bathtub.value_range.high_setting = 0
        plot_bathtub.value_range.low_setting = -18
        plot_bathtub.value_axis.tick_interval = 3
        plot_bathtub.title = title
        plot_bathtub.index_axis.title = "Time (ps)"
        plot_bathtub.value_axis.title = "Log10(P(Transition occurs inside.))"
        container_bathtub.add(plot_bathtub)
    return container_bathtub
