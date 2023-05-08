"""Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from enable.component_editor import ComponentEditor
from pyface.image_resource import ImageResource
from traitsui.api import (  # CloseAction,
    Action,
    CheckListEditor,
    FileEditor,
    Group,
    HGroup,
    Item,
    Menu,
    MenuBar,
    NoButtons,
    ObjectColumn,
    Separator,
    TableEditor,
    TextEditor,
    VGroup,
    View,
    spring,
)

from pybert.gui.handler import MyHandler

# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
            HGroup(
                VGroup(
                    HGroup(  # Simulation Control
                        VGroup(
                            Item(
                                name="bit_rate",
                                label="Bit Rate (Gbps)",
                                tooltip="bit rate",
                                show_label=True,
                                enabled_when="True",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                            ),
                            Item(
                                name="nbits",
                                label="Nbits",
                                tooltip="# of bits to run",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="nspb",
                                label="Nspb",
                                tooltip="# of samples per bit",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="mod_type",
                                label="Modulation",
                                tooltip="line signalling/modulation scheme",
                                editor=CheckListEditor(values=[(0, "NRZ"), (1, "Duo-binary"), (2, "PAM-4")]),
                            ),
                        ),
                        VGroup(
                            Item(name="do_sweep", label="Do Sweep", tooltip="Run parameter sweeps."),
                            Item(
                                name="sweep_aves",
                                label="SweepAves",
                                tooltip="# of trials, per sweep, for averaging.",
                                enabled_when="do_sweep == True",
                            ),
                            HGroup(
                                Item(
                                    name="pattern",
                                    label="Pattern",
                                    tooltip="pattern to use to construct bit stream",
                                    # editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                                ),
                                Item(
                                    name="seed",
                                    label="Seed",
                                    tooltip="LFSR seed. 0 means new random seed for each run.",
                                ),
                            ),
                            Item(
                                name="eye_bits",
                                label="EyeBits",
                                tooltip="# of bits to use to form eye diagrams",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                        ),
                        VGroup(
                            Item(name="vod", label="Vod (V)", tooltip="Tx output voltage into matched load"),
                            Item(name="rn", label="Rn (V)", tooltip="standard deviation of random noise"),
                            Item(name="pn_mag", label="Pn (V)", tooltip="peak magnitude of periodic noise"),
                            Item(name="pn_freq", label="f(Pn) (MHz)", tooltip="frequency of periodic noise"),
                        ),
                    ),
                    label="Simulation Control",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="thresh",
                        label="Pj Threshold (sigma)",
                        tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)",
                    ),
                    Item(
                        name="impulse_length",
                        label="Impulse Response Length (ns)",
                        tooltip="Manual impulse response length override",
                    ),
                    Item(name="debug", label="Debug", tooltip="Enable to log extra information to console."),
                    label="Analysis Parameters",
                    show_border=True,
                ),
            ),
            HGroup(
                VGroup(
                    Item("tx", style="custom", show_label=False),
                    label="Tx",
                    show_border=True,
                ),
                VGroup(  # Interconnect
                    Item("channel", style="custom", show_label=False),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(
                    Item("rx", style="custom", show_label=False),
                    label="Rx",
                    show_border=True,
                ),
                label="Channel",
                show_border=True,
            ),
            # spring,
            label="Config.",
            id="config",
        ),
        # "Optimizer" tab.
        VGroup(
            HGroup(
                Group(
                    Item(
                        name="tx_tap_tuners",
                        editor=TableEditor(
                            columns=[
                                ObjectColumn(name="name", editable=False),
                                ObjectColumn(name="enabled"),
                                ObjectColumn(name="min_val"),
                                ObjectColumn(name="max_val"),
                                ObjectColumn(name="value", format="%+05.3f"),
                            ],
                            configurable=False,
                            reorderable=False,
                            sortable=False,
                            selection_mode="cell",
                            auto_size=False,
                            rows=4,
                            orientation="horizontal",
                            is_grid_cell=True,
                        ),
                        show_label=False,
                    ),
                    label="Tx Equalization",
                    show_border=True,
                    springy=True,
                ),
                # HGroup(
                VGroup(
                    HGroup(
                        Item(
                            name="peak_mag_tune",
                            label="CTLE: boost (dB)",
                            tooltip="CTLE peaking magnitude (dB)",
                            format_str="%4.1f",
                        ),
                        Item(
                            name="max_mag_tune",
                            label="Max boost (dB)",
                            tooltip="CTLE maximum peaking magnitude (dB)",
                            format_str="%4.1f",
                        ),
                    ),
                    HGroup(
                        Item(name="peak_freq_tune", label="fp (GHz)", tooltip="CTLE peaking frequency (GHz)"),
                        Item(
                            name="rx_bw_tune",
                            label="BW (GHz)",
                            tooltip="unequalized signal path bandwidth (GHz).",
                        ),
                    ),
                    HGroup(
                        Item(name="ctle_mode_tune", label="mode", tooltip="CTLE Operating Mode"),
                        Item(
                            name="ctle_offset_tune",
                            tooltip="CTLE d.c. offset (dB)",
                            show_label=False,
                            enabled_when='ctle_mode_tune == "Manual"',
                        ),
                    ),
                    HGroup(
                        Item(name="use_dfe_tune", label="DFE: Enable", tooltip="Include ideal DFE in optimization."),
                        Item(name="n_taps_tune", label="Taps", tooltip="Number of DFE taps."),
                    ),
                    label="Rx Equalization",
                    show_border=True,
                ),
                # ),
                VGroup(
                    Item(
                        name="max_iter",
                        label="Max. Iterations",
                        tooltip="Maximum number of iterations to allow, during optimization.",
                    ),
                    Item(
                        name="rel_opt",
                        label="Rel. Opt.:",
                        format_str="%7.4f",
                        tooltip="Relative optimization metric.",
                        style="readonly",
                    ),
                    Item(
                        name="przf_err",
                        label="PRZF Err.:",
                        format_str="%5.3f",
                        tooltip="Pulse Response Zero Forcing approximation error.",
                        style="readonly",
                    ),
                    label="Tuning Options",
                    show_border=True,
                ),
                springy=False,
            ),
            Item(
                label="Note: Only CTLE boost will be optimized; please, set peak frequency, bandwidth, and mode appropriately.",
            ),
            Item("plot_h_tune", editor=ComponentEditor(high_resolution=False), show_label=False, springy=True),
            HGroup(
                Item("btn_rst_eq", show_label=False, tooltip="Reset all values to those on the 'Config.' tab."),
                Item("btn_save_eq", show_label=False, tooltip="Store all values to 'Config.' tab."),
                Item("btn_opt_tx", show_label=False, tooltip="Run Tx tap weight optimization."),
                Item("btn_opt_rx", show_label=False, tooltip="Run Rx CTLE optimization."),
                Item("btn_coopt", show_label=False, tooltip="Run co-optimization."),
                Item("btn_abort", show_label=False, tooltip="Abort all optimizations."),
            ),
            label="Optimizer",
            id="eq_tune",
        ),
        Group(  # Responses
            Group(
                Item("plots_h", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Impulses",
                id="plots_h",
            ),
            Group(
                Item("plots_s", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Steps",
                id="plots_s",
            ),
            Group(
                Item("plots_p", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Pulses",
                id="plots_p",
            ),
            Group(
                Item("plots_H", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Freq. Resp.",
                id="plots_H",
            ),
            layout="tabbed",
            label="Responses",
            id="responses",
        ),
        Group(  # Results
            Group(
                Item("plots_dfe", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="DFE",
                id="plots_dfe",
            ),
            Group(
                Item("plots_out", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Outputs",
                id="plots_out",
            ),
            Group(
                Item("plots_eye", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Eyes",
                id="plots_eye",
            ),
            Group(
                Item("plots_bathtub", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Bathtubs",
                id="plots_bathtub",
            ),
            Group(Item("sweep_info", style="readonly", show_label=False), label="Sweep Info"),
            layout="tabbed",
            label="Results",
            id="results",
        ),
        Group(  # Jitter
            Group(
                Item("plots_jitter_dist", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Jitter Dist.",
                id="plots_jitter_dist",
            ),
            Group(
                Item("plots_jitter_spec", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Jitter Spec.",
                id="plots_jitter_spec",
            ),
            Group(Item("jitter_info", style="readonly", show_label=False), label="Jitter Info"),
            layout="tabbed",
            label="Jitter",
            id="jitter",
        ),
        Group(  # Info
            Group(
                Item("perf_info", style="readonly", show_label=False),
                label="Performance",
            ),
            Group(Item("instructions", style="readonly", show_label=False), label="User's Guide"),
            Group(Item("console_log", style="readonly", show_label=False), label="Console", id="console"),
            layout="tabbed",
            label="Info",
            id="info",
        ),
        layout="tabbed",
        springy=True,
        id="tabs",
    ),
    resizable=True,
    handler=MyHandler(),
    menubar=MenuBar(
        Menu(
            Action(name="Load Config.", action="do_load_cfg", accelerator="Ctrl+O"),
            Action(name="Load Results", action="do_load_data"),
            Separator(),
            Action(name="Save Config.", action="do_save_cfg", accelerator="Ctrl+S"),
            Action(name="Save Config. As...", action="do_save_cfg_as", accelerator="Ctrl+Shift+S"),
            Action(name="Save Results", action="do_save_data"),
            Separator(),
            Action(name="&Quit", action="close_app", accelerator="Ctrl+Q"),  # CloseAction()
            id="file",
            name="&File",
        ),
        Menu(
            Action(
                name="Debug Mode",
                action="toggle_debug_clicked",
                accelerator="Ctrl+`",
                style="toggle",
                checked_when="debug == True",
            ),
            Action(
                name="Clear Loaded Waveforms",
                action="do_clear_data",
            ),
            id="view",
            name="&View",
        ),
        Menu(
            Action(name="Run", action="do_run_simulation", accelerator="Ctrl+R"),
            Action(name="Abort", action="do_stop_simulation"),
            id="simulation",
            name="Simulation",
        ),
        Menu(
            Action(name="Getting Started", action="getting_started_clicked"),
            Action(name="&About", action="show_about_clicked"),
            id="help",
            name="&Help",
        ),
    ),
    buttons=NoButtons,
    statusbar="status_str",
    title="PyBERT",
    # width=0.95,
    # height=0.9,
    icon=ImageResource("icon.png"),
)
