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
        Item("optimizer", style="custom", label="Optimizer"),
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
