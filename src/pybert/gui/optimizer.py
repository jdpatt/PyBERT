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

OPTIMIZER_VIEW = VGroup(
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
            ))
