
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


RX_VIEW = View(VGroup(  VGroup(
                        HGroup(
                            Item(
                                name="ibis_filepath",
                                label="File",
                                springy=True,
                                editor=FileEditor(dialog_style="open", filter=["*.ibs"]),
                            )
                                                    ),
                        HGroup(
                            Item(name="use_ibis", label="Use IBIS"),
                            Item(name="btn_select_model", show_label=False),
                            Item(name="btn_view_model", show_label=False),
                            Item(
                                name="use_ondie_sparameters",
                                label="Use on-die S-parameters.",
                                enabled_when="use_ibis and ibis_model.has_ondie_sparameters",
                            ),
                            enabled_when="ibis_model",
                        ),

                        HGroup(
                            VGroup(
                                HGroup(
                                    Item(name="ami_filepath", label="AMI File:", style="readonly", springy=True),
                                    # Item(name="ami_valid", label="Valid", style="simple", enabled_when="False"),
                                ),
                                HGroup(
                                    Item(name="dll_filepath", label="DLL File:", style="readonly", springy=True),
                                    # Item(name="dll_valid", label="Valid", style="simple", enabled_when="False"),
                                ),
                            ),
                            VGroup(
                                Item(
                                    name="use_ami",
                                    label="Use AMI",
                                    tooltip="You must select both files, first.",
                                    enabled_when="ami_config and ami_model",
                                ),
                                Item(
                                    name="use_getwave",
                                    label="Use GetWave",
                                    tooltip="Use the model's GetWave() function.",
                                    enabled_when="use_ami and ami_config.has_getwave",
                                ),
                                Item(
                                    "btn_ami_config",
                                    show_label=False,
                                    tooltip="Configure Rx AMI parameters.",
                                    enabled_when="ami_config",
                                ),
                            ),
                        ),
                        ),

                        label="IBIS",
                        show_border=True,
                    ),
                    VGroup(
                        Item(
                            name="resistance",
                            label="Rx_Rin (Ohms)",
                            tooltip="Rx differential input impedance",
                        ),
                        Item(
                            name="capacitance",
                            label="Rx_Cin (pF)",
                            tooltip="Rx parasitic input capacitance (each pin)",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(
                            name="coupling_capacitance",
                            label="Rx_Cac (uF)",
                            tooltip="Rx a.c. coupling capacitance (each pin)",
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="use_ibis == False",
                    ),
                    VGroup(
                    # VGroup(
                    #     VGroup(
                    #         VGroup(
                    #             HGroup(
                    #                 Item(
                    #                     name="use_ctle_file",
                    #                     label="fromFile",
                    #                     tooltip="Select CTLE impulse/step response from file.",
                    #                 ),
                    #                 Item(
                    #                     name="ctle_file",
                    #                     label="Filename",
                    #                     enabled_when="use_ctle_file == True",
                    #                     editor=FileEditor(dialog_style="open"),
                    #                 ),
                    #             ),
                    #             HGroup(
                    #                 Item(
                    #                     name="peak_freq",
                    #                     label="CTLE fp (GHz)",
                    #                     tooltip="CTLE peaking frequency (GHz)",
                    #                     enabled_when="use_ctle_file == False",
                    #                 ),
                    #                 Item(
                    #                     name="rx_bw",
                    #                     label="Bandwidth (GHz)",
                    #                     tooltip="unequalized signal path bandwidth (GHz).",
                    #                     enabled_when="use_ctle_file == False",
                    #                 ),
                    #             ),
                    #             HGroup(
                    #                 Item(
                    #                     name="peak_mag",
                    #                     label="CTLE boost (dB)",
                    #                     tooltip="CTLE peaking magnitude (dB)",
                    #                     format_str="%4.1f",
                    #                     enabled_when="use_ctle_file == False",
                    #                 ),
                    #                 Item(
                    #                     name="ctle_mode",
                    #                     label="CTLE mode",
                    #                     tooltip="CTLE Operating Mode",
                    #                     enabled_when="use_ctle_file == False",
                    #                 ),
                    #                 Item(
                    #                     name="ctle_offset",
                    #                     tooltip="CTLE d.c. offset (dB)",
                    #                     show_label=False,
                    #                     enabled_when='ctle_mode == "Manual"',
                    #                 ),
                    #             ),
                    #             label="CTLE",
                    #             show_border=True,
                    #             enabled_when="rx_use_ami == False",
                    #         ),
                    #     ),
                    #     HGroup(
                    #         VGroup(
                    #             Item(
                    #                 name="delta_t",
                    #                 label="Delta-t (ps)",
                    #                 tooltip="magnitude of CDR proportional branch",
                    #             ),
                    #             Item(name="alpha", label="Alpha", tooltip="relative magnitude of CDR integral branch"),
                    #             Item(
                    #                 name="n_lock_ave",
                    #                 label="Lock Nave.",
                    #                 tooltip="# of UI estimates to average, when determining lock",
                    #             ),
                    #             Item(
                    #                 name="rel_lock_tol",
                    #                 label="Lock Tol.",
                    #                 tooltip="relative tolerance for determining lock",
                    #             ),
                    #             Item(
                    #                 name="lock_sustain",
                    #                 label="Lock Sus.",
                    #                 tooltip="length of lock determining hysteresis vector",
                    #             ),
                    #             label="CDR",
                    #             show_border=True,
                    #         ),
                    #         VGroup(
                    #             HGroup(
                    #                 Item(
                    #                     name="use_dfe",
                    #                     label="Use DFE",
                    #                     tooltip="Include DFE in simulation.",
                    #                 ),
                    #                 Item(
                    #                     name="sum_ideal",
                    #                     label="Ideal",
                    #                     tooltip="Use ideal DFE. (performance boost)",
                    #                     enabled_when="use_dfe == True",
                    #                 ),
                    #             ),
                    #             VGroup(
                    #                 Item(name="n_taps", label="Taps", tooltip="# of taps"),
                    #                 Item(name="gain", label="Gain", tooltip="error feedback gain"),
                    #                 Item(name="decision_scaler", label="Level", tooltip="target output magnitude"),
                    #                 Item(
                    #                     name="n_ave", label="Nave.", tooltip="# of CDR adaptations per DFE adaptation"
                    #                 ),
                    #                 Item(
                    #                     name="sum_bw",
                    #                     label="BW (GHz)",
                    #                     tooltip="summing node bandwidth",
                    #                     enabled_when="sum_ideal == False",
                    #                 ),
                    #                 enabled_when="use_dfe == True",
                    #             ),
                    #             label="DFE",
                    #             show_border=True,
                    #         ),
                    #     ),
                    #     label="Native",
                    #     show_border=True,
                    #     enabled_when="rx_use_ami == False",
                    # ),
                    label="Equalization",
                    show_border=True,))
