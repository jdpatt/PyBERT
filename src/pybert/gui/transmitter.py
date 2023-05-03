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


TX_VIEW = View(
                VGroup(
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
                                enabled_when="ami_model and ami_config",
                            ),
                            Item(
                                name="use_getwave",
                                label="Use GetWave",
                                tooltip="Use the model's GetWave() function.",
                                enabled_when="use_ami", # and has_getwave",
                            ),
                            Item(
                                "btn_ami_config",
                                show_label=False,
                                tooltip="Configure Tx AMI parameters.",
                                enabled_when="ami_model",
                            ),
                        ),
                    ),
                    label="IBIS",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="impedance",
                        label="Tx_Rs (Ohms)",
                        tooltip="Tx differential source impedance",
                    ),
                    Item(
                        name="capacitance",
                        label="Tx_Cout (pF)",
                        tooltip="Tx parasitic output capacitance (each pin)",
                        editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                        label="Native",
                        show_border=True,
                        enabled_when="use_ibis == False",
                        ),
                    VGroup(
                    # Item(
                    #     name="taps",
                    #     editor=TableEditor(
                    #         columns=[
                    #             ObjectColumn(name="name", editable=False),
                    #             ObjectColumn(name="enabled", style="simple"),
                    #             ObjectColumn(name="min_val", horizontal_alignment="center"),
                    #             ObjectColumn(name="max_val", horizontal_alignment="center"),
                    #             ObjectColumn(name="value", format="%+05.3f", horizontal_alignment="center"),
                    #             ObjectColumn(name="steps", horizontal_alignment="center"),
                    #         ],
                    #         configurable=False,
                    #         reorderable=False,
                    #         sortable=False,
                    #         selection_mode="cell",
                    #         # auto_size=True,
                    #         rows=4,
                    #     ),
                    #     show_label=False,
                    #     ),
                        label="Equalization",
                        show_border=True,
                        enabled_when="use_ami == False",
                    ),
                )
