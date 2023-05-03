from traitsui.api import FileEditor, HGroup, Item, VGroup, View, spring

CHANNEL_VIEW = View(
    VGroup(  # From File
        VGroup(
            HGroup(
                Item(
                    name="filepath",
                    label="File",
                    springy=True,
                    editor=FileEditor(dialog_style="open"),
                ),
            ),
            HGroup(
                Item(
                    name="use_ch_file",
                    label="Use file",
                ),
                spring,
                enabled_when="filepath",
                # Item(name="padded",   label="Zero-padded", enabled_when="use_ch_file == True"),
                # Item(name="windowed", label="Windowed",    enabled_when="use_ch_file == True"),
            ),
        ),
        HGroup(
            Item(
                name="f_step",
                label="f_step",
                tooltip="Frequency step to use in generating H(f).",
            ),
            Item(label="MHz"),
            enabled_when="use_ch_file == True",
        ),
        label="From File",
        show_border=True,
    ),
    HGroup(  # Native (i.e. - Howard Johnson's) interconnect model.
        VGroup(
            Item(
                name="length",
                label="Length (m)",
                tooltip="interconnect length",
            ),
            Item(
                name="loss_tangent",
                label="Loss Tan.",
                tooltip="dielectric loss tangent",
            ),
            Item(
                name="Z0",
                label="Z0 (Ohms)",
                tooltip="characteristic differential impedance",
            ),
            Item(
                name="v0",
                label="v_rel (c)",
                tooltip="normalized propagation velocity",
            ),
        ),
        VGroup(
            Item(
                name="dc_resistance",
                label="Rdc (Ohms)",
                tooltip="d.c. resistance",
            ),
            Item(
                name="w0",
                label="w0 (rads./s)",
                tooltip="transition frequency",
            ),
            Item(
                name="skin_resistance",
                label="R0 (Ohms)",
                tooltip="skin effect resistance",
            ),
        ),
        label="Native",
        show_border=True,
        enabled_when="use_ch_file == False",
    ),
)
