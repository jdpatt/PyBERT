from traitsui.api import CheckListEditor, HGroup, Item, TextEditor, VGroup, View

CONTROL_VIEW = View(
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
            label="Analysis Parameters",
            show_border=True,
        ),
    )
)
