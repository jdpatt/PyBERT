"""
Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. 

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""


class WaveformData:
    """
    PyBERT simulation results data encapsulation class.

    This class is used to encapsulate that subset of the results
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Results" button.
    """

    _item_names = [
        "chnl_h",
        "tx_out_h",
        "ctle_out_h",
        "dfe_out_h",
        "chnl_s",
        "tx_s",
        "ctle_s",
        "dfe_s",
        "tx_out_s",
        "ctle_out_s",
        "dfe_out_s",
        "chnl_p",
        "tx_out_p",
        "ctle_out_p",
        "dfe_out_p",
        "chnl_H",
        "tx_H",
        "ctle_H",
        "dfe_H",
        "tx_out_H",
        "ctle_out_H",
        "dfe_out_H",
    ]

    def __init__(self, the_PyBERT):
        """
        Copy just that subset of the supplied PyBERT instance's
        'plotdata' attribute, which should be saved during pickling.
        """
        self.data_file = File("", entries=5, filter=["*.pybert_data"])
        plotdata = the_PyBERT.plotdata

        the_data = ArrayPlotData()

        for item_name in self._item_names:
            the_data.set_data(item_name, plotdata.get_data(item_name))

        self.the_data = the_data

    def save(self):
        """YAML out all the generated data."""
        the_pybert = info.object
        dlg = FileDialog(
            action="save as", wildcard="*.pybert_data", default_path=the_pybert.data_file
        )
        if dlg.open() == OK:
            plotdata = PyBertData(the_pybert)
            with open(dlg.path, "w") as the_file:
                yaml.dump(plotdata, the_file)
            the_pybert.data_file = dlg.path

    def load(self):
        """Read in the YAML data.'"""
        the_pybert = info.object
        dlg = FileDialog(
            action="open", wildcard="*.pybert_data", default_path=the_pybert.data_file
        )
        if dlg.open() == OK:
            with open(dlg.path, "r") as the_file:
                the_plotdata = yaml.full_load(the_file)
            if not isinstance(the_plotdata, PyBertData):
                raise Exception("The data structure read in is NOT of type: ArrayPlotData!")
            for prop, value in the_plotdata.the_data.arrays.items():
                the_pybert.plotdata.set_data(prop + "_ref", value)
            the_pybert.data_file = dlg.path

            # Add reference plots, if necessary.
            # - time domain
            for (container, suffix, has_both) in [
                (the_pybert.plots_h.component_grid.flat, "h", False),
                (the_pybert.plots_s.component_grid.flat, "s", True),
                (the_pybert.plots_p.component_grid.flat, "p", False),
            ]:
                if "Reference" not in container[0].plots:
                    (ix, prefix) = (0, "chnl")
                    item_name = prefix + "_" + suffix + "_ref"
                    container[ix].plot(
                        ("t_ns_chnl", item_name), type="line", color="darkcyan", name="Inc_ref"
                    )
                    for (ix, prefix) in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                        item_name = prefix + "_out_" + suffix + "_ref"
                        container[ix].plot(
                            ("t_ns_chnl", item_name),
                            type="line",
                            color="darkmagenta",
                            name="Cum_ref",
                        )
                    if has_both:
                        for (ix, prefix) in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                            item_name = prefix + "_" + suffix + "_ref"
                            container[ix].plot(
                                ("t_ns_chnl", item_name),
                                type="line",
                                color="darkcyan",
                                name="Inc_ref",
                            )

            # - frequency domain
            for (container, suffix, has_both) in [
                (the_pybert.plots_H.component_grid.flat, "H", True)
            ]:
                if "Reference" not in container[0].plots:
                    (ix, prefix) = (0, "chnl")
                    item_name = prefix + "_" + suffix + "_ref"
                    container[ix].plot(
                        ("f_GHz", item_name),
                        type="line",
                        color="darkcyan",
                        name="Inc_ref",
                        index_scale="log",
                    )
                    for (ix, prefix) in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                        item_name = prefix + "_out_" + suffix + "_ref"
                        container[ix].plot(
                            ("f_GHz", item_name),
                            type="line",
                            color="darkmagenta",
                            name="Cum_ref",
                            index_scale="log",
                        )
                    if has_both:
                        for (ix, prefix) in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                            item_name = prefix + "_" + suffix + "_ref"
                            container[ix].plot(
                                ("f_GHz", item_name),
                                type="line",
                                color="darkcyan",
                                name="Inc_ref",
                                index_scale="log",
                            )
