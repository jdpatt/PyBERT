"""
Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. 

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""
from pathlib import Path

import yaml
from PySide2.QtCore import QObject
from PySide2.QtWidgets import QFileDialog


class Waveforms:
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

    def __init__(self, parent):
        """
        Copy just that subset of the supplied PyBERT instance's
        'plotdata' attribute, which should be saved during pickling.
        """
        self.data_file = None
        self.parent = parent

    def save_to_file(self):
        """Yaml out the current waveforms."""
        if self.data_file:
            directory = self.data_file
        else:
            directory = ""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent,
            self.parent.tr("Save waveforms"),
            directory,
            self.parent.tr("PyBERT Data Files (*.data.yaml)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "w") as out_file:
                yaml.dump(self, out_file)
            self.data_file = filename

    def load_from_file(self):
        """Read in the YAML waveforms."""
        if self.data_file:
            directory = self.data_file
        else:
            directory = ""

        filename, _ = QFileDialog.getOpenFileName(
            self.parent,
            self.parent.tr("Load waveforms"),
            directory,
            self.parent.tr("PyBERT Data Files (*.data.yaml)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "r") as in_file:
                config = yaml.full_load(in_file)
