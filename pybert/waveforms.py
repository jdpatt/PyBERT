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

    def __init__(self, parent):
        """
        Copy just that subset of the supplied PyBERT instance's
        'plotdata' attribute, which should be saved during pickling.
        """
        self.data_file = None
        self.parent = parent

    def save_to_file(self, results):
        """yaml out the current waveforms."""
        if self.data_file:
            directory = self.data_file
        else:
            directory = ""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent,
            self.parent.tr("Save waveforms"),
            directory,
            self.parent.tr("PyBERT Data Files (*.pybert_data)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "w") as out_file:
                yaml.dump(results, out_file)
            self.data_file = filename

    def load_from_file(self):
        """Read in the yaml waveforms."""
        if self.data_file:
            directory = self.data_file
        else:
            directory = ""

        filename, _ = QFileDialog.getOpenFileName(
            self.parent,
            self.parent.tr("Load waveforms"),
            directory,
            self.parent.tr("PyBERT Data Files (*.pybert_data)"),
        )
        if filename:
            filename = Path(filename)
            with open(filename, "r") as in_file:
                results = yaml.full_load(in_file)
        return results
