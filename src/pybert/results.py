"""Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
result could be saved and later restored, as a reference waveform.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from pybert import __version__
from pybert.bert import SimulationPerf

RESULTS_FILEDIALOG_WILDCARD = "PyBERT Results (*.pybert_data);;All Files (*.*)"
"""This sets the supported file types in the GUI's save-as or loading dialog."""


@dataclass
class Results:
    """PyBERT simulation results data encapsulation class.

    This class is used to encapsulate that subset of the results data
    for a PyBERT instance, which is to be saved when the user clicks the
    "Save Results" button.
    """

    date_created: str = time.asctime()
    version: str = __version__
    performance: SimulationPerf = field(default_factory=SimulationPerf)
    results: dict = field(default_factory=dict)

    def save(self, filepath: Path | str):
        """Save all of the plot data out to a file.

        Args:
            filepath: The full filepath including the extension to save too.
        """
        with open(filepath, "wb") as the_file:
            pickle.dump(self, the_file)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "Results":
        """Recall all the results from a file and load them as reference plots.

        Confirms that the file actually exists and attempts to load back the
        graphs as reference plots in pybert.

        Args:
            filepath: The full filepath including the extension to save too.
            pybert: instance of the main app
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        # Right now the loads deserialize back into a `PyBertData` class.
        with open(filepath, "rb") as the_file:
            user_results = pickle.load(the_file)
        if not isinstance(user_results, cls):
            raise ValueError("The data structure read in is NOT of type: ArrayPlotData!")
        return user_results
