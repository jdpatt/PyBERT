"""Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
result could be saved and later restored, as a reference waveform.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from pybert import __version__

logger = logging.getLogger(__name__)

RESULTS_FILEDIALOG_WILDCARD = "PyBERT Results (*.pybert_data);;All Files (*.*)"
"""This sets the supported file types in the GUI's save-as or loading dialog."""


@dataclass
class SimulationPerfResults:
    """Performance metrics for the simulation."""

    start_time: float = 0.0
    end_time: float = 0.0
    channel: float = 0.0
    tx: float = 0.0
    ctle: float = 0.0
    dfe: float = 0.0
    jitter: float = 0.0
    plotting: float = 0.0
    total: float = 0.0

    def __str__(self):
        return (
            f"Performance Metrics: (Msmpls./min) "
            f"Channel: {self.channel * 6e-05:6.3f}  "
            f"Tx Preemphasis: {self.tx * 6e-05:6.3f}  "
            f"CTLE: {self.ctle * 6e-05:6.3f}  "
            f"DFE: {self.dfe * 6e-05:6.3f}  "
            f"Jitter: {self.jitter * 6e-05:6.3f}  "
            f"Plotting: {self.plotting * 6e-05:6.3f}  "
            f"Total: {self.total * 6e-05:6.3f}"
        )


@dataclass
class Results:
    """PyBERT simulation results data encapsulation class.

    This class is used to encapsulate that subset of the results data
    for a PyBERT instance, which is to be saved when the user clicks the
    "Save Results" button.
    """

    date_created: str = time.asctime()
    version: str = __version__
    performance: SimulationPerfResults = field(default_factory=SimulationPerfResults)
    data: dict = field(default_factory=dict)

    def save(self, filepath: Path | str):
        """Save all of the plot data out to a file.

        Args:
            filepath: The full filepath including the extension to save too.
        """
        try:
            with open(filepath, "wb") as the_file:
                pickle.dump(self, the_file)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to save results to file.")
            logger.exception(str(err))
        else:
            logger.error("No results to save. Please run a simulation first.")

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "Results | None":
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
        try:
            with open(filepath, "rb") as the_file:
                user_results = pickle.load(the_file)
            if not isinstance(user_results, cls):
                raise ValueError("The data structure read in is NOT of type: Configuration!")
            logger.info("Loaded results.")
            return user_results
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load results from file.")
            logger.exception(str(err))
            return None
