"""This file contains all the widgets that make up each tab of the GUI."""
from PySide2.QtCore import *
from PySide2.QtWidgets import *


class ConfigWidget(QWidget):
    """This is where everything is setup and configured for the simulation."""

    def __init__(self, parent):
        super().__init__()
        self.title = "Config."


class DFEWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "DFE"


class EQTuneWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "EQ Tune"


class ImpulseWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Impulses"


class StepWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Steps"

class PulsesWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Pulses"

class FrequencyWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Frequency Responses"


class OutputWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Outputs"


class EyeDiagramWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Eyes"


class JitterDistributionsWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Jitter Dist."


class JitterSpectrumsWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Jitter Spec."


class BathtubCurvesWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Bathtubs"


class JitterInfoWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.title = "Jitter Info"


#  The order matters.  This is the order that they will be in the GUI.
#  If we add the ability to allow plugins or adatper pattern, this is probably an easy place
#  to insert additional tabs or content.
TABS = {
    "config_wig": ConfigWidget,
    "dfe_wig": DFEWidget,
    "eq_wig": EQTuneWidget,
    "impulse_wig": ImpulseWidget,
    "step_wig": StepWidget,
    "pulse_wig": PulsesWidget,
    "frequency_wig": FrequencyWidget,
    "output_wig": OutputWidget,
    "eyes_wig": EyeDiagramWidget,
    "jitter_dist_wig": JitterDistributionsWidget,
    "jitter_spect_wig": JitterSpectrumsWidget,
    "bathtub_wig": BathtubCurvesWidget,
    "jitter_info_wig": JitterInfoWidget,
}
