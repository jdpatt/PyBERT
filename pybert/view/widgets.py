"""This file contains all the widgets that make up each major tab of the GUI."""
import pyqtgraph as pg
from pubsub import pub
from PySide2.QtCore import *
from PySide2.QtWidgets import *

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class ConfigWidget(QWidget):
    """This is where everything is setup and configured for the simulation."""

    def __init__(self):
        super().__init__()
        self.title = "Config."
        layout = QGridLayout()

        self.sim_control = QGroupBox(self.tr("Simulation Control"))
        self.channel = QGroupBox(self.tr("Channel Parameters"))
        self.tx = QGroupBox(self.tr("Tx Equalization"))
        self.rx = QGroupBox(self.tr("Rx Equalization"))
        self.cdr = QGroupBox(self.tr("CDR Parameters"))
        self.dfe = QGroupBox(self.tr("DFE Parameters"))

        layout.addWidget(self.sim_control, 0, 0)
        layout.addWidget(self.channel, 0, 1)
        layout.addWidget(self.tx, 1, 0)
        layout.addWidget(self.rx, 1, 1)
        layout.addWidget(self.cdr, 2, 0)
        layout.addWidget(self.dfe, 2, 1)
        self.setLayout(layout)


class DFEWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "DFE"
        layout = QGridLayout()

        self.cdr_adapt = pg.PlotWidget(
            title="CDR Adaptation", labels={"left": "UI (ps)", "bottom": "Time (ns)"}
        )
        self.dfe_adapt = pg.PlotWidget(title="DFE Adaptation")
        self.cdr_histo = pg.PlotWidget(
            title="CDR Clock Period Histogram",
            labels={"left": "Bin Count", "bottom": "Clock Period (ps)"},
        )
        self.cdr_spect = pg.PlotWidget(
            title="CDR Adaptation",
            labels={"left": "|H(f)| (dB mean)", "bottom": "Frequency (bit rate)"},
        )
        layout.addWidget(self.cdr_adapt, 0, 0)  # Upper Left
        layout.addWidget(self.dfe_adapt, 0, 1)  # Upper Right
        layout.addWidget(self.cdr_histo, 1, 0)  # Lower Left
        layout.addWidget(self.cdr_spect, 1, 1)  # Lower Right
        self.setLayout(layout)


class EQTuneWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "EQ Tune"
        layout = QGridLayout()
        self.tune = pg.PlotWidget(
            title="Channel + Tx Preemphasis + CTLE + Ideal DFE",
            labels={"left": "Post-CTLE Pulse Response (V)", "bottom": "Time (ns)"},
        )

        vbox = QVBoxLayout()
        self.txeq = QTableWidget(4, 5)
        self.txeq.setHorizontalHeaderLabels(["Name", "Enabled", "Min Val", "Max Val", "Value"])
        self.txeq.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vbox.addWidget(self.txeq)
        txeq_box = QGroupBox(self.tr("Tx Equalization"))
        txeq_box.setLayout(vbox)

        self.rxeq = QFormLayout()
        line = QLineEdit()
        line.setStatusTip(self.tr("CTLE peaking frequency (GHz)"))
        self.rxeq.addRow(self.tr("CTLE fp (GHz): "), line)
        line2 = QLineEdit()
        line2.setStatusTip(self.tr("Unequalized signal path bandwidth (GHz)"))
        self.rxeq.addRow(self.tr("Bandwidth (GHz): "), line2)
        line3 = QLineEdit()
        line3.setStatusTip(self.tr("CTLE peaking magnitude (dB)"))
        self.rxeq.addRow(self.tr("CTLE Boost (dB): "), line3)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(self.tr("CTLE Mode: ")))
        hbox.addWidget(QComboBox())
        line4 = QLineEdit()
        line4.setStatusTip(self.tr("CTLE D.C. offset (dB)"))
        hbox.addWidget(line4)
        self.rxeq.addRow(hbox)
        hbox = QHBoxLayout()
        check = QCheckBox()
        check.setStatusTip(self.tr("Include ideal DFE in optimization."))
        hbox.addWidget(QLabel(self.tr("Use DFE: ")))
        hbox.addWidget(check)
        hbox.addWidget(QLabel(self.tr("Taps: ")))
        line = QLineEdit()
        line.setStatusTip(self.tr("Number of ideal DFE taps."))
        hbox.addWidget(line)
        self.rxeq.addRow(hbox)
        self.rxeq.addRow(
            QLabel(
                self.tr(
                    "Note: Only peaking magnitude will be optimized; please set"
                    " peak fequency, bandwidth and mode appropriately."
                )
            )
        )
        rxeq_box = QGroupBox(self.tr("Rx Equalization"))
        rxeq_box.setLayout(self.rxeq)

        self.tune_options = QFormLayout()
        line = QLineEdit()
        line.setStatusTip(self.tr("Maximum number of iterations to allow, during optimization."))
        self.tune_options.addRow(self.tr("Max Iterations: "), line)
        line = QLabel()
        line.setStatusTip(self.tr("Relative optimization metric."))
        self.tune_options.addRow(self.tr("Relative Opt.: "), line)
        line = QLabel()
        line.setStatusTip(self.tr("Pulse Response Zero Forcing approximation error."))
        self.tune_options.addRow(self.tr("PRZF Error: "), line)
        tune_opt_box = QGroupBox(self.tr("Tunning Options"))
        tune_opt_box.setLayout(self.tune_options)

        # Setup the buttons and group the slots under one QButtonGroup
        self.tune_buttons = QButtonGroup()
        reset = QPushButton("&Reset", self)
        reset.setStatusTip(self.tr("Reset the Equalization back to Default."))
        save = QPushButton("&Save", self)
        save.setStatusTip(self.tr("Save the Equalization as Default."))
        opt_tx = QPushButton("&Opt Tx", self)
        opt_tx.setStatusTip(self.tr("Optimize the Tx Equalization."))
        opt_rx = QPushButton("&Opt Rx", self)
        opt_rx.setStatusTip(self.tr("Optimize the Rx Equalization."))
        opt_co = QPushButton("&Co-Opt", self)
        opt_co.setStatusTip(self.tr("Optimize the Tx & Rx Equalization together."))
        opt_abort = QPushButton("&Abort Opt", self)
        opt_abort.setStatusTip(self.tr("Abort the Current Equalization."))

        # Add the button to a group and set an ID; so that we don't have to parse their text
        self.tune_buttons.addButton(reset, 0)
        self.tune_buttons.addButton(save, 1)
        self.tune_buttons.addButton(opt_tx, 2)
        self.tune_buttons.addButton(opt_rx, 3)
        self.tune_buttons.addButton(opt_co, 4)
        self.tune_buttons.addButton(opt_abort, 5)

        self.tune_buttons.buttonClicked[int].connect(self.hanndle_button)

        # Actually add the buttons to the GUI
        self.buttons = QHBoxLayout()
        self.buttons.addWidget(reset)
        self.buttons.addWidget(save)
        self.buttons.addWidget(opt_tx)
        self.buttons.addWidget(opt_rx)
        self.buttons.addWidget(opt_co)
        self.buttons.addWidget(opt_abort)

        layout.addWidget(txeq_box, 0, 0)
        layout.addWidget(rxeq_box, 0, 1)
        layout.addWidget(tune_opt_box, 0, 2)
        layout.addWidget(self.tune, 1, 0, 1, -1)
        layout.addLayout(self.buttons, 2, 0)
        self.setLayout(layout)

    def hanndle_button(self, button_id):
        """Send a message based off which button is pressed."""
        if button_id == 0:  # Reset EQ
            pub.sendMessage("eq.reset")
        elif button_id == 1:  # Save EQ
            pub.sendMessage("eq.save")
        elif button_id == 2:  # Start Tx Opt
            pub.sendMessage("eq.start", opt="tx")
        elif button_id == 3:  # Start Rx Opt
            pub.sendMessage("eq.start", opt="rx")
        elif button_id == 4:  # Start Co Opt
            pub.sendMessage("eq.start", opt="co")
        elif button_id == 5:  # Abort Opt
            pub.sendMessage("eq.abort")


TITLES = [
    "Channel",
    "Channel + Tx Preemphasis",
    "Channel + Tx Preemphasis + CTLE",
    "Channel + Tx Preemphasis + CTLE + DFE",
]


class ImpulseWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Impulse Response (V/ns)", x_axis="Time (ns)"):
        super().__init__()
        self.title = "Impulses"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class StepWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Step Response (V)", x_axis="Time (ns)"):
        super().__init__()
        self.title = "Steps"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class PulsesWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Pulse Response (V)", x_axis="Time (ns)"):
        super().__init__()
        self.title = "Pulses"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class FrequencyWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Frequency Response (dB)", x_axis="Frequency (GHz)"):
        super().__init__()
        self.title = "Frequency Responses"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class OutputWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Output (V)", x_axis="Time (ns)"):
        super().__init__()
        self.title = "Outputs"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class EyeDiagramWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Signal Level (V)", x_axis="Time (ps)"):
        super().__init__()
        self.title = "Eyes"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class JitterDistributionsWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Count", x_axis="Time (ps)"):
        super().__init__()
        self.title = "Jitter Dist."

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class JitterSpectrumsWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="|FFT(TIE)| (dBui)", x_axis="Frequency (MHz)"):
        super().__init__()
        self.title = "Jitter Spec."

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class BathtubCurvesWidget(pg.GraphicsLayoutWidget):
    def __init__(self, y_axis="Log10(P(Transition occurs inside.))", x_axis="Time (ps)"):
        super().__init__()
        self.title = "Bathtubs"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": y_axis, "bottom": x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": y_axis, "bottom": x_axis}
        )


class JitterInfoWidget(QWidget):
    def __init__(self):
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
