"""This file contains all the widgets that make up each major tab of the GUI."""
import pyqtgraph as pg

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

TITLES = [
    "Channel",
    "Channel + Tx Pre-emphasis",
    "Channel + Tx Pre-emphasis + CTLE",
    "Channel + Tx Pre-emphasis + CTLE + DFE",
]


class QuadGraphicsLayout(pg.GraphicsLayoutWidget):
    def __init__(self, _):
        super().__init__()
        self.y_axis = "left"
        self.x_axis = "bottom"

        self.channel = self.addPlot(
            row=0, col=0, title=TITLES[0], labels={"left": self.y_axis, "bottom": self.x_axis}
        )
        self.channel_tx = self.addPlot(
            row=0, col=1, title=TITLES[1], labels={"left": self.y_axis, "bottom": self.x_axis}
        )
        self.channel_ctle = self.addPlot(
            row=1, col=0, title=TITLES[2], labels={"left": self.y_axis, "bottom": self.x_axis}
        )
        self.channel_dfe = self.addPlot(
            row=1, col=1, title=TITLES[3], labels={"left": self.y_axis, "bottom": self.x_axis}
        )

    def enable_legend(self):
        """Enable the legend for all four plots."""
        self.channel.addLegend()
        self.channel_tx.addLegend()
        self.channel_ctle.addLegend()
        self.channel_dfe.addLegend()

    def set_axis_labels(self, y_axis, x_axis):
        """Set the Labels for all four plots."""
        self.y_axis = y_axis
        self.x_axis = x_axis

    def update_plots(self, x_data, y_data_chnl, y_data_tx, y_data_ctle, y_data_dfe):
        """Update the four plots in this view."""
        self.channel.plot(x_data, y_data_chnl, pen="b", clear=True)
        self.channel_tx.plot(x_data, y_data_tx, pen="b", clear=True)
        self.channel_ctle.plot(x_data, y_data_ctle, pen="b", clear=True)
        self.channel_dfe.plot(x_data, y_data_dfe, pen="b", clear=True)
