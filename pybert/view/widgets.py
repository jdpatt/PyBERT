"""This file contains all the widgets that make up each major tab of the GUI."""
import pyqtgraph as pg

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
# pg.setConfigOption("leftButtonPan", False)

TITLES = [
    "Channel",
    "Channel + Tx Pre-emphasis",
    "Channel + Tx Pre-emphasis + CTLE",
    "Channel + Tx Pre-emphasis + CTLE + DFE",
]


class QuadGraphicsLayout(pg.GraphicsLayoutWidget):
    def __init__(self, _):
        super().__init__()
        self.channel = self.addPlot(row=0, col=0, title=TITLES[0])
        self.channel_tx = self.addPlot(row=0, col=1, title=TITLES[1])
        self.channel_ctle = self.addPlot(row=1, col=0, title=TITLES[2])
        self.channel_dfe = self.addPlot(row=1, col=1, title=TITLES[3])

    def enable_legends(self):
        """Enable the legend for all four plots."""
        self.channel.addLegend()
        self.channel_tx.addLegend()
        self.channel_ctle.addLegend()
        self.channel_dfe.addLegend()

    def enable_log_scale(self, x=False, y=False):
        """Enable log scale for both x and y for all four plots."""
        self.channel.setLogMode(x=x, y=y)
        self.channel_tx.setLogMode(x=x, y=y)
        self.channel_ctle.setLogMode(x=x, y=y)
        self.channel_dfe.setLogMode(x=x, y=y)

    def set_axis_labels(self, y_axis, x_axis):
        """Set the Labels for all four plots."""
        self.channel.setLabels(left=y_axis, bottom=x_axis)
        self.channel_tx.setLabels(left=y_axis, bottom=x_axis)
        self.channel_ctle.setLabels(left=y_axis, bottom=x_axis)
        self.channel_dfe.setLabels(left=y_axis, bottom=x_axis)

    def link_x_axes(self):
        self.channel_tx.setXLink(self.channel)
        self.channel_ctle.setXLink(self.channel)
        self.channel_dfe.setXLink(self.channel)

    def set_y_range(self, min_val, max_val):
        """Pass the arguments to each plot's setRange."""
        self.channel.setYRange(min_val, max_val)
        self.channel_tx.setYRange(min_val, max_val)
        self.channel_ctle.setYRange(min_val, max_val)
        self.channel_dfe.setYRange(min_val, max_val)

    def set_x_range(self, min_val, max_val):
        """Pass the arguments to each plot's setRange."""
        self.channel.setXRange(min_val, max_val)
        self.channel_tx.setXRange(min_val, max_val)
        self.channel_ctle.setXRange(min_val, max_val)
        self.channel_dfe.setXRange(min_val, max_val)
