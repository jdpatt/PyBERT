"""Jitter tab for PyBERT GUI.

This tab shows jitter analysis results including distributions, spectrums and statistics.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTextEdit
from PySide6.QtCore import Qt

import pyqtgraph as pg


class JitterTab(QWidget):
    """Tab for displaying jitter analysis results."""

    def __init__(self, parent=None):
        """Initialize the jitter tab.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget for different jitter views
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Create tabs for each view
        self.dist_tab = self._create_dist_tab()
        self.spec_tab = self._create_spec_tab()
        self.info_tab = self._create_info_tab()

        tab_widget.addTab(self.dist_tab, "Jitter Dist.")
        tab_widget.addTab(self.spec_tab, "Jitter Spec.")
        tab_widget.addTab(self.info_tab, "Jitter Info")

    def _create_dist_tab(self):
        """Create the jitter distributions tab.

        Returns:
            QWidget: Widget containing the distribution plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of distribution plots
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.dist_plots = []

        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("PDF")
            plot.getAxis("bottom").setLabel("Time", units="ps")
            plot.addLegend()

            # Add curves for total and data-independent jitter
            total_curve = plot.plot(pen=pg.mkPen("b"), name="Total")
            di_curve = plot.plot(pen=pg.mkPen("r"), name="Data-Ind.")

            self.dist_plots.append((total_curve, di_curve))

        return widget

    def _create_spec_tab(self):
        """Create the jitter spectrum tab.

        Returns:
            QWidget: Widget containing the spectrum plots
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create plot grid
        plot_grid = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_grid)

        # Create 2x2 grid of spectrum plots
        titles = [
            "Channel",
            "+ Tx De-emphasis & Noise",
            "+ CTLE (& IBIS-AMI DFE if apropos)",
            "+ PyBERT Native DFE if enabled",
        ]
        self.spec_plots = []

        for i in range(4):
            row = i // 2
            col = i % 2

            plot = plot_grid.addPlot(row=row, col=col)
            plot.showGrid(x=True, y=True)
            plot.setTitle(titles[i])
            plot.getAxis("left").setLabel("|FFT(TIE)|", units="dBui")
            plot.getAxis("bottom").setLabel("Frequency", units="MHz")
            plot.addLegend()

            # Add curves for total, data-independent and threshold
            total_curve = plot.plot(pen=pg.mkPen("b"), name="Total")
            di_curve = plot.plot(pen=pg.mkPen("r"), name="Data Independent")
            thresh_curve = plot.plot(pen=pg.mkPen("m"), name="Pj Threshold")

            self.spec_plots.append((total_curve, di_curve, thresh_curve))

            # Add pan/zoom tools
            plot.setMouseEnabled(x=True, y=True)

        return widget

    def _create_info_tab(self):
        """Create the jitter info tab.

        Returns:
            QWidget: Widget containing the jitter statistics
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Create text display for jitter statistics
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)

        return widget

    def update_dist_plots(self, jitter_bins, jitter_data, jitter_ext_data):
        """Update jitter distribution plots.

        Args:
            jitter_bins: Time points for distributions
            jitter_data: List of total jitter PDFs
            jitter_ext_data: List of data-independent jitter PDFs
        """
        for (total_curve, di_curve), total, di in zip(self.dist_plots, jitter_data, jitter_ext_data):
            total_curve.setData(jitter_bins, total)
            di_curve.setData(jitter_bins, di)

    def update_spec_plots(self, f_MHz, jitter_spectrum, jitter_ind_spectrum, thresh):
        """Update jitter spectrum plots.

        Args:
            f_MHz: Frequency points in MHz
            jitter_spectrum: List of total jitter spectrums
            jitter_ind_spectrum: List of data-independent jitter spectrums
            thresh: List of threshold curves
        """
        for (total_curve, di_curve, thresh_curve), total, di, th in zip(
            self.spec_plots, jitter_spectrum, jitter_ind_spectrum, thresh
        ):
            total_curve.setData(f_MHz, total)
            di_curve.setData(f_MHz, di)
            thresh_curve.setData(f_MHz, th)

    def update_info_text(self, info_text):
        """Update jitter statistics text.

        Args:
            info_text: String containing jitter statistics
        """
        self.info_text.setText(info_text)
