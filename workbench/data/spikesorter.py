# newly organized
import numpy as np
from scipy.signal import find_peaks
import sys
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from typing import Union
from scipy.signal import firwin, filtfilt, find_peaks
from sklearn.decomposition import PCA

from workbench.data.sorterhelp import PCARubberbandSelector

PathLike = Union[str, Path]


class spikesorter(QMainWindow):

    def __init__(self, *args):
        super().__init__()

        if args:
            data = args[0]
            self.__filter_trace(data)
        else:
            self.fs = 25000  # Hz
            self.t = np.linspace(0, 1, self.fs)
            self.trace = np.sin(
                2*np.pi*10*self.t) + 0.3 * np.random.randn(len(self.t))

        self.threshold = None

        # Create central widget
        self.central = QWidget()
        self.layout = QGridLayout(self.central)
        self.setCentralWidget(self.central)

        # call thresholding view
        self.setupThresholdingView()

    def mouseMoved(self, evt):
        """
        handle the movement of the mouse within the widget
        also draws the crosshair
        """

        pos = evt
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine.setPos(x)
            self.hLine.setPos(y)

    def mouseClicked(self, event):
        """
        handles mouse clicks inside the widgets
        """
        if self.state == 'threshold':
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.scenePos()
                vb = self.plot_widget.plotItem.vb
                if vb.sceneBoundingRect().contains(pos):
                    mousePoint = vb.mapSceneToView(pos)
                    y = mousePoint.y()
                    self.threshold = y
                    self.thresholdLine.setPos(y)
                    self.thresholdLine.show()
                    print(f"Threshold set to: {y:.4f}")
                else:
                    return
        else:
            pos = event.scenePos()
            spikes_vb = self.spikes_plot.plotItem.vb
            pca_vb = self.pca_plot.plotItem.vb
            if spikes_vb.sceneBoundingRect().contains(pos):
                mousePoint = spikes_vb.mapSceneToView(pos)

                x = mousePoint.x()
                y = mousePoint.y()

                # set timevector to find closest sample to the x value
                x_idx = np.argmin(np.abs(self.time_vector-x))

                if -.5 <= x <= 1.5:
                    self.undo_stack.append(self.filtered_mask.copy())
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.filtered_mask &= self.snippets[:, x_idx] >= y
                    elif event.button() == Qt.MouseButton.RightButton:
                        self.filtered_mask &= self.snippets[:, x_idx] <= y

                    self.updatePlots()

            elif pca_vb.sceneBoundingRect().contains(pos):
                mousePoint = pca_vb.mapSceneToView(pos)

                self.updatePlots()
            else:
                return

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and self.state == 'threshold' and self.threshold is not None:
            print(f"Threshold locked at: {self.threshold:.4f}")
            self.detectSpikes()
            self.showAnalysisPlots()

        elif event.key() == Qt.Key.Key_R and self.state == 'analysis':
            print("Returning to thresholding view.")
            self.setupThresholdingView()

        elif event.key() == Qt.Key.Key_S and self.state == 'analysis':
            self.saveSortedData()

        elif event.key() == Qt.Key.Key_G and self.state == 'analysis':
            self.removeThroughRoi()

        elif event.key() == Qt.Key.Key_E and self.state == 'analysis':
                    if hasattr(self, 'pca_selector'):
                        mask = self.pca_selector.get_mask()
                        self.undo_stack.append(self.filtered_mask.copy())
                        self.filtered_mask &= ~mask
                        print(f"Removed {np.sum(mask)} spikes via PCA selection.")
                        self.updatePlots()
                        self.pca_selector.clear()

        elif event.key() == Qt.Key.Key_B and self.state == 'analysis':
            if self.undo_stack:
                self.filtered_mask = self.undo_stack.pop()
                print("Undo: reverted to previous")
                self.updatePlots()
            else:
                print("Undo stack empty")

    def detectSpikes(self):
        self.state = 'analysis'
        # Parameters
        refractory_samples = int(0.001 * self.fs)  # 1 ms
        pre_window = round(0.0005 * self.fs)
        post_window = round(0.0015 * self.fs)
        prominence = 0.1                           # You can tune this

        # Spike detection using find_peaks
        peaks, _ = find_peaks(
            self.trace,
            height=self.threshold,
            distance=refractory_samples,
            prominence=prominence
        )

        print(f"Detected {len(peaks)} spikes")

        # Extract spike snippets
        snippets = []
        self.valid_peaks = []

        for p in peaks:
            if p - pre_window >= 0 and p + post_window < len(self.trace):
                snippet = self.trace[p - pre_window: p + post_window + 1]
                snippets.append(snippet)
                self.valid_peaks.append(p)

        snippets = np.array(snippets)
        self.snippets = snippets
        self.time_vector = np.linspace(-0.5,
                                       1.5, self.snippets.shape[1])
        self.filtered_mask = np.ones(snippets.shape[0], dtype=bool)

    def updatePlots(self):
        # insert the plots
        self.plotSpikeOverlay()
        self.plotPcaPlaceholder()
        self.plotIsi()

    def removeThroughRoi(self):
        print('removing datapoints within selected roi')

    def saveSortedData(self):
        print('saving data')

    def setupThresholdingView(self):
        self.state = 'threshold'
        # empty undo stack in case threshold is reset
        self.undo_stack = []
        # Clear layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.setParent(None)

        # Create thresholding plot
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget, 0, 0)

        # Plot the trace
        curve = pg.PlotCurveItem(
            self.t, self.trace, pen='y', downsample=20, clipToView=True)
        self.plot_widget.addItem(curve)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True)

        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='w')
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen='w')
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        # Add threshold line (hidden initially)
        self.thresholdLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.thresholdLine)
        if self.threshold is None:
            self.thresholdLine.hide()
        else:
            self.thresholdLine.setPos(self.threshold)
            self.thresholdLine.show()

        # Connect signals
        self.plot_widget.scene().sigMouseMoved.connect(self.mouseMoved)
        self.plot_widget.scene().sigMouseClicked.connect(self.mouseClicked)

    def showAnalysisPlots(self):
        # Clear layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.setParent(None)

        # Top: spikes across full width
        self.spikes_plot = pg.PlotWidget(title="Aligned Spikes")
        self.spikes_plot.plotItem.setMenuEnabled(False)
        self.spikes_plot.scene().sigMouseClicked.connect(self.mouseClicked)

        self.layout.addWidget(self.spikes_plot, 0, 0, 1,
                              2)  # row 0, col 0, span 1x2

        # Bottom left: PCA placeholder
        self.pca_plot = pg.PlotWidget(title="PCA Placeholder")
        self.pca_plot.setMouseEnabled(x=False, y=False)
        self.pca_plot.plotItem.setMenuEnabled(False)
        self.pca_plot.scene().sigMouseClicked.connect(self.mouseClicked)
        self.layout.addWidget(self.pca_plot, 1, 0)

        # Bottom right: ISI histogram
        self.isi_plot = pg.PlotWidget(title="ISI Histogram")
        self.isi_plot.plotItem.setMenuEnabled(False)
        self.layout.addWidget(self.isi_plot, 1, 1)

        # insert the plots
        self.plotSpikeOverlay()
        self.plotPcaPlaceholder()
        self.plotIsi()

    def plotSpikeOverlay(self):
        snippets = self.snippets[self.filtered_mask]
        if len(snippets) == 0:
            return

        # generate singular vector of appended snippets
        # snippets are separated in vector by nan
        # this allows to draw singular line, better for performance
        n_spikes, n_samples = snippets.shape
        snippets_nan = np.full((n_spikes, n_samples + 1), np.nan)
        snippets_nan[:, :-1] = snippets

        x_nan = np.full((n_spikes, n_samples + 1), np.nan)
        x_nan[:, :-1] = np.tile(self.time_vector, (n_spikes, 1))

        self.spikes_plot.clear()
        multi_curve = pg.PlotDataItem(
            x=x_nan.flatten(),
            y=snippets_nan.flatten(),
            pen=pg.mkPen((200, 200, 200, 50))
        )
        self.spikes_plot.addItem(multi_curve)
        self.spikes_plot.setLabel('bottom', 'Time (ms)')
        self.spikes_plot.setLabel('left', 'Voltage')

    def plotPcaPlaceholder(self):
        self.pca_data = PCA(n_components=2).fit_transform(
            self.snippets[self.filtered_mask])
        self.pca_plot.clear()

        self.pca_points = pg.ScatterPlotItem(
            x=self.pca_data[:, 0],
            y=self.pca_data[:, 1],
            pen=None,
            brush=(100, 100, 255, 100),
            size=5
        )
        self.pca_plot.addItem(self.pca_points)
        self.pca_plot.setLabel('bottom', 'PC1')
        self.pca_plot.setLabel('left', 'PC2')

        self.pca_selector = PCARubberbandSelector(
            pca_plot=self.pca_plot,
            pca_data=self.pca_data,
            on_selection_changed=self.updatePcaSelectionDisplay
        )

    def plotIsi(self):
        valid_peaks = np.array(self.valid_peaks)
        if len(valid_peaks) > 1:
            isis = np.diff(valid_peaks) / self.fs * 1000  # in ms
            y, xedges = np.histogram(isis, bins=50, range=(0, 100))
            self.isi_plot.clear()
            self.isi_plot.plot(
                xedges,
                y,
                stepMode=True,
                fillLevel=0,
                brush=(0, 0, 255, 50)
            )
            self.isi_plot.setLabel('bottom', 'ISI (ms)')
            self.isi_plot.setLabel('left', 'Count')

    def updatePcaSelectionDisplay(self, mask):
        brushes = [
            pg.mkBrush(255, 0, 0, 180) if selected else pg.mkBrush(100, 100, 255, 100)
            for selected in mask
        ]
        self.pca_points.setBrush(brushes)

    def __filter_trace(self, data):
        self.fs = 25000
        CUTOFF = 300
        N = 2 ** 8
        N |= 1
        filterb = firwin(N, CUTOFF, fs=self.fs, pass_zero=False)
        trace = data.raw_data['Ch1']['values']
        self.trace = filtfilt(filterb, 1, trace, axis=0)
        self.t = data.raw_data['Ch1']['times'][0]
        return self


if __name__ == '__main__':
    print('not meant to run as main')
