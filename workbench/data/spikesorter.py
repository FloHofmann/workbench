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

PathLike = Union[str, Path]


class spikesorter(QMainWindow):

    def __init__(self, *args):
        super().__init__()

        if args:
            # ToDo Here's the next point of attack
            data = args[0]
            self.__filter_trace(data)
        else:
            self.fs = 25000  # Hz
            self.t = np.linspace(0, 1, self.fs)
            self.trace = np.sin(
                2 * np.pi * 10 * self.t) + 0.3 * np.random.randn(len(self.t))

        self.threshold = None

        # State: 'threshold' or 'analysis'
        self.state = 'threshold'

        # Create central widget
        self.central = QWidget()
        self.layout = QGridLayout(self.central)
        self.setCentralWidget(self.central)

        # Setup thresholding view initially
        self.setupThresholdingView()

    def setupThresholdingView(self):
        self.state = 'threshold'
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

    def mouseMoved(self, evt):
        pos = evt
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine.setPos(x)
            self.hLine.setPos(y)

    def mouseClicked(self, event):
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

        elif self.state == 'analysis':
            pos = event.scenePos()
            vb = self.spikes_plot.plotItem.vb
            pcab = self.pca_plot.plotItem.vb
            if vb.sceneBoundingRect().contains(pos):
                # spike overlay logic
                mousePoint = vb.mapSceneToView(pos)
                # todo this does not work yet, i need a workaround for finding the correct snippet to remove
                x = mousePoint.x()
                y = mousePoint.y()
                # reuse your snippet_t
                time_vector = np.linspace(-0.5, 1.5, self.snippets.shape[1])
                # find closest time index
                x_idx = np.argmin(np.abs(time_vector - x))

                if -.5 <= x <= 1.5:
                    self.undo_stack.append(self.filtered_mask.copy())
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.filtered_mask &= self.snippets[:, x_idx] >= y
                    elif event.button() == Qt.MouseButton.RightButton:
                        self.filtered_mask &= self.snippets[:, x_idx] <= y

                    self.updateSpikeOverlay()
            elif pcab.sceneBoundingRect().contains(pos):
                # PCA roi logic
                self.startPcaRoi()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and self.state == 'threshold' and self.threshold is not None:
            print(f"Threshold locked at: {self.threshold:.4f}")
            self.showAnalysisPlots()

        elif event.key() == Qt.Key.Key_R and self.state == 'analysis':
            print("Returning to thresholding view.")
            self.setupThresholdingView()
        elif event.key() == Qt.Key.Key_B and self.state == 'analysis':
            if self.undo_stack:
                self.filtered_mask = self.undo_stack.pop()
                print("Undo: reverted to previous")
                self.updateSpikeOverlay()
            else:
                print("Undo stack empty")

    def updateSpikeOverlay(self):
        snippets = self.snippets[self.filtered_mask]
        if len(snippets) == 0:
            return

        n_spikes, n_samples = snippets.shape
        snippets_nan = np.full((n_spikes, n_samples + 1), np.nan)
        snippets_nan[:, :-1] = snippets

        x_template = np.linspace(-0.5, 1.5, n_samples)
        x_nan = np.full((n_spikes, n_samples + 1), np.nan)
        x_nan[:, :-1] = np.tile(x_template, (n_spikes, 1))

        # clear and re-add overlay
        self.spikes_plot.clear()
        multi_curve = pg.PlotDataItem(
            x=x_nan.flatten(),
            y=snippets_nan.flatten(),
            pen=pg.mkPen((200, 200, 200, 50))
        )
        self.spikes_plot.addItem(multi_curve)

    def showAnalysisPlots(self):
        self.state = 'analysis'

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

        # Perform analysis
        self.detectAndPlotSpikes()

    def detectAndPlotSpikes(self):
        # Parameters
        refractory_samples = int(0.001 * self.fs)  # 1 ms
        pre_window = round(0.0005 * self.fs)
        post_window = round(0.0015 * self.fs)
        prominence = 0.1                           # You can tune this

        # Spike detection using find_peaks
        peaks, properties = find_peaks(
            self.trace,
            height=self.threshold,
            distance=refractory_samples,
            prominence=prominence
        )

        print(f"Detected {len(peaks)} spikes")

        # Extract spike snippets
        snippets = []
        valid_peaks = []

        for p in peaks:
            if p - pre_window >= 0 and p + post_window < len(self.trace):
                snippet = self.trace[p - pre_window: p + post_window + 1]
                snippets.append(snippet)
                valid_peaks.append(p)

        snippets = np.array(snippets)
        self.snippets = snippets
        self.filtered_mask = np.ones(snippets.shape[0], dtype=bool)

        # NaN-interleaved spike overlay
        n_spikes, n_samples = snippets.shape
        snippets_nan = np.full((n_spikes, n_samples + 1), np.nan)
        snippets_nan[:, :-1] = snippets

        # Time vector from -0.5 ms to +1.5 ms
        snippet_t = np.linspace(-pre_window / self.fs * 1000,
                                post_window/self.fs * 1000,
                                pre_window+post_window + 1)
        x_nan = np.full((n_spikes, n_samples + 1), np.nan)
        x_nan[:, :-1] = np.tile(snippet_t, (n_spikes, 1))

        self.spikes_plot.clear()
        multi_curve = pg.PlotDataItem(
            x=x_nan.flatten(),
            y=snippets_nan.flatten(),
            pen=pg.mkPen((200, 200, 200, 50))
        )
        self.spikes_plot.addItem(multi_curve)
        self.spikes_plot.setLabel('bottom', 'Time (ms)')
        self.spikes_plot.setLabel('left', 'Voltage')

        # PCA placeholder
        self.pca_data = PCA(n_components=2).fit_transform(self.snippets)
        self.pca_plot.plot(self.pca_data, pen=None, symbol='o')
        self.pca_plot.setLabel('bottom', 'PC1')
        self.pca_plot.setLabel('left', 'PC2')

        # ISI histogram
        valid_peaks = np.array(valid_peaks)
        if len(valid_peaks) > 1:
            isis = np.diff(valid_peaks) / self.fs * 1000  # in ms
            y, xedges = np.histogram(isis, bins=50, range=(0, 100))
            self.isi_plot.plot(
                xedges,
                y,
                stepMode=True,
                fillLevel=0,
                brush=(0, 0, 255, 50)
            )
            self.isi_plot.setLabel('bottom', 'ISI (ms)')
            self.isi_plot.setLabel('left', 'Count')

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
