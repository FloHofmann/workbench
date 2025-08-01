from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QRubberBand
from PyQt6.QtCore import Qt, QRect, QSize, QPointF, QObject
from typing import Union
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, firwin, filtfilt
from sklearn.decomposition import PCA
import pyqtgraph as pg

PathLike = Union[str, Path]


class PcaRubberbandSelector(QObject):
    def __init__(self, plot_widget: pg.PlotWidget, parent):
        super().__init__()
        self.plot_widget = plot_widget
        self.parent = parent
        self.viewport = self.plot_widget.viewport()
        self.rubberband = QRubberBand(
            QRubberBand.Shape.Rectangle, self.viewport)
        self.origin = None
        self.active = False
        self.viewport.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is not self.viewport:
            return False

        if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubberband.setGeometry(QRect(self.origin, QSize()))
            self.rubberband.show()
            self.active = True
            return True

        elif event.type() == event.Type.MouseMove and self.active:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubberband.setGeometry(rect)
            return True

        elif event.type() == event.Type.MouseButtonRelease and self.active:
            self.rubberband.hide()
            self.active = False

            rb_rect = self.rubberband.geometry()
            scene_top_left = self.plot_widget.mapToScene(rb_rect.topLeft())
            scene_bottom_right = self.plot_widget.mapToScene(
                rb_rect.bottomRight())

            p1 = self.plot_widget.plotItem.vb.mapSceneToView(scene_top_left)
            p2 = self.plot_widget.plotItem.vb.mapSceneToView(
                scene_bottom_right)

            x0, x1 = sorted([p1.x(), p2.x()])
            y0, y1 = sorted([p1.y(), p2.y()])

            epsilon = 1e-6
            x0 += epsilon
            x1 -= epsilon
            y0 += epsilon
            y1 -= epsilon

            mask = (
                (self.parent.pca_data[:, 0] >= x0) & (self.parent.pca_data[:, 0] <= x1) &
                (self.parent.pca_data[:, 1] >= y0) & (
                    self.parent.pca_data[:, 1] <= y1)
            )
            self.parent.pca_selector_mask = mask
            self.parent.updatePcaSelectionDisplay(mask)
            return True

        return False


class SpikeSorter(QMainWindow):
    def __init__(self, *args):
        super().__init__()
        self.fs = 25000
        if args:
            self.__filter_trace(args[0])
        else:
            self.t = np.linspace(0, 1, self.fs)
            self.trace = np.sin(2 * np.pi * 10 * self.t) + \
                0.3 * np.random.randn(len(self.t))

        self.threshold = None
        self.central = QWidget()
        self.layout = QGridLayout(self.central)
        self.setCentralWidget(self.central)
        self.setupThresholdingView()

    def __filter_trace(self, data):
        CUTOFF = 300
        N = 2 ** 8 | 1
        filterb = firwin(N, CUTOFF, fs=self.fs, pass_zero=False)
        self.trace = filtfilt(
            filterb, 1, data.raw_data['Ch1']['values'], axis=0)
        self.t = data.raw_data['Ch1']['times'][0]

    def setupThresholdingView(self):
        self.state = 'threshold'
        self.undo_stack = []
        self._clearLayout()

        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget, 0, 0)

        curve = pg.PlotCurveItem(
            self.t, self.trace, pen='y', downsample=20, clipToView=True)
        self.plot_widget.addItem(curve)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True)

        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='w')
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen='w')
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        self.thresholdLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.thresholdLine)
        if self.threshold is None:
            self.thresholdLine.hide()
        else:
            self.thresholdLine.setPos(self.threshold)
            self.thresholdLine.show()

        self.plot_widget.scene().sigMouseMoved.connect(self.onMouseMoved)
        self.plot_widget.scene().sigMouseClicked.connect(self.onMouseClicked)

    def detectSpikes(self):
        self.state = 'analysis'
        refractory = int(0.001 * self.fs)
        pre, post = round(0.0005 * self.fs), round(0.0015 * self.fs)

        peaks, _ = find_peaks(self.trace, height=self.threshold,
                              distance=refractory, prominence=0.1)
        print(f"Detected {len(peaks)} spikes")

        snippets, self.valid_peaks = [], []
        for p in peaks:
            if p - pre >= 0 and p + post < len(self.trace):
                snippets.append(self.trace[p - pre: p + post + 1])
                self.valid_peaks.append(p)

        self.snippets = np.array(snippets)
        self.time_vector = np.linspace(-0.5, 1.5, self.snippets.shape[1])
        self.filtered_mask = np.ones(self.snippets.shape[0], dtype=bool)

    def showAnalysisPlots(self):
        self._clearLayout()

        self.spikes_plot = pg.PlotWidget(title="Aligned Spikes")
        self.spikes_plot.plotItem.setMenuEnabled(False)
        self.spikes_plot.scene().sigMouseClicked.connect(self.onMouseClicked)
        self.layout.addWidget(self.spikes_plot, 0, 0, 1, 2)

        self.pca_plot = pg.PlotWidget(title="PCA Placeholder")
        self.pca_plot.setMouseEnabled(False, False)
        self.pca_plot.plotItem.setMenuEnabled(False)
        self.pca_plot.scene().sigMouseClicked.connect(self.onMouseClicked)
        self.layout.addWidget(self.pca_plot, 1, 0)

        self.pca_selector = PcaRubberbandSelector(self.pca_plot, self)
        self.pca_selector_mask = np.zeros_like(self.filtered_mask, dtype=bool)

        self.isi_plot = pg.PlotWidget(title="ISI Histogram")
        self.isi_plot.plotItem.setMenuEnabled(False)
        self.layout.addWidget(self.isi_plot, 1, 1)

        self.updatePlots()

    def updatePlots(self):
        self.plotSpikeOverlay()
        self.plotPca()
        self.plotIsi()

    def plotSpikeOverlay(self):
        snippets = self.snippets[self.filtered_mask]
        if len(snippets) == 0:
            return

        n_spikes, n_samples = snippets.shape
        snippets_nan = np.full((n_spikes, n_samples + 1), np.nan)
        x_nan = np.full((n_spikes, n_samples + 1), np.nan)

        snippets_nan[:, :-1] = snippets
        x_nan[:, :-1] = np.tile(self.time_vector, (n_spikes, 1))

        self.spikes_plot.clear()
        self.spikes_plot.addItem(pg.PlotDataItem(
            x=x_nan.flatten(), y=snippets_nan.flatten(), pen=pg.mkPen((255, 255, 0, 100))
        ))
        self.spikes_plot.setLabel('bottom', 'Time (ms)')
        self.spikes_plot.setLabel('left', 'Voltage')

    def plotPca(self):
        self.pca_indices = np.flatnonzero(self.filtered_mask)
        self.pca_data = PCA(n_components=2).fit_transform(
            self.snippets[self.filtered_mask])

        self.pca_plot.clear()
        self.pca_points = pg.ScatterPlotItem(
            x=self.pca_data[:, 0], y=self.pca_data[:, 1], pen=None,
            brush=pg.mkBrush(0, 255, 255, 180), size=5
        )
        self.pca_plot.addItem(self.pca_points)
        self.pca_plot.setLabel('bottom', 'PC1')
        self.pca_plot.setLabel('left', 'PC2')

    def plotIsi(self):
        peaks = np.array(self.valid_peaks)
        if len(peaks) > 1:
            isis = np.diff(peaks) / self.fs * 1000
            y, xedges = np.histogram(isis, bins=50, range=(0, 100))
            self.isi_plot.clear()
            self.isi_plot.plot(xedges, y, stepMode=True,
                               fillLevel=0, brush=(0, 0, 255, 50))
            self.isi_plot.setLabel('bottom', 'ISI (ms)')
            self.isi_plot.setLabel('left', 'Count')

    def updatePcaSelectionDisplay(self, mask):
        brushes = [pg.mkBrush(255, 0, 0, 180) if m else pg.mkBrush(
            100, 100, 255, 100) for m in mask]
        self.pca_points.setBrush(brushes)

    def onMouseMoved(self, pos):
        if self.state != 'threshold':
            return
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            point = vb.mapSceneToView(pos)
            self.vLine.setPos(point.x())
            self.hLine.setPos(point.y())

    def onMouseClicked(self, event):
        if self.state == 'threshold' and event.button() == Qt.MouseButton.LeftButton:
            vb = self.plot_widget.plotItem.vb
            if vb.sceneBoundingRect().contains(event.scenePos()):
                y = vb.mapSceneToView(event.scenePos()).y()
                self.threshold = y
                self.thresholdLine.setPos(y)
                self.thresholdLine.show()
                print(f"Threshold set to: {y:.4f}")

        elif self.state == 'analysis':
            pos = event.scenePos()
            spikes_vb = self.spikes_plot.plotItem.vb
            if spikes_vb.sceneBoundingRect().contains(pos):
                mousePoint = spikes_vb.mapSceneToView(pos)
                x = mousePoint.x()
                y = mousePoint.y()

                if -0.5 <= x <= 1.5:
                    x_idx = np.argmin(np.abs(self.time_vector - x))
                    self.undo_stack.append(self.filtered_mask.copy())

                    if event.button() == Qt.MouseButton.LeftButton:
                        self.filtered_mask &= self.snippets[:, x_idx] >= y
                    elif event.button() == Qt.MouseButton.RightButton:
                        self.filtered_mask &= self.snippets[:, x_idx] <= y

                    self.updatePlots()
            return

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Return and self.state == 'threshold' and self.threshold is not None:
            print(f"Threshold locked at: {self.threshold:.4f}")
            self.detectSpikes()
            self.showAnalysisPlots()

        elif key == Qt.Key.Key_E and self.state == 'analysis':
            local_mask = self.pca_selector_mask
            global_mask = np.zeros_like(self.filtered_mask, dtype=bool)
            global_mask[self.pca_indices] = local_mask
            self.undo_stack.append(self.filtered_mask.copy())
            self.filtered_mask &= ~global_mask
            print(f"Removed {np.sum(local_mask)} spikes via PCA selection.")
            self.pca_selector_mask[:] = False
            self.updatePlots()

        elif key == Qt.Key.Key_R and self.state == 'analysis':
            self.setupThresholdingView()

        elif key == Qt.Key.Key_B and self.state == 'analysis' and self.undo_stack:
            self.filtered_mask = self.undo_stack.pop()
            print("Undo: reverted to previous")
            self.updatePlots()

        elif key == Qt.Key.Key_S and event.modifiers() and Qt.KeyboardModifier.ControlModifier and self.state == 'analysis':
            self.saveStoredData()

    def saveStoredData(self):
        print('save stored data')
        self.close()

    def _clearLayout(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.setParent(None)


if __name__ == '__main__':
    print('not meant to run as main')
