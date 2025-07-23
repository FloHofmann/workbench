import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Sampling parameters
        self.fs = 10000  # Hz
        self.t = np.linspace(0, 1, self.fs)
        self.trace = np.sin(2 * np.pi * 10 * self.t) + 0.3 * np.random.randn(len(self.t))
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

        # Clear layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.setParent(None)

        # Create thresholding plot
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget, 0, 0)

        # Plot the trace
        self.plot_widget.plot(self.t, self.trace, pen='y')
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True)

        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='w')
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen='w')
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        # Add threshold line (hidden initially)
        self.thresholdLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2))
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and self.state == 'threshold' and self.threshold is not None:
            print(f"Threshold locked at: {self.threshold:.4f}")
            self.showAnalysisPlots()

        elif event.key() == Qt.Key.Key_R and self.state == 'analysis':
            print("Returning to thresholding view.")
            self.setupThresholdingView()

    def showAnalysisPlots(self):
        self.state = 'analysis'

        # Clear layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.setParent(None)

        # Top: spikes across full width
        self.spikes_plot = pg.PlotWidget(title="Aligned Spikes")
        self.layout.addWidget(self.spikes_plot, 0, 0, 1, 2)  # row 0, col 0, span 1x2

        # Bottom left: PCA placeholder
        self.pca_plot = pg.PlotWidget(title="PCA Placeholder")
        self.layout.addWidget(self.pca_plot, 1, 0)

        # Bottom right: ISI histogram
        self.isi_plot = pg.PlotWidget(title="ISI Histogram")
        self.layout.addWidget(self.isi_plot, 1, 1)

        # Perform analysis
        self.detectAndPlotSpikes()

    def detectAndPlotSpikes(self):
        # Detect spikes
        above = np.where(self.trace > self.threshold)[0]
        refractory = int(0.001 * self.fs)  # 1 ms refractory
        peaks = []

        last_peak = -np.inf
        for idx in above:
            if idx - last_peak > refractory:
                peaks.append(idx)
                last_peak = idx

        peaks = np.array(peaks)
        print(f"Detected {len(peaks)} spikes")

        # Extract waveforms
        window = int(0.0015 * self.fs)  # 1.5 ms
        snippets = []
        for p in peaks:
            if p - window >= 0 and p + window < len(self.trace):
                snippets.append(self.trace[p-window:p+window+1])
        snippets = np.array(snippets)

        # Time vector for snippets
        snippet_t = np.linspace(-1.5, 1.5, snippets.shape[1])  # in ms

        # Plot all snippets
        for s in snippets:
            self.spikes_plot.plot(snippet_t, s, pen=pg.mkPen(0.5))

        self.spikes_plot.setLabel('bottom', 'Time (ms)')
        self.spikes_plot.setLabel('left', 'Voltage')

        # PCA placeholder
        self.pca_plot.plot([0, 1, 2], [0, 1, 0], pen=None, symbol='o')
        self.pca_plot.setLabel('bottom', 'PC1')
        self.pca_plot.setLabel('left', 'PC2')

        # ISI histogram
        if len(peaks) > 1:
            isis = np.diff(peaks) / self.fs * 1000  # in ms
            y, xedges = np.histogram(isis, bins=50, range=(0, 100))
            self.isi_plot.plot(
                xedges,
                y,
                stepMode=True,
                fillLevel=0,
                brush=(0,0,255,50)
            )
            self.isi_plot.setLabel('bottom', 'ISI (ms)')
            self.isi_plot.setLabel('left', 'Count')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

