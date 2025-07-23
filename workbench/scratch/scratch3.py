import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Generate fake data: 1 second of a noisy sine wave
        t = np.linspace(0, 1, 10000)
        trace = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

        # Create a plot widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Plot the trace
        self.plot_widget.plot(t, trace, pen='y')

        # Enable interactive features (default)
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
        self.thresholdLine.hide()

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
                self.thresholdLine.setPos(y)
                self.thresholdLine.show()
                print(f"Threshold set to: {y:.4f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

