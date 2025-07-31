from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np


class RubberBandViewBox(pg.ViewBox):
    def __init__(self, scatter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scatter = scatter
        self.rubberBand = QtWidgets.QGraphicsRectItem()
        self.rubberBand.setPen(QtGui.QPen(
            QtCore.Qt.GlobalColor.red, 1, QtCore.Qt.PenStyle.DashLine))
        self.rubberBand.setBrush(QtGui.QBrush(
            QtCore.Qt.GlobalColor.transparent))
        self.addItem(self.rubberBand)
        self.rubberBand.setZValue(1000)
        self.origin = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubberBand.setRect(QtCore.QRectF(self.origin, self.origin))
            self.rubberBand.show()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDragEvent(self, event):
        if self.origin is not None:
            rect = QtCore.QRectF(self.origin, event.pos()).normalized()
            self.rubberBand.setRect(rect)
            event.accept()
        else:
            super().mouseDragEvent(event)

    def mouseReleaseEvent(self, event):
        if self.origin is not None:
            rect = QtCore.QRectF(self.origin, event.pos()).normalized()
            self.rubberBand.hide()
            self.selectPointsInRect(rect)
            self.origin = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def selectPointsInRect(self, rect):
        p1 = self.mapSceneToView(rect.topLeft())
        p2 = self.mapSceneToView(rect.bottomRight())
        xmin, xmax = sorted([p1.x(), p2.x()])
        ymin, ymax = sorted([p1.y(), p2.y()])

        selected_spots = []
        for spot in self.scatter.points():
            x, y = spot.pos().x(), spot.pos().y()
            if xmin <= x <= xmax and ymin <= y <= ymax:
                selected_spots.append(spot)

        for s in self.scatter.points():
            s.setBrush(pg.mkBrush("w"))  # reset
        for s in selected_spots:
            s.setBrush(pg.mkBrush("r"))  # highlight selected


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Rubberband Selection")
        self.resize(800, 600)

        # Create scatter item first
        self.scatter = pg.ScatterPlotItem()

        # Create a custom ViewBox that knows about the scatter
        self.custom_vb = RubberBandViewBox(self.scatter)

        # Pass it to the PlotWidget
        self.plotWidget = pg.PlotWidget(viewBox=self.custom_vb)
        self.setCentralWidget(self.plotWidget)
        self.plotWidget.addItem(self.scatter)

        # Generate data
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        spots = [{'pos': (x_, y_), 'brush': pg.mkBrush('w')}
                 for x_, y_ in zip(x, y)]
        self.scatter.addPoints(spots)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
