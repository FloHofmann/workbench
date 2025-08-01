from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import sys


class ScatterSelectionWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Scatter points
        self.points = np.random.rand(100, 2) * 10
        self.sp_item = pg.ScatterPlotItem(
            x=self.points[:, 0],
            y=self.points[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10,
        )
        self.addItem(self.sp_item)

        # Rubberband variables
        self.rubberband = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle, self)
        self.origin = QtCore.QPoint()
        self.view_box = self.getViewBox()

        # Mouse tracking
        self.setMouseTracking(True)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubberband.setGeometry(
                QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberband.show()

    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            rect = QtCore.QRect(self.origin, event.pos()).normalized()
            self.rubberband.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.rubberband.isVisible():
            self.rubberband.hide()

            # Get rubberband geometry in scene coordinates
            rb_rect = self.rubberband.geometry()
            topLeft = self.mapToScene(rb_rect.topLeft())
            bottomRight = self.mapToScene(rb_rect.bottomRight())

            # Convert to data coordinates
            p1 = self.view_box.mapSceneToView(topLeft)
            p2 = self.view_box.mapSceneToView(bottomRight)

            x0, x1 = sorted([p1.x(), p2.x()])
            y0, y1 = sorted([p1.y(), p2.y()])

            # Select points in bounding box
            mask = (
                (self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1)
            )

            selected_points = self.points[mask]
            print(f"Selected {selected_points.shape[0]} points")

            # Update color to show selection
            brushes = [pg.mkBrush(255, 0, 0) if m else pg.mkBrush(
                100, 100, 255, 120) for m in mask]
            self.sp_item.setBrush(brushes)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    plot = ScatterSelectionWidget()
    win.setCentralWidget(plot)
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
