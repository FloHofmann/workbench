import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QPointF
import numpy as np

class PCARubberbandSelector:
    def __init__(self, pca_plot, pca_data, on_selection_changed=None):
        super().__init__()
        self.pca_plot = pca_plot  # pg.PlotWidget
        self.pca_data = pca_data  # np.array (N, 2)
        self.selected_mask = np.zeros(pca_data.shape[0], dtype=bool)

        self.origin = None
        self.rubberband = None
        self.selecting = False

        self.on_selection_changed = on_selection_changed  # callback

        # Connect to scene
        self.pca_plot.scene().sigMouseClicked.connect(self.handle_mouse_click)
        self.pca_plot.scene().sigMouseMoved.connect(self.handle_mouse_move)

    def handle_mouse_click(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        vb = self.pca_plot.plotItem.vb
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return

        point = vb.mapSceneToView(pos)

        if event.type() == pg.QtCore.QEvent.Type.GraphicsSceneMousePress:
            self.origin = point
            self.selecting = True

            if self.rubberband:
                self.pca_plot.scene().removeItem(self.rubberband)

            self.rubberband = pg.QtWidgets.QGraphicsRectItem()
            self.rubberband.setPen(pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            self.pca_plot.scene().addItem(self.rubberband)

        elif event.type() == pg.QtCore.QEvent.Type.GraphicsSceneMouseRelease and self.selecting:
            self.finalize_selection()

    def handle_mouse_move(self, pos):
        if not self.selecting or self.origin is None:
            return

        vb = self.pca_plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            return

        point = vb.mapSceneToView(pos)
        x0, y0 = self.origin.x(), self.origin.y()
        x1, y1 = point.x(), point.y()

        rect = QRectF(QPointF(min(x0, x1), min(y0, y1)),
                      QPointF(max(x0, x1), max(y0, y1)))

        self.rubberband.setRect(rect)

    def finalize_selection(self):
        if not self.selecting or self.origin is None:
            return

        rect = self.rubberband.rect()
        self.pca_plot.scene().removeItem(self.rubberband)
        self.rubberband = None
        self.selecting = False

        xmin, xmax = rect.left(), rect.right()
        ymin, ymax = rect.top(), rect.bottom()

        self.selected_mask = (
            (self.pca_data[:, 0] >= xmin) & (self.pca_data[:, 0] <= xmax) &
            (self.pca_data[:, 1] >= ymin) & (self.pca_data[:, 1] <= ymax)
        )

        if self.on_selection_changed:
            self.on_selection_changed(self.selected_mask)

    def clear(self):
        self.selected_mask[:] = False
        if self.on_selection_changed:
            self.on_selection_changed(self.selected_mask)

    def get_mask(self):
        return self.selected_mask.copy()

