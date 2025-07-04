import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
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

        # Optional: add crosshair, region selector, etc.


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
