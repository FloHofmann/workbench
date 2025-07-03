import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg

# Set PyQtGraph theme
pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 'w')


class SpikeSorterGUI(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Spike Sorter with PyQtGraph")
        self.resize(1200, 800)

        # Extract data
        self.time = data['time']
        self.filtered_trace = data['filtered_trace']
        self.spike_traces = data['spike_traces']  # shape: (n_spikes, n_points)
        self.spike_time = data['spike_time']
        self.pca_out = data['pca_out']
        self.isis = data['isis']

        # Central widget and layout
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Create graphics layout
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)

        # Panels
        self.plot_ephys_trace()
        self.plot_spike_traces()
        self.plot_pca()
        self.plot_isi()

    def plot_ephys_trace(self):
        p1 = self.graphics_layout.addPlot(
            row=0, col=0, colspan=2, title="Filtered Trace with Spikes")
        p1.plot(self.time, self.filtered_trace, pen='w')
        # Mark spike times
        spike_indices = np.searchsorted(self.time, self.spike_time)
        # Check index boundaries
        spike_indices = spike_indices[spike_indices < len(self.filtered_trace)]
        p1.plot(self.spike_time[:len(spike_indices)], self.filtered_trace[spike_indices],
                pen=None, symbol='o', symbolBrush='r', symbolSize=6)

    def plot_spike_traces(self):
        p2 = self.graphics_layout.addPlot(
            row=1, col=0, title="Overlaid Spike Traces")
        n_spikes, n_points = self.spike_traces.shape
        t_spike = np.linspace(0, 1.5e-3, n_points) * 1000  # ms

        for i in range(n_spikes):
            p2.plot(t_spike, self.spike_traces[i], pen=pg.mkPen(
                width=0.5, color=(100, 100, 255, 100)))

        p2.setLabel("bottom", "Time (ms)")
        p2.setLabel("left", "Voltage (mV)")

    def plot_pca(self):
        p3 = self.graphics_layout.addPlot(row=1, col=1, title="PCA Scatter")
        # Use scatterPlotItem for better performance
        scatter = pg.ScatterPlotItem(
            x=self.pca_out[0], y=self.pca_out[1], pen=None, brush='g', size=7, symbol='o')
        p3.addItem(scatter)
        p3.setLabel("bottom", "PCA1")
        p3.setLabel("left", "PCA2")

    def plot_isi(self):
        p4 = self.graphics_layout.addPlot(
            row=2, col=0, colspan=2, title="ISI Histogram")
        # Check for nonpositive ISI values
        isis = self.isis[self.isis > 0]
        if len(isis) == 0:
            p4.setTitle("ISI Histogram (no valid ISIs)")
            return
        y, x = np.histogram(isis, bins=np.logspace(
            np.log10(min(isis)), np.log10(max(isis)), 50))
        # Step-like histogram plot
        x_mid = 0.5 * (x[:-1] + x[1:])
        p4.plot(x_mid, y, pen='b', symbol=None)
        p4.setLogMode(x=True)
        p4.setLabel("bottom", "ISI (ms)")
        p4.setLabel("left", "Count")


if __name__ == "__main__":
    # Mock data for testing
    N = 2000
    T = 150
    np.random.seed(0)

    time = np.linspace(0, 10, 250000)
    filtered_trace = 0.1 * np.random.randn(len(time))
    spike_times = np.sort(np.random.choice(time, N, replace=False))
    spike_traces = np.random.randn(N, T) * 0.1
    pca_out = np.random.randn(2, N)
    isis = np.diff(spike_times) * 1000

    data = dict(
        time=time,
        filtered_trace=filtered_trace,
        spike_traces=spike_traces,
        spike_time=spike_times,
        pca_out=pca_out,
        isis=isis
    )

    app = QApplication(sys.argv)
    win = SpikeSorterGUI(data)
    win.show()
    sys.exit(app.exec_())
