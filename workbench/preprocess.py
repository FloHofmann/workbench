from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from neo.io import CedIO
from scipy.signal import firwin, filtfilt, find_peaks
from sklearn.decomposition import PCA


# %% spikesorting function
def spikesort_smrx(smrx_path: str) -> object:
    """
    This function will take the path to a .smrx file and start the spike-sorting process for it. Output is a
    spikesorter object, which can be further spikesorted or the data can be accessed to retrieve the post sorting
    data. It will create a Figure with 2 interactive subplots, one allowing for deletion of individual spike traces
    by setting a threshold with left (downwards) or right (upwards) mouseclick. This can be initiated by pressing 't'
    while viewing the figure. Alternatively, a rectangle can be drawn in the PCA plot after pressing 'e'. All
    Datapoints within the rectangle will be deleted. 'u' can be pressed do undo the latest changes.
    """
    # call class and return it
    spksorter: object = SpikeSorter(smrx_path)
    plt.show()
    return spksorter


class SpikeSorter:
    def __init__(self, path: str) -> None:
        #  load file
        test = CedIO(str(Path(path)))

        # read() reads the entire file into memory at once.
        # If you only want to access part of the data,
        # you can do so using Neo's lazy data loading see docs
        a = test.read()

        #  Create filtered signal from raw
        signal = a[0].segments[0].analogsignals[0]
        trace = signal.magnitude.ravel()
        ch2 = a[0].segments[0].analogsignals[1].magnitude.ravel()
        ch6 = a[0].segments[0].analogsignals[5].magnitude.ravel()
        ch3 = a[0].segments[0].analogsignals[2].magnitude.ravel()
        ch4 = a[0].segments[0].analogsignals[3].magnitude.ravel()
        ch5 = a[0].segments[0].analogsignals[4].magnitude.ravel()
        time = signal.times.magnitude

        # This signal needs to be filtered
        SR = int(signal.sampling_rate)
        CUTOFF = 300
        N = 2 ** 8
        N |= 1
        filterb = firwin(N, CUTOFF, fs=SR, pass_zero=False)
        filtered = filtfilt(filterb, 1, trace, axis=0)

        #  first interactive plot
        plt.plot(signal.times, filtered)
        plt.ylabel('voltage mV')
        plt.xlabel("time")
        coordinates = plt.ginput(1)
        print(coordinates)
        plt.close("all")

        #  Threshold trace and get peaks
        MS = 1e-3
        pre_spike_time = 0.5 * MS
        post_spike_time = 1.5 * MS

        idx_range_pre = round(pre_spike_time * SR)
        idx_range_post = round(post_spike_time * SR)
        idx_range = idx_range_pre + idx_range_post

        spike_refractory = 1 * MS
        idx_spike_refractory = round(spike_refractory * SR)

        peaks = find_peaks(filtered, height=coordinates[0][1], distance=idx_spike_refractory)
        peakheight = peaks[1]['peak_heights']
        peakloc = peaks[0]

        #  create spike sorting window

        # Remove peaks right after the recording starting
        peakheight = np.delete(peakheight, (peakloc < idx_range_pre))
        peakloc = np.delete(peakloc, (peakloc < idx_range_post))

        # Remove peaks right before the recording stops
        peakheight = np.delete(peakheight, (peakloc > len(filtered) - idx_range_post))
        peakloc = np.delete(peakloc, (peakloc > len(filtered) - idx_range_post))

        num_spikes = len(peakheight)  # get number of spikes from peaks

        spike_indices_pre = np.subtract(peakloc, idx_range_pre)
        spike_indices_post = np.add(peakloc, idx_range_post)

        # Get spike traces and calculate PCA

        spk_trace = np.zeros((num_spikes, idx_range + 1))

        for i in range(num_spikes):
            spk_trace[i, :] = filtered[spike_indices_pre[i]:spike_indices_post[i] + 1]

        spike_trace_time = np.divide(np.array(list(range(0, idx_range + 1))), SR)
        self.pca = PCA(n_components=2, random_state=16)
        output = self.pca.fit_transform(spk_trace).transpose()

        spike_times = time[peakloc]  # use original times

        spk_trace = spk_trace.transpose()

        # calculate isi distribution
        isis = np.diff(spike_times) * 1000
        self.data = {
            'recording': {
                'raw_trace': trace,
                'filtered_raw': filtered,
                'session_time': time,
                'session_peaks': peakheight,
                'session_spiketimes': spike_times
            },
            'processed': {
                'spk_trace': spk_trace,
                'spk_trace_time': spike_trace_time,
                'pca_out': output,
                'isis': isis
            },
            'raw_channels': {
                'ch2': ch2,
                'ch3': ch3,
                'ch4': ch4,
                'ch5': ch5,
                'ch6': ch6,
            }
        }

        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 10))
        self.threshold = None
        self.threshold_mode = False
        self.pca_selector_mode = False
        self.latest_change = []
        self.last_operation = []
        self.rect_selector = None

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.plot_data()

    def plot_data(self):
        self.fig.suptitle('''Press "t" to threshold within the traces
                            Press "e" to draw a rectangle in the PCA
                            Press "u" to undo the latest changes''')
        ax_ephys = self.ax[0, 1]
        ax_ephys.clear()
        ax_spikes = self.ax[0, 0]
        ax_spikes.clear()
        ax_isi = self.ax[1, 1]
        ax_isi.clear()
        ax_pca = self.ax[1, 0]
        ax_pca.clear()

        # plot ephys trace
        ax_ephys.plot(self.data['recording']['session_time'], self.data['recording']['filtered_raw'])
        ax_ephys.plot(self.data['recording']['session_spiketimes'], self.data['recording']['session_peaks'],
                      'r.')
        ax_ephys.set_xlabel('Time (s)')
        ax_ephys.set_ylabel('Voltage (V)')

        # plot spike trace
        spktimes_expanded = np.repeat(np.expand_dims(self.data['processed']['spk_trace_time'].transpose(), 0),
                                      self.data['processed']['spk_trace'].shape[1], axis=0).transpose()
        ax_spikes.plot(spktimes_expanded, self.data['processed']['spk_trace'])
        ax_spikes.set_xlabel('Time (s)')
        ax_spikes.set_ylabel('Voltage (V)')
        ax_spikes.set_xlim([0, max(self.data['processed']['spk_trace_time'])])

        # plot pca scatter
        ax_pca.scatter(self.data['processed']['pca_out'][0, :], self.data['processed']['pca_out'][1, :], marker='.')
        ax_pca.set_xlabel('PCA 1')
        ax_pca.set_ylabel('PCA 2')

        # plot isi
        MIN, MAX = min(self.data['processed']['isis']), max(
            self.data['processed']['isis'])  # retrieve min max to set range for bins
        ax_isi.hist(self.data['processed']['isis'],
                    bins=10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))  # calc logarithmic bins
        ax_isi.set_xscale("log")
        ax_isi.set_xlabel('Time [ms]')

        if self.rect_selector:
            self.rect_selector.set_active(False)
            self.rect_selector = False
        plt.draw()

    def on_key_press(self, event):
        if event.key == 't':
            print("Press 't' to select threshold.")
            self.threshold_mode = not self.threshold_mode
            if not self.threshold_mode:
                print("Threshold mode exited.")
            self.last_operation.append('t')
        elif event.key == 'u':
            self.undo_latest_change()
        elif event.key == 'e':
            print("Press 'e' key to select and delete data points.")
            self.select_and_delete_data()
            self.last_operation.append('e')
        elif event.key == 'x':
            plt.close()
            return self

    def on_mouse_click(self, event):
        if self.threshold_mode and event.inaxes:
            if event.button == 1:
                self.threshold = event.ydata
                self.threshold_mode = False
                # Add here the latest change
                below_thresh = \
                    np.where(self.data['processed']['spk_trace'][round(event.xdata * 25000), :] <= self.threshold)[
                        0]
                removed_entries = {
                    'del_trace': self.data['processed']['spk_trace'][:, below_thresh],
                    'del_peaks': self.data['recording']['session_peaks'][below_thresh],
                    'del_times': self.data['recording']['session_spiketimes'][below_thresh]
                }
                self.data['processed']['spk_trace'] = np.delete(self.data['processed']['spk_trace'], below_thresh,
                                                                axis=1)
                self.data['recording']['session_peaks'] = np.delete(self.data['recording']['session_peaks'],
                                                                    below_thresh)
                self.data['recording']['session_spiketimes'] = np.delete(
                    self.data['recording']['session_spiketimes'], below_thresh)
                self.data['processed']['pca_out'] = self.pca.fit_transform(
                    self.data['processed']['spk_trace'].transpose()).transpose()
                self.latest_change.append(removed_entries)
            elif event.button == 3:
                self.threshold = event.ydata
                self.threshold_mode = False

                # Add the latest change
                above_thresh = \
                    np.where(self.data['processed']['spk_trace'][round(event.xdata * 25000), :] >= self.threshold)[
                        0]
                removed_entries = {'del_trace': self.data['processed']['spk_trace'][:, above_thresh],
                                   'del_peaks': self.data['recording']['session_peaks'][above_thresh],
                                   'del_times': self.data['recording']['session_spiketimes'][above_thresh]
                                   }
                self.data['processed']['spk_trace'] = np.delete(self.data['processed']['spk_trace'], above_thresh,
                                                                axis=1)
                self.data['recording']['session_peaks'] = np.delete(self.data['recording']['session_peaks'],
                                                                    above_thresh)
                self.data['recording']['session_spiketimes'] = np.delete(
                    self.data['recording']['session_spiketimes'], above_thresh)
                # Adjust the other values
                self.data['processed']['pca_out'] = self.pca.fit_transform(
                    self.data['processed']['spk_trace']).transpose()
                self.data['processed']['isis'] = np.diff(self.data['recording']['session_spiketimes']) * 1000
                self.latest_change.append(removed_entries)
            self.plot_data()

    def undo_latest_change(self):
        if not self.last_operation:
            print('No last changes')
        else:
            print("undoing latest change")

            self.data['processed']['spk_trace'] = np.append(self.data['processed']['spk_trace'],
                                                            self.latest_change[-1]['del_trace'], axis=1)
            self.data['recording']['session_peaks'] = np.append(self.data['recording']['session_peaks'],
                                                                self.latest_change[-1]['del_peaks'])
            self.data['recording']['session_spiketimes'] = np.append(
                self.data['recording']['session_spiketimes'], self.latest_change[-1]['del_times'])
            self.data['recording']['session_peaks'] = np.array([x for _, x in sorted(
                zip(self.data['recording']['session_spiketimes'], self.data['recording']['session_peaks']))])
            self.data['recording']['session_spiketimes'] = sorted(self.data['recording']['session_spiketimes'])
            self.data['processed']['pca_out'] = self.pca.fit_transform(
                self.data['processed']['spk_trace'].transpose()).transpose()
            self.latest_change = self.latest_change[:-1]
            self.last_operation = self.last_operation[:-1]

            self.plot_data()

    # find a way to only enable the methods on specific subplots
    def on_select(self, eclick, erelease):
        # Callback for RectangleSelector selection
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        # Identify selected data points based on the rectangel coordinates
        selected_datapoints = np.where((self.data['processed']['pca_out'][0, :] <= x2) &
                                       (self.data['processed']['pca_out'][0, :] >= x1) &
                                       (self.data['processed']['pca_out'][1, :] <= y2) &
                                       (self.data['processed']['pca_out'][1, :] >= y1))[0]

        # store datapoints that are about to be deleted
        removed_entries = {
            'del_trace': self.data['processed']['spk_trace'][:, selected_datapoints],
            'del_times': self.data['recording']['session_spiketimes'],
            'del_peaks': self.data['recording']['session_peaks']
        }
        # delete the datapoints from all datasets
        self.data['recording']['session_peaks'] = np.delete(self.data['recording']['session_peaks'],
                                                            selected_datapoints)
        self.data['recording']['session_spiketimes'] = np.delete(self.data['recording']['session_spiketimes'],
                                                                 selected_datapoints)
        self.data['processed']['spk_trace'] = np.delete(self.data['processed']['spk_trace'], selected_datapoints,
                                                        axis=1)
        self.data['processed']['pca_out'] = np.delete(self.data['processed']['pca_out'], selected_datapoints, axis=1)

        self.latest_change.append(removed_entries)
        self.plot_data()

    def select_and_delete_data(self):
        print("select a region to delete datapoints using the mouse")
        self.rect_selector = RectangleSelector(self.ax[1, 0], self.on_select,
                                               useblit=True,
                                               button=[1],
                                               minspanx=0.05,
                                               minspany=0.05,
                                               spancoords='data',
                                               interactive=False)


# %% Process recording function. This calls the spikesorting function above

def process_rec():
    pass
    # TODO write the preprocess function according to the matlab original
