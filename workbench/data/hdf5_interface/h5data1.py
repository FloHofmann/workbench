"""
This class will be filled up with the data from the hdf5 files
"""
import h5py
import sys
from typing import Union
from pathlib import Path
from PyQt6.QtWidgets import QApplication


from workbench.data.spikesorter import SpikeSorter


class loadmat:
    def __init__(self, path: Union[Path, str]):
        """
        initializes the class.
        Will load the corresponding h5 file and start
        unpacking and loading the class.
        """

        # initialize data storage
        self.info = {}
        self.raw_data = {}
        self.processed_data = {}
        self.waveform = {}
        print(path)
        # extract metadata from path

        if isinstance(path, (str)):
            path = Path(path)
        self.info['path'] = str(path)

        # retrieval of raw data from the exported mat file
        with h5py.File(path, "r") as f:

            file_keys = f.keys()
            # load the raw_data
            for key in file_keys:
                # find out the origin of the data and assign it accordingly
                if 'ch1' in key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)

                elif 'ch31' == key.lower():
                    # marker channel
                    self.raw_data['codes'] = f[key]['codes'][0][:]
                    self.raw_data['times'] = f[key]['times'][:].flatten()
                    self.raw_data['title'] = ''.join(
                        f[key]['title'][:].astype('uint32')
                        .view('U1').flatten())
                elif 'ch32' == key.lower():
                    # marker channel
                    self.raw_data['codes'] = f[key]['codes'][:]
                    self.raw_data['times'] = f[key]['times'][:]
                    self.raw_data['title'] = ''.join(
                        f[key]['title'][:].astype('uint32')
                        .view('U1').flatten())
                elif 'ch2' == key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)
                elif 'ch3' == key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)
                elif 'ch4' in key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)
                elif 'ch5' in key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)
                elif 'ch6' in key.lower():
                    self.raw_data[key] = self.assign_value_dict_signal(f, key)

    def assign_value_dict_signal(self, f, key):
        # this just assigns the extracted values from the hdf5 file to a dict
        temp_dict = {}
        temp_dict['interval'] = f[key]['interval'][:]
        temp_dict['length'] = f[key]['length'][:]
        temp_dict['offset'] = f[key]['offset'][:]
        temp_dict['scale'] = f[key]['scale'][:]
        temp_dict['start'] = f[key]['start'][:]
        temp_dict['times'] = f[key]['times'][:]  # in seconds
        temp_dict['title'] = ''.join(
            f[key]['title'][:].astype('uint32').view('U1').flatten())
        temp_dict['units'] = f[key]['units'][:]
        temp_dict['values'] = f[key]['values'][:].flatten()
        return temp_dict

    def export_processed_data(self):
        """
        the loaded data from the .mat file is routed through the spikesorter.
        also looks for any baseline videos to perform tracking on.
        """
        # this function should export the processed data to a h5file/numpy
        # file.
        # tbd.... this will be saved in the same folder as the raw_data h5file
        app = QApplication(sys.argv)
        window = SpikeSorter(self)
        window.show()
        app.exec()

        return window


def save_processed_data(
    folderpath,
    *,
    sorted_spikes=None,
    angles=None,
    nframes=None,
    x_values=None,
    y_value=None,
    event_times=None,
    raw_path=None,
    sampling_rate=None
):
    """
    saves the processed/spikesorted data/event data
    """
    folderpath = Path(folderpath)
    folderpath.mkdir(parents=True, exist_ok=True)
    file_path = folderpath / "exp_data.h5"

    with h5py.File(file_path, "w") as f:
        # === Groups ===
        infos = f.create_group("infos")
        processed = f.create_group("processed_data")
        analysis = f.create_group("analysis")

        # === Infos ===
        if nframes is not None:
            infos.create_dataset("nframes", data=nframes)
        if x_values is not None:
            infos.create_dataset("x_values", data=np.asarray(x_values))
        if y_value is not None:
            infos.create_dataset("y_value", data=np.asarray(y_value))
        if raw_path is not None:
            infos.create_dataset("raw_data_path", data=str(raw_path))
        if sampling_rate is not None:
            infos.create_dataset("sampling_rate", data=sampling_rate)

        # presence flags
        infos.attrs["has_spikes"] = sorted_spikes is not None
        infos.attrs["has_events"] = event_times is not None
        infos.attrs["has_tracking"] = angles is not None

        # === Processed ===
        if sorted_spikes is not None:
            if "times" in sorted_spikes:
                processed.create_dataset("spike_times", data=np.asarray(sorted_spikes["times"]))
            if "traces" in sorted_spikes:
                processed.create_dataset("spike_traces", data=np.asarray(sorted_spikes["traces"]))

        if event_times is not None:
            processed.create_dataset("event_times", data=np.asarray(event_times))

        # === Analysis ===
        if angles is not None:
            analysis.create_dataset("angles", data=np.asarray(angles))

def load_processed_data(file_path):
    import h5py

    with h5py.File(file_path, "r") as f:
        data = {}

        # === Infos ===
        infos = f["infos"]
        data["infos"] = {
            "nframes": infos.get("nframes", default=None),
            "x_values": infos.get("x_values", default=None),
            "y_value": infos.get("y_value", default=None),
            "raw_data_path": infos.get("raw_data_path", default=None),
            "sampling_rate": infos.get("sampling_rate", default=None),
            "has_spikes": infos.attrs.get("has_spikes", False),
            "has_events": infos.attrs.get("has_events", False),
            "has_tracking": infos.attrs.get("has_tracking", False),
        }

        # Load values if they exist
        for key in ["nframes", "x_values", "y_value", "raw_data_path", "sampling_rate"]:
            if key in infos:
                data["infos"][key] = infos[key][()] if infos[key].shape == () else infos[key][:]

        # === Processed Data ===
        processed = f["processed_data"]
        data["processed_data"] = {}
        if "spike_times" in processed:
            data["processed_data"]["spike_times"] = processed["spike_times"][:]
        if "spike_traces" in processed:
            data["processed_data"]["spike_traces"] = processed["spike_traces"][:]
        if "event_times" in processed:
            data["processed_data"]["event_times"] = processed["event_times"][:]

        # === Analysis ===
        analysis = f["analysis"]
        data["analysis"] = {}
        if "angles" in analysis:
            data["analysis"]["angles"] = analysis["angles"][:]

    return data

if __name__ == '__main__':
    print("not supposed to run as main")
