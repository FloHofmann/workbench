import h5py
import numpy as np
from pathlib import Path
from typing import Union


class loadmat:
    def __init__(self, path: Union[Path, str]):
        """
        Initializes the class and loads data from the given HDF5 file path.
        """
        self.info = {}
        self.raw_data = {}
        self.processed_data = {}
        self.waveform = {}

        if isinstance(path, str):
            path = Path(path)
        self.info['path'] = str(path)

        with h5py.File(path, "r") as f:
            for key in f.keys():
                k = key.lower()
                if 'ch1' in k:
                    self.raw_data[key] = self._assign_signal(f, key)
                elif k in ['ch31', 'ch32']:
                    self.raw_data['codes'] = f[key]['codes'][...]
                    self.raw_data['times'] = f[key]['times'][...]
                    self.raw_data['title'] = ''.join(
                        f[key]['title'][:].astype('uint32').view('U1').flatten())
                elif any(c in k for c in ['ch2', 'ch3', 'ch4', 'ch5', 'ch6']):
                    self.raw_data[key] = self._assign_signal(f, key)

    def _assign_signal(self, f, key):
        """
        Assigns a signal dictionary from HDF5 structure.
        """
        return {
            'interval': f[key]['interval'][...],
            'length': f[key]['length'][...],
            'offset': f[key]['offset'][...],
            'scale': f[key]['scale'][...],
            'start': f[key]['start'][...],
            'times': f[key]['times'][...],
            'title': ''.join(f[key]['title'][:].astype('uint32').view('U1').flatten()),
            'units': f[key]['units'][...],
            'values': f[key]['values'][...].flatten()
        }

    def export_processed_data(self):
        """
        Opens the SpikeSorter GUI with the loaded raw data.
        """
        import sys
        from PyQt6.QtWidgets import QApplication
        from workbench.data.spikesorter import SpikeSorter

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
    Saves processed spike/event/tracking data into a structured HDF5 file.
    """
    folderpath = Path(folderpath)
    folderpath.mkdir(parents=True, exist_ok=True)
    file_path = folderpath / "exp_data.h5"

    with h5py.File(file_path, "w") as f:
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
    """
    Loads processed spike/event/tracking data from an HDF5 file.
    """
    with h5py.File(file_path, "r") as f:
        data = {
            "infos": {},
            "processed_data": {},
            "analysis": {}
        }

        infos = f["infos"]
        keys = ["nframes", "x_values", "y_value", "raw_data_path", "sampling_rate"]
        for key in keys:
            if key in infos:
                data["infos"][key] = infos[key][()] if infos[key].shape == () else infos[key][:]

        data["infos"].update({
            "has_spikes": infos.attrs.get("has_spikes", False),
            "has_events": infos.attrs.get("has_events", False),
            "has_tracking": infos.attrs.get("has_tracking", False)
        })

        processed = f["processed_data"]
        if "spike_times" in processed:
            data["processed_data"]["spike_times"] = processed["spike_times"][:]
        if "spike_traces" in processed:
            data["processed_data"]["spike_traces"] = processed["spike_traces"][:]
        if "event_times" in processed:
            data["processed_data"]["event_times"] = processed["event_times"][:]

        analysis = f["analysis"]
        if "angles" in analysis:
            data["analysis"]["angles"] = analysis["angles"][:]

    return data

if __name__ == '__main__':
    print("not supposed to run as main")
