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

        if isinstance(path, str):
            path = Path(path)
        self.info['path'] = str(path)

        with h5py.File(path, "r") as f:
            for key in f.keys():
                k = key.lower()
                if "_" in k:
                    keysplit = k.split("_")
                    keysplit = keysplit[-1]
                else:
                    keysplit = k
                if 'ch1' in keysplit:
                    self.raw_data[key] = self._assign_signal(f, key)
                elif keysplit in ['ch31', 'ch32']:
                    self.raw_data[key] = dict()
                    self.raw_data[key]['codes'] = f[key]['codes'][...]
                    self.raw_data[key]['times'] = f[key]['times'][...]
                    self.raw_data[key]['title'] = ''.join(
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

        return self.info, self.raw_data, window


def save_processed_data(
    folderpath,
    *,
    sorted_spikes=None,
    angles=None,
    nframes=None,
    x_values=None,
    y_value=None,
    n_ttls=None,
    ttl_times=None,
    event_times=None,
    event_codes=None,
    sampling_rate=None,
    ch4_stims=None,
    ch5_stims=None
):
    """
    Saves processed spike/event/tracking data into a structured HDF5 file.
    """
    folderpath = Path(folderpath)
    folderpath.mkdir(parents=True, exist_ok=True)
    file_path = folderpath / "exp_data.h5"
    print(str(file_path))

    with h5py.File(file_path, "w") as f:
        infos = f.create_group("infos")
        processed = f.create_group("processed_data")
        analysis = f.create_group("analysis")

        #  Infos
        if nframes is not None:
            infos.create_dataset("nframes", data=nframes)
        if sampling_rate is not None:
            infos.create_dataset("sampling_rate", data=sampling_rate)
        else:
            infos.create_dataset("sampling_rate", data=25000)

        infos.attrs["has_spikes"] = sorted_spikes is not None
        infos.attrs["has_events"] = event_times is not None
        infos.attrs["has_tracking"] = angles is not None

        #  Processed
        if sorted_spikes is not None:
            if "times" in sorted_spikes:
                processed.create_dataset(
                    "spike_times", data=np.asarray(sorted_spikes["times"]))
            if "traces" in sorted_spikes:
                processed.create_dataset(
                    "spike_traces", data=np.asarray(sorted_spikes["traces"]))

        if ch4_stims is not None:
            processed.create_dataset("ch4_stims", data=np.asarray(ch4_stims))
        if ch5_stims is not None:
            processed.create_dataset("ch5_stims", data=np.asarray(ch5_stims))

        if event_times is not None:
            processed.create_dataset(
                "event_times", data=np.asarray(event_times))

        if event_codes is not None:
            processed.create_dataset(
                "event_codes", data=event_codes)

        #  Analysis
        if angles is not None:
            analysis.create_dataset("angles", data=np.asarray(angles))

        if ttl_times is not None:
            analysis.create_dataset("ttl_times", data=np.asarray(ttl_times))
        if n_ttls is not None:
            analysis.create_dataset("n_ttls", data=n_ttls)
        if x_values is not None:
            analysis.create_dataset("x_values", data=np.asarray(x_values))
        if y_value is not None:
            analysis.create_dataset("y_value", data=np.asarray(y_value))


def load_processed_data(file_path):
    """
    Loads data saved by `save_processed_data`.

    Returns:
        dict: {
            'infos': dict,
            'processed_data': dict,
            'analysis': dict
        }
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    with h5py.File(file_path, "r") as f:
        infos = {key: f["infos"][key][()] for key in f["infos"]}
        infos.update({k: f["infos"].attrs[k] for k in f["infos"].attrs})

        processed = {
            key: f["processed_data"][key][()]
            for key in f.get("processed_data", {})
        }

        analysis = {
            key: f["analysis"][key][()]
            for key in f.get("analysis", {})
        }

    return {
        "infos": infos,
        "processed_data": processed,
        "analysis": analysis
    }


def combTableCreate(datatable, datapath, filename):
    """
    Creates the comb tables and saves them as h5 files to the datapath folder under the specified filename
    Input is a pandas dataframe loaded from sqlite
    """
    return 'Hi'


def parseComb():
    """
    This function is used to parse the saved h5 file back to a usable dict
    """
    return 'Hi'


if __name__ == '__main__':
    print("not supposed to run as main")
