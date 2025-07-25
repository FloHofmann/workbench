"""
This class will be filled up with the data from the hdf5 files
"""
import h5py
import re
from typing import Union
from pathlib import Path

from workbench.data.spikesorter import spikesorter


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
        import sys
        from PyQt6.QtWidgets import QApplication

        # this function should export the processed data to a h5file/numpy
        # file.
        # tbd.... this will be saved in the same folder as the raw_data h5file
        app = QApplication(sys.argv)
        window = spikesorter(self)
        window.show()
        sys.exit(app.exec())

        return 'exported'


if __name__ == '__main__':
    print("not supposed to run as main")
