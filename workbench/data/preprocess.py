from workbench.data.spikesorter import SpikeSorter
from workbench.data.hdf5_interface import loadmat

import h5py
from pathlib import Path


def process_exp(datapath: Path):
    data = loadmat(datapath)
    data.export_processed_data()
    sorted = SpikeSorter(trace)


if __name__ == "__main__":
    process_exp()
