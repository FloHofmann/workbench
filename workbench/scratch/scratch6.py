from workbench.data.hdf5_interface.h5data import loadmat
import os
print(os.getcwd())
datapath = r"data/Data10-tones-95.mat"

a = loadmat(datapath)
info, raw, sorted_spikes = a.export_processed_data()
exp = sorted_spikes.exportData()
breakpoint()
print(exp)
