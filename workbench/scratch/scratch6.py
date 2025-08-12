from workbench.data.hdf5_interface.h5data import loadmat
import os
print(os.getcwd())
datapath = r"data/Data6.mat"

a = loadmat(datapath)
info, raw, sorted = a.export_processed_data()
exp = sorted.exportData()
breakpoint()
print(exp)
