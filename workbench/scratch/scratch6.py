from workbench.data.hdf5_interface.h5data import loadmat

datapath = r"data/Data6.mat"

a = loadmat(datapath)
a.export_processed_data()
