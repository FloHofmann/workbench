from workbench.data.hdf5_interface.h5data import loadmat

datapath = r"\\172.25.250.112\burgalossi\lab share\Data\Florian\ADN\ADNFH3\analysis\Data6\ramp\Data6_ramp.mat"

a = loadmat(datapath)
a.export_processed_data()
