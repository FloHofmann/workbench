from pathlib import Path
from git import Repo
import IPython

from workbench.data.hdf5_interface.h5data import h5data


root = Repo('.', search_parent_directories=True).working_tree_dir
data = Path(root, "data/Data6.mat")

a = h5data(data)
IPython.embed()
