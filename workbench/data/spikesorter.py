import numpy as np
import sys
from pathlib import Path
from PyQt6 import QtCore, QtGraphsWidgets, QtDataVisualization, QtWidgets
from typing import Union
from scipy.signal import firwin, filtfilt, find_peaks
from sklearn.decomposition import PCA

from workbench.data.hdf5_interface.h5data import h5data

PathLike = Union[str, Path]


def spikesort(inp: Union[PathLike, h5data]) -> object:
    """
    Call the spikesorter object.
    Parameters
    ---------------------
    inp: str | pathlib.Path | object
        if str or Path, treated as path.
        otherwise loads a custom hdf5 data object
    """

    # check for input type. If str or path create the h5 file first
    if isinstance(inp, (str, Path)):
        h5 = h5data(inp)

    else:
        h5 = inp

    spsorter = spikesorter(h5)
    return spsorter


class spikesorter:
    def __init__(self, h5: h5data):
        self.h5 = h5
