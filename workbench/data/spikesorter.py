from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from neo.io import CedIO
from scipy.signal import firwin, filtfilt, find_peaks
from sklearn.decomposition import PCA
