# %%
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

from workbench.data.preprocess import combTableCreate, expand_dict_columns