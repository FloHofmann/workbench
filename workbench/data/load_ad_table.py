"""
load_ad_table.py
-----------------
Load MATLAB v7.3 AD combined table into pandas.

MATLAB table objects are serialised as opaque HDF5 Datasets that h5py/mat73
cannot parse directly. The MATLAB save step therefore also writes
`table_struct = table2struct(out_table, 'ToScalar', true)`, which is a plain
MATLAB struct. Structs are stored as HDF5 Groups (one child per field) and are
trivially readable.

HDF5 encoding of struct fields:
  numeric scalar column  → float64 Dataset  shape (1, nrows)  [MATLAB col-major]
  cell-array column      → object  Dataset  shape (1, nrows)  each elem is HDF5 ref
  strings                → uint16  Dataset  shape (n, 1)
  empty matrix           → uint64  Dataset  shape (2,)  values [0, 0]

Install:  pip install h5py numpy pandas
"""

from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_DEFAULT_PATH = Path(r"C:\Users\FloHofmann\Documents\AD_combined_table.mat")


# ── public container ──────────────────────────────────────────────────────

class ADData:
    """Returned by load(). Holds the DataFrame plus HD index arrays."""

    def __init__(self, df, hd_logical, hd_cell_ids, hd_animal_ids, unique_ids):
        self.df            = df
        self.hd_logical    = np.asarray(hd_logical,  dtype=bool).flatten()
        self.hd_cell_ids   = np.asarray(hd_cell_ids, dtype=float).flatten()
        self.hd_animal_ids = list(hd_animal_ids)
        self.unique_ids    = np.asarray(unique_ids,  dtype=float).flatten()

    def __repr__(self):
        return (f"ADData  rows={len(self.df)}  "
                f"cells={len(self.unique_ids)}  "
                f"HD={self.hd_logical.sum()}")


# ── entry point ───────────────────────────────────────────────────────────

def load(mat_path: str | Path = _DEFAULT_PATH,
         struct_name: str = "table_struct") -> ADData:
    """
    Load the AD combined table from a MATLAB v7.3 .mat file.

    Reads `table_struct` (plain MATLAB struct, HDF5 Group) not `out_table`
    (MATLAB table object, stored as opaque HDF5 Dataset).

    Returns
    -------
    ADData
        .df            – DataFrame, one row per (Cell_Id, Condition)
        .hd_logical    – bool (ncells,)  True = HD cell
        .hd_cell_ids   – Cell_Id values of HD cells
        .hd_animal_ids – Animal_Id strings of HD cells
        .unique_ids    – all unique Cell_Id values
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    with h5py.File(mat_path, "r") as f:
        if struct_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"{struct_name!r} not found. "
                f"Available keys: {available}\n"
                f"Re-run Build_AD_Table.m to regenerate the .mat file "
                f"(table2struct export was added after the original run)."
            )

        df            = _read_struct(f, struct_name)
        hd_logical    = _flat_array(f, "hd_logical",    bool)
        hd_cell_ids   = _flat_array(f, "hd_cell_ids",   float)
        unique_ids    = _flat_array(f, "unique_ids",     float)
        hd_animal_ids = _cell_strings(f, "hd_animal_ids")

    return ADData(df, hd_logical, hd_cell_ids, hd_animal_ids, unique_ids)


# ── struct → DataFrame ────────────────────────────────────────────────────

def _read_struct(f: h5py.File, var_name: str) -> pd.DataFrame:
    grp   = f[var_name]
    names = list(grp.keys())
    nrows = _nrows(f, grp, names)

    data = {}
    for name in names:
        try:
            data[name] = _read_col(f, grp[name], nrows)
        except Exception as e:
            print(f"  [warn] field {name!r}: {e}")
            data[name] = [None] * nrows

    return pd.DataFrame(data)


def _nrows(f: h5py.File, grp: h5py.Group, names: list[str]) -> int:
    """Infer row count: prefer object-dtype (cell) columns over scalar numeric."""
    for prefer_obj in (True, False):
        for name in names:
            item = grp[name]
            if not isinstance(item, h5py.Dataset):
                continue
            is_obj = item.dtype == object
            if is_obj == prefer_obj and item.size > 1:
                return item.size
    return 0


def _read_col(f: h5py.File,
              item: h5py.Dataset | h5py.Group,
              nrows: int) -> list:
    if isinstance(item, h5py.Dataset):
        arr  = item[:]
        flat = arr.flatten()

        if arr.dtype == object:
            # cell-array column: each element is an HDF5 reference
            return [_deref(f, r) for r in flat]

        # numeric column stored as (1, nrows) due to MATLAB column-major
        if flat.size == nrows:
            return flat.tolist()
        if flat.size == 1:
            return [float(flat[0])] * nrows
        return flat.tolist()

    if isinstance(item, h5py.Group):
        keys = list(item.keys())
        for sub in ("data", "value"):
            if sub in keys:
                return _read_col(f, item[sub], nrows)
        if keys:
            return _read_col(f, item[keys[0]], nrows)

    return [None] * nrows


# ── HDF5 reference decoder ────────────────────────────────────────────────

def _deref(f: h5py.File, ref) -> object:
    """Decode one HDF5 object reference to Python data."""
    if not isinstance(ref, h5py.Reference):
        return float(ref) if np.isscalar(ref) else np.asarray(ref)

    obj = f[ref]

    if isinstance(obj, h5py.Group):
        keys = list(obj.keys())
        if "data" in keys:
            return _deref(f, obj["data"])
        if len(keys) == 1:
            return _deref(f, obj[keys[0]])
        # Multi-field MATLAB struct → return as dict
        return {k: _deref(f, obj[k]) for k in keys} if keys else None

    arr = obj[:]

    # Empty matrix sentinel: uint64 (2,) = [0, 0]
    if arr.dtype == np.uint64 and arr.size == 2:
        return np.array([], dtype=float)

    # MATLAB string: uint16
    if arr.dtype == np.uint16:
        return "".join(chr(c) for c in arr.flatten())

    # Nested cell (object references)
    if arr.dtype == object:
        return [_deref(f, r) for r in arr.flatten()]

    # Numeric array — squeeze removes MATLAB's extra size-1 dimension
    sq = arr.squeeze()
    if sq.size == 0:
        return np.array([], dtype=float)
    return sq


# ── simple array / string-cell helpers ───────────────────────────────────

def _flat_array(f: h5py.File, name: str, dtype=float) -> np.ndarray:
    if name not in f:
        return np.array([], dtype=dtype)
    return f[name][:].flatten().astype(dtype)


def _cell_strings(f: h5py.File, name: str) -> list[str]:
    if name not in f:
        return []
    arr = f[name][:]
    if arr.dtype == object:
        return [_deref(f, r) for r in arr.flatten()]
    return [str(x) for x in arr.flatten()]


# ── convenience selectors ─────────────────────────────────────────────────

def baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Condition"] == "Baseline"].copy()

def stim_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Condition"] == "PD + sound") &
              (df["stim_numspikes"].astype(float) > 0)].copy()

def hd_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Condition"] == "Baseline") & (df["hd_class"] == "HD")].copy()


# ── array helpers ─────────────────────────────────────────────────────────

def stack_curves(df: pd.DataFrame, col: str = "HDRateSmooth") -> np.ndarray:
    arrs  = [np.asarray(v).squeeze() for v in df[col]]
    valid = [a for a in arrs if a.ndim == 1 and a.size > 0]
    return np.vstack(valid) if valid else np.empty((0,))


def get_trigger_time(df: pd.DataFrame) -> np.ndarray:
    for v in df.loc[df["Condition"] == "PD + sound", "trigger_time"]:
        a = np.asarray(v).squeeze()
        if a.size > 0:
            return a
    raise ValueError("No trigger_time found in stim rows.")


def stack_behavioural(df: pd.DataFrame, col: str = "mot_avg") -> np.ndarray:
    arrs  = [np.asarray(v).squeeze() for v in df[col]]
    valid = [a for a in arrs if a.ndim == 1 and a.size > 0]
    return np.vstack(valid) if valid else np.empty((0,))


# ── diagnostic ────────────────────────────────────────────────────────────

def inspect_hdf5(mat_path: str | Path, depth: int = 2) -> None:
    """Print top-level structure and first-level children of each variable."""
    with h5py.File(Path(mat_path), "r") as f:
        print(f"Top-level keys: {list(f.keys())}\n")

        def _v(name, obj):
            if name.count("/") >= depth:
                return
            indent = "  " * name.count("/")
            info   = (f"  [{obj.dtype}  {obj.shape}]"
                      if isinstance(obj, h5py.Dataset) else "  [Group]")
            print(f"{indent}/{name}{info}")

        f.visititems(_v)


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATH
    data = load(path)
    df   = data.df

    print(data)
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df["Condition"].value_counts().to_string(), "\n")
    print(f"HD logical : {data.hd_logical.sum()} / {len(data.hd_logical)} cells")
    print(f"HD cell IDs: {data.hd_cell_ids}")

    hd = hd_cells(df)
    if len(hd):
        curves = stack_curves(hd)
        print(f"\nHD tuning-curve matrix : {curves.shape}  (cells × bins)")

    sr = stim_rows(df)
    if len(sr):
        t  = get_trigger_time(df)
        wh = stack_behavioural(sr)
        print(f"Whisker mean matrix    : {wh.shape}  (t: {t[0]:.2f}→{t[-1]:.2f} s)")
