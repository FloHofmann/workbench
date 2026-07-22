"""
vivo_target.py
--------------
Single source of truth for the in-vivo population PSTH target.

Extracted verbatim (same speaker-sort, 6 ms bins, sigma=3 smooth) from the
data-loading block of DoVM_model.py so the mechanistic model's loss and its
diagnostic plots fit/compare against the SAME real data instead of a hand-drawn
caricature.

Usage:
    from vivo_target import load_vivo_psth
    bins_plot, mean_vivo_rate, mean_vivo_smooth = load_vivo_psth()

`bins_plot` is the bin-center time axis in ms (stimulus at t=0). The two PSTH
arrays are the population mean across cells of the speaker-closest-to-PD trace:
raw and Gaussian-smoothed (sigma=3 bins).
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d

# ── geometry / binning (mirrors DoVM_model.py) ───────────────────────────────
SPEAKER_POSITION = {"a": 93, "w": 178, "e": 272, "r": 356}
SPEAKERS = ["a", "w", "e", "r"]

HALF_WINDOW = 1500
TIME_BIN = 0.006  # seconds (6 ms bins)
NBINS = int(np.round(HALF_WINDOW / (TIME_BIN * 1000)))
RASTER_EDGES = np.linspace(-HALF_WINDOW, HALF_WINDOW, NBINS * 2 + 1)

_centers = np.median(np.vstack([RASTER_EDGES[:-1], RASTER_EDGES[1:]]), axis=0)
BINS_PLOT = np.concatenate([[RASTER_EDGES[0]], _centers])[1:]

SMOOTH_SIGMA = 3

_LOCAL_PARQUET = Path(__file__).resolve().parent / "soso_comb.parquet"
_SHARE_PARQUET = Path(
    r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet"
)
# Prefer the local copy; fall back to the lab share when it isn't present.
DEFAULT_PARQUET = _LOCAL_PARQUET if _LOCAL_PARQUET.exists() else _SHARE_PARQUET


def load_vivo_percell(parquet_path=DEFAULT_PARQUET, min_stims=40):
    """Per-cell speaker-pooled in-vivo PSTHs (the building block of the population mean).

    Returns
    -------
    bins_plot : (n_bins,) ms, bin centers, stimulus at t=0
    per_cell : (n_cells, n_bins) Hz, each cell's PSTH pooled over all speaker locations
    """
    comb_table = pl.read_parquet(Path(parquet_path))
    comb_table = comb_table.filter(pl.col("n_stims") > min_stims)

    raster_times = comb_table["RasterTimes"]
    raster_rows = comb_table["RasterRows"]
    stim_keys = list(raster_times[0].keys())
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}
    speaker_idx = [stim_key_to_idx[k] for k in SPEAKERS]

    new_rate = np.zeros((len(raster_times), len(RASTER_EDGES) - 1, len(stim_keys)))
    has_data = np.zeros((len(raster_times), len(stim_keys)), dtype=bool)
    for i in range(len(raster_times)):
        for j in stim_keys:
            if len(raster_times[i][j]) != 0:
                counts, _ = np.histogram(raster_times[i][j], bins=RASTER_EDGES)
                new_rate[i, :, stim_key_to_idx[j]] = counts / (
                    np.max(raster_rows[i][j]) * TIME_BIN
                )
                has_data[i, stim_key_to_idx[j]] = True

    new_rate = new_rate[:, :, speaker_idx]
    has_data = has_data[:, speaker_idx]

    # POOL ACROSS ALL FOUR SPEAKER LOCATIONS.
    # The soso preparation is HEAD-FIXED: the cell sits at its preferred direction for
    # the whole recording, and the speakers are sound-SOURCE locations, not head
    # directions. Responses do not differ significantly across source location, so
    # speaker identity carries no head-direction information and pooling is the correct
    # treatment -- it also keeps ~4x the trials.
    # (Previously this kept ONLY the speaker whose azimuth was closest to the cell's
    # HDAngle, discarding 3/4 of the data on a criterion that is meaningless here. See
    # DATA.md.)
    per_cell = np.nanmean(
        np.where(has_data[:, None, :], new_rate, np.nan), axis=2
    )  # (cells, bins), NaN-safe if a speaker had no trials
    return BINS_PLOT.copy(), per_cell


def load_vivo_psth(parquet_path=DEFAULT_PARQUET, min_stims=40, smooth_sigma=SMOOTH_SIGMA):
    """Load the population-mean in-vivo PSTH from the combined-table parquet.

    Returns
    -------
    bins_plot : (n_bins,) ms, bin centers, stimulus at t=0
    mean_vivo_rate : (n_bins,) Hz, mean across cells of the speaker-pooled PSTH
    mean_vivo_smooth : (n_bins,) Hz, Gaussian-smoothed (sigma=smooth_sigma bins)
    """
    bins, per_cell = load_vivo_percell(parquet_path, min_stims)
    mean_vivo_rate = np.nanmean(per_cell, axis=0)
    mean_vivo_smooth = gaussian_filter1d(mean_vivo_rate.astype(float), sigma=smooth_sigma)
    return bins, mean_vivo_rate, mean_vivo_smooth


# ── AD dataset (the 77-cell target the model is now fit to) ──────────────────
# "ad"   = AD_combined_table, 77 HD cells, head-fixed, ONE fixed sound location.
#          baseline ~24.1 Hz, FC ~161.8, SC ~89.0; bootstrap noise ceiling 0.973.
# "soso" = SoundSource, 13 cells (after n_stims>40), 4 sound-source locations pooled.
#          baseline ~36.5 Hz, FC ~192.7, SC ~101.1; bootstrap noise ceiling 0.889.
# Both are head-fixed with the cell at its PD throughout -> neither has a head-direction
# axis. See DATA.md. Binning is IDENTICAL (500 bins, same edges), so the two are
# drop-in interchangeable.
DATASET = "ad"

_AD_NPZ = Path(__file__).resolve().parent / "vivo_percell.npz"


def load_ad_percell(npz_path=None):
    """Per-cell PSTHs of the 77-cell AD dataset. Returns (bins, per_cell (77, n_bins)).

    Uses `pd_rate` only. `anti_rate` in that file is an ALL-ZERO placeholder (single
    sound location -> no anti-PD condition exists); never use it. See DATA.md.
    """
    d = np.load(Path(npz_path) if npz_path else _AD_NPZ, allow_pickle=True)
    return d["bins"].copy(), d["pd_rate"]


def load_ad_psth(npz_path=None, smooth_sigma=SMOOTH_SIGMA):
    """Population-mean PSTH of the 77-cell AD dataset.

    Same (bins, rate, smooth) contract as load_vivo_psth.
    """
    bins, per_cell = load_ad_percell(npz_path)
    rate = np.nanmean(per_cell, axis=0)
    return bins, rate, gaussian_filter1d(rate.astype(float), sigma=smooth_sigma)


def load_target(dataset=None):
    """CANONICAL entry point for the population fit target -- use this, not the
    dataset-specific loaders, so every consumer switches together."""
    ds = (dataset or DATASET).lower()
    if ds == "ad":
        return load_ad_psth()
    if ds == "soso":
        return load_vivo_psth()
    raise ValueError(f"unknown dataset {ds!r} (expected 'ad' or 'soso')")


def load_percell(dataset=None):
    """Per-cell traces for the active dataset -> (bins, per_cell). For cross-validation."""
    ds = (dataset or DATASET).lower()
    if ds == "ad":
        return load_ad_percell()
    if ds == "soso":
        return load_vivo_percell()
    raise ValueError(f"unknown dataset {ds!r} (expected 'ad' or 'soso')")


if __name__ == "__main__":
    for ds in ("ad", "soso"):
        bins, raw, smooth = load_target(ds)
        _, pc = load_percell(ds)
        base = raw[(bins >= -50) & (bins < 0)].mean()
        tag = " <- active" if ds == DATASET else ""
        print(f"{ds:5s} cells={pc.shape[0]:3d} bins={len(bins)} "
              f"baseline={base:5.1f} peak={raw.max():6.1f} Hz{tag}")
