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

DEFAULT_PARQUET = Path(
    r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet"
)


def load_vivo_psth(parquet_path=DEFAULT_PARQUET, min_stims=40, smooth_sigma=SMOOTH_SIGMA):
    """Load the population-mean in-vivo PSTH from the combined-table parquet.

    Returns
    -------
    bins_plot : (n_bins,) ms, bin centers, stimulus at t=0
    mean_vivo_rate : (n_bins,) Hz, population mean of the PD-closest speaker trace
    mean_vivo_smooth : (n_bins,) Hz, Gaussian-smoothed (sigma=smooth_sigma bins)
    """
    comb_table = pl.read_parquet(Path(parquet_path))
    comb_table = comb_table.filter(pl.col("n_stims") > min_stims)

    raster_times = comb_table["RasterTimes"]
    raster_rows = comb_table["RasterRows"]
    stim_keys = list(raster_times[0].keys())
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}
    speaker_idx = [stim_key_to_idx[k] for k in SPEAKERS]

    new_rate = np.zeros((len(raster_times), len(RASTER_EDGES) - 1, len(stim_keys)))
    for i in range(len(raster_times)):
        for j in stim_keys:
            if len(raster_times[i][j]) != 0:
                counts, _ = np.histogram(raster_times[i][j], bins=RASTER_EDGES)
                new_rate[i, :, stim_key_to_idx[j]] = counts / (
                    np.max(raster_rows[i][j]) * TIME_BIN
                )

    new_rate = new_rate[:, :, speaker_idx]

    # Sort each cell's speakers by angular proximity to its preferred direction,
    # then take the closest (index 0) -> the "PD speaker" trace.
    preferred = np.array(comb_table["HDAngle"])
    speaker_angles = np.array(list(SPEAKER_POSITION.values()))
    diff = preferred[:, None] - speaker_angles[None, :]
    wrapped = (diff + 180) % 360 - 180
    idx_sorted = np.argsort(np.abs(wrapped), axis=1)
    new_rate_sort = np.take_along_axis(new_rate, idx_sorted[:, None, :], axis=2)

    mean_vivo_rate = new_rate_sort[:, :, 0].mean(axis=0)
    mean_vivo_smooth = gaussian_filter1d(mean_vivo_rate.astype(float), sigma=smooth_sigma)

    return BINS_PLOT.copy(), mean_vivo_rate, mean_vivo_smooth


if __name__ == "__main__":
    bins, raw, smooth = load_vivo_psth()
    print(f"bins: {len(bins)}, range {bins[0]:.0f}..{bins[-1]:.0f} ms")
    print(f"raw    FR: {raw.min():.1f}..{raw.max():.1f} Hz")
    print(f"smooth FR: {smooth.min():.1f}..{smooth.max():.1f} Hz")
