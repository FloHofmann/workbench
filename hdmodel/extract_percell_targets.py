"""
extract_percell_targets.py
--------------------------
One-time LOCAL extraction of per-cell PSTH targets for the mechanistic per-cell
fit. Reads the AD combined table (same source as df.csv), builds each cell's
PD (speaker closest to HDAngle) and anti-PD (farthest) firing-rate PSTH with the
SAME 6 ms bins / sigma-3 smoothing as vivo_target.py, and writes one portable
artifact:

    vivo_percell.npz

so the cluster fit job (fit_cell_mech.py) needs NO network share, no .mat, and no
`workbench.data` import at runtime -- only this npz + optimization_engine.

Run once, locally, from the hdmodel/ directory:
    python extract_percell_targets.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

# repo root on path so `workbench.data...` (imported transitively) resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# reuse the proven binning constants (do not duplicate)
from DoVM_model_slurm import (
    RASTER_EDGES,
    BINS_PLOT,
    SPEAKER_POSITION,
    SPEAKERS,
    TIME_BIN,
    SMOOTH_SIGMA,
)
from workbench.data.load_ad_table import load, stim_rows, hd_cells


def get_cell_list(df):
    """Cells with both stim and HD-baseline data, as (Animal_Id, Cell_Id) rows.
    Same logic as DoVM_model_slurm.get_cell_list but takes an already-loaded df
    (whose load() used the default Documents path)."""
    sr, hd = stim_rows(df), hd_cells(df)
    stim_ids = set(zip(sr["Animal_Id"], sr["Cell_Id"].astype(str)))
    hd_ids = set(zip(hd["Animal_Id"], hd["Cell_Id"].astype(str)))
    valid = stim_ids & hd_ids
    cells = sr[["Animal_Id", "Cell_Id"]].drop_duplicates().reset_index(drop=True)
    mask = [(r["Animal_Id"], str(r["Cell_Id"])) in valid for _, r in cells.iterrows()]
    return cells[mask].reset_index(drop=True)


def _speaker_matrix(raster_times, raster_rows, parity=None):
    """(n_bins, n_valid_speakers) rate matrix + the valid speaker labels, dict path.

    parity=None uses all trials; parity=0/1 keeps only even/odd trial rows (for the
    split-half reliability check). Trial COUNT for a half is derived from the row
    range, not from rows present, so silent trials still count in the denominator.
    """
    valid = [k for k in SPEAKERS if k in raster_times]
    mat = np.zeros((len(RASTER_EDGES) - 1, len(valid)))
    for col, k in enumerate(valid):
        times, rows_k = np.asarray(raster_times[k]).ravel(), np.asarray(raster_rows[k]).ravel()
        if times.size == 0 or rows_k.size == 0:
            continue
        max_row = int(np.max(rows_k))
        if parity is None:
            sel = np.ones(times.shape, dtype=bool)
            n_trials = max_row
        else:
            sel = (rows_k.astype(int) % 2) == parity
            n_trials = sum(1 for r in range(1, max_row + 1) if r % 2 == parity)
        if n_trials > 0 and sel.any():
            counts, _ = np.histogram(times[sel], bins=RASTER_EDGES)
            mat[:, col] = counts / (n_trials * TIME_BIN)
    return mat, valid


def _pd_anti(row, parity=None):
    """Return (pd_rate, anti_rate) raw PSTHs for one cell, both (n_bins,).

    !! anti_rate IS NOT DATA FOR THIS DATASET -- IT IS ZEROS. !!
    AD_combined_table is head-fixed with a SINGLE fixed sound location, so the cell
    sits at its PD throughout and there is no anti-PD condition to measure. Such rows
    take the ARRAY path below, which returns anti = np.zeros(nbins) -- a placeholder
    kept only for schema compatibility with downstream loaders. Every `sc_anti_data`
    derived from it is identically 0.0 and must NEVER be cited as a measurement.
    (It previously was, yielding a vacuous "0.0 predicted / 0.0 measured" validation.)

    The dict/speaker path below is DEAD for this dataset; it dates from the soso table
    (4 sound-source locations). Even there it would be wrong: speaker azimuth is a
    SOUND-SOURCE location, not a head direction, so sorting speakers by
    |HDAngle - azimuth| does not produce an anti-preferred-direction condition.
    See DATA.md.

    parity=0/1 restricts to even/odd trials (split-half).
    """
    rt, rr = row["RasterTimes"], row["RasterRows"]
    hd_angle = float(row["HDAngle"])
    nbins = len(RASTER_EDGES) - 1

    if isinstance(rt, dict):
        mat, valid = _speaker_matrix(rt, rr, parity)
        if not valid:
            return np.zeros(nbins), np.zeros(nbins)
        angles = np.array([SPEAKER_POSITION[k] for k in valid])
        wrapped = (hd_angle - angles + 180) % 360 - 180
        order = np.argsort(np.abs(wrapped))  # 0 = PD-closest ... -1 = anti-PD
        return mat[:, order[0]], mat[:, order[-1]]

    # array path: flat combined spikes -> PD only
    times = np.asarray(rt).flatten()
    rows_arr = np.asarray(rr).flatten()
    if rows_arr.size == 0 or times.size == 0:
        return np.zeros(nbins), np.zeros(nbins)
    max_row = int(np.max(rows_arr))
    if parity is None:
        sel = np.ones(times.shape, dtype=bool)
        n_trials = max_row
    else:
        sel = (rows_arr.astype(int) % 2) == parity
        n_trials = sum(1 for r in range(1, max_row + 1) if r % 2 == parity)
    if n_trials == 0 or not sel.any():
        return np.zeros(nbins), np.zeros(nbins)
    counts, _ = np.histogram(times[sel], bins=RASTER_EDGES)
    return counts / (n_trials * TIME_BIN), np.zeros(nbins)


def main(out_path=str(Path(__file__).resolve().parent / "vivo_percell.npz")):
    df = load().df  # default path: C:\Users\FloHofmann\Documents\AD_combined_table.mat
    cells = get_cell_list(df)
    sr, hd = stim_rows(df), hd_cells(df)
    print(f"{len(cells)} candidate cells (stim + HD baseline)")

    pd_r, pd_s, an_r, an_s = [], [], [], []
    animals, cell_ids, hd_angles = [], [], []
    skipped = 0

    for _, c in cells.iterrows():
        a, cid = c["Animal_Id"], c["Cell_Id"]
        stim = sr[(sr["Animal_Id"] == a) & (sr["Cell_Id"] == cid)]
        base = hd[(hd["Animal_Id"] == a) & (hd["Cell_Id"] == cid)]
        if stim.empty or base.empty:
            skipped += 1
            continue
        row = stim.iloc[0].copy()
        row["HDAngle"] = float(base.iloc[0]["HDAngle"])

        pd_rate, anti_rate = _pd_anti(row)
        if pd_rate.max() < 1.0:  # near-zero PSTH -> unusable target
            skipped += 1
            continue

        pd_r.append(pd_rate)
        pd_s.append(gaussian_filter1d(pd_rate.astype(float), SMOOTH_SIGMA))
        an_r.append(anti_rate)
        an_s.append(gaussian_filter1d(anti_rate.astype(float), SMOOTH_SIGMA))
        animals.append(str(a))
        cell_ids.append(float(cid))
        hd_angles.append(float(row["HDAngle"]))

    np.savez(
        out_path,
        bins=BINS_PLOT,
        pd_rate=np.array(pd_r),
        pd_smooth=np.array(pd_s),
        anti_rate=np.array(an_r),
        anti_smooth=np.array(an_s),
        animal_ids=np.array(animals),
        cell_ids=np.array(cell_ids),
        hd_angles=np.array(hd_angles),
    )
    print(f"kept {len(pd_r)} cells, skipped {skipped} -> {out_path}")
    if pd_r:
        ex = np.array(pd_r[0])
        print(f"  cell0 {animals[0]}/{cell_ids[0]}: PD peak {ex.max():.0f} Hz "
              f"@ {BINS_PLOT[ex.argmax()]:.0f} ms")


def main_split(out_path=str(Path(__file__).resolve().parent / "vivo_percell_split.npz")):
    """Same cells / same order as vivo_percell.npz, but each cell's PD PSTH built
    from EVEN vs ODD stimulus trials separately -> split-half reliability of the
    per-cell fits. Cell inclusion still uses the FULL PSTH so the row order matches."""
    df = load().df
    cells = get_cell_list(df)
    sr, hd = stim_rows(df), hd_cells(df)

    a_r, a_s, b_r, b_s, animals, cell_ids = [], [], [], [], [], []
    for _, c in cells.iterrows():
        a, cid = c["Animal_Id"], c["Cell_Id"]
        stim = sr[(sr["Animal_Id"] == a) & (sr["Cell_Id"] == cid)]
        base = hd[(hd["Animal_Id"] == a) & (hd["Cell_Id"] == cid)]
        if stim.empty or base.empty:
            continue
        row = stim.iloc[0].copy()
        row["HDAngle"] = float(base.iloc[0]["HDAngle"])
        if _pd_anti(row)[0].max() < 1.0:      # same inclusion test as main()
            continue
        pd_a, _ = _pd_anti(row, parity=0)
        pd_b, _ = _pd_anti(row, parity=1)
        a_r.append(pd_a); a_s.append(gaussian_filter1d(pd_a.astype(float), SMOOTH_SIGMA))
        b_r.append(pd_b); b_s.append(gaussian_filter1d(pd_b.astype(float), SMOOTH_SIGMA))
        animals.append(str(a)); cell_ids.append(float(cid))

    np.savez(out_path, bins=BINS_PLOT,
             pd_rate_a=np.array(a_r), pd_smooth_a=np.array(a_s),
             pd_rate_b=np.array(b_r), pd_smooth_b=np.array(b_s),
             animal_ids=np.array(animals), cell_ids=np.array(cell_ids))
    print(f"split-half: {len(a_r)} cells -> {out_path}")
    if a_r:
        ca = np.corrcoef(np.array(a_r).ravel(), np.array(b_r).ravel())[0, 1]
        print(f"  half-to-half PSTH correlation (all cells pooled): r = {ca:.3f}")


if __name__ == "__main__":
    import sys
    main_split() if "--split" in sys.argv else main()
