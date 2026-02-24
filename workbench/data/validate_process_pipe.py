"""
validate_pipeline.py

Goal:
- Provide a lightweight "checksum report" for your head-direction / speaker analysis.
- Catch silent convention drift (bin edges, trial normalization, smoothing meaning, PD definition).

How to use:
1) Make sure you can load/build these objects in Python:
   - comb_table (your final joined/analysis table)
   - new_rate (n_cells x n_bins x 4) OR enough to recompute it
   - bins_plot (bin centers in ms for PSTH)
   - time_bin (seconds)
   - DIRECTIONS (e.g. np.linspace(0,360,37) for HD tuning)
   - speakers_deg (array of speaker angles in degrees, e.g. [93,178,272,356])
   - preferred_deg (array of PD in degrees, one per cell)

2) Import and call validate(...). See "Example usage" at bottom.

This script assumes:
- RasterRows represent trial indices (may be 0- or 1-based).
- Trial normalization should use n_unique_trials, NOT max(row).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

try:
    from scipy.stats import friedmanchisquare
except Exception:  # pragma: no cover
    friedmanchisquare = None


# -----------------------------
# Utilities
# -----------------------------

def wrap360(a: np.ndarray) -> np.ndarray:
    return np.mod(a, 360.0)

def circ_dist_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute circular distance in degrees between arrays a and b."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)

def signed_circ_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a-b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def pd_from_rate(
    rate_36_or_37: np.ndarray,
    directions_edges_deg: np.ndarray,
    use: str = "centers",
) -> float:
    """
    Compute preferred direction from an HD tuning curve using complex vector average.

    rate_36_or_37:
        Either 36 bins or 37 entries. If 37, last entry often duplicates wrap.
    directions_edges_deg:
        37 edges [0,10,...,360]
    use:
        'centers' (recommended) or 'edges' (legacy-like).
    """
    r = np.asarray(rate_36_or_37, dtype=float).copy()
    dirs = np.asarray(directions_edges_deg, dtype=float)

    if r.size == dirs.size:
        # drop last value so we have 36 bins matching dirs[:-1]
        r = r[:-1]
    elif r.size != (dirs.size - 1):
        raise ValueError(f"Rate length {r.size} does not match directions {dirs.size}.")

    if use == "edges":
        ang = dirs[:-1]
    elif use == "centers":
        ang = 0.5 * (dirs[:-1] + dirs[1:])
    else:
        raise ValueError("use must be 'edges' or 'centers'")

    vec = np.nansum(r * np.exp(1j * np.deg2rad(ang)))
    return float(wrap360(np.array([np.rad2deg(np.angle(vec))]))[0])

def bh_fdr_adjust(p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values with monotonicity).
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    q = p_sorted * (m / ranks)
    # enforce monotone non-decreasing q from largest to smallest
    q_mon = q.copy()
    for i in range(m - 2, -1, -1):
        q_mon[i] = min(q_mon[i], q_mon[i + 1])
    q_mon = np.clip(q_mon, 0.0, 1.0)
    out = np.empty_like(q_mon)
    out[order] = q_mon
    return out


# -----------------------------
# Report structures
# -----------------------------

@dataclass
class PDShiftSensitivity:
    offset_deg: float
    frac_cells_any_order_change: float
    frac_cells_closest_change: float

@dataclass
class ValidationReport:
    n_cells: int
    speakers_deg: np.ndarray
    time_bin_s: float

    # trial normalization
    trial_index_min: Optional[float]
    trial_index_max: Optional[float]
    median_n_unique_trials: Optional[float]
    median_max_trial_index: Optional[float]
    suspicious_trialcount_mismatch_frac: Optional[float]

    # PD diagnostics
    pd_edges: Optional[np.ndarray]
    pd_centers: Optional[np.ndarray]
    pd_edges_minus_centers_deg: Optional[np.ndarray]  # signed in (-180,180]
    pd_shift_sensitivity: Optional[List[PDShiftSensitivity]]

    # sorted AUC friedman results
    friedman_sorted_auc_p: Optional[float]
    friedman_sorted_auc_stat: Optional[float]

    notes: List[str]


def _safe_unique_trials(rr: np.ndarray) -> int:
    rr = np.asarray(rr)
    if rr.size == 0:
        return 0
    return int(np.unique(rr).size)

def _safe_minmax(rr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    rr = np.asarray(rr)
    if rr.size == 0:
        return None, None
    return float(np.min(rr)), float(np.max(rr))

def _get_row_like(obj: Any, i: int) -> Any:
    """Best-effort row access for polars/pandas/list-like."""
    try:
        return obj[i]
    except Exception:
        return obj[i, :]


# -----------------------------
# Main validator
# -----------------------------

def validate(
    *,
    auc_resp: np.ndarray,
    preferred_deg: np.ndarray,
    speakers_deg: np.ndarray,
    directions_edges_deg: np.ndarray,
    hd_rate_smooth: Optional[np.ndarray] = None,
    raster_rows: Optional[Any] = None,
    raster_times: Optional[Any] = None,
    time_bin_s: float,
    offsets_to_test: Iterable[float] = (-10, -5, 0, 5, 10),
) -> ValidationReport:
    """
    Parameters
    ----------
    auc_resp:
        (n_cells, 4) AUC per speaker in seconds*Hz (or spikes, depending on definition).
    preferred_deg:
        (n_cells,) PD angle degrees used by your pipeline (whatever you currently use).
    speakers_deg:
        (4,) speaker angles in degrees.
    directions_edges_deg:
        (37,) e.g. np.linspace(0,360,37)
    hd_rate_smooth:
        Optional. If provided, (n_cells, 36) or (n_cells, 37) HD tuning curve used to compute PD.
        If present, report will compute PD_edges vs PD_centers and their difference.
    raster_rows / raster_times:
        Optional raw raster structures for trial-count sanity checks. Can be any container where
        raster_rows[i][k] gives an array of trial indices for cell i and speaker k.
        If provided, report checks whether using max(rows) would be wrong.

    Returns
    -------
    ValidationReport
    """
    auc_resp = np.asarray(auc_resp, dtype=float)
    preferred_deg = np.asarray(preferred_deg, dtype=float).reshape(-1)
    speakers_deg = np.asarray(speakers_deg, dtype=float).reshape(-1)
    directions_edges_deg = np.asarray(directions_edges_deg, dtype=float).reshape(-1)

    n_cells = auc_resp.shape[0]
    notes: List[str] = []

    if auc_resp.shape[1] != 4 or speakers_deg.size != 4:
        raise ValueError("Expected auc_resp shape (n_cells, 4) and speakers_deg length 4.")

    # -------------------------
    # Trial normalization checks
    # -------------------------
    trial_index_min = trial_index_max = None
    median_n_unique = median_max_idx = None
    suspicious_frac = None

    if raster_rows is not None:
        mins, maxs, nunq, maxidx = [], [], [], []
        mismatch_flags = []

        # Attempt to handle two common formats:
        # 1) list of dicts: raster_rows[i][speaker_key]
        # 2) 2D table-like: raster_rows[i, speaker_index]
        speaker_keys = ["a", "w", "e", "r"]
        for i in range(n_cells):
            for s in range(4):
                rr = None
                try:
                    # dict-like
                    rr = _get_row_like(raster_rows, i)[speaker_keys[s]]
                except Exception:
                    try:
                        # table-like (i, s)
                        rr = raster_rows[i, s]
                    except Exception:
                        # try list-of-lists
                        rr = _get_row_like(raster_rows, i)[s]

                rr = np.asarray(rr)
                if rr.size == 0:
                    continue

                mn, mx = _safe_minmax(rr)
                nu = _safe_unique_trials(rr)

                mins.append(mn)
                maxs.append(mx)
                nunq.append(nu)
                maxidx.append(mx)

                # If you used max(rr) as #trials, you are wrong when:
                # - rr is 0-based contiguous: n_trials should be max+1 (nu == max+1)
                # - rr is 1-based contiguous: n_trials should be max (nu == max)
                # - rr is missing trials or non-contiguous: max says nothing; must use nu
                # We'll flag if nu differs from either max or max+1
                ok = (nu == mx) or (nu == mx + 1)
                mismatch_flags.append(not ok)

        if mins:
            trial_index_min = float(np.min(mins))
            trial_index_max = float(np.max(maxs))
            median_n_unique = float(np.median(nunq)) if nunq else None
            median_max_idx = float(np.median(maxidx)) if maxidx else None
            suspicious_frac = float(np.mean(mismatch_flags)) if mismatch_flags else None

            if suspicious_frac is not None and suspicious_frac > 0.05:
                notes.append(
                    f"⚠️ Trial-count inference via max(rows) looks unreliable for "
                    f"{suspicious_frac*100:.1f}% of cell×speaker entries. Use n_unique_trials."
                )

            if trial_index_min == 0.0:
                notes.append("RasterRows include 0 → likely 0-based trial indexing present somewhere.")
            if trial_index_min == 1.0:
                notes.append("RasterRows min is 1 → likely 1-based indexing (MATLAB-style) for at least some entries.")
        else:
            notes.append("raster_rows provided but no non-empty entries found; skipped trial-count checks.")
    else:
        notes.append("No raster_rows provided; skipping trial-count normalization checks.")

    # -------------------------
    # PD diagnostics (edges vs centers)
    # -------------------------
    pd_edges = pd_centers = pd_delta = None
    sens: List[PDShiftSensitivity] = []

    if hd_rate_smooth is not None:
        hd_rate_smooth = np.asarray(hd_rate_smooth, dtype=float)
        if hd_rate_smooth.shape[0] != n_cells:
            notes.append("hd_rate_smooth row count does not match auc_resp; PD edge/center check skipped.")
        else:
            pd_edges = np.array([pd_from_rate(hd_rate_smooth[i], directions_edges_deg, "edges") for i in range(n_cells)])
            pd_centers = np.array([pd_from_rate(hd_rate_smooth[i], directions_edges_deg, "centers") for i in range(n_cells)])
            pd_delta = signed_circ_diff_deg(pd_edges, pd_centers)

            notes.append(
                f"PD edge-vs-center: median Δ={np.median(pd_delta):.2f}° "
                f"(IQR {np.percentile(pd_delta,25):.2f}..{np.percentile(pd_delta,75):.2f})."
            )

            # Sensitivity of ordering to PD offsets
            # Compare ordering for preferred_deg (your current PD) vs shifted PDs
            base_pd = wrap360(preferred_deg)
            base_dist = circ_dist_deg(base_pd[:, None], speakers_deg[None, :])
            base_idx = np.argsort(base_dist, axis=1)
            base_closest = np.argmin(base_dist, axis=1)

            for off in offsets_to_test:
                pd_shift = wrap360(base_pd + float(off))
                dist = circ_dist_deg(pd_shift[:, None], speakers_deg[None, :])
                idx = np.argsort(dist, axis=1)
                closest = np.argmin(dist, axis=1)

                sens.append(
                    PDShiftSensitivity(
                        offset_deg=float(off),
                        frac_cells_any_order_change=float(np.mean(np.any(idx != base_idx, axis=1))),
                        frac_cells_closest_change=float(np.mean(closest != base_closest)),
                    )
                )
    else:
        notes.append("No hd_rate_smooth provided; skipping PD edge/center and sensitivity diagnostics.")

    # -------------------------
    # Sorted AUC Friedman
    # -------------------------
    fried_p = fried_stat = None
    if friedmanchisquare is None:
        notes.append("scipy not available; skipping Friedman test.")
    else:
        # sort by distance to preferred_deg
        dist = circ_dist_deg(preferred_deg[:, None], speakers_deg[None, :])
        idx = np.argsort(dist, axis=1)
        auc_sorted = np.take_along_axis(auc_resp, idx, axis=1)

        try:
            fried_stat, fried_p = friedmanchisquare(
                auc_sorted[:, 0], auc_sorted[:, 1], auc_sorted[:, 2], auc_sorted[:, 3]
            )
            fried_stat = float(fried_stat)
            fried_p = float(fried_p)
        except Exception as e:
            notes.append(f"Friedman test failed: {e!r}")

    return ValidationReport(
        n_cells=n_cells,
        speakers_deg=speakers_deg,
        time_bin_s=float(time_bin_s),

        trial_index_min=trial_index_min,
        trial_index_max=trial_index_max,
        median_n_unique_trials=median_n_unique,
        median_max_trial_index=median_max_idx,
        suspicious_trialcount_mismatch_frac=suspicious_frac,

        pd_edges=pd_edges,
        pd_centers=pd_centers,
        pd_edges_minus_centers_deg=pd_delta,
        pd_shift_sensitivity=sens if sens else None,

        friedman_sorted_auc_p=fried_p,
        friedman_sorted_auc_stat=fried_stat,

        notes=notes,
    )


def print_report(r: ValidationReport, max_lines: int = 200) -> None:
    lines: List[str] = []
    lines.append("=== Pipeline Validation Report ===")
    lines.append(f"n_cells: {r.n_cells}")
    lines.append(f"speakers_deg: {r.speakers_deg.tolist()}")
    lines.append(f"time_bin_s: {r.time_bin_s}")

    lines.append("")
    lines.append("Trial indexing / normalization:")
    if r.trial_index_min is None:
        lines.append("  (skipped)")
    else:
        lines.append(f"  trial_index_min: {r.trial_index_min}")
        lines.append(f"  trial_index_max: {r.trial_index_max}")
        lines.append(f"  median_n_unique_trials: {r.median_n_unique_trials}")
        lines.append(f"  median_max_trial_index: {r.median_max_trial_index}")
        lines.append(f"  suspicious max(rows) mismatch frac: {r.suspicious_trialcount_mismatch_frac}")

    lines.append("")
    lines.append("PD diagnostics:")
    if r.pd_edges is None:
        lines.append("  (skipped)")
    else:
        d = r.pd_edges_minus_centers_deg
        lines.append(f"  PD_edges vs PD_centers: median Δ={np.median(d):.2f}°, mean Δ={np.mean(d):.2f}°")
        lines.append(f"  ΔPD percentiles (5/25/50/75/95): {np.percentile(d, [5,25,50,75,95]).round(2).tolist()}")
        lines.append("  Sensitivity to PD offsets (order change rates):")
        for s in r.pd_shift_sensitivity or []:
            lines.append(
                f"    offset {s.offset_deg:+.1f}°: "
                f"any-order change {s.frac_cells_any_order_change*100:.1f}%, "
                f"closest change {s.frac_cells_closest_change*100:.1f}%"
            )

    lines.append("")
    lines.append("Sorted-AUC Friedman:")
    if r.friedman_sorted_auc_p is None:
        lines.append("  (skipped)")
    else:
        lines.append(f"  chi2={r.friedman_sorted_auc_stat:.6g}, p={r.friedman_sorted_auc_p:.6g}")

    lines.append("")
    lines.append("Notes:")
    for n in r.notes:
        lines.append(f"  - {n}")

    print("\n".join(lines[:max_lines]))


# -----------------------------
# Example usage (edit to your objects)
# -----------------------------
if __name__ == "__main__":
    """
    Replace the placeholders below with your own variables.

    Minimal requirements:
      auc_resp: (n_cells, 4)
      preferred: (n_cells,)
      speakers_deg: (4,)
      DIRECTIONS: (37,)
      time_bin: scalar (s)

    Optional but recommended:
      hdRateSmooth: (n_cells, 36 or 37)
      raster_rows: raw raster row indices for trial checks
    """
    # from your pipeline import/build these:
    # auc_resp = ...
    # preferred = ...
    # speakers_deg = np.array([93,178,272,356], float)
    # DIRECTIONS = np.linspace(0,360,37)
    # time_bin = 0.002
    # hdRateSmooth = ...   # optional
    # raster_rows = ...    # optional

    raise SystemExit(
        "Edit validate_pipeline.py __main__ section to pass your variables, "
        "or import validate/print_report from this file."
    )