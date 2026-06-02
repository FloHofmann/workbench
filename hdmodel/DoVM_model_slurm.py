"""
DoVM_model_slurm.py
-------------------
Per-cell CANN optimization, designed for SLURM array jobs.

Usage:
    python DoVM_model_slurm.py $SLURM_ARRAY_TASK_ID

Each job processes one cell: loads its RasterTimes/RasterRows from the
AD combined table, computes a per-cell PSTH, runs differential_evolution
(workers=1), refines at N=360, and writes results to results/.

To find the array bound for the SBATCH script, run:
    python DoVM_model_slurm.py --count
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, minimize

from workbench.data.load_ad_table import load, stim_rows, hd_cells

# ==========================================
# Constants
# ==========================================
N = 100
THETA = np.linspace(-np.pi, np.pi, N, endpoint=False)
SIGMA_PHASIC = 10

DT = 0.2
T_BURN_IN = -300
T_END = 600
TIME = np.arange(T_BURN_IN, T_END, DT)
T_STIM = 100

COS_D_THETA = np.cos(np.angle(np.exp(1j * (THETA[:, None] - THETA[None, :]))))
IDX_0 = np.argmin(np.abs(THETA))
IDX_180 = np.argmin(np.abs(THETA - np.pi))

SPEAKER_POSITION = {"a": 93, "w": 178, "e": 272, "r": 356}
SPEAKERS = ["a", "w", "e", "r"]

HALF_WINDOW = 1500
TIME_BIN = 0.006  # seconds (6 ms bins)
NBINS = int(np.round(HALF_WINDOW / (TIME_BIN * 1000)))
RASTER_EDGES = np.linspace(-HALF_WINDOW, HALF_WINDOW, NBINS * 2 + 1)

_centers = np.median(np.vstack([RASTER_EDGES[:-1], RASTER_EDGES[1:]]), axis=0)
BINS_PLOT = np.concatenate([[RASTER_EDGES[0]], _centers])[1:]

SMOOTH_SIGMA = 3
WIN_LO, WIN_HI = -50, 300

OPT_BOUNDS = [
    (5, 60),       # 0  I_baseline
    (200, 3000),   # 1  A_fast
    (10, 300),     # 2  A_phasic
    (20, 500),     # 3  tau_decay
    (1.0, 30.0),   # 4  J1
    (2.0, 20.0),   # 5  KAPPA
    (0.1, 20.0),   # 6  J0
    (-40, -1),     # 7  U_floor
    (102, 150),    # 8  T_phasic
    (10, 50.0),    # 9  tau_neural
    (1.0, 5.0),    # 10 fc_stim_duration
    (0, 60),       # 11 t_delay
]

PARAM_NAMES = [
    "I_baseline", "A_fast", "A_phasic", "tau_decay",
    "J1", "KAPPA", "J0", "U_floor",
    "T_phasic", "tau_neural", "fc_stim_duration", "t_delay",
]


# ==========================================
# Model
# ==========================================
def run_model_temporal(
    I_baseline, A_fast, A_phasic, tau_decay, J1, KAPPA, J0,
    U_floor, T_phasic, tau_neural, fc_stim_duration,
):
    W_local = (J1 * np.exp(KAPPA * (COS_D_THETA - 1.0)) - J0) / N

    u = 40.0 * np.maximum(0, np.cos(THETA))
    r = np.maximum(0, u)
    rates = np.zeros((len(TIME), N))

    sigma_phasic_rad = np.deg2rad(SIGMA_PHASIC)

    for step, t in enumerate(TIME):
        I_ext = np.ones(N) * I_baseline

        if T_STIM <= t < (T_STIM + fc_stim_duration):
            I_ext += A_fast
        elif t >= T_phasic:
            decay = np.exp(-(t - T_phasic) / tau_decay)
            I_ext += A_phasic * decay * np.exp(-0.5 * (THETA / sigma_phasic_rad) ** 2)

        u += (-u + W_local @ r + I_ext) * (DT / tau_neural)
        u = np.maximum(U_floor, u)
        r = np.maximum(0, u)
        rates[step, :] = r

    mask = TIME >= 0
    return TIME[mask] - T_STIM, rates[mask, :]


# ==========================================
# Per-cell PSTH
# ==========================================
def compute_cell_psth(row):
    """
    Compute firing-rate PSTH for one cell from its RasterTimes/RasterRows dicts.

    row : pandas Series with columns RasterTimes (dict), RasterRows (dict), HDAngle.
    Returns (bins_plot, raw_psth, smoothed_psth) aligned to closest speaker.
    """
    raster_times_dict = row["RasterTimes"]
    raster_rows_dict = row["RasterRows"]
    hd_angle = float(row["HDAngle"])

    stim_keys = list(raster_times_dict.keys())
    stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}

    valid_speakers = [k for k in SPEAKERS if k in stim_key_to_idx]
    if not valid_speakers:
        raise ValueError(f"No known speaker keys found in RasterTimes: {stim_keys}")

    speaker_idx = [stim_key_to_idx[k] for k in valid_speakers]

    cell_rate = np.zeros((len(RASTER_EDGES) - 1, len(stim_keys)))
    for j in stim_keys:
        times = raster_times_dict[j]
        rows_j = raster_rows_dict[j]
        if len(times) == 0:
            continue
        counts, _ = np.histogram(times, bins=RASTER_EDGES)
        n_trials = np.max(rows_j)
        if n_trials > 0:
            cell_rate[:, stim_key_to_idx[j]] = counts / (n_trials * TIME_BIN)

    cell_rate = cell_rate[:, speaker_idx]

    speaker_angles = np.array([SPEAKER_POSITION[k] for k in valid_speakers])
    diff = hd_angle - speaker_angles
    wrapped = (diff + 180) % 360 - 180
    idx_sorted = np.argsort(np.abs(wrapped))
    cell_psth = cell_rate[:, idx_sorted[0]]

    cell_smooth = gaussian_filter1d(cell_psth.astype(float), sigma=SMOOTH_SIGMA)
    return BINS_PLOT.copy(), cell_psth, cell_smooth


# ==========================================
# Loss (closure over per-cell PSTH)
# ==========================================
def make_loss(bins_plot, vivo_rate, vivo_smooth):
    def loss_temporal(params):
        (
            I_baseline, A_fast, A_phasic, tau_decay, J1, KAPPA, J0,
            U_floor, T_phasic, tau_neural, fc_stim_duration, t_delay,
        ) = params

        t_model_rel, rates = run_model_temporal(
            I_baseline, A_fast, A_phasic, tau_decay, J1, KAPPA, J0,
            U_floor, T_phasic, tau_neural, fc_stim_duration,
        )

        if np.any(np.isnan(rates)) or np.any(np.isinf(rates)):
            return 1e10

        pd_trace = rates[:, IDX_0]
        anti_pd_trace = rates[:, IDX_180]

        t_vivo_rel = bins_plot.astype(float) - t_delay
        vivo_mask = (t_vivo_rel >= WIN_LO) & (t_vivo_rel <= WIN_HI)
        model_mask = (t_model_rel >= WIN_LO) & (t_model_rel <= WIN_HI)

        if vivo_mask.sum() < 3 or model_mask.sum() < 3:
            return 1e10

        interp_fn = interp1d(
            t_model_rel[model_mask], pd_trace[model_mask],
            bounds_error=False, fill_value="extrapolate",
        )

        matched_times = t_vivo_rel[vivo_mask]
        model_at_vivo = interp_fn(matched_times)

        early_mask = matched_times <= 20.0
        late_mask = matched_times > 20.0
        hybrid_target = np.zeros_like(matched_times)
        hybrid_target[early_mask] = vivo_rate[vivo_mask][early_mask]
        hybrid_target[late_mask] = vivo_smooth[vivo_mask][late_mask]

        raw_errors = (model_at_vivo - hybrid_target) ** 2
        weights = np.ones_like(raw_errors)
        weights[early_mask] = 5.0
        weighted_mse = np.average(raw_errors, weights=weights)

        baseline_mask = (t_model_rel >= WIN_LO) & (t_model_rel < 0)
        anti_pd_baseline = np.mean(anti_pd_trace[baseline_mask])
        background_error = (anti_pd_baseline - 0.0) ** 2

        baseline_profile = np.mean(rates[baseline_mask, :], axis=0)
        peak = np.max(baseline_profile)
        if peak > 1.0:
            half_max = peak / 2.0
            active_bins = np.sum(baseline_profile >= half_max)
            fwhm_baseline = active_bins * (360.0 / N)
        else:
            fwhm_baseline = 360.0

        fwhm_penalty = 0.0
        if fwhm_baseline < 60.0:
            fwhm_penalty = (60.0 - fwhm_baseline) ** 2
        elif fwhm_baseline > 90.0:
            fwhm_penalty = (fwhm_baseline - 90.0) ** 2

        total_loss = weighted_mse + (background_error * 50.0) + (fwhm_penalty * 5.0)

        if np.isnan(total_loss) or np.isinf(total_loss):
            return 1e10
        return float(total_loss)

    return loss_temporal


# ==========================================
# Main
# ==========================================
def get_cell_list():
    data = load()
    df = data.df
    sr = stim_rows(df)
    hd = hd_cells(df)
    # Only cells with both stim and HD baseline data
    stim_ids = set(zip(sr["Animal_Id"], sr["Cell_Id"].astype(str)))
    hd_ids = set(zip(hd["Animal_Id"], hd["Cell_Id"].astype(str)))
    valid_ids = stim_ids & hd_ids
    cells = (
        sr[["Animal_Id", "Cell_Id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    mask = [
        (row["Animal_Id"], str(row["Cell_Id"])) in valid_ids
        for _, row in cells.iterrows()
    ]
    return cells[mask].reset_index(drop=True), df


if __name__ == "__main__":
    # --count mode: print how many cells exist (use to set SLURM --array bound)
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        cells, _ = get_cell_list()
        print(len(cells))
        sys.exit(0)

    try:
        task_id = int(sys.argv[1])
    except IndexError:
        task_id = 0
        print("No SLURM Array ID found. Defaulting to task_id = 0.")

    cells, df = get_cell_list()

    if task_id >= len(cells):
        print(f"Task {task_id} out of bounds ({len(cells)} cells). Exiting.")
        sys.exit(0)

    animal_id = cells.loc[task_id, "Animal_Id"]
    cell_id = cells.loc[task_id, "Cell_Id"]
    print(f"--- Processing Animal: {animal_id}, Cell: {cell_id} ---")

    # Merge stim + HD baseline into one row for compute_cell_psth
    sr = stim_rows(df)
    hd = hd_cells(df)

    cell_stim = sr[(sr["Animal_Id"] == animal_id) & (sr["Cell_Id"] == cell_id)]
    cell_hd = hd[(hd["Animal_Id"] == animal_id) & (hd["Cell_Id"] == cell_id)]

    stim_row = cell_stim.iloc[0].copy()
    stim_row["HDAngle"] = float(cell_hd.iloc[0]["HDAngle"])

    bins_plot, vivo_rate, vivo_smooth = compute_cell_psth(stim_row)

    if vivo_rate.max() < 1.0:
        print(f"Cell {cell_id}: near-zero PSTH, skipping.")
        sys.exit(0)

    loss_fn = make_loss(bins_plot, vivo_rate, vivo_smooth)

    # --- N=100 global search ---
    print("Running differential_evolution (N=100)...")
    opt_100 = differential_evolution(
        loss_fn,
        OPT_BOUNDS,
        seed=42,
        maxiter=1000,
        popsize=30,
        tol=1e-5,
        mutation=(0.8, 1.9),
        recombination=0.5,
        polish=True,
        disp=False,
        workers=1,  # SLURM handles parallelism — do not use workers=-1
    )
    print(f"N=100 loss: {opt_100.fun:.4f}")

    # --- N=360 local refinement ---
    N = 360
    THETA = np.linspace(-np.pi, np.pi, N, endpoint=False)
    COS_D_THETA = np.cos(np.angle(np.exp(1j * (THETA[:, None] - THETA[None, :]))))
    IDX_0 = np.argmin(np.abs(THETA))
    IDX_180 = np.argmin(np.abs(THETA - np.pi))

    print("Refining with L-BFGS-B (N=360)...")
    opt_360 = minimize(
        loss_fn,
        x0=opt_100.x,
        bounds=OPT_BOUNDS,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100},
    )
    print(f"N=360 loss: {opt_360.fun:.4f}")

    # R² on final fit
    params_final = opt_360.x
    t_delay_final = params_final[11]
    t_model_rel, rates_final = run_model_temporal(*params_final[:11])

    t_vivo_rel = bins_plot - t_delay_final
    vivo_mask = (t_vivo_rel >= WIN_LO) & (t_vivo_rel <= WIN_HI)
    model_mask = (t_model_rel >= WIN_LO) & (t_model_rel <= WIN_HI)

    interp_fn = interp1d(
        t_model_rel[model_mask], rates_final[model_mask, IDX_0],
        bounds_error=False, fill_value="extrapolate",
    )
    model_at_vivo = interp_fn(t_vivo_rel[vivo_mask])
    bio = vivo_rate[vivo_mask]

    ss_res = np.sum((bio - model_at_vivo) ** 2)
    ss_tot = np.sum((bio - np.mean(bio)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"R²: {r_squared:.4f}")

    # Save
    Path("results").mkdir(exist_ok=True)
    out = {
        "animal_id": str(animal_id),
        "cell_id": float(cell_id),
        "hd_angle": float(stim_row["HDAngle"]),
        "loss_100N": float(opt_100.fun),
        "loss_360N": float(opt_360.fun),
        "r_squared": r_squared,
        "params": {n: float(v) for n, v in zip(PARAM_NAMES, params_final)},
    }
    safe_animal = str(animal_id).replace("/", "-").replace("\\", "-")
    out_path = Path("results") / f"cell_{safe_animal}_{cell_id}.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved → {out_path}")
