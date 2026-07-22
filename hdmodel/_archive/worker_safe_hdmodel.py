# %% [markdown]
# # Exploration of the HD model of sound responses of thalamic HD cells

# %%
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

from workbench.data.preprocess import combTableCreate, expand_dict_columns

# ==========================================
# 1. Constants & Pre-computations
# ==========================================
N = 100
THETA = np.linspace(-np.pi, np.pi, N, endpoint=False)

DT = 0.2
T_BURN_IN = -300
T_END = 600
TIME = np.arange(T_BURN_IN, T_END, DT)

# Initial/Checkpoint Parameters
I_BASELINE = 34.27
I_TUNING_CUE = 5
SIGMA_CUE = np.deg2rad(15)

FC_STIM_DURATION = 2
A_FAST = 800
A_PHASIC = 60
SIGMA_PHASIC = np.deg2rad(10)

TAU_NEURAL = 10
TAU_PHASIC_DECAY = 100

T_STIM = 100
T_PHASIC = 110

J1 = 8.27
J0 = 2.01
KAPPA = 9

COS_D_THETA = np.cos(np.angle(np.exp(1j * (THETA[:, None] - THETA[None, :]))))
IDX_0 = np.argmin(np.abs(THETA))

W = (J1 * np.exp(KAPPA * (COS_D_THETA - 1.0)) - J0) / N

times_to_plot = [90, 101, 110, 150]

# ==========================================
# 2. Function Definitions
# ==========================================
def plot_mexican_hat(J1=4.0, J0=1.5):
    theta_deg = np.linspace(-180, 180, 500)
    theta_rad = np.radians(theta_deg)
    W_kernel = J1 * np.cos(theta_rad) - J0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta_deg, W_kernel, color='black', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    ax.fill_between(theta_deg, W_kernel, 0, where=(W_kernel > 0), interpolate=True,
                    color='tomato', alpha=0.4, label='Local Excitation (W > 0)')
    ax.fill_between(theta_deg, W_kernel, 0, where=(W_kernel <= 0), interpolate=True,
                    color='royalblue', alpha=0.4, label='Global Inhibition (W < 0)')
    ax.set_title('Mexican Hat Synaptic Kernel\n$W(\\Delta\\theta) = J_1 \\cos(\\Delta\\theta) - J_0$', fontsize=14)
    ax.set_xlabel('Angular Distance $\\Delta\\theta$ (Degrees)', fontsize=12)
    ax.set_ylabel('Synaptic Weight ($W$)', fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(-max(10, J1) - 2, max(10, J1) + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.show()

def run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J1, J0):
    """Run CANN; return (time_rel_to_stim, rates [time x N])."""
    W_local = (J1 * np.exp(KAPPA * (COS_D_THETA - 1.0)) - J0) / N

    u = 40.0 * np.maximum(0, np.cos(THETA))
    r = np.maximum(0, u)
    rates = np.zeros((len(TIME), N))

    # --- THE FIX: Biological Hyperpolarization Limit ---
    # Prevents the massive population spike from driving the network 
    # into an unrecoverable mathematical abyss.
    U_FLOOR = -15.0 

    for step, t in enumerate(TIME):
        I_ext = np.ones(N) * I_baseline
        
        # Phase 1: The Ubiquitous Population Spike (Back in the attractor!)
        if T_STIM <= t < (T_STIM + FC_STIM_DURATION):
            I_ext += A_fast
            
        # Phase 2: The Phasic Rescue
        elif t >= T_PHASIC:
            decay = np.exp(-(t - T_PHASIC) / tau_decay)
            I_ext += A_phasic * decay * np.exp(-0.5 * (THETA / SIGMA_PHASIC) ** 2)
            
        # 1. Standard CANN Integration
        u += (-u + W_local @ r + I_ext) * (DT / TAU_NEURAL)
        
        # 2. Apply the biological floor
        u = np.maximum(U_FLOOR, u)
        
        # 3. Output firing rate
        r = np.maximum(0, u)
        rates[step, :] = r

    mask = TIME >= 0
    return TIME[mask] - T_STIM, rates[mask, :]

def plot_model_full(I_baseline, A_fast, A_phasic, tau_decay, J1, J0, title_suffix=""):
    """Run CANN with given params and produce the full diagnostic plot."""
    t_rel, rates_plot = run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J1, J0)
    time_plot = t_rel + T_STIM  

    idx_0   = np.argmin(np.abs(THETA))
    idx_90  = np.argmin(np.abs(THETA - np.pi / 2))
    idx_180 = np.argmin(np.abs(THETA - np.pi))

    # ==========================================
    # Calculate FWHM (Tuning Width) over time
    # ==========================================
    fwhm = np.zeros(len(time_plot))
    for i in range(len(time_plot)):
        profile = rates_plot[i, :]
        peak = np.max(profile)
        
        # Only calculate if the network is actively firing
        if peak > 1.0: 
            half_max = peak / 2.0
            # Count how many bins (neurons) are firing above the half-max threshold
            active_bins = np.sum(profile >= half_max)
            # Convert bins to spatial degrees
            fwhm[i] = active_bins * (360.0 / N)
        else:
            fwhm[i] = 0.0

    # ==========================================
    # Expanded Visualization (3 Rows)
    # ==========================================
    fig = plt.figure(figsize=(12, 14)) # Increased height to accommodate the new row
    gs = fig.add_gridspec(3, 2)

    # --- Row 1: Heatmap ---
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(
        rates_plot.T, aspect='auto',
        origin='lower',
        extent=[0, T_END, -180, 180],
        cmap='magma', vmin=0,
    )
    ax1.set_ylabel("Preferred Direction (deg)")
    ax1.set_title(f"Network Activity{title_suffix}")
    ax1.axvline(T_STIM, color='white', linestyle='--', alpha=0.5, label=f"Stim {T_STIM - 10} ms")
    ax1.axvline(T_PHASIC, color='cyan', linestyle='--', alpha=0.5, label=f"Rescue {T_PHASIC} ms")
    ax1.legend(loc='upper right')
    fig.colorbar(im, ax=ax1, label="Firing Rate (Hz)")

    # --- Row 2 Left: PSTH ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_plot, rates_plot[:, idx_0],   label="PD Cell (0°)",   color="red",    lw=2)
    ax2.plot(time_plot, rates_plot[:, idx_90],  label="PD Cell (90°)",  color="orange", lw=2)
    ax2.plot(time_plot, rates_plot[:, idx_180], label="PD Cell (180°)", color="blue",   lw=2)
    ax2.axvspan(0, T_STIM, color="gray", alpha=0.1, label="100ms Baseline")
    ax2.axvspan(T_STIM + FC_STIM_DURATION, T_PHASIC, color="red", alpha=0.1, label="Inhibitory Dip")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_title(f"Temporal Traces (PSTH){title_suffix}")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Row 2 Right: Snapshots ---
    ax3 = fig.add_subplot(gs[1, 1])
    snap_colors = ['black', 'red', 'blue', 'green']
    base_names = ['Baseline', 'FC', 'Inhibitory Dip', 'SC']
    snap_labels = [f"{name} ({t} ms)" for name, t in zip(base_names, times_to_plot)]
    for t_target, color, label in zip(times_to_plot, snap_colors, snap_labels):
        ax3.plot(np.rad2deg(THETA), rates_plot[int(t_target / DT), :], label=label, color=color, lw=2)
    ax3.set_xlabel("Preferred Direction (°)")
    ax3.set_ylabel("Firing Rate (Hz)")
    ax3.set_title(f"Spatial Tuning Snapshots{title_suffix}")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # --- Row 3: Temporal FWHM Trace ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time_plot, fwhm, color='purple', lw=2, label="Tuning Width (FWHM)")
    ax4.axvspan(0, T_STIM, color="gray", alpha=0.1)
    ax4.axvspan(T_STIM + FC_STIM_DURATION, T_PHASIC, color="red", alpha=0.1)
    ax4.axvline(T_STIM, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(T_PHASIC, color='cyan', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("FWHM (Degrees)")
    ax4.set_title(f"Spatial Tuning Width Over Time{title_suffix}")
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. Data Loading (Kept Global for Workers)
# ==========================================
conn = sqlite3.connect(
    r"\\172.25.250.112\burgalossi\lab share\Data\Florian\Recordings_FH.db"
)
sql = """
SELECT * FROM Recordings WHERE (Animal_Id, Cell_Id) IN (
SELECT Animal_Id, Cell_Id FROM Recordings WHERE Condition IN ("Baseline", "soso") GROUP BY Animal_Id, Cell_Id
HAVING COUNT(DISTINCT Condition) >= 2)
AND Condition IN ("Baseline","soso") AND use = 1 AND Folders_generated = 1
"""
datatable = pd.read_sql_query(sql, conn)
conn.close()

try:
    comb_table = pl.read_parquet(
        Path(r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet")
    )
except Exception:
    print("comb_table doesn't exist, creating...")
    comb_table = combTableCreate(datatable)
    comb_df = pd.DataFrame(comb_table)
    comb_df = expand_dict_columns(
        comb_df, dict_columns=["pupil_psth", "whisk_psth", "eye_psth"], flatten_2d=False
    )
    comb_table = pl.from_dataframe(comb_df)
    conditions = comb_table["Condition"].unique().to_list()
    comb_a = comb_table.filter(pl.col("Condition") == conditions[0]).drop("Condition")
    comb_b = comb_table.filter(pl.col("Condition") == conditions[1]).drop("Condition")
    comb_joined = comb_a.join(comb_b, on=["Animal_Id", "Cell_Id"], how="inner")
    comb_joined = comb_joined[
        [s.name for s in comb_joined if not (s.null_count() == comb_joined.height)]
    ]
    comb_table = comb_joined.rename(lambda c: c[:-6] if c.endswith("_right") else c)
    comb_table.write_parquet(
        r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet",
        use_pyarrow=True,
    )

comb_table = comb_table.filter(pl.col("n_stims") > 40)

speaker_position = {"a": 93, "w": 178, "e": 272, "r": 356}
speakers = ["a", "w", "e", "r"]

half_window = 1500 
time_bin = 0.006
nbins = int(np.round(half_window / (time_bin * 1000)))
raster_edges = np.linspace(-half_window, half_window, nbins * 2 + 1)

bins_plot = np.concatenate(
    [
        [raster_edges[0]],
        np.median(np.vstack([raster_edges[:-1], raster_edges[1:]]), axis=0),
    ]
)
bins_plot = bins_plot[1:]

raster_times = comb_table["RasterTimes"]
raster_rows  = comb_table["RasterRows"]
stim_keys = list(raster_times[0].keys())
stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}
speaker_idx = [stim_key_to_idx[k] for k in speakers]

new_rate = np.zeros((len(raster_times), len(raster_edges) - 1, len(stim_keys)))
for i in range(len(raster_times)):
    for j in stim_keys:
        if len(raster_times[i][j]) != 0:
            counts, _ = np.histogram(raster_times[i][j], bins=raster_edges)
            new_rate[i, :, stim_key_to_idx[j]] = counts / (np.max(raster_rows[i][j]) * time_bin)

new_rate = new_rate[:, :, speaker_idx]

preferred = np.array(comb_table["HDAngle"])
speaker_angles = np.array(list(speaker_position.values()))
diff = preferred[:, None] - speaker_angles[None, :]
wrapped = (diff + 180) % 360 - 180
idx_sorted = np.argsort(np.abs(wrapped), axis=1)
new_rate_sort = np.take_along_axis(new_rate, idx_sorted[:, None, :], axis=2)

mean_vivo_rate = new_rate_sort[:, :, 0].mean(axis=0)

SMOOTH_SIGMA = 3
mean_vivo_smooth = gaussian_filter1d(mean_vivo_rate.astype(float), sigma=SMOOTH_SIGMA)
WIN_LO, WIN_HI = -50, 300

# ==========================================
# 4. Objective Function (Kept Global)
# ==========================================
def loss_temporal(params):
    I_baseline, A_fast, A_phasic, tau_decay, J1, J0, t_delay = params

    t_model_rel, rates = run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J1, J0)
    pd_trace = rates[:, IDX_0]

    t_vivo_rel = bins_plot.astype(float) - t_delay
    vivo_mask  = (t_vivo_rel  >= WIN_LO) & (t_vivo_rel  <= WIN_HI)
    model_mask = (t_model_rel >= WIN_LO) & (t_model_rel <= WIN_HI)

    if vivo_mask.sum() < 3 or model_mask.sum() < 3:
        return 1e10

    interp_fn = interp1d(
        t_model_rel[model_mask], pd_trace[model_mask],
        bounds_error=False, fill_value='extrapolate'
    )
    
    matched_times = t_vivo_rel[vivo_mask]
    model_at_vivo = interp_fn(matched_times)
    
    # 1. Calculate General Shape Error (Weighted MSE)
    raw_errors = (model_at_vivo - mean_vivo_smooth[vivo_mask]) ** 2
    weights = np.ones_like(raw_errors) 
    critical_zone = (matched_times >= 0) & (matched_times <= 40)
    weights[critical_zone] = 10.0 # Kept moderately high
    weighted_mse = np.average(raw_errors, weights=weights)
    
    # --- 2. THE NEW FIX: EXPLICIT PEAK PENALTY ---
    # Find the absolute highest firing rate in the biological data and the model
    max_vivo = np.max(mean_vivo_smooth[vivo_mask])
    max_model = np.max(model_at_vivo)
    
    # Calculate how far off the model's peak is from the biological peak
    peak_error = (max_model - max_vivo) ** 2
    
    # Combine them. We multiply the peak error by a massive factor (e.g., 50.0) 
    # so the optimizer CANNOT ignore it.
    total_loss = weighted_mse + (peak_error * 50.0)
    
    return float(total_loss)

# ==========================================
# 5. EXECUTION BLOCK (Safe for Multiprocessing)
# ==========================================
if __name__ == '__main__':
    
    print(f'In vivo bins: {len(bins_plot)}, range {bins_plot[0]}..{bins_plot[-1]} ms')
    print(f'FR range raw:      {mean_vivo_rate.min():.1f}..{mean_vivo_rate.max():.1f} Hz')
    print(f'FR range smoothed: {mean_vivo_smooth.min():.1f}..{mean_vivo_smooth.max():.1f} Hz')

    # Optional: Initial Exploratory Plots 
    # (These will only run once on the main thread now!)
    plt.imshow(W, origin='lower', vmin=0)
    plt.xlabel('Neurons')
    plt.ylabel('Neurons')
    plt.colorbar(label='Connection Weight')
    plt.title('Initial Weight Matrix')
    plt.show()

    # --- Run Optimization ---
    OPT_BOUNDS = [
        (5,    60),   # I_baseline 
        (200, 3000),  # A_fast 
        (10,  300),   # A_phasic 
        (20,  500),   # tau_decay 
        (2,    25),   # J1 
        (0.1,  15),   # J0 
        (0,    60),   # t_delay 
    ]

    print('\nRunning differential_evolution (Workers Unleashed!)...')
    opt_result = differential_evolution(
        loss_temporal,
        OPT_BOUNDS,
        seed=42,
        maxiter=500,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=True,
        workers=-1, # Safe to use now!
    )

    I_opt, Af_opt, Ap_opt, tau_opt, J1_opt, J0_opt, td_opt = opt_result.x
    print(f'\n--- Optimization complete (Weighted MSE = {opt_result.fun:.2f}) ---')
    print(f'  I_BASELINE       = {I_opt:.2f}')
    print(f'  A_FAST           = {Af_opt:.2f}')
    print(f'  A_PHASIC         = {Ap_opt:.2f}')
    print(f'  TAU_PHASIC_DECAY = {tau_opt:.2f} ms')
    print(f'  J1               = {J1_opt:.2f}')
    print(f'  J0               = {J0_opt:.2f}')
    print(f'  t_delay          = {td_opt:.2f} ms')

    # --- Post-Optimization Visualizations ---
    t_model_rel, rates_opt = run_model_temporal(I_opt, Af_opt, Ap_opt, tau_opt, J1_opt, J0_opt)
    t_vivo_rel = bins_plot.astype(float) - td_opt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t_vivo_rel, mean_vivo_rate, color='gray', lw=1, alpha=0.4, label='In vivo (raw)')
    ax.plot(t_vivo_rel, mean_vivo_smooth, 'k-', lw=2, label='In vivo (smoothed)')
    ax.plot(t_model_rel, rates_opt[:, IDX_0], 'r-', lw=2, label='Model PD (0 deg)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.6, label='Stim onset')
    ax.axvspan(WIN_LO, WIN_HI, color='lightblue', alpha=0.15, label='Fit window')
    ax.set_xlim(-150, 450)
    ax.set_xlabel('Time relative to stim (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Optimized Model vs In Vivo PD (0 deg)  [latency={td_opt:.1f} ms]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    im = ax2.imshow(
        rates_opt.T, aspect='auto', origin='lower',
        extent=[t_model_rel[0], t_model_rel[-1], -180, 180],
        cmap='magma', vmin=0,
    )
    ax2.axvline(0, color='white', linestyle='--', alpha=0.5, label='Stim onset')
    ax2.axvline(T_PHASIC - T_STIM, color='cyan', linestyle='--', alpha=0.5, label='SC onset')
    ax2.set_xlabel('Time relative to stim (ms)')
    ax2.set_ylabel('Preferred Direction (deg)')
    ax2.set_title('Network Activity (Optimized)')
    ax2.legend(loc='upper right')
    fig.colorbar(im, ax=ax2, label='Firing Rate (Hz)')

    plt.tight_layout()
    plt.show()

    # Final Diagnostic Plot
    plot_model_full(I_opt, Af_opt, Ap_opt, tau_opt, J1_opt, J0_opt, title_suffix=" (Optimized)")