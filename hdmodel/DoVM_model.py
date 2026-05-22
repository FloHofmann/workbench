# %% [markdown]
# # Exploration of the HD model of sound responses of thalamic HD cells (DoVM Architecture)

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

U_FLOOR = -15

# Initial DoVM Guesses 
J_EXC_INIT = 12
K_EXC_INIT = 8
J_INH_INIT = 4
K_INH_INIT = 1

# Creation of the Connectivity Matrix and their assigned preferred directions
COS_D_THETA = np.cos(np.angle(np.exp(1j * (THETA[:, None] - THETA[None, :]))))
IDX_0 = np.argmin(np.abs(THETA))
IDX_180 = np.argmin(np.abs(THETA - np.pi))

W = (J_EXC_INIT * np.exp(K_EXC_INIT * (COS_D_THETA - 1)) - J_INH_INIT * np.exp(K_INH_INIT * (COS_D_THETA - 1))) / N

# Timestamps in ms to plot the snapshots of the attractor activity
times_to_plot = [90, 101, 110, 150]

# visualization of the DoVM 
def plot_mexican_hat(J_exc, K_exc, J_inh, K_inh):
    theta_deg = np.linspace(-180, 180, 500)
    theta_rad = np.radians(theta_deg)
    
    W_kernel = (J_exc * np.exp(K_exc * (np.cos(theta_rad) - 1)) - 
                J_inh * np.exp(K_inh * (np.cos(theta_rad) - 1)))
                
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta_deg, W_kernel, color='black', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.fill_between(theta_deg, W_kernel, 0, where=(W_kernel > 0), interpolate=True,
                    color='tomato', alpha=0, label='Local Excitation')
    ax.fill_between(theta_deg, W_kernel, 0, where=(W_kernel <= 0), interpolate=True,
                    color='royalblue', alpha=0, label='Lateral Inhibition')
    ax.set_title('Difference of Von Mises (DoVM) Synaptic Kernel', fontsize=14)
    ax.set_xlabel('Angular Distance $\\Delta\\theta$ (Degrees)', fontsize=12)
    ax.set_ylabel('Synaptic Weight ($W$)', fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.show()

# Run the model
def run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J_exc, K_exc, J_inh, K_inh, U_FLOOR):
    """
    Runs a comparison between 
    """
    W_local = (J_exc * np.exp(K_exc * (COS_D_THETA - 1.0)) - J_inh * np.exp(K_inh * (COS_D_THETA - 1.0))) / N

    u = 40.0 * np.maximum(0, np.cos(THETA))
    r = np.maximum(0, u)
    rates = np.zeros((len(TIME), N))

    for step, t in enumerate(TIME):
        I_ext = np.ones(N) * I_baseline
        
        if T_STIM <= t < (T_STIM + FC_STIM_DURATION):
            I_ext += A_fast
            
        elif t >= T_PHASIC:
            decay = np.exp(-(t - T_PHASIC) / tau_decay)
            I_ext += A_phasic * decay * np.exp(-0.5 * (THETA / SIGMA_PHASIC) ** 2)
            
        u += (-u + W_local @ r + I_ext) * (DT / TAU_NEURAL)
        u = np.maximum(U_FLOOR, u)
        
        r = np.maximum(0, u)
        rates[step, :] = r

    mask = TIME >= 0
    return TIME[mask] - T_STIM, rates[mask, :]

# full plotting of the model
def plot_model_full(I_baseline, A_fast, A_phasic, tau_decay, J_exc, K_exc, J_inh, K_inh, U_FLOOR, title_suffix=""):
    """Run CANN with given params and produce the full diagnostic plot."""
    t_rel, rates_plot = run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J_exc, K_exc, J_inh, K_inh, U_FLOOR)
    time_plot = t_rel + T_STIM  

    idx_0   = np.argmin(np.abs(THETA))
    idx_90  = np.argmin(np.abs(THETA - np.pi / 2))
    idx_180 = np.argmin(np.abs(THETA - np.pi))

    fwhm = np.zeros(len(time_plot))
    for i in range(len(time_plot)):
        profile = rates_plot[i, :]
        peak = np.max(profile)
        if peak > 1.0: 
            half_max = peak / 2.0
            active_bins = np.sum(profile >= half_max)
            fwhm[i] = active_bins * (360.0 / N)
        else:
            fwhm[i] = 0.0

    fig = plt.figure(figsize=(12, 14)) 
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(
        rates_plot.T, aspect='auto', origin='lower',
        extent=[0, T_END, -180, 180], cmap='magma', vmin=0,
    )
    ax1.set_ylabel("Preferred Direction (deg)")
    ax1.set_title(f"Network Activity{title_suffix}")
    ax1.axvline(T_STIM, color='white', linestyle='--', alpha=0.5)
    ax1.axvline(T_PHASIC, color='cyan', linestyle='--', alpha=0.5)
    fig.colorbar(im, ax=ax1, label="Firing Rate (Hz)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_plot, rates_plot[:, idx_0],   label="PD Cell (0°)",   color="red",    lw=2)
    ax2.plot(time_plot, rates_plot[:, idx_90],  label="PD Cell (90°)",  color="orange", lw=2)
    ax2.plot(time_plot, rates_plot[:, idx_180], label="PD Cell (180°)", color="blue",   lw=2)
    ax2.axvspan(0, T_STIM, color="gray", alpha=0.1)
    ax2.axvspan(T_STIM + FC_STIM_DURATION, T_PHASIC, color="red", alpha=0.1)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_title(f"Temporal Traces (PSTH){title_suffix}")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

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

# load data to optimize the model towards
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

# loss function for optimizer
def loss_temporal(params):
    I_baseline, A_fast, A_phasic, tau_decay, J_exc, K_exc, J_inh, K_inh, U_floor, t_delay = params

    t_model_rel, rates = run_model_temporal(I_baseline, A_fast, A_phasic, tau_decay, J_exc, K_exc, J_inh, K_inh, U_floor)
    
    # 1. Grab the PD cell AND the cell on the exact opposite side of the ring
    pd_trace = rates[:, IDX_0]
    anti_pd_trace = rates[:, IDX_180] 

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
    
    # Penalty 1: Shape Error (Weighted MSE)
    raw_errors = (model_at_vivo - mean_vivo_smooth[vivo_mask]) ** 2
    weights = np.ones_like(raw_errors) 
    critical_zone = (matched_times >= 0) & (matched_times <= 40)
    weights[critical_zone] = 10.0 
    weighted_mse = np.average(raw_errors, weights=weights)
    
    # Penalty 2: Peak Penalty
    max_vivo = np.max(mean_vivo_smooth[vivo_mask])
    max_model = np.max(model_at_vivo)
    peak_error = (max_model - max_vivo) ** 2
    
    # --- Penalty 3: THE BACKGROUND SUPPRESSION PENALTY ---
    # Look at the baseline period (before the stimulus hits at t=0)
    baseline_mask = (t_model_rel >= WIN_LO) & (t_model_rel < 0)
    
    # Calculate the average firing rate of the 180-degree cell during this time
    anti_pd_baseline_rate = np.mean(anti_pd_trace[baseline_mask])
    
    # We want the opposite side of the ring to be dead silent (~0 Hz)
    # If the optimizer tries to flood the network, it gets hit with a massive penalty.
    background_error = (anti_pd_baseline_rate - 0.0) ** 2
    
    # Combine all three penalties
    total_loss = weighted_mse + (peak_error * 50.0) + (background_error * 50.0)
    
    return float(total_loss)

if __name__ == '__main__':
    
    print(f'In vivo bins: {len(bins_plot)}, range {bins_plot[0]}..{bins_plot[-1]} ms')
    print(f'FR range raw:      {mean_vivo_rate.min():.1f}..{mean_vivo_rate.max():.1f} Hz')
    print(f'FR range smoothed: {mean_vivo_smooth.min():.1f}..{mean_vivo_smooth.max():.1f} Hz')

    # DoVM visualization
    #plot_mexican_hat(J_EXC_INIT, K_EXC_INIT, J_INH_INIT, K_INH_INIT)

    # --- Run Optimization with 9 Parameters ---
    OPT_BOUNDS = [
        (5,    60),   # I_baseline 
        (200, 3000),  # A_fast 
        (10,  300),   # A_phasic 
        (20,  500),   # tau_decay 
        (1.0, 30.0),  # J_exc (Excitation Amplitude)
        (2.0, 20.0),  # K_exc (Excitation Sharpness - Forced narrow)
        (0.1, 20.0),  # J_inh (Inhibition Amplitude)
        (0.1, 5.0),   # K_inh (Inhibition Sharpness - Forced wide)
        (-40, -1),    # U_FLOOR (Sets hyperpolarization limit)
        (0,    60),   # t_delay 
    ]

    print('\nRunning differential_evolution with DoVM (Workers Unleashed!)...')
    opt_result = differential_evolution(
        loss_temporal,
        OPT_BOUNDS,
        seed = 42,
        maxiter = 1000,
        popsize = 30,
        tol = 1e-5,
        mutation = (0.8, 1.9),
        recombination = 0.5,
        polish = True,
        disp = True,
        workers = -1, 
    )

    I_opt, Af_opt, Ap_opt, tau_opt, Jexc_opt, Kexc_opt, Jinh_opt, Kinh_opt, U_floor_opt, td_opt = opt_result.x
    
    print(f'\n--- Optimization complete (Weighted Loss = {opt_result.fun:.2f}) ---')
    print(f'  I_BASELINE       = {I_opt:.2f}')
    print(f'  A_FAST           = {Af_opt:.2f}')
    print(f'  A_PHASIC         = {Ap_opt:.2f}')
    print(f'  TAU_PHASIC_DECAY = {tau_opt:.2f} ms')
    print(f'  J_EXC (Amp)      = {Jexc_opt:.2f}')
    print(f'  K_EXC (Sharp)    = {Kexc_opt:.2f}')
    print(f'  J_INH (Amp)      = {Jinh_opt:.2f}')
    print(f'  K_INH (Sharp)    = {Kinh_opt:.2f}')
    print(f'  U_FLOOR          = {U_floor_opt:.2f}')
    print(f'  t_delay          = {td_opt:.2f} ms')

    # --- Post-Optimization Visualizations ---
    t_model_rel, rates_opt = run_model_temporal(I_opt, Af_opt, Ap_opt, tau_opt, Jexc_opt, Kexc_opt, Jinh_opt, Kinh_opt, U_floor_opt)
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
    ax.set_title(f'Optimized DoVM vs In Vivo PD (0 deg)  [latency={td_opt:.1f} ms]')
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
    ax2.set_title('Network Activity (Optimized DoVM)')
    ax2.legend(loc='upper right')
    fig.colorbar(im, ax=ax2, label='Firing Rate (Hz)')

    plt.tight_layout()
    plt.show()

    # Final Diagnostic Plot (3 Rows including FWHM)
    plot_model_full(I_opt, Af_opt, Ap_opt, tau_opt, Jexc_opt, Kexc_opt, Jinh_opt, Kinh_opt, U_floor_opt, title_suffix=" (Optimized DoVM)")

    # 1. Re-align the arrays using the optimized temporal delay
    t_vivo_rel_final = bins_plot.astype(float) - td_opt
    vivo_mask_final  = (t_vivo_rel_final  >= WIN_LO) & (t_vivo_rel_final  <= WIN_HI)
    model_mask_final = (t_model_rel >= WIN_LO) & (t_model_rel <= WIN_HI)
    
    # 2. Interpolate the model to match the exact biological time bins
    interp_fn_final = interp1d(
        t_model_rel[model_mask_final], rates_opt[model_mask_final, IDX_0],
        bounds_error=False, fill_value='extrapolate'
    )
    matched_times_final = t_vivo_rel_final[vivo_mask_final]
    model_at_vivo_final = interp_fn_final(matched_times_final)
    biology_final = mean_vivo_smooth[vivo_mask_final]
    
    # 3. Calculate Pure MSE (No weights, no peak penalties, no background penalties)
    pure_mse = np.mean((model_at_vivo_final - biology_final) ** 2)
    
    # 4. Calculate R-squared (Coefficient of Determination)
    ss_res = np.sum((biology_final - model_at_vivo_final) ** 2) # Residual sum of squares
    ss_tot = np.sum((biology_final - np.mean(biology_final)) ** 2) # Total sum of squares
    r_squared = 1.0 - (ss_res / ss_tot)
    
    print(f"  Pure MSE (Unweighted): {pure_mse:.2f} Hz^2")
    print(f"  R-Squared (R^2):       {r_squared:.4f}")
    
    if r_squared > 0.90:
        print("  Verdict: EXCELLENT fit.")
    elif r_squared > 0.70:
        print("  Verdict: GOOD fit.")
    else:
        print("  Verdict: POOR fit.")
    print("==========================================\n")

    # Increase N to 360
    # ==========================================
    # 7. HIGH-RESOLUTION REFINEMENT (N=360)
    # ==========================================
    print("\n" + "="*42)
    print("   UPSCALING TO N=360 FOR FINE-TUNING   ")
    print("="*42)
    
    # 1. Update Globals for N=360 Spatial Resolution
    N = 360
    THETA = np.linspace(-np.pi, np.pi, N, endpoint=False)
    COS_D_THETA = np.cos(np.angle(np.exp(1j * (THETA[:, None] - THETA[None, :]))))
    IDX_0 = np.argmin(np.abs(THETA))
    IDX_180 = np.argmin(np.abs(THETA - np.pi))
    
    # 2. Setup the "Warm Start" 
    # We take the perfect parameters from the N=100 swarm as our starting point
    best_100N_params = opt_result.x
    
    print('\nRefining N=360 model using Local Gradient Descent (L-BFGS-B)...')
    from scipy.optimize import minimize
    
    opt_result_360 = minimize(
        loss_temporal,
        x0=best_100N_params, 
        bounds=OPT_BOUNDS,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 100}
    )

    # 3. Unpack the high-resolution parameters
    I_h, Af_h, Ap_h, tau_h, Jexc_h, Kexc_h, Jinh_h, Kinh_h, U_floor_h, td_h = opt_result_360.x
    
    print(f'\n--- N=360 Optimization complete (Weighted Loss = {opt_result_360.fun:.2f}) ---')
    print(f'  I_BASELINE       = {I_h:.2f}')
    print(f'  A_FAST           = {Af_h:.2f}')
    print(f'  A_PHASIC         = {Ap_h:.2f}')
    print(f'  TAU_PHASIC_DECAY = {tau_h:.2f} ms')
    print(f'  J_EXC (Amp)      = {Jexc_h:.2f}')
    print(f'  K_EXC (Sharp)    = {Kexc_h:.2f}')
    print(f'  J_INH (Amp)      = {Jinh_h:.2f}')
    print(f'  K_INH (Sharp)    = {Kinh_h:.2f}')
    print(f'  U_FLOOR          = {U_floor_h:.2f}')
    print(f'  t_delay          = {td_h:.2f} ms')

    # 4. Generate High-Resolution Visualizations
    plot_model_full(I_h, Af_h, Ap_h, tau_h, Jexc_h, Kexc_h, Jinh_h, Kinh_h, U_floor_h, title_suffix=" (N=360 Fine-Tuned)")

    # 5. Final High-Resolution Biological Fit Assessment
    t_model_rel_h, rates_opt_h = run_model_temporal(I_h, Af_h, Ap_h, tau_h, Jexc_h, Kexc_h, Jinh_h, Kinh_h, U_floor_h)
    
    t_vivo_rel_h = bins_plot.astype(float) - td_h
    vivo_mask_h  = (t_vivo_rel_h  >= WIN_LO) & (t_vivo_rel_h  <= WIN_HI)
    model_mask_h = (t_model_rel_h >= WIN_LO) & (t_model_rel_h <= WIN_HI)
    
    interp_fn_h = interp1d(
        t_model_rel_h[model_mask_h], rates_opt_h[model_mask_h, IDX_0],
        bounds_error=False, fill_value='extrapolate'
    )
    model_at_vivo_h = interp_fn_h(t_vivo_rel_h[vivo_mask_h])
    biology_h = mean_vivo_smooth[vivo_mask_h]
    
    pure_mse_h = np.mean((model_at_vivo_h - biology_h) ** 2)
    
    ss_res_h = np.sum((biology_h - model_at_vivo_h) ** 2)
    ss_tot_h = np.sum((biology_h - np.mean(biology_h)) ** 2) 
    r_squared_h = 1.0 - (ss_res_h / ss_tot_h)
    
    print("\n==========================================")
    print("   N=360 TRUE BIOLOGICAL FIT ASSESSMENT   ")
    print("==========================================")
    print(f"  Pure MSE (Unweighted): {pure_mse_h:.2f} Hz^2")
    print(f"  R-Squared (R^2):       {r_squared_h:.4f}")
    print("==========================================\n")