import numpy as np
import matplotlib.pyplot as plt

times_to_plot = [90, 101, 104, 130]
# Network parameters
N = 100
dt = 0.5
T = 200 # decay time for phasic response
t_burn_in = -50
t_end = 250
time = np.arange(t_burn_in, t_end, dt)

theta = np.linspace(-np.pi, np.pi, N, endpoint = False)

# Stimulus and Tonic parameters
FC_DURATION = 2 # 2 ms
I_baseline = 10 # Tonic excitatory drive
A_fast = 600
A_phasic = 40
sigma_phasic = np.deg2rad(10) # spatial width for the phasic response

tau = 10
tau_decay = 40

t_stim = 100 # Stimulus onset pushed back to 50 ms
t_phasic = 110 # Phasic onset at 55 ms

# Connectivity Matrix
J1 = 4
J0 = 1.5
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d_theta = np.angle(np.exp(1j * (theta[i] - theta[j])))
        W[i, j] = J1 * np.cos(d_theta) - J0
W = W / N

# initialize the model
u = 40 * np.maximum(0, np.cos(theta))
r = np.maximum(0, u)

rates_history = np.zeros((len(time), N))

for step, t in enumerate(time):
    I_ext = np.ones(N) * I_baseline

    # Phase 1: Fast Ubiquitous Spike (2ms duration)
    if t_stim <= t < (t_stim + FC_DURATION):
        I_ext += A_fast

    # Phase 2: Slow spatially restricted component
    elif t >= t_phasic:
        decay_factor = np.exp(-(t - t_phasic) / tau_decay)
        I_ext += A_phasic * decay_factor * np.exp(-0.5 * (theta / sigma_phasic) ** 2)

    # CANN Equation
    du = (-u + W @ r + I_ext) * (dt / tau)
    u += du
    r = np.maximum(0, u)
    rates_history[step, :] = r

# cut initialization time
plot_mask = time >= 0
time_plot = time[plot_mask]
rates_plot = rates_history[plot_mask, :]

# Visualization
fig = plt.figure(figsize = (12, 10))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, :])
im = ax1.imshow(
        rates_plot.T, aspect = 'auto',
        origin = 'lower',
        extent = [0, t_end, -180, 180],
        cmap = 'magma',
        vmin = 0,
        )
ax1.set_ylabel("Preferred Direction (deg)")
ax1.set_title("Network Activity")
ax1.axvline(
        t_stim, color = 'white', linestyle = '--',
        alpha = 0.5, label = 'Spike (100ms)'
        )
ax1.axvline(
        t_phasic, color = 'cyan',
        linestyle = '--',
        alpha = 0.5,
        label = 'Rescue (105 ms)',
        )
ax1.legend(loc = 'upper right')
fig.colorbar(im, ax=ax1, label = "Firing Rate (Hz)")

# Single Cell temporal Trace
ax2 = fig.add_subplot(gs[1, 0])
idx_0 = np.argmin(np.abs(theta))
idx_90 = np.argmin(np.abs(theta - np.pi/2))
idx_180 = np.argmin(np.abs(theta - np.pi))

ax2.plot(time_plot, rates_plot[:, idx_0],
         label = "PD Cell (0°)",
         color = "red",
         lw = 2)
ax2.plot(time_plot, rates_plot[:, idx_90],
         label='Orthogonal (90°)',
         color = 'orange',
         lw=2)
ax2.plot(time_plot, rates_plot[:, idx_180],
         label='Anti-PD (180°)',
         color='blue',
         lw=2)

ax2.axvspan(0, t_stim, color='gray', alpha=0.1, label='100ms Baseline')
ax2.axvspan(t_stim+2, t_phasic, color='red', alpha=0.1, label='Inhibitory Quench')

ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Firing Rate (Hz)")
ax2.set_title("Temporal Traces (PSTH Match)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Panel 3: Spatial Snapshots ---
ax3 = fig.add_subplot(gs[1, 1])
# Adjusted snapshot times to fit the new timeline
colors = ['black', 'red', 'blue', 'green']

# Define just the descriptive part of the labels
base_names = ['Baseline', 'Fast Spike', 'Inhibitory Dip', 'Phasic Rescue']

# Dynamically combine them
labels = [f"{name} ({time}ms)" for name, time in zip(base_names, times_to_plot)]

for t_target, color, label in zip(times_to_plot, colors, labels):
    # Calculate index relative to the sliced plot array
    step_idx = int(t_target / dt) 
    ax3.plot(np.rad2deg(theta), rates_plot[step_idx, :], label=label, color=color, lw=2)

ax3.set_xlabel("Preferred Direction (°)")
ax3.set_ylabel("Firing Rate (Hz)")
ax3.set_title("Spatial Tuning Snapshots")
ax3.legend(loc = 'upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

