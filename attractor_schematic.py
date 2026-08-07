import numpy as np
import matplotlib.pyplot as plt

# Parameters for the tuning curves
kappa = 6.0          # Tuning width (concentration parameter)
mu = np.pi / 2       # Preferred direction (set to 90 degrees/North)
base_amp = 100.0     # Baseline amplitude
inc_amp = 230.0      # Adjusted increased amplitude
noise_floor = 12.0   # Baseline firing rate for standard tuning (anti-PD activity)
noise_std = 4.0      # Standard deviation of the standard tuning noise

# 1. Standard attractor tuning curve (von Mises + baseline noise)
theta = np.linspace(0, 2 * np.pi, 72)

# Generate smooth von Mises component for baseline
smooth_tuning = np.exp(kappa * np.cos(theta - mu))
smooth_tuning = (base_amp - noise_floor) * (smooth_tuning / np.max(smooth_tuning))

# Generate random background noise
np.random.seed(42)
std_noise = np.random.normal(loc=noise_floor, scale=noise_std, size=len(theta) - 1)
std_noise = np.append(std_noise, std_noise[0])
std_noise = np.clip(std_noise, 0, None)

standard_tuning = smooth_tuning + std_noise

# 2. Noisy, coarse 360-degree tuning
theta_coarse = np.linspace(0, 2 * np.pi, 36)
coarse_noise = np.random.normal(loc=inc_amp, scale=15.0, size=len(theta_coarse) - 1)
noisy_coarse_tuning = np.append(coarse_noise, coarse_noise[0])
noisy_coarse_tuning = np.clip(noisy_coarse_tuning, 0, None)  

# 3. Standard tuning with amplitude increased (Noisy inside bounds, 0 outside)
# Base smooth curve scaled to 230% (no noise floor added)
smooth_tuning_3 = np.exp(kappa * np.cos(theta - mu))
smooth_tuning_3 = inc_amp * (smooth_tuning_3 / np.max(smooth_tuning_3))

# Generate zero-mean noise, scaled up proportionally to the new amplitude
scaled_noise_std = noise_std * (inc_amp / base_amp)
noise_3 = np.random.normal(loc=0.0, scale=scaled_noise_std, size=len(theta) - 1)
noise_3 = np.append(noise_3, noise_3[0])

# Define "attractor bounds" (e.g., where the tuning curve is > 2% of peak amplitude)
in_bounds = smooth_tuning_3 > (0.02 * inc_amp)

# Apply noise inside the bounds; strictly zero out everything outside
increased_tuning = np.where(in_bounds, smooth_tuning_3 + noise_3, 0.0)
increased_tuning = np.clip(increased_tuning, 0, None)

# Setup 1x4 figure with polar projections
fig, axs = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(16, 4))
r_max = inc_amp + 30  

# Column 1: Standard tuning
axs[0].plot(theta, standard_tuning, color='blue', linewidth=1.5, marker='.', markersize=4)
axs[0].set_title(f"1. Standard Tuning ({int(base_amp)}%)", pad=15)

# Column 2: 360-degree coarse/noisy tuning
axs[1].plot(theta_coarse, noisy_coarse_tuning, color='red', linewidth=1, marker='o', markersize=3)
axs[1].set_title(f"2. Noisy/Coarse ({int(inc_amp)}%)", pad=15)

# Column 3: Standard tuning (Increased Amp, strictly bounded)
axs[2].plot(theta, increased_tuning, color='green', linewidth=1.5, marker='.', markersize=4)
axs[2].set_title(f"3. Bounded Tuning ({int(inc_amp)}%)", pad=15)

# Column 4: Standard tuning (Back to baseline)
std_noise_2 = np.random.normal(loc=noise_floor, scale=noise_std, size=len(theta) - 1)
std_noise_2 = np.append(std_noise_2, std_noise_2[0])
std_noise_2 = np.clip(std_noise_2, 0, None)
standard_tuning_trial2 = smooth_tuning + std_noise_2

axs[3].plot(theta, standard_tuning_trial2, color='blue', linewidth=1.5, marker='.', markersize=4)
axs[3].set_title(f"4. Standard Tuning ({int(base_amp)}%)", pad=15)

# Standardize formatting across all axes
for ax in axs:
    ax.set_rmax(r_max)
    ax.set_rticks([base_amp, inc_amp])
    ax.set_yticklabels([f'{int(base_amp)}', f'{int(inc_amp)}'], color='grey', size=8)

plt.tight_layout()
plt.show()
