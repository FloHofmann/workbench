"""
Basic noisy ring-attractor model for the head-direction system
with optional sound-evoked input.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Step 1: Define basic simulation parameters
# ============================================================

# Number of head-direction neurons on the ring
N = 60

# Simulation time (seconds) and time-step (seconds)
T = 2.0          # total simulation time
dt = 0.001       # time step = 1 ms
n_steps = int(T / dt)

# Time constant tau (seconds) for firing rate dynamics
tau = 0.05       # 50 ms

# Noise strength (controls stochasticity of firing rate)
sigma_noise = 2.0    # adjust this to change noise level

# For convenience, precompute a time array
time = np.arange(0, T, dt)

# ============================================================
# Step 2: Place neurons on a ring (preferred head directions)
# ============================================================

# Each neuron i has a preferred angle theta_i in [0, 2π)
# Here we space them evenly around the ring
theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

# ============================================================
# Step 3: Define recurrent connectivity (Mexican-hat / cosine)
# ============================================================

# We define weights: W_ij = J0 + J1 * cos(theta_i - theta_j)
# J0 is global inhibition; J1 gives local excitation
J0 = -1.0
J1 = 2.0

# Compute pairwise angle differences
dtheta = theta[:, None] - theta[None, :]
# Ensure angles wrap around the circle correctly
# (for cosine this is not strictly necessary, but it's good practice)
# dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

# Weight matrix (N x N)
W = J0 + J1 * np.cos(dtheta)

# ============================================================
# Step 4: Define firing-rate dynamics and nonlinearity
# ============================================================


def phi(x):
    """
    Static nonlinearity: rectified linear function (ReLU).
    phi(x) = max(0, x)

    You can replace this with a sigmoid if you like.
    """
    return np.maximum(0.0, x)


# Initialize firing rates r_i(0)
# We start with a small "bump" of activity around some angle
r = np.zeros(N)

# Let's initialize activity centered at angle 0
# with a Gaussian bump in the index space (for simplicity)
bump_center_angle = 0.0    # radians
bump_width = np.pi / 6.0   # controls how wide the bump is

# Compute distance of each neuron's preferred angle from bump center
d0 = np.angle(np.exp(1j * (theta - bump_center_angle)))  # wrap to [-π, π]
r_init = np.exp(-0.5 * (d0 / bump_width) ** 2)
r = 10.0 * r_init  # scale up to set initial firing rates

# For storing firing rates over time (for visualization)
r_history = np.zeros((n_steps, N))

# ============================================================
# Step 5: Define sound-evoked input (optional)
# ============================================================

# Let's pretend we deliver a brief sound pulse at t_snd
t_snd = 0.5      # seconds (sound onset)
snd_duration = 0.1  # seconds
snd_amp = 5.0       # amplitude of sound input (constant for all neurons)

# Create a time course S(t) for the sound (simple box pulse)
S = np.zeros(n_steps)
snd_on = int(t_snd / dt)
snd_off = int((t_snd + snd_duration) / dt)
S[snd_on:snd_off] = 1.0

# Option 1: direction-independent sound input:
#   same extra input for all neurons when sound is on.
# Option 2: direction-dependent (e.g., localized around some angle)
# Here we start with option 1 for simplicity.


def I_sound(t_index):
    """
    Sound-evoked input at time step t_index.
    Direction-independent (same for all neurons).
    """
    return snd_amp * S[t_index] * np.ones(N)


# ============================================================
# Step 6: Run the simulation (Euler–Maruyama integration)
# ============================================================

# For decoding the bump angle over time
decoded_angle = np.zeros(n_steps)

for t_idx in range(n_steps):
    # Store current rates
    r_history[t_idx, :] = r

    # ----- 6a. Decode bump position via population vector -----
    # HD estimate: θ_hat(t) = arg( sum_i r_i e^{i θ_i} )
    complex_vector = np.sum(r * np.exp(1j * theta))
    decoded_angle[t_idx] = np.angle(complex_vector)  # in [-π, π]

    # ----- 6b. Compute total synaptic input I_i(t) -----
    # Recurrent input from the network
    I_rec = W @ r   # matrix-vector product

    # Sound-evoked input
    I_snd = I_sound(t_idx)

    # Total input to each neuron
    I_total = I_rec + I_snd
    # (You could add baseline input or velocity input here if desired.)

    # ----- 6c. Update firing rates with Euler–Maruyama -----
    # Rate equation:  τ dr/dt = -r + phi(I_total) + σ * η(t)
    # => r(t+dt) = r(t) + dt/τ[-r + phi(I_total)] + sqrt(dt)*σ * ξ
    noise = np.sqrt(dt) * sigma_noise * np.random.randn(N)

    dr = (dt / tau) * (-r + phi(I_total)) + noise
    r = r + dr

    # Enforce non-negative rates (numerical safety)
    r = np.maximum(0.0, r)

# ============================================================
# Step 7: Plot results
# ============================================================

# Convert decoded_angle from [-π, π] to degrees in [0, 360)
decoded_angle_deg = np.degrees(decoded_angle) % 360.0

# Plot firing rates of a subset of neurons as an image
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.imshow(r_history.T, aspect='auto', origin='lower',
           extent=[0, T, 0, N])
plt.colorbar(label='Firing rate (a.u.)')
plt.xlabel('Time (s)')
plt.ylabel('Neuron index')
plt.title('Ring attractor activity (bump + noise + sound input)')
# Mark sound interval
plt.axvspan(t_snd, t_snd + snd_duration, color='white', alpha=0.3,
            label='Sound')
plt.legend(loc='upper right')

# Plot decoded bump angle over time
plt.subplot(2, 1, 2)
plt.plot(time, decoded_angle_deg, 'k')
plt.axvspan(t_snd, t_snd + snd_duration, color='gray', alpha=0.3,
            label='Sound')
plt.ylim(0, 360)
plt.xlabel('Time (s)')
plt.ylabel('Decoded HD (deg)')
plt.title('Decoded head direction (population vector)')
plt.legend()

plt.tight_layout()
plt.show()
