import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters for the Network and Dynamics ---
N = 100  # Number of HD neurons
tau = 0.050  # Time constant (50 ms)
sigma = 0.1  # Noise amplitude

f_max = 100.0  # Max firing rate (Hz)
beta = 0.1   # Sigmoid steepness (kept smooth)
x_half = 5.0  # Input at 50% max rate

# Gaussian recurrent connectivity
J_E = 25.0        # Local excitatory strength
sigma_W_E = np.pi / 10  # Width of excitatory Gaussian (radians)
# ADJUSTED: Global inhibitory component (from -8.0 to -0.5 for stability)
J_I = -0.5

# External (sensory) input parameters
A_ext = 10.0      # Amplitude of external input bump
sigma_ext = np.pi / 12  # Width of external input bump (radians)

# ADJUSTED: Small constant bias current (from 0.0 to 0.5 to prevent too low activity)
I_bias = 0.5

dt = 0.001        # Time step (1 ms)
T_sim = 2.0       # Total simulation time (2 s)
num_steps = int(T_sim / dt)

# --- 2. HD neuron preferred directions on the ring ---
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# --- 3. Sigmoid activation function ---


def phi(x, f_max, beta, x_half):
    # Ensure numerical stability for extreme inputs
    # If x is very small, -beta * (x - x_half) becomes very large positive, can cause exp overflow.
    # Clip the exponent to prevent this. exp(700) is approx float max.
    exponent = -beta * (x - x_half)
    # Clip upper bound to prevent overflow
    exponent_clipped = np.clip(exponent, None, 500)
    return f_max / (1 + np.exp(exponent_clipped))


# --- 4. Gaussian recurrent connectivity W_ij ---
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        angle_diff = theta[i] - theta[j]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(
            angle_diff))  # wrap to [-pi, pi]
        # Gaussian excitatory connectivity + global inhibition
        W[i, j] = J_E * np.exp(-(angle_diff**2) / (2 * sigma_W_E**2)) + J_I

# --- 5. Angular velocity and head direction trajectory ---
# Define angular velocity omega(t):
# 0–0.5 s: 0 rad/s (no turn)
# 0.5–1.0 s: +pi rad/s (turn 180 degrees over 0.5 s)
# 1.0–2.0 s: 0 rad/s
omega = np.zeros(num_steps)  # rad/s
t = np.arange(num_steps) * dt
omega[(t >= 0.5) & (t < 1.0)] = np.pi  # constant CW turn

# Integrate omega to get head angle theta_head(t)
theta_head = np.zeros(num_steps)
for k in range(1, num_steps):
    theta_head[k] = theta_head[k-1] + omega[k] * dt
    theta_head[k] = np.mod(theta_head[k], 2 * np.pi)  # Wrap to [0, 2pi)

# --- 6. Simulation setup ---
r = np.random.rand(N) * 0.1  # initial firing rates

time_points = t.copy()
firing_rate_history = np.zeros((num_steps, N))
input_history = np.zeros((num_steps, N))

# --- 7. Simulation loop ---
for t_step in range(num_steps):
    # 1. Recurrent input
    I_rec = W @ r

    # 2. External Gaussian input centered at current head direction
    center = theta_head[t_step]
    angle_diff = theta - center
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(
        angle_diff))  # circular difference
    I_ext = A_ext * np.exp(-(angle_diff**2) / (2 * sigma_ext**2))

    # 3. Total input
    I_total = I_rec + I_ext + I_bias

    # 4. Activation + noise
    phi_I = phi(I_total, f_max, beta, x_half)
    noise_term = sigma * np.random.randn(N)

    # 5. Euler update of firing rates
    dr_dt = (1.0 / tau) * (-r + phi_I + noise_term)
    r = r + dr_dt * dt
    r = np.maximum(0, r)  # no negative rates

    # 6. Store
    firing_rate_history[t_step, :] = r
    input_history[t_step, :] = I_total

# --- 8. Visualization ---
plt.figure(figsize=(14, 6))

# (A) Selected neurons over time
plt.subplot(1, 2, 1)
plt.plot(time_points, firing_rate_history[:, 0], label='Neuron 1')
plt.plot(time_points, firing_rate_history[:, N // 4], label='Neuron N/4')
plt.plot(time_points, firing_rate_history[:, N // 2], label='Neuron N/2')
plt.axvspan(0.5, 1.0, color='gray', alpha=0.2, label='Turn interval')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rates of Selected Neurons Over Time')
plt.legend()
plt.grid(True)

# (B) Population heatmap + head direction trajectory
plt.subplot(1, 2, 2)
plt.imshow(
    firing_rate_history.T,
    aspect='auto',
    origin='lower',
    extent=[0, T_sim, 0, 2 * np.pi],
    cmap='hot',
    vmin=0,
    vmax=f_max
)
plt.xlabel('Time (s)')
plt.ylabel('Preferred Direction (radians)')
plt.title('Population Firing Rates with Head Turn')
cbar = plt.colorbar(label='Firing Rate (Hz)')

# Overlay head direction path (scaled to y-axis in radians)
plt.plot(time_points, theta_head, color='cyan',
         linewidth=2, label='Head direction')
plt.legend(loc='upper right')
plt.yticks(ticks=np.linspace(0, 2 * np.pi, 5),
           labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.tight_layout()
plt.show()
