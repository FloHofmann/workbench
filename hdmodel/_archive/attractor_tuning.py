import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup Parameters
# ==========================================
N = 100
tau = 10.0
dt = 0.5
T_settle = 100.0  # Allow enough time for the network to reach steady state
time = np.arange(0, T_settle, dt)
theta = np.linspace(-np.pi, np.pi, N, endpoint=False)

# Network Weights
J1 = 4.0
J0 = 1.5
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d_theta = np.angle(np.exp(1j * (theta[i] - theta[j])))
        W[i, j] = J1 * np.cos(d_theta) - J0
W = W / N

# ==========================================
# 2. Stimulus Parameters for Tuning
# ==========================================
I_base = 10.0        # Constant tonic drive
I_cue = 5.0          # A small anchoring sensory drive to dictate the heading
sigma_cue = np.deg2rad(20)

# Initialize the empty matrix to store the heatmap
# Rows = Neuron PD, Columns = Current Heading
tuning_heatmap = np.zeros((N, N))

print("Generating tuning curves across all headings...")

# ==========================================
# 3. Sweep Across All Headings
# ==========================================
for h_idx, heading in enumerate(theta):
    
    # Initialize the network flat for every new heading
    u = np.zeros(N)
    r = np.zeros(N)
    
    # Calculate the targeted sensory cue for this specific heading
    # We use the complex plane trick again to ensure the cue wraps perfectly
    d_theta_cue = np.angle(np.exp(1j * (theta - heading)))
    I_ext = I_base + I_cue * np.exp(-0.5 * (d_theta_cue / sigma_cue)**2)
    
    # Run integration loop until the bump stabilizes at the heading
    for t in time:
        u += (-u + W @ r + I_ext) * (dt / tau)
        r = np.maximum(0, u)
        
    # Store the final, stabilized firing rate of the entire population
    tuning_heatmap[:, h_idx] = r

# ==========================================
# 4. Visualization
# ==========================================
fig, ax = plt.subplots(figsize = (10, 8))

# Plot the heatmap
im = ax.imshow(tuning_heatmap, origin = 'lower', aspect = 'auto', 
               extent=[-180, 180, -180, 180], cmap = 'magma')

# Add a dashed white line along the diagonal to represent perfect tuning
ax.plot([-180, 180], [-180, 180], color='white', linestyle = '--', alpha = 0.5, label = 'Perfect Tuning Match')

ax.set_title("Network Tuning Curve Heatmap\n(Steady-State Firing Rates)", fontsize = 14)
ax.set_xlabel("Current Heading / Stimulus Angle (°)", fontsize = 12)
ax.set_ylabel("Neuron Preferred Direction (PD) (°)", fontsize = 12)

fig.colorbar(im, ax = ax, label = "Firing Rate (Hz)")
ax.legend(loc = 'upper left')

plt.tight_layout()
plt.show()
