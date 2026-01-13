
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 180
theta = np.linspace(0, 2*np.pi, N, endpoint=False)

J0 = -0.6    # global inhibition (more negative = more inhibition)
J1 = 1.4    # local excitation strength (bump support)
tau = 10.0
dt = 0.5
T = 2000.0
steps = int(T / dt)

# Velocity drive gain: how strongly omega shifts the bump
g_vel = 1.0

# Landmark (optional) - set A_landmark=0.0 to disable
A_landmark = 0.2
kappa_landmark = 6.0
theta_landmark = 1.0  # radians

# --- Connectivity (cosine on ring) ---
dtheta = theta[:, None] - theta[None, :]
W = (J0 + J1 * np.cos(dtheta)) / N


def phi(x):
    return np.maximum(0.0, x)  # ReLU


def landmark_input(theta, theta_L, A, kappa):
    return A * np.exp(kappa * np.cos(theta - theta_L))


I_land = landmark_input(theta, theta_landmark, A_landmark, kappa_landmark)

# --- Helper: read out represented head direction from population vector ---


def decode_hd(r, theta):
    z = np.sum(r * np.exp(1j * theta))
    return float(np.angle(z) % (2*np.pi))


# --- Initial condition: seed a bump ---
r = 0.05 * np.random.rand(N)
r += 0.5 * np.exp(10.0 * np.cos(theta - np.pi))  # bump near pi

# --- Define a "true" head direction trajectory + angular velocity ---
# Example: constant turn for a while, then stop, then reverse
psi_true = 0.0
psi_true_hist = []
psi_hat_hist = []


def omega_of_time(t):
    if t < 600:
        return 0.01  # rad per time unit
    elif t < 1200:
        return 0.0
    else:
        return -0.008


# --- Sim loop ---
for k in range(steps):
    t = k * dt
    omega = omega_of_time(t)

    # update true head direction (for comparison)
    psi_true = (psi_true + omega * dt) % (2*np.pi)

    # antisymmetric shift operator ~ derivative on ring
    Kr = (np.roll(r, -1) - np.roll(r, 1)) / 2.0

    x = W @ r + I_land + g_vel * omega * Kr
    dr = (-r + phi(x)) / tau
    r = r + dt * dr

    if k % 5 == 0:
        psi_true_hist.append(psi_true)
        psi_hat_hist.append(decode_hd(r, theta))

psi_true_hist = np.array(psi_true_hist)
psi_hat_hist = np.array(psi_hat_hist)

# --- Plot decoding ---
plt.figure()
plt.plot(psi_true_hist, label="true HD")
plt.plot(psi_hat_hist,  label="decoded HD (network)")
plt.xlabel("time (downsampled)")
plt.ylabel("angle (rad)")
plt.legend()
plt.show()

# --- Plot final firing pattern on the ring ---
plt.figure()
plt.plot(theta, r)
plt.xlabel("preferred direction theta (rad)")
plt.ylabel("activity r")
plt.title("Final population activity (should be a bump)")
plt.show()
