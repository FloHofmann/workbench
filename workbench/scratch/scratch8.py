import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat
import matplotlib.pyplot as plt


def alpha(x, A, tau_r, tau_d):
    return A * (jnp.exp(-x / tau_d) - jnp.exp(-x / tau_r))


def k_fast(t):
    Apos, tr1, td1 = 58.8398486, 0.00227383226, 0.00300001189
    Aneg, tr2, td2 = 42.8674114, 0.00519790233, 0.005
    d = 0.01192995120
    x = jnp.maximum(t - d, 0.0)
    return alpha(x, Apos, tr1, td1) - alpha(x, Aneg, tr2, td2)


def k_slow(t):
    As, tau, n, d = 72.1121761, 0.0845112753, 1.0, 0.0
    x = jnp.maximum(t - d, 0.0)
    return As * (x ** n) * jnp.exp(-x / tau)


def normalize(k):
    return k / jnp.max(jnp.abs(k))


mat = loadmat(r"C:\Users\FloHofmann\OneDrive\hdsignal.mat",
              squeeze_me=True, struct_as_record=False)
hdsignal = mat["hdsignal"]

# adjust field names if needed
t = jnp.asarray(np.asarray(hdsignal.time).ravel())
y = jnp.asarray(np.asarray(hdsignal.signal).ravel())

# mask approach
t = t.reshape(-1)
mask = (t >= 0.0) & (t <= 0.2)
t_post = t[mask]

kf = k_fast(t_post)
ks = k_slow(t_post)
kf_n = normalize(kf)
ks_n = normalize(ks)

ks_n = normalize(ks)
# baseline = mean from -0.5 to -0.05 s
baseline_mask = (t >= -0.5) & (t <= -0.05)
baseline = jnp.mean(y[baseline_mask])

y_bs = y - baseline

# restrict signal to same post-stim window
y_post = y_bs[mask]

# ---- kernels ----
kf = k_fast(t_post)
ks = k_slow(t_post)
ksum = kf + ks

# ---- normalize for visual comparison only ----


def norm(x):
    return x / jnp.max(jnp.abs(x))


y_plot = norm(y_post)
kf_plot = norm(kf)
ks_plot = norm(ks)
ksum_plot = norm(ksum)

# ---- plot ----
plt.figure(figsize=(6, 4))

plt.plot(t_post * 1e3, y_plot,  color="black", lw=2, label="data")
plt.plot(t_post * 1e3, kf_plot, color="tab:blue", lw=2, label="fast kernel")
plt.plot(t_post * 1e3, ks_plot, color="tab:purple", lw=2, label="slow kernel")
plt.plot(t_post * 1e3, ksum_plot, color="tab:red",
         lw=2, ls="--", label="fast + slow")

plt.axvline(0, color="gray", ls=":", lw=1)

plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
