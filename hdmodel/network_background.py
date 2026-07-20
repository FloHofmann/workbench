"""
network_background.py
---------------------
Run the population (average) network ONCE and record the synaptic drivers a single
readout cell receives. Run this before fitting; it produces network_drivers.npz.

The ring is held at the population fit -- the "average HD activity" the model was
validated against. Only the readout cell is heterogeneous (see readout_cell.py).

All drivers are DERIVED from the returned rates + population params, so no engine
edit is needed:
  (K_E@r_E)_i(t)  -- from the STALE rate sequence (the loop computes rec_E from the
                     previous step's r_E, optimization_engine.py:312)
  (K_I@r_I)_i(t)  -- the interneuron ring re-integrated offline (it depends only on
                     r_E: optimization_engine.py:378-380)
  mean(r_E)(t)    -- ring mean of the stale rates
  a(t)            -- the global adaptation state (optimization_engine.py:372-376)

Recorded at BOTH the PD cell (_IDX_0) and the antipode (_IDX_180); the anti-PD
drivers give a held-out test of emergent tuning (fit on PD, predict anti-PD).

    python network_background.py        # writes npz + runs the exactness test
"""

import json
from pathlib import Path

import numpy as np

import optimization_engine as oe
from readout_cell import CELL_PARAMS, simulate_readout

HERE = Path(__file__).resolve().parent
POP = HERE / "_full_fit_result.json"
OUT = HERE / "network_drivers.npz"


def build(pop=None):
    p = pop or json.load(open(POP))["params"]
    x = [p[k] for k in oe.PARAM_NAMES]
    KAPPA_E, KAPPA_I, W_EI = p["KAPPA_E"], p["KAPPA_I"], p["W_EI"]
    tau_I, tau_a = p["tau_I"], p["tau_a"]

    # the average network, run once
    time, rates = oe.run_model_stim(*x)

    # kernels exactly as the engine builds them (optimization_engine.py:270-271)
    K_E = np.exp(KAPPA_E * (oe.COS_D_THETA - 1.0)) / oe.N
    K_I = np.exp(KAPPA_I * (oe.COS_D_THETA - 1.0)) / oe.N

    # STALE rate sequence: rec_E at step k uses r_E from before step k
    u_E0_vec = 40.0 * np.exp(KAPPA_E * (np.cos(oe.THETA) - 1.0))
    r_E_init = np.minimum(oe.R_MAX, oe.GAIN_SLOPE * np.maximum(0.0, u_E0_vec))
    r_E_stale = np.vstack([r_E_init[None, :], rates[:-1, :]])

    KE_R = r_E_stale @ K_E                      # (T, N), K_E symmetric
    mean_rE = r_E_stale.mean(axis=1)

    # interneuron ring, re-integrated offline (r_I is updated BEFORE du_E)
    u_I = W_EI * (K_E @ r_E_init)
    r_I_all = np.zeros_like(rates)
    for k in range(len(time)):
        u_I = u_I + (-u_I + W_EI * KE_R[k]) * (oe.DT / tau_I)
        r_I_all[k] = np.minimum(500.0, np.maximum(0.0, u_I))
    KI_RI = r_I_all @ K_I

    # global adaptation state (integrated only from stim onset)
    stim_onset = oe.T_STIM + p["stim_delay"]
    a = 0.0
    a_net = np.zeros(len(time))
    for k, t in enumerate(time):
        if t >= stim_onset:
            a += (mean_rE[k] - a) * (oe.DT / tau_a)
            a_net[k] = a

    out = dict(
        time=time, mean_rE=mean_rE, a_net=a_net,
        J1=p["J1"], W_IE=p["W_IE"], W_GLOBAL=p["W_GLOBAL"],
    )
    for site, i in [("pd", oe._IDX_0), ("anti", oe._IDX_180)]:
        out[f"kE_r_{site}"] = KE_R[:, i]
        out[f"kI_rI_{site}"] = KI_RI[:, i]
        out[f"u_E0_{site}"] = u_E0_vec[i]
        out[f"hd_shape_{site}"] = oe.HD_SHAPE[i]
    return out, rates, p


def exactness_test(drv, rates, p):
    """Re-simulate the readout cell with the POPULATION cellular params and confirm
    it reproduces the full ring's PD trace. If this fails, the decomposition is
    wrong and nothing downstream can be trusted."""
    cell = np.array([p[k] for k in CELL_PARAMS], dtype=np.float64)
    y = simulate_readout(
        cell, drv["time"], drv["kE_r_pd"], drv["kI_rI_pd"], drv["mean_rE"], drv["a_net"],
        drv["J1"], drv["W_IE"], drv["W_GLOBAL"], drv["u_E0_pd"], drv["hd_shape_pd"], 0)
    ref = rates[:, oe._IDX_0]
    err = np.max(np.abs(y - ref))
    rel = err / max(ref.max(), 1e-9)
    print(f"exactness: max|readout - ring_PD| = {err:.3e} Hz  (rel {rel:.2e}, peak {ref.max():.1f} Hz)")
    return err, rel


if __name__ == "__main__":
    drv, rates, p = build()
    err, rel = exactness_test(drv, rates, p)
    np.savez(OUT, **drv)
    print(f"saved -> {OUT}  ({len(drv['time'])} steps)")
    if rel < 1e-6:
        print("PASS: single-cell decomposition is exact.")
    else:
        print("FAIL: drivers do not reproduce the ring -- do not fit until this passes.")
