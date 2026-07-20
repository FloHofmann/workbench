"""
probe_heterogeneous_ring.py
---------------------------
Retires the last idealization: in the readout model the target cell is heterogeneous
but its NEIGHBOURS are still the homogeneous population cell. Here we build a ring
in which EVERY cell is a different fitted cell, sampled from the 77 per-cell fits,
and ask:

  1. Does a genuinely heterogeneous attractor still form a single stable bump, and
     does it recover (width + rate) like the homogeneous one?
  2. Does its emergent population-mean PSTH still match the measured population PSTH
     the homogeneous model was fit to?

If yes, the mean-field decomposition is not just self-consistent on the mean (the
R2=0.966 check) but robust to real cell-to-cell variability. If no, the homogeneous
population model is only valid as a description of the average, not of the circuit.

    python probe_heterogeneous_ring.py [--gate network|own] [--seeds 5]
-> probe_heterogeneous_ring.png + printed summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_engine as oe
import readout_cell as rc

HERE = Path(__file__).resolve().parent
N, DT = oe.N, oe.DT


@njit(fastmath=True, cache=True)
def simulate_hetero(P, K_E, K_I, HD_SHAPE, J1, W_IE, W_GLOBAL, W_EI, tau_I, tau_a, time):
    """Ring where every cell has its OWN 24 cellular params. P is (N, 24) in
    readout_cell.CELL_PARAMS order. Network connectivity stays scalar."""
    tau_E = P[:, 0]; I_baseline = P[:, 1]; I_HD = P[:, 2]; revpot = P[:, 3]
    A_fast = P[:, 4]; fc_dur = P[:, 5]; stim_delay = P[:, 6]
    g_T = P[:, 7]; V_half_T = P[:, 8]; k_T = P[:, 9]; tau_hT = P[:, 10]; E_Ca = P[:, 11]
    g_h = P[:, 12]; V_half_h = P[:, 13]; tau_h_on = P[:, 14]; tau_h_off = P[:, 15]
    g_a = P[:, 16]
    A_sc = P[:, 17]; tau_sc_on = P[:, 18]; tau_sc_off = P[:, 19]
    tau_sc_off2 = P[:, 20]; w_sc = P[:, 21]; sc_delay = P[:, 22]; tau_scg = P[:, 23]

    n = P.shape[0]
    u_E = 40.0 * np.exp(oe.KAPPA_HD * 0.0) * HD_SHAPE * 0.0 + 40.0 * HD_SHAPE
    r_E = np.minimum(oe.R_MAX, oe.GAIN_SLOPE * np.maximum(0.0, u_E))
    u_I = W_EI * (K_E @ r_E)
    r_I = np.maximum(0.0, u_I)
    h_T = np.zeros(n); m_h = np.zeros(n); sc_lp = np.zeros(n)
    a = 0.0

    I_base = I_baseline + I_HD * HD_SHAPE
    onset = oe.T_STIM + stim_delay
    end = onset + fc_dur
    sc_on = onset + sc_delay
    t_first = onset.min()

    rates = np.zeros((len(time), n))
    for s in range(len(time)):
        t = time[s]
        rec_E = J1 * (K_E @ r_E)
        rec_gate = rec_E / (rec_E + oe.REC_HALF)
        sc_lp += (rec_E - sc_lp) * (DT / tau_scg)
        sc_gate = sc_lp / (sc_lp + oe.REC_HALF)

        act = (t >= onset).astype(np.float64)
        m_T = 1.0 / (1.0 + np.exp(-(u_E - V_half_T) / k_T))
        h_inf = 1.0 / (1.0 + np.exp((u_E - V_half_T) / k_T))
        h_T += (h_inf - h_T) * (DT / tau_hT) * act
        I_T = g_T * m_T * h_T * (E_Ca - u_E) * rec_gate * act

        mh_inf = 1.0 / (1.0 + np.exp((u_E - V_half_h) / oe.K_H_SLOPE))
        tau_mh = np.where(mh_inf > m_h, tau_h_on, tau_h_off)
        m_h += (mh_inf - m_h) * (DT / tau_mh) * act
        I_h = g_h * m_h * act

        I_ext = I_base + A_fast * ((t >= onset) & (t < end)).astype(np.float64)
        dts = np.maximum(0.0, t - sc_on)
        decay = w_sc * np.exp(-dts / tau_sc_off) + (1.0 - w_sc) * np.exp(-dts / tau_sc_off2)
        g_sc = decay * (1.0 - np.exp(-dts / tau_sc_on))
        I_ext += A_sc * g_sc * sc_gate * (t >= sc_on).astype(np.float64)

        mean_rE = np.sum(r_E) / n
        if t >= t_first:
            a += (mean_rE - a) * (DT / tau_a)
        I_adapt = g_a * a * act

        u_I += (-u_I + W_EI * (K_E @ r_E)) * (DT / tau_I)
        r_I = np.minimum(500.0, np.maximum(0.0, u_I))

        du_E = (-u_E + rec_E - W_IE * (K_I @ r_I) - W_GLOBAL * mean_rE
                + I_ext + I_T + I_h - I_adapt)
        u_E += du_E * (DT / tau_E)
        u_E = np.maximum(u_E, revpot)
        r_E = np.minimum(oe.R_MAX, oe.GAIN_SLOPE * np.maximum(0.0, u_E))
        rates[s, :] = r_E
    return rates


def metrics(rates, t):
    i600 = np.argmin(np.abs(t - 600))
    p6 = rates[i600, :]
    prof = np.mean(rates[(t > 100) & (t < 700), :], axis=0)
    pre = rates[t < -50, :].mean(axis=0) if (t < -50).any() else rates[0, :]
    return dict(
        pre_pk=pre.max(),
        pre_fwhm=(pre >= pre.max() / 2).sum() * 360 / rates.shape[1] if pre.max() > 1 else 360.,
        fwhm600=(p6 >= p6.max() / 2).sum() * 360 / rates.shape[1] if p6.max() > 1 else 360.,
        offlobe=prof[oe.OFF_LOBE].max() / prof.max() if prof.max() > 0 else np.nan,
        pd600=rates[i600, oe._IDX_0], finite=bool(np.all(np.isfinite(rates))))


def main():
    gate = "network"
    if "--gate" in sys.argv:
        gate = sys.argv[sys.argv.index("--gate") + 1]
    nseed = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5

    pop = __import__("json").load(open(HERE / "_full_fit_result.json"))["params"]
    df = pd.read_parquet(HERE / f"optimized_cells_readout_{gate}.parquet")
    fitted = df[rc.CELL_PARAMS].to_numpy(dtype=np.float64)

    K_E = np.exp(pop["KAPPA_E"] * (oe.COS_D_THETA - 1.0)) / N
    K_I = np.exp(pop["KAPPA_I"] * (oe.COS_D_THETA - 1.0)) / N
    args = (K_E, K_I, oe.HD_SHAPE, pop["J1"], pop["W_IE"], pop["W_GLOBAL"],
            pop["W_EI"], pop["tau_I"], pop["tau_a"], oe.TIME)
    t = oe.TIME

    # homogeneous reference: every cell = the population parameter set
    P_hom = np.tile(np.array([pop[k] for k in rc.CELL_PARAMS]), (N, 1))
    R_hom = simulate_hetero(P_hom, *args)
    m_hom = metrics(R_hom, t)
    print("homogeneous reference:", {k: (round(v, 2) if isinstance(v, float) else v)
                                     for k, v in m_hom.items()})

    npz = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    b, data_mean = npz["bins"], npz["pd_rate"].mean(axis=0)
    w = (b >= oe.WIN_LO) & (b <= oe.WIN_HI)

    print(f"\nheterogeneous rings ({nseed} random draws from the {len(fitted)} fitted cells)")
    outs, r2s = [], []
    for s in range(nseed):
        rng = np.random.default_rng(s)
        P = fitted[rng.integers(0, len(fitted), N)].copy()
        R = simulate_hetero(P, *args)
        m = metrics(R, t)
        # emergent population PSTH = mean over cells, each read at its own PD.
        # A cell at ring position i has its PD there, so the population-average
        # evoked response is the mean over cells of each cell's own trace.
        pop_psth = R.mean(axis=1)
        mv = np.interp(b[w], t, pop_psth)
        r2 = 1 - np.sum((data_mean[w] - mv) ** 2) / np.sum((data_mean[w] - data_mean[w].mean()) ** 2)
        r2s.append(r2); outs.append((m, pop_psth))
        print(f"  seed {s}: bump pk={m['pre_pk']:5.1f} fwhm={m['pre_fwhm']:5.0f} "
              f"offlobe={m['offlobe']:.2f} fwhm600={m['fwhm600']:5.0f} "
              f"pd600={m['pd600']:5.1f} finite={m['finite']} | R2 vs measured mean {r2:6.3f}")

    ok = [m for m, _ in outs if m["finite"] and m["offlobe"] < 0.15 and m["pre_fwhm"] < 130]
    print(f"\n  stable single-bump rings: {len(ok)}/{nseed}")
    print(f"  R2 of emergent population PSTH vs measured mean: "
          f"median {np.median(r2s):.3f} (homogeneous model achieves ~0.92 on the target)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(np.rad2deg(oe.THETA), R_hom[t < -50].mean(axis=0), color="k", lw=2, label="homogeneous")
    for s, (m, _) in enumerate(outs):
        rng = np.random.default_rng(s)
        P = fitted[rng.integers(0, len(fitted), N)].copy()
        R = simulate_hetero(P, *args)
        ax[0].plot(np.rad2deg(oe.THETA), R[t < -50].mean(axis=0), lw=.9, alpha=.7)
    ax[0].set_xlabel("preferred direction (deg)"); ax[0].set_ylabel("baseline rate (Hz)")
    ax[0].set_title("does a heterogeneous ring still form a bump?"); ax[0].legend(fontsize=8)

    ax[1].plot(b[w], data_mean[w], color="0.5", lw=1.6, label="measured population mean")
    for _, psth in outs:
        ax[1].plot(t, psth, lw=.9, alpha=.75)
    ax[1].set_xlim(-50, 700); ax[1].set_xlabel("time rel. stim (ms)"); ax[1].set_ylabel("Hz")
    ax[1].set_title("emergent population PSTH"); ax[1].legend(fontsize=8)

    ax[2].bar(range(len(r2s)), r2s, color="steelblue")
    ax[2].axhline(np.median(r2s), color="crimson", ls="--")
    ax[2].set_xlabel("random draw"); ax[2].set_ylabel("R² vs measured mean")
    ax[2].set_title("consistency across draws")
    fig.tight_layout(); fig.savefig(HERE / "probe_heterogeneous_ring.png", dpi=130)
    print("saved -> probe_heterogeneous_ring.png")


if __name__ == "__main__":
    main()
