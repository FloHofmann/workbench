"""
probe_split_half.py
-------------------
Are the 24 per-cell params biology or noise-fitting?

Fits each cell INDEPENDENTLY on even trials (half A) and odd trials (half B), then:
  1. NOISE CEILING  - split-half correlation of the two half-PSTHs, Spearman-Brown
     corrected to the full PSTH. This bounds the R2 ANY model can reach. Without it,
     "model R2 = 0.70" is uninterpretable.
  2. CROSS-PREDICTION - fit on A, score on B (and vice versa). True out-of-sample
     generalization; the gap vs in-sample R2 is the overfitting penalty of 24 params.
  3. PARAMETER RELIABILITY - correlate each param's A-estimate against its B-estimate
     across the 77 cells. Params that don't replicate across halves of the SAME cell
     are not interpretable, however pretty their distribution looks.

    python probe_split_half.py [--gate network|own]
-> probe_split_half.csv + probe_split_half.png
"""

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_engine as oe
import readout_cell as rc
from fit_cell_readout import fit_one

HERE = Path(__file__).resolve().parent


def _r2(y, target):
    ss = np.sum((target - y) ** 2)
    st = np.sum((target - target.mean()) ** 2)
    return float(1 - ss / st) if st > 0 else np.nan


def main():
    gate = "network"
    if "--gate" in sys.argv:
        gate = sys.argv[sys.argv.index("--gate") + 1]
    gate_own = 1 if gate == "own" else 0

    d = np.load(HERE / "vivo_percell_split.npz", allow_pickle=True)
    full = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    drv = rc.load_drivers(HERE / "network_drivers.npz", "pd")
    import json
    pop = json.load(open(HERE / "_full_fit_result.json"))["params"]
    x0 = np.array([pop[k] for k in rc.CELL_PARAMS])

    bins = d["bins"]
    win = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    n = len(d["cell_ids"])
    rows = []
    t0 = _time.time()

    for i in range(n):
        A, As = d["pd_rate_a"][i], d["pd_smooth_a"][i]
        B, Bs = d["pd_rate_b"][i], d["pd_smooth_b"][i]

        # 1. noise ceiling: split-half r, Spearman-Brown -> reliability of the FULL psth
        r_ab = np.corrcoef(A[win], B[win])[0, 1]
        sb = 2 * r_ab / (1 + r_ab) if (1 + r_ab) != 0 else np.nan
        ceiling = sb ** 2 if np.isfinite(sb) else np.nan

        # 2. independent fits on each half
        pA, r2_AA = fit_one(bins, A, As, drv, x0, gate_own, seed=i)
        pB, r2_BB = fit_one(bins, B, Bs, drv, x0, gate_own, seed=i + 1000)

        # cross-prediction (true out-of-sample)
        yA = rc.simulate_readout(np.asarray(pA, float), *drv, gate_own)
        yB = rc.simulate_readout(np.asarray(pB, float), *drv, gate_own)
        mA = np.interp(bins[win], drv[0], yA)
        mB = np.interp(bins[win], drv[0], yB)
        r2_AB = _r2(mA, B[win])      # fit on A, tested on B
        r2_BA = _r2(mB, A[win])

        row = dict(Animal_Id=str(d["animal_ids"][i]), Cell_Id=float(d["cell_ids"][i]),
                   r_split=r_ab, ceiling=ceiling,
                   r2_in=np.nanmean([r2_AA, r2_BB]),
                   r2_out=np.nanmean([r2_AB, r2_BA]))
        for k, va, vb in zip(rc.CELL_PARAMS, pA, pB):
            row[f"{k}__A"] = float(va)
            row[f"{k}__B"] = float(vb)
        rows.append(row)
        print(f"[{i:2d}] {row['Animal_Id']}/{row['Cell_Id']:<7} ceiling={ceiling:5.3f} "
              f"in={row['r2_in']:6.3f} out={row['r2_out']:6.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "probe_split_half.csv", index=False)

    # 3. parameter reliability across halves
    rel = []
    for k in rc.CELL_PARAMS:
        a, b = df[f"{k}__A"], df[f"{k}__B"]
        r = a.corr(b)
        rel.append(dict(param=k, r=r, cv=a.std() / abs(a.mean()) if a.mean() else np.nan))
    rel = pd.DataFrame(rel).sort_values("r", ascending=False)
    rel.to_csv(HERE / "probe_split_half_param_reliability.csv", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(df["ceiling"], df["r2_in"], s=20, alpha=.75, label="in-sample")
    ax[0].scatter(df["ceiling"], df["r2_out"], s=20, alpha=.75, label="out-of-sample")
    lim = [0, 1]
    ax[0].plot(lim, lim, "k--", alpha=.5, label="noise ceiling")
    ax[0].set_xlim(lim); ax[0].set_ylim(-0.2, 1)
    ax[0].set_xlabel("noise ceiling (Spearman-Brown R²)"); ax[0].set_ylabel("model R²")
    ax[0].set_title("is the model at the ceiling?"); ax[0].legend(fontsize=8)

    ax[1].hist([df["r2_in"], df["r2_out"]], bins=18, label=["in-sample", "out-of-sample"])
    ax[1].set_xlabel("R²"); ax[1].set_ylabel("# cells"); ax[1].legend(fontsize=8)
    ax[1].set_title("overfitting gap (median %.3f)" % (df["r2_in"] - df["r2_out"]).median())

    y = np.arange(len(rel))
    ax[2].barh(y, rel["r"], color="steelblue")
    ax[2].set_yticks(y); ax[2].set_yticklabels(rel["param"], fontsize=7)
    ax[2].axvline(0, color="k", lw=.8); ax[2].set_xlabel("split-half r (A vs B)")
    ax[2].set_title("parameter reliability"); ax[2].invert_yaxis()
    fig.tight_layout(); fig.savefig(HERE / "probe_split_half.png", dpi=130)

    print(f"\n=== split-half ({gate} gate, {n} cells, {(_time.time()-t0)/60:.1f} min) ===")
    print(f"  noise ceiling      median {df['ceiling'].median():.3f}")
    print(f"  R2 in-sample       median {df['r2_in'].median():.3f}")
    print(f"  R2 out-of-sample   median {df['r2_out'].median():.3f}")
    print(f"  overfit gap        median {(df['r2_in']-df['r2_out']).median():.3f}")
    print(f"  cells at/above ceiling (out-of-sample): "
          f"{int((df['r2_out'] >= df['ceiling']).sum())}/{n}")
    print("\n  most reliable params:", ", ".join(
        f"{r.param}({r.r:.2f})" for r in rel.head(6).itertuples()))
    print("  least reliable params:", ", ".join(
        f"{r.param}({r.r:.2f})" for r in rel.tail(6).itertuples()))
    print("saved -> probe_split_half.csv / _param_reliability.csv / .png")


if __name__ == "__main__":
    main()
