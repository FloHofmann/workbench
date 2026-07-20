"""
probe_ablation_cell.py
----------------------
"Does the model need this part?" -- answered by deleting it and refitting.

This is the justification backbone for the minimal model. Curvature and split-half
reliability answer a DIFFERENT question ("can we read this parameter off the data?").
Ablation answers the one a reader actually asks.

TWO KINDS OF REMOVAL, labelled separately, because conflating them is what made the
earlier cuts hard to defend:

  Type M (mechanism off)   set a conductance/amplitude to 0 and refit the rest.
                           -> "do we need this part of the model at all?"
  Type F (freedom removed) a time constant or half-activation cannot be zeroed, so
                           fix it at ONE shared value (the median across cells) and
                           refit the rest.
                           -> "does this need to be a free per-cell parameter, or can
                              it be a constant?"

Judged on FOUR criteria, not R2 alone -- a parameter may only be cut if removing it
damages none of them. The already-completed no-gate run is the worked example of why:
R2 was UNCHANGED (0.709 vs 0.692) while direction-selectivity collapsed
(anti-PD 0.0 -> 8.2 Hz). A single-metric study would have called the gate useless.

  1 selectivity  anti-PD SC peak (should stay ~0)
  2 shape        median R2
  3 recovery     rate at 600 ms relative to the cell's own pre-stim baseline (~1)
  4 heterogeneity  handled in the dossier via split-half reliability

    python probe_ablation_cell.py            # tier 1 (instant, no refit)
    python probe_ablation_cell.py --tier2    # + refit on a stratified subset
-> probe_ablation_cell_tier{1,2}.csv
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import optimization_engine as oe
import readout_cell_v3 as r3

HERE = Path(__file__).resolve().parent
P = r3.CELL_PARAMS

# Type M: amplitudes / conductances -> 0 removes the mechanism outright
TYPE_M = ["A_fast", "g_T", "g_a", "A_sc", "I_HD", "I_baseline"]
# Type F: shape params -> fix at the across-cell median (one shared constant)
TYPE_F = [p for p in P if p not in TYPE_M]


def criteria(y, t, bins, pd_rate, y_anti):
    """(R2, anti-PD SC peak, recovery ratio) for one simulated cell."""
    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[m], t, y)
    ss = np.sum((pd_rate[m] - mv) ** 2)
    st = np.sum((pd_rate[m] - pd_rate[m].mean()) ** 2)
    r2 = float(1 - ss / st) if st > 0 else np.nan
    sc = (t > 40) & (t < 300)
    base = y[(t >= -50) & (t < 0)].mean()
    rec = y[np.argmin(np.abs(t - 600))] / base if base > 1 else np.nan
    return r2, float(y_anti[sc].max()), rec


def run_tier1(df, d, drv, drv_anti, med):
    """Apply each ablation with NO refit -> upper bound on the cost."""
    rows = []
    base = [("(none) full model", "-", {}, 0), ("SC gate", "M", {}, 1)]
    abl = base + [(p, "M", {p: 0.0}, 0) for p in TYPE_M] \
                + [(p, "F", {p: med[p]}, 0) for p in TYPE_F]

    for name, typ, fixes, gate in abl:
        r2s, antis, recs = [], [], []
        for i, (_, r) in enumerate(df.iterrows()):
            x = np.array([float(r[k]) for k in P])
            for k, v in fixes.items():
                x[P.index(k)] = v
            y = r3.simulate_readout_v3(x, *drv, gate)
            ya = r3.simulate_readout_v3(x, *drv_anti, gate)
            a, b, c = criteria(y, drv[0], d["bins"], d["pd_rate"][i], ya)
            r2s.append(a); antis.append(b); recs.append(c)
        rows.append(dict(param=name, type=typ, r2=np.median(r2s),
                         anti=np.median(antis), recov=np.nanmedian(recs)))
    return pd.DataFrame(rows)


def fit_ablated(bins, pr, ps, drv, x0, gate, fixes, seed, maxiter):
    """Refit the remaining free params with `fixes` held."""
    fix_i = {P.index(k): v for k, v in fixes.items()}
    free = [i for i in range(len(P)) if i not in fix_i]
    bounds = [r3.CELL_BOUNDS[i] for i in free]

    def L(xf):
        x = np.array(x0, float)
        x[free] = xf
        for i, v in fix_i.items():
            x[i] = v
        return r3.loss(x, drv, bins, pr, ps, gate)

    de = differential_evolution(L, bounds, seed=seed, maxiter=maxiter, popsize=14,
                                tol=1e-6, init="sobol", mutation=(0.7, 1.9),
                                recombination=0.5, polish=True, disp=False, workers=1)
    rp = minimize(L, de.x, bounds=bounds, method="L-BFGS-B", options={"maxiter": 300})
    best = rp.x if rp.fun < de.fun else de.x
    x = np.array(x0, float); x[free] = best
    for i, v in fix_i.items():
        x[i] = v
    return x


def run_tier2(df, d, drv, drv_anti, med, subset, maxiter=80):
    """Refit after ablation on a stratified subset -> realistic cost."""
    rows = []
    abl = [("(none) full model", "-", {}, 0), ("SC gate", "M", {}, 1)] \
        + [(p, "M", {p: 0.0}, 0) for p in TYPE_M] \
        + [(p, "F", {p: med[p]}, 0) for p in TYPE_F]

    for name, typ, fixes, gate in abl:
        t0 = _time.time()
        r2s, antis, recs = [], [], []
        for i in subset:
            r = df.iloc[i]
            x0 = np.array([float(r[k]) for k in P])
            pr, ps = d["pd_rate"][i], d["pd_smooth"][i]
            x = fit_ablated(d["bins"], pr, ps, drv, x0, gate, fixes, i, maxiter)
            y = r3.simulate_readout_v3(x, *drv, gate)
            ya = r3.simulate_readout_v3(x, *drv_anti, gate)
            a, b, c = criteria(y, drv[0], d["bins"], pr, ya)
            r2s.append(a); antis.append(b); recs.append(c)
        rows.append(dict(param=name, type=typ, r2=np.median(r2s),
                         anti=np.median(antis), recov=np.nanmedian(recs)))
        print(f"  {name:20s} [{typ}] R2={np.median(r2s):6.3f} "
              f"anti={np.median(antis):6.1f} recov={np.nanmedian(recs):5.2f} "
              f"({_time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    d = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    drv = r3.load_drivers(HERE / "network_drivers.npz", "pd")
    drv_anti = r3.load_drivers(HERE / "network_drivers.npz", "anti")
    df = pd.read_parquet(HERE / "optimized_cells_v3_mg.parquet")
    med = {p: float(df[p].median()) for p in P}

    t1 = run_tier1(df, d, drv, drv_anti, med)
    full = t1[t1["param"].str.startswith("(none)")].iloc[0]
    t1["dR2"] = full["r2"] - t1["r2"]
    t1 = t1.sort_values("dR2", ascending=False)
    t1.to_csv(HERE / "probe_ablation_cell_tier1.csv", index=False)
    print("=== TIER 1: ablate, no refit (upper bound on cost) ===")
    print(f"  full model: R2={full['r2']:.3f} anti={full['anti']:.1f} recov={full['recov']:.2f}\n")
    print(t1[["param", "type", "dR2", "r2", "anti", "recov"]].to_string(index=False))

    if "--tier2" not in sys.argv:
        print("\n(run with --tier2 to refit after each ablation)")
        return

    # stratified subset spanning the baseline range
    b = d["bins"]
    base = d["pd_rate"][:, (b >= -50) & (b < 0)].mean(axis=1)
    order = np.argsort(base)
    subset = list(order[:: max(1, len(order) // 20)][:20])
    print(f"\n=== TIER 2: refit after ablation, {len(subset)} stratified cells ===")
    t2 = run_tier2(df, d, drv, drv_anti, med, subset)
    full2 = t2[t2["param"].str.startswith("(none)")].iloc[0]
    t2["dR2"] = full2["r2"] - t2["r2"]
    t2 = t2.sort_values("dR2", ascending=False)
    t2.to_csv(HERE / "probe_ablation_cell_tier2.csv", index=False)
    print("\n" + t2[["param", "type", "dR2", "r2", "anti", "recov"]].to_string(index=False))
    print("saved -> probe_ablation_cell_tier1.csv / _tier2.csv")


if __name__ == "__main__":
    main()
