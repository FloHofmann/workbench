"""
fit_cell_min.py
---------------
The MINIMAL model. Same dynamics as v3 (readout_cell_v3.simulate_readout_v3), but
only the parameters whose ABLATION costs something stay free per cell; everything
else is a shared constant. Nothing overwritten -- v3 stays intact.

Decision rule: a parameter stays FREE only if it both MATTERS (ablation cost) and
REPLICATES (split-half reliability). Everything else is a shared constant or cut.

  FREE per cell (7):
      I_HD     baseline bump height   (ablation 0.051; reliability 0.41)
      A_fast   FC spike amplitude     (ablation 0.215; reliability 0.47)
      A_sc     SC drive amplitude     (ablation 0.143; reliability 0.36)
      g_a      adaptation strength    (ablation 0.108; sets recovery)
      g_T      T-type conductance     (ablation 0.107; reliability 0.41)
      V_half_T I_T activation voltage (reliability 0.72 -- the MOST replicable param)
      k_T      I_T activation slope   (reliability 0.31)
  V_half_T + k_T are free because SHARING them lets I_T fire at the antipode of strong
  cells -> the SC leaks to anti-PD (direction-selectivity, criterion 1, breaks). They
  are also the most reliable shape params, so freeing them is principled, not a patch.

  SHARED constant -- every remaining shape/timing param (fixing them costs <= 0 and,
  crucially, freeing mg_conc/gate params REOPENS the gate at anti-PD, so they MUST be
  shared): fc_stim_duration, stim_delay, tau_hT, E_Ca, tau_E, tau_sc_on, tau_sc_off,
  sc_delay, tau_dend, mg_conc.
  CUT: I_baseline (ablation NEGATIVE -0.012; degenerate with I_HD) + everything already
  gone in v3 (I_h x4, tau_sc_off2, w_sc, reversal_potential).

Shared constants = median of the v3 fits.

    python fit_cell_min.py --all
    python fit_cell_min.py 12
-> results_min/cell_{animal}_{cell}.json
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import readout_cell_v3 as r3

HERE = Path(__file__).resolve().parent
P = r3.CELL_PARAMS
FREE = ["I_HD", "A_fast", "A_sc", "g_a", "g_T", "V_half_T", "k_T"]
FREE_I = [P.index(k) for k in FREE]
FREE_BOUNDS = [r3.CELL_BOUNDS[i] for i in FREE_I]


def shared_constants():
    """Median of the v3 fits for every non-free param (I_baseline set to 0 -- cut)."""
    df = pd.read_parquet(HERE / "optimized_cells_v3_mg.parquet")
    c = {k: float(df[k].median()) for k in P}
    c["I_baseline"] = 0.0
    return c


def _assemble(free_vals, const):
    x = np.array([const[k] for k in P], float)
    for i, v in zip(FREE_I, free_vals):
        x[i] = v
    return x


def fit_one(bins, pr, ps, drv, const, seed):
    def L(f):
        return r3.loss(_assemble(f, const), drv, bins, pr, ps, 0)
    de = differential_evolution(L, FREE_BOUNDS, seed=seed, maxiter=150, popsize=16,
                                tol=1e-6, init="sobol", mutation=(0.7, 1.9),
                                recombination=0.5, polish=True, disp=False, workers=1)
    rp = minimize(L, de.x, bounds=FREE_BOUNDS, method="L-BFGS-B", options={"maxiter": 400})
    best = rp.x if rp.fun < de.fun else de.x
    x = _assemble(best, const)
    return x, r3.r2(x, drv, bins, pr, 0)


def run_cell(i, d, drv, drv_anti, const, outdir):
    bins = d["bins"]; pr, ps = d["pd_rate"][i], d["pd_smooth"][i]
    animal, cell = str(d["animal_ids"][i]), float(d["cell_ids"][i])
    x, r2 = fit_one(bins, pr, ps, drv, const, seed=i)
    t = drv[0]
    y = r3.simulate_readout_v3(x, *drv, 0); ya = r3.simulate_readout_v3(x, *drv_anti, 0)
    sc = (t > 40) & (t < 300); scb = (bins > 40) & (bins < 300)
    outdir.mkdir(exist_ok=True)
    safe = animal.replace("/", "-").replace("\\", "-")
    json.dump({
        "animal_id": animal, "cell_id": cell, "hd_angle": float(d["hd_angles"][i]),
        "r2": r2, "sc_pd": float(y[sc].max()), "sc_anti": float(ya[sc].max()),
        "sc_anti_data": float(d["anti_rate"][i][scb].max()),
        "free": {k: float(x[P.index(k)]) for k in FREE},
        "params": {k: float(v) for k, v in zip(P, x)},
    }, open(outdir / f"cell_{safe}_{cell}.json", "w"), indent=2)
    return r2, float(ya[sc].max())


def main():
    argv = sys.argv[1:]
    d = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    if "--count" in argv:
        print(len(d["cell_ids"])); return
    drv = r3.load_drivers(HERE / "network_drivers.npz", "pd")
    drv_anti = r3.load_drivers(HERE / "network_drivers.npz", "anti")
    const = shared_constants()
    outdir = HERE / "results_min"

    if "--all" in argv:
        t0 = _time.time()
        res = [run_cell(i, d, drv, drv_anti, const, outdir) for i in range(len(d["cell_ids"]))]
        r2 = np.array([a for a, _ in res]); anti = np.array([b for _, b in res])
        print(f"MINIMAL ({len(FREE)} free/cell): {len(r2)} cells in {(_time.time()-t0)/60:.1f} min")
        print(f"  R2 median={np.median(r2):.3f} mean={r2.mean():.3f} min={r2.min():.3f}")
        print(f"  held-out anti-PD SC median={np.median(anti):.1f} Hz | "
              f"{int((anti>5).sum())}/{len(anti)} leak >5 Hz")
        return
    tasks = [int(a) for a in argv if a.lstrip("-").isdigit()]
    i = tasks[0] if tasks else 0
    r2, anti = run_cell(i, d, drv, drv_anti, const, outdir)
    print(f"cell {i}: R2={r2:.3f} anti-PD SC={anti:.1f} Hz")


if __name__ == "__main__":
    main()
