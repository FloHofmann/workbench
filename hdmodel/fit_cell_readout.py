"""
fit_cell_readout.py
-------------------
Fit the 24 CELLULAR params of a heterogeneous readout cell against a FIXED
average-network background (see readout_cell.py / network_background.py).

The network stays at the population fit; only this cell is its own. Because the
single-cell sim is ~100x faster than the full ring, all 77 cells fit locally.

Usage:
    python fit_cell_readout.py --count
    python fit_cell_readout.py --gate network --all      # all 77 cells
    python fit_cell_readout.py --gate own 12             # one cell (or SLURM task id)
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import optimization_engine as oe
import readout_cell as rc

HERE = Path(__file__).resolve().parent
NPZ = HERE / "vivo_percell.npz"
POP = HERE / "_full_fit_result.json"

MAXITER = 200
POPSIZE = 14


def fit_one(bins, pd_rate, pd_smooth, drv, x0, gate_own, seed=0):
    args = (drv, bins, pd_rate, pd_smooth, gate_own)
    cands = []

    ra = minimize(rc.loss, x0=x0, bounds=rc.CELL_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(ra.x)

    de = differential_evolution(
        rc.loss, rc.CELL_BOUNDS, args=args, seed=seed, maxiter=MAXITER,
        popsize=POPSIZE, tol=1e-6, init="sobol", mutation=(0.7, 1.9),
        recombination=0.5, polish=True, disp=False, workers=1)
    cands.append(de.x)
    rp = minimize(rc.loss, x0=de.x, bounds=rc.CELL_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(rp.x)

    scored = [(x, rc.r2(x, drv, bins, pd_rate, gate_own)) for x in cands]
    return max(scored, key=lambda s: s[1])


def run_cell(task, d, drv, x0, gate, gate_own, outdir):
    bins = d["bins"]
    pd_rate, pd_smooth = d["pd_rate"][task], d["pd_smooth"][task]
    animal, cell = str(d["animal_ids"][task]), float(d["cell_ids"][task])

    t0 = _time.time()
    x, r2 = fit_one(bins, pd_rate, pd_smooth, drv, x0, gate_own, seed=task)
    y = rc.simulate_readout(np.asarray(x, float), *drv, gate_own)
    t = drv[0]
    fc_model = y[(t >= 0) & (t < 25)].max()
    fc_data = pd_rate[(bins >= 0) & (bins < 25)].max()

    outdir.mkdir(exist_ok=True)
    safe = animal.replace("/", "-").replace("\\", "-")
    json.dump({
        "animal_id": animal, "cell_id": cell,
        "hd_angle": float(d["hd_angles"][task]),
        "r2": r2, "gate": gate,
        "fc_model": float(fc_model), "fc_data": float(fc_data),
        "params": {k: float(v) for k, v in zip(rc.CELL_PARAMS, x)},
    }, open(outdir / f"cell_{safe}_{cell}.json", "w"), indent=2)

    print(f"[{task:2d}] {animal}/{cell:<7} R2={r2:6.3f}  FC {fc_model:5.0f} vs {fc_data:5.0f} data"
          f"  ({_time.time()-t0:.1f}s)", flush=True)
    return r2


def main():
    argv = sys.argv[1:]
    gate = "network"
    if "--gate" in argv:
        gate = argv[argv.index("--gate") + 1]
    if gate not in ("network", "own"):
        print("--gate must be 'network' or 'own'")
        return
    gate_own = 1 if gate == "own" else 0

    d = np.load(NPZ, allow_pickle=True)
    n = len(d["cell_ids"])
    if "--count" in argv:
        print(n)
        return

    drv = rc.load_drivers(HERE / "network_drivers.npz", site="pd")
    p = json.load(open(POP))["params"]
    x0 = np.array([p[k] for k in rc.CELL_PARAMS], dtype=np.float64)
    outdir = HERE / f"results_readout_{gate}"

    if "--all" in argv:
        t0 = _time.time()
        r2s = [run_cell(i, d, drv, x0, gate, gate_own, outdir) for i in range(n)]
        r2s = np.array(r2s)
        print(f"\n{gate} gate: {n} cells in {(_time.time()-t0)/60:.1f} min | "
              f"R2 median={np.median(r2s):.3f} mean={r2s.mean():.3f} "
              f"min={r2s.min():.3f} max={r2s.max():.3f}")
        return

    tasks = [int(a) for a in argv if a.lstrip("-").isdigit() and not a.startswith("--")]
    task = tasks[0] if tasks else 0
    if task >= n:
        print(f"task {task} out of bounds ({n} cells)")
        return
    run_cell(task, d, drv, x0, gate, gate_own, outdir)


if __name__ == "__main__":
    main()
