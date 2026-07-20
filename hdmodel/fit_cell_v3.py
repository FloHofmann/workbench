"""
fit_cell_v3.py
--------------
Fit the 18 cellular params of the v3 readout cell (Jahr-Stevens Mg gate,
single-exponential SC, no I_h). v2's fit_cell_readout.py is untouched.

    python fit_cell_v3.py --count
    python fit_cell_v3.py --gate mg     --all     # Mg block (default)
    python fit_cell_v3.py --gate nogate --all     # necessity control: sc_gate == 1
    python fit_cell_v3.py --gate mg 12            # one cell

-> results_v3_{gate}/cell_{animal}_{cell}.json
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import readout_cell_v3 as r3

HERE = Path(__file__).resolve().parent
NPZ = HERE / "vivo_percell.npz"
POP = HERE / "_full_fit_result.json"

MAXITER = 200
POPSIZE = 16
GATES = {"mg": 0, "nogate": 1}


def fit_one(bins, pd_rate, pd_smooth, drv, x0, gate_mode, seed=0):
    args = (drv, bins, pd_rate, pd_smooth, gate_mode)
    cands = []
    ra = minimize(r3.loss, x0=x0, bounds=r3.CELL_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(ra.x)
    de = differential_evolution(
        r3.loss, r3.CELL_BOUNDS, args=args, seed=seed, maxiter=MAXITER,
        popsize=POPSIZE, tol=1e-6, init="sobol", mutation=(0.7, 1.9),
        recombination=0.5, polish=True, disp=False, workers=1)
    cands.append(de.x)
    rp = minimize(r3.loss, x0=de.x, bounds=r3.CELL_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(rp.x)
    scored = [(x, r3.r2(x, drv, bins, pd_rate, gate_mode)) for x in cands]
    return max(scored, key=lambda s: s[1])


def run_cell(task, d, drv, drv_anti, x0, gate, gate_mode, outdir):
    bins = d["bins"]
    pr, ps = d["pd_rate"][task], d["pd_smooth"][task]
    animal, cell = str(d["animal_ids"][task]), float(d["cell_ids"][task])
    t0 = _time.time()
    x, r2 = fit_one(bins, pr, ps, drv, x0, gate_mode, seed=task)

    t = drv[0]
    y = r3.simulate_readout_v3(np.asarray(x, float), *drv, gate_mode)
    y_anti = r3.simulate_readout_v3(np.asarray(x, float), *drv_anti, gate_mode)
    sc = (t > 40) & (t < 300)
    scb = (bins > 40) & (bins < 300)

    outdir.mkdir(exist_ok=True)
    safe = animal.replace("/", "-").replace("\\", "-")
    json.dump({
        "animal_id": animal, "cell_id": cell,
        "hd_angle": float(d["hd_angles"][task]),
        "r2": r2, "gate": gate,
        "fc_model": float(y[(t >= 0) & (t < 25)].max()),
        "fc_data": float(pr[(bins >= 0) & (bins < 25)].max()),
        "sc_pd": float(y[sc].max()), "sc_anti": float(y_anti[sc].max()),
        "sc_anti_data": float(d["anti_rate"][task][scb].max()),
        "params": {k: float(v) for k, v in zip(r3.CELL_PARAMS, x)},
    }, open(outdir / f"cell_{safe}_{cell}.json", "w"), indent=2)

    print(f"[{task:2d}] {animal}/{cell:<7} R2={r2:6.3f}  SC pd={y[sc].max():5.1f} "
          f"anti={y_anti[sc].max():5.1f} (data anti {d['anti_rate'][task][scb].max():4.1f})"
          f"  ({_time.time()-t0:.1f}s)", flush=True)
    return r2, y_anti[sc].max()


def main():
    argv = sys.argv[1:]
    gate = argv[argv.index("--gate") + 1] if "--gate" in argv else "mg"
    if gate not in GATES:
        print("--gate must be one of:", ", ".join(GATES))
        return
    gate_mode = GATES[gate]

    d = np.load(NPZ, allow_pickle=True)
    n = len(d["cell_ids"])
    if "--count" in argv:
        print(n)
        return

    drv = r3.load_drivers(HERE / "network_drivers.npz", "pd")
    drv_anti = r3.load_drivers(HERE / "network_drivers.npz", "anti")
    x0 = r3.seed_from_v2(json.load(open(POP))["params"])
    outdir = HERE / f"results_v3_{gate}"

    if "--all" in argv:
        t0 = _time.time()
        res = [run_cell(i, d, drv, drv_anti, x0, gate, gate_mode, outdir) for i in range(n)]
        r2s = np.array([r for r, _ in res]); anti = np.array([a for _, a in res])
        print(f"\nv3 [{gate}]: {n} cells in {(_time.time()-t0)/60:.1f} min | "
              f"R2 median={np.median(r2s):.3f} mean={r2s.mean():.3f} "
              f"min={r2s.min():.3f} max={r2s.max():.3f}")
        print(f"  held-out anti-PD SC: median {np.median(anti):.1f} Hz | "
              f"{int((anti > 5).sum())}/{n} cells leak >5 Hz")
        return

    tasks = [int(a) for a in argv if a.lstrip("-").isdigit() and not a.startswith("--")]
    run_cell(tasks[0] if tasks else 0, d, drv, drv_anti, x0, gate, gate_mode, outdir)


if __name__ == "__main__":
    main()
