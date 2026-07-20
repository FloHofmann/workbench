"""
fit_cell_mech.py
----------------
Per-cell fit of the 32-param MECHANISTIC model, designed for SLURM array jobs.

Each cell independently fits ALL 32 params (its own attractor geometry + evoked
response) to its own PD-speaker PSTH, warm-started from the population best
(_full_fit_result.json). This mirrors the DoVM per-cell workflow: the population
attractor is a ~40 Hz-regime network and goes degenerate (multi-lobe) at the low
drive that low-baseline cells need, so geometry must be per-cell too.

Runtime deps: vivo_percell.npz + _full_fit_result.json + optimization_engine.
No workbench.data, no network share, no .mat.

Usage:
    python fit_cell_mech.py --count          # -> n_cells (set SLURM --array bound)
    python fit_cell_mech.py $SLURM_ARRAY_TASK_ID
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import optimization_engine as oe

HERE = Path(__file__).resolve().parent
NPZ = HERE / "vivo_percell.npz"
POP = HERE / "_full_fit_result.json"
OUTDIR = HERE / "results_mech"

MAXITER = 300
POPSIZE = 18


def _weights(mt):
    """Time-window weighting (FC 5x, rebound 4x, tail 3x, decay 2x) -- same scheme
    as oe.loss_stage2, but nothing else is borrowed (that loss hardcodes the
    population 40 Hz baseline, which does not apply per cell)."""
    w = np.where(mt <= 20.0, 5.0, 1.0)
    w = np.where((mt > 25.0) & (mt <= 75.0), 4.0, w)
    w = np.where((mt > 80.0) & (mt <= 200.0), 3.0, w)
    w = np.where((mt > 200.0) & (mt <= 700.0), 2.0, w)
    return w


def _objective(x, bins, pd_rate, pd_smooth):
    """Per-cell loss: weighted MSE to THIS cell's PD trace + scale-free structural
    guards (anti-PD silence, single-lobe topology). No absolute-Hz assumption -> the
    cell's own baseline/dip/SC/recovery are set by the MSE, so per-cell baselines
    (0-53 Hz) fit."""
    t, rr = oe.run_model_stim(*x[:10], *x[10:])
    if not np.all(np.isfinite(rr)):
        return 1e6
    pd, anti = rr[:, oe._IDX_0], rr[:, oe._IDX_180]

    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mt = bins[m]
    if mt.size < 3:
        return 1e6
    mv = np.interp(mt, t, pd)
    target = np.where(mt <= 20.0, pd_rate[m], pd_smooth[m])  # raw FC, smooth tail
    loss = np.average((mv - target) ** 2, weights=_weights(mt))

    sc = (t > 40.0) & (t < 700.0)
    loss += np.mean(anti[sc] ** 2) * 2.0                    # anti-PD ~silent
    prof = np.mean(rr[(t > 100.0) & (t < 700.0), :], axis=0)
    if prof.max() > 0.5:                                    # single lobe (no 2nd bump)
        loss += (np.sum(prof[oe.OFF_LOBE]) / (np.sum(prof) + 1e-9)) ** 2 * 30.0

    return float(loss) if np.isfinite(loss) else 1e6


def _gate(x, bins, pd_rate, pd_smooth):
    """(R2, valid) with a PER-CELL baseline (not the population 33-47 Hz window).
    Rejects degenerate blowups; the bump height itself is free per cell."""
    _, ri = oe.run_model_idle(*x[:10])
    pf = ri[-1, :]
    pk = pf.max()
    valid = np.all(np.isfinite(pf)) and pk >= 2
    if valid:
        base_fwhm = (pf >= pk / 2).sum() * 360 / oe.N
        if pf[oe.OFF_LOBE].max() / pk > 0.15 or base_fwhm > 130:   # single, not-too-broad bump
            valid = False
        else:
            t, rr = oe.run_model_stim(*x[:10], *x[10:])
            if not np.all(np.isfinite(rr)):
                valid = False
            else:
                m = (t > 40) & (t < 700)
                i600 = np.argmin(np.abs(t - 600))
                p6 = rr[i600, :]
                if rr[:, oe._IDX_180][m].max() > max(6.0, 0.25 * pk):        # anti-PD silent
                    valid = False
                elif rr[i600, oe._IDX_0] > 1.6 * pk:                         # rate recovers
                    valid = False
                elif (p6 >= p6.max() / 2).sum() * 360 / oe.N > 1.6 * base_fwhm:  # width recovers
                    valid = False

    t, rr = oe.run_model_stim(*x[:10], *x[10:])
    win = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[win], t, rr[:, oe._IDX_0])
    ss = np.sum((pd_rate[win] - mv) ** 2)
    st = np.sum((pd_rate[win] - pd_rate[win].mean()) ** 2)
    r2 = float(1 - ss / st) if st > 0 else float("nan")
    return r2, bool(valid)


def fit_cell(bins, pd_rate, pd_smooth, x0, seed=0):
    args = (bins, pd_rate, pd_smooth)
    cands = []

    # warm anchor: refine from the population best (good for near-population cells)
    ra = minimize(_objective, x0=x0, bounds=oe.JOINT_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(ra.x)

    # global search over all 32 dims (finds the cell's own basin when far from pop)
    de = differential_evolution(
        _objective, oe.JOINT_BOUNDS, args=args, seed=seed, maxiter=MAXITER,
        popsize=POPSIZE, tol=1e-6, init="sobol", mutation=(0.7, 1.9),
        recombination=0.5, polish=True, disp=False, workers=1)
    cands.append(de.x)
    rp = minimize(_objective, x0=de.x, bounds=oe.JOINT_BOUNDS, args=args,
                  method="L-BFGS-B", options={"maxiter": 500})
    cands.append(rp.x)

    scored = [(x, *_gate(x, bins, pd_rate, pd_smooth)) for x in cands]
    valid = [s for s in scored if s[2]]
    return max(valid or scored, key=lambda s: s[1])  # (x32, r2, valid)


def main():
    d = np.load(NPZ, allow_pickle=True)
    n = len(d["cell_ids"])

    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(n)
        return

    task = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if task >= n:
        print(f"task {task} out of bounds ({n} cells)")
        return

    x0 = np.array([json.load(open(POP))["params"][k] for k in oe.PARAM_NAMES])
    bins = d["bins"]
    pd_rate, pd_smooth = d["pd_rate"][task], d["pd_smooth"][task]
    animal, cell = str(d["animal_ids"][task]), float(d["cell_ids"][task])
    print(f"--- cell {task}: {animal}/{cell} (PD peak {pd_rate.max():.0f} Hz) ---", flush=True)

    x, r2, valid = fit_cell(bins, pd_rate, pd_smooth, x0, seed=task)
    _, ri = oe.run_model_idle(*x[:10])
    print(f"R2={r2:.3f} valid={valid} | idle bump pk={ri[-1,:].max():.1f} Hz", flush=True)

    OUTDIR.mkdir(exist_ok=True)
    out = {
        "animal_id": animal,
        "cell_id": cell,
        "hd_angle": float(d["hd_angles"][task]),
        "r2": r2,
        "valid": bool(valid),
        "params": {k: float(v) for k, v in zip(oe.PARAM_NAMES, x)},
    }
    safe = animal.replace("/", "-").replace("\\", "-")
    path = OUTDIR / f"cell_{safe}_{cell}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
