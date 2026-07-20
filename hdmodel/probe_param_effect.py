"""
probe_param_effect.py
---------------------
The most communicable artifact: for every parameter, sweep it across its range and
draw the resulting PD trace. No statistics -- you just SEE what each knob does
("this one sets SC height", "this one shifts the dip"). One multi-panel figure per
model. This is what makes the parameter list legible.

    python probe_param_effect.py
-> probe_param_effect_cell.png  (v3 readout, 18 params)
-> probe_param_effect_ring.png  (population ring, 32 params)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_engine as oe
import readout_cell_v3 as r3

HERE = Path(__file__).resolve().parent
NSWEEP = 5


def _grid(ncols, n):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.7, nrows * 2.1))
    return fig, np.atleast_1d(axes).ravel()


def sweep_cell():
    drv = r3.load_drivers(HERE / "network_drivers.npz", "pd")
    npz = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    df = __import__("pandas").read_parquet(HERE / "optimized_cells_v3_mg.parquet")
    # median-param reference cell
    x0 = np.array([float(df[k].median()) for k in r3.CELL_PARAMS])
    t = drv[0]
    b, data = npz["bins"], npz["pd_rate"].mean(axis=0)

    fig, axes = _grid(5, len(r3.CELL_PARAMS))
    for ax, (j, k) in zip(axes, enumerate(r3.CELL_PARAMS)):
        lo, hi = r3.CELL_BOUNDS[j]
        vals = np.linspace(max(lo, x0[j] * 0.3 if x0[j] > 0 else lo),
                           min(hi, x0[j] * 1.8 if x0[j] > 0 else hi), NSWEEP)
        if not np.ptp(vals):
            vals = np.linspace(lo, hi, NSWEEP)
        ax.plot(b[(b >= -50) & (b <= 400)], data[(b >= -50) & (b <= 400)],
                color="0.8", lw=3, zorder=0)
        for v, c in zip(vals, plt.cm.viridis(np.linspace(0, 1, NSWEEP))):
            x = x0.copy(); x[j] = v
            y = r3.simulate_readout_v3(x, *drv, 0)
            ax.plot(t[(t >= -50) & (t <= 400)], y[(t >= -50) & (t <= 400)], color=c, lw=1)
        ax.set_title(k, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(r3.CELL_PARAMS):]:
        ax.axis("off")
    fig.suptitle("v3 readout cell: what each parameter does to the PD trace "
                 "(grey = data mean; dark→bright = low→high)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "probe_param_effect_cell.png", dpi=130)
    print("saved -> probe_param_effect_cell.png")


def sweep_ring():
    pop = json.load(open(HERE / "_full_fit_result.json"))["params"]
    x0 = np.array([pop[k] for k in oe.PARAM_NAMES])
    from vivo_target import load_vivo_psth
    b, vr, _ = load_vivo_psth()
    bounds = dict(zip(oe.PARAM_NAMES, oe.JOINT_BOUNDS))

    fig, axes = _grid(6, len(oe.PARAM_NAMES))
    for ax, (j, k) in zip(axes, enumerate(oe.PARAM_NAMES)):
        lo, hi = bounds[k]
        vals = np.linspace(max(lo, x0[j] * 0.5 if x0[j] > 0 else lo),
                           min(hi, x0[j] * 1.5 if x0[j] > 0 else hi), NSWEEP)
        if not np.ptp(vals):
            vals = np.linspace(lo, hi, NSWEEP)
        ax.plot(b[(b >= -50) & (b <= 400)], vr[(b >= -50) & (b <= 400)],
                color="0.8", lw=3, zorder=0)
        for v, c in zip(vals, plt.cm.viridis(np.linspace(0, 1, NSWEEP))):
            x = x0.copy(); x[j] = v
            try:
                t, rr = oe.run_model_stim(*x[:10], *x[10:])
                if np.all(np.isfinite(rr)):
                    ax.plot(t[(t >= -50) & (t <= 400)], rr[(t >= -50) & (t <= 400), oe._IDX_0],
                            color=c, lw=1)
            except Exception:
                pass
        ax.set_title(k, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(oe.PARAM_NAMES):]:
        ax.axis("off")
    fig.suptitle("population ring: what each parameter does to the PD trace "
                 "(grey = population data)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "probe_param_effect_ring.png", dpi=130)
    print("saved -> probe_param_effect_ring.png")


if __name__ == "__main__":
    sweep_cell()
    sweep_ring()
