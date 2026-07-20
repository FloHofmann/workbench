"""
probe_gate_identifiability.py
-----------------------------
Closes a loophole in the claim "per-cell SC amplitude is independent of the cell's
own excitability" (which rested on A_sc spread being unchanged between gate modes).

That inference is only valid if the own-Vm gate actually VARIES across cells. If it
saturates near 1 for everyone, the gate does no work, A_sc carries all the variance
for a trivial reason, and the conclusion is right by accident.

  1. Reconstruct sc_gate(t) per cell for BOTH modes and report its value at the SC
     peak: does it vary, or is it pinned at 1? (u_E is recoverable from the returned
     rate: r = 6*[u_E]+ , so max(0,u_E) = r/6.)
  2. Does the own-gate track the cell's own depolarization (it should, by construction)?
  3. Compensation ridge: for one cell, sweep A_sc under both gate modes. If the two
     modes are trade-off equivalent, the achievable loss is the same and only the
     optimal A_sc shifts -> the data cannot localize the gate.

    python probe_gate_identifiability.py
-> probe_gate_identifiability.png + printed summary
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_engine as oe
import readout_cell as rc

HERE = Path(__file__).resolve().parent


def gate_trace(params, drv, gate_own):
    """Reconstruct sc_gate(t) exactly as simulate_readout computes it."""
    p = dict(zip(rc.CELL_PARAMS, params))
    t, kE_r, J1 = drv[0], drv[1], drv[5]
    y = rc.simulate_readout(np.asarray(params, float), *drv, gate_own)
    drive = np.maximum(0.0, y / oe.GAIN_SLOPE) if gate_own else J1 * kE_r
    sc_lp = np.zeros_like(drive)
    acc = 0.0
    for k in range(len(drive)):
        acc += (drive[k] - acc) * (oe.DT / p["tau_scg"])
        sc_lp[k] = acc
    return t, sc_lp / (sc_lp + oe.REC_HALF), y


def main():
    drv = rc.load_drivers(HERE / "network_drivers.npz", "pd")
    npz = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    net = pd.read_parquet(HERE / "optimized_cells_readout_network.parquet")
    own = pd.read_parquet(HERE / "optimized_cells_readout_own.parquet")
    t = drv[0]
    sc = (t > 40) & (t < 300)

    rows = []
    for df, mode, go in [(net, "network", 0), (own, "own", 1)]:
        for _, r in df.iterrows():
            p = [float(r[k]) for k in rc.CELL_PARAMS]
            _, g, y = gate_trace(p, drv, go)
            pk = y[sc].argmax()
            rows.append(dict(mode=mode, Animal_Id=r["Animal_Id"], Cell_Id=r["Cell_Id"],
                             gate_at_scpeak=g[sc][pk], gate_max=g.max(), gate_min=g[sc].min(),
                             A_sc=r["A_sc"], sc_peak=y[sc].max()))
    g = pd.DataFrame(rows)

    print("=== does sc_gate actually vary across cells? ===")
    for mode in ["network", "own"]:
        s = g[g["mode"] == mode]["gate_at_scpeak"]
        print(f"  {mode:8s} gate at SC peak: median {s.median():.3f}  "
              f"range {s.min():.3f}-{s.max():.3f}  CV {s.std()/s.mean():.3f}  "
              f"| saturated(>0.95): {int((s>0.95).sum())}/{len(s)}")

    # does the own-gate track the cell's own depolarisation?
    o = g[g["mode"] == "own"].reset_index(drop=True)
    n = g[g["mode"] == "network"].reset_index(drop=True)
    print(f"  own-gate vs its cell's SC peak rate: r = {o['gate_at_scpeak'].corr(o['sc_peak']):.3f}")
    print(f"  A_sc CV: network {n['A_sc'].std()/n['A_sc'].mean():.3f} | "
          f"own {o['A_sc'].std()/o['A_sc'].mean():.3f}")

    # ---- compensation ridge on one cell ----
    row = net.loc[net["R_Squared"].idxmax()]
    ids = list(zip([str(a) for a in npz["animal_ids"]], [float(c) for c in npz["cell_ids"]]))
    i = ids.index((str(row["Animal_Id"]), float(row["Cell_Id"])))
    bins, pr, ps = npz["bins"], npz["pd_rate"][i], npz["pd_smooth"][i]
    base = np.array([float(row[k]) for k in rc.CELL_PARAMS])
    ja = rc.CELL_PARAMS.index("A_sc")
    grid = np.linspace(5, 300, 60)
    curves = {}
    for mode, go in [("network", 0), ("own", 1)]:
        L = []
        for a in grid:
            q = base.copy(); q[ja] = a
            L.append(rc.loss(q, drv, bins, pr, ps, go))
        curves[mode] = np.array(L)
        print(f"  [{row['Animal_Id']}/{row['Cell_Id']}] {mode:8s}: best A_sc={grid[np.argmin(L)]:.0f} "
              f"min loss={min(L):.1f}")
    print("  => if min losses match, the two gate modes are trade-off equivalent"
          " and the data cannot localize the gate.")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for mode, c in [("network", "crimson"), ("own", "navy")]:
        ax[0].hist(g[g["mode"] == mode]["gate_at_scpeak"], bins=20, alpha=.6, color=c, label=mode)
    ax[0].axvline(0.95, color="k", ls="--", lw=.8, label="saturation")
    ax[0].set_xlabel("sc_gate at SC peak"); ax[0].set_ylabel("# cells")
    ax[0].set_title("does the gate vary, or is it pinned?"); ax[0].legend(fontsize=8)

    ax[1].scatter(o["sc_peak"], o["gate_at_scpeak"], s=20, alpha=.75, color="navy", label="own")
    ax[1].scatter(n["sc_peak"], n["gate_at_scpeak"], s=20, alpha=.75, color="crimson", label="network")
    ax[1].set_xlabel("cell's SC peak rate (Hz)"); ax[1].set_ylabel("sc_gate at SC peak")
    ax[1].set_title("does the gate track depolarisation?"); ax[1].legend(fontsize=8)

    for mode, c in [("network", "crimson"), ("own", "navy")]:
        ax[2].plot(grid, curves[mode], color=c, label=mode)
        ax[2].scatter([grid[np.argmin(curves[mode])]], [curves[mode].min()], color=c, zorder=5)
    ax[2].set_xlabel("A_sc"); ax[2].set_ylabel("loss")
    ax[2].set_title(f"A_sc compensation ({row['Animal_Id']}/{row['Cell_Id']})"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "probe_gate_identifiability.png", dpi=130)
    print("saved -> probe_gate_identifiability.png")


if __name__ == "__main__":
    main()
