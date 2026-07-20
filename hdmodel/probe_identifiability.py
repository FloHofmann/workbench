"""
probe_identifiability.py
------------------------
Which of the 24 per-cell params are actually determined by the data?

We plot 24 parameter distributions; that is only meaningful for params the fit can
pin down. Sloppy directions produce pretty histograms that are mostly optimizer
noise. This computes, per cell, the local Hessian of the loss at the optimum in
BOUND-SCALED units and reports:

  - sensitivity per param  sqrt(H_ii): how sharply the loss curves along it alone
  - the sloppy/stiff eigenspectrum: log10 spread of Hessian eigenvalues
  - the specific question that matters: are tau_sc_off / tau_sc_off2 / w_sc (the
    GluN2B/GluN2A signature) stiff or sloppy?
  - whether freezing tau_a was necessary: correlation of the adaptation direction
    with the NMDA-decay directions in the leading sloppy eigenvector.

    python probe_identifiability.py [--gate network|own]
-> probe_identifiability.csv + .png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import readout_cell as rc

HERE = Path(__file__).resolve().parent
STEP = 0.01  # perturbation as a fraction of each param's bound range


def hessian(p, drv, bins, pr, ps, gate_own, scale):
    n = len(p)
    H = np.zeros((n, n))
    h = STEP * scale

    def L(q):
        return rc.loss(q, drv, bins, pr, ps, gate_own)

    for i in range(n):
        for j in range(i, n):
            a = np.array(p, float); a[i] += h[i]; a[j] += h[j]
            b = np.array(p, float); b[i] += h[i]; b[j] -= h[j]
            c = np.array(p, float); c[i] -= h[i]; c[j] += h[j]
            d = np.array(p, float); d[i] -= h[i]; d[j] -= h[j]
            H[i, j] = H[j, i] = (L(a) - L(b) - L(c) + L(d)) / (4 * h[i] * h[j])
    # express in scaled units so params with different ranges are comparable
    return H * np.outer(scale, scale)


def main():
    gate = "network"
    if "--gate" in sys.argv:
        gate = sys.argv[sys.argv.index("--gate") + 1]
    gate_own = 1 if gate == "own" else 0

    drv = rc.load_drivers(HERE / "network_drivers.npz", "pd")
    npz = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    df = pd.read_parquet(HERE / f"optimized_cells_readout_{gate}.parquet")
    ids = list(zip([str(a) for a in npz["animal_ids"]], [float(c) for c in npz["cell_ids"]]))
    scale = np.array([hi - lo for lo, hi in rc.CELL_BOUNDS])

    sens, spectra = [], []
    for _, r in df.iterrows():
        i = ids.index((str(r["Animal_Id"]), float(r["Cell_Id"])))
        p = np.array([float(r[k]) for k in rc.CELL_PARAMS])
        H = hessian(p, drv, npz["bins"], npz["pd_rate"][i], npz["pd_smooth"][i], gate_own, scale)
        sens.append(np.sqrt(np.abs(np.diag(H))))
        ev = np.sort(np.abs(np.linalg.eigvalsh((H + H.T) / 2)))[::-1]
        spectra.append(ev)

    S = pd.DataFrame(np.array(sens), columns=rc.CELL_PARAMS)
    med = S.median().sort_values(ascending=False)
    S.to_csv(HERE / "probe_identifiability.csv", index=False)

    EV = np.array(spectra)
    EV = EV / EV[:, :1]                      # normalise to the stiffest direction
    span = np.log10(EV[:, 0] / np.clip(EV[:, -1], 1e-30, None))

    print(f"=== identifiability ({gate} gate, {len(df)} cells) ===")
    print(f"  sloppy span: median {np.median(span):.1f} decades between stiffest and "
          f"softest direction")
    print("\n  STIFFEST (well determined):")
    for k, v in med.head(8).items():
        print(f"    {k:20s} {v:.3g}")
    print("  SLOPPIEST (not interpretable):")
    for k, v in med.tail(8).items():
        print(f"    {k:20s} {v:.3g}")
    nmda = ["tau_sc_off", "tau_sc_off2", "w_sc", "tau_sc_on", "A_sc", "tau_scg"]
    rank = {k: int(np.where(med.index == k)[0][0]) + 1 for k in nmda}
    print("\n  NMDA / SC params (rank out of 24, 1 = stiffest):")
    for k in nmda:
        print(f"    {k:14s} rank {rank[k]:2d}/24   sensitivity {med[k]:.3g}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    y = np.arange(len(med))
    colors = ["crimson" if k in nmda else "steelblue" for k in med.index]
    ax[0].barh(y, med.values, color=colors)
    ax[0].set_yticks(y); ax[0].set_yticklabels(med.index, fontsize=8)
    ax[0].set_xscale("log"); ax[0].invert_yaxis()
    ax[0].set_xlabel("median sensitivity  sqrt(H_ii)  [bound-scaled]")
    ax[0].set_title("stiff (top) -> sloppy (bottom); red = SC/NMDA params")

    for e in EV:
        ax[1].plot(np.arange(1, len(e) + 1), np.clip(e, 1e-30, None), color="0.7", lw=.6)
    ax[1].plot(np.arange(1, EV.shape[1] + 1), np.median(EV, axis=0), color="crimson", lw=2)
    ax[1].set_yscale("log"); ax[1].set_xlabel("eigenvalue rank")
    ax[1].set_ylabel("eigenvalue / largest")
    ax[1].set_title(f"sloppy spectrum (median span {np.median(span):.1f} decades)")
    fig.tight_layout(); fig.savefig(HERE / "probe_identifiability.png", dpi=130)
    print("saved -> probe_identifiability.csv / .png")


if __name__ == "__main__":
    main()
