"""
probe_pharmacology.py
---------------------
Turn the fitted per-cell model into quantitative, graded experimental predictions.
Each manipulation isolates a different claim:

  APV (full NMDA block)      A_sc -> 0        tests whether the SC is NMDA-DRIVEN.
                                              Prediction: SC abolished, FC+dip intact.
  partial block              A_sc x 0.75/0.5/0.25   dose-response per cell.
  Mg-relief (low [Mg2+])     sc_gate == 1     tests whether the gate is VOLTAGE-
                                              DEPENDENT. Prediction: SC appears at
                                              anti-PD -> direction-selectivity lost.
                                              This is the discriminative experiment;
                                              APV cannot test the gate, only the drive.
  ifenprodil (GluN2B block)  w_sc -> 0        removes the slow decay component ->
                                              prediction: SC decays faster, smaller AUC.

    python probe_pharmacology.py [--gate network|own]
-> probe_pharmacology.csv + .png
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


def sc_stats(y, t):
    m = (t > 40) & (t < 700)
    seg = y[m]
    peak = seg.max()
    auc = np.trapezoid(seg, t[m]) if hasattr(np, "trapezoid") else np.trapz(seg, t[m])
    # decay time: time from peak to half-peak within the SC window
    i = seg.argmax()
    half = np.where(seg[i:] <= peak / 2)[0]
    thalf = (t[m][i + half[0]] - t[m][i]) if len(half) else np.nan
    return peak, auc, thalf


def main():
    gate = "network"
    if "--gate" in sys.argv:
        gate = sys.argv[sys.argv.index("--gate") + 1]
    go = 1 if gate == "own" else 0

    drv = rc.load_drivers(HERE / "network_drivers.npz", "pd")
    drv_anti = rc.load_drivers(HERE / "network_drivers.npz", "anti")
    df = pd.read_parquet(HERE / f"optimized_cells_readout_{gate}.parquet")

    # APV blocks NMDA on EVERY cell, so the network background must also be
    # re-recorded with A_sc = 0 -- otherwise the readout cell still inherits the
    # network's SC through rec_E and "APV" is only a partial block.
    import json
    import network_background as nb
    pop_apv = dict(json.load(open(HERE / "_full_fit_result.json"))["params"])
    pop_apv["A_sc"] = 0.0
    dd, _, _ = nb.build(pop_apv)
    drv_apv = (dd["time"], dd["kE_r_pd"], dd["kI_rI_pd"], dd["mean_rE"], dd["a_net"],
               dd["J1"], dd["W_IE"], dd["W_GLOBAL"], dd["u_E0_pd"], dd["hd_shape_pd"])
    t = drv[0]
    ia = rc.CELL_PARAMS.index("A_sc")
    iw = rc.CELL_PARAMS.index("w_sc")

    rows = []
    for _, r in df.iterrows():
        p = np.array([float(r[k]) for k in rc.CELL_PARAMS])
        ctl_pk, ctl_auc, ctl_th = sc_stats(rc.simulate_readout(p, *drv, go), t)

        rec = dict(Animal_Id=r["Animal_Id"], Cell_Id=r["Cell_Id"],
                   ctl_peak=ctl_pk, ctl_auc=ctl_auc, ctl_thalf=ctl_th)

        # APV + partial block
        for f in [0.0, 0.25, 0.5, 0.75]:
            q = p.copy(); q[ia] = p[ia] * f
            pk, auc, _ = sc_stats(rc.simulate_readout(q, *drv, go), t)
            rec[f"peak_block{int((1-f)*100)}"] = pk
            rec[f"auc_block{int((1-f)*100)}"] = auc

        # TRUE APV: NMDA blocked on the network AND on this cell
        q = p.copy(); q[ia] = 0.0
        rec["apv_full_peak"], rec["apv_full_auc"], _ = sc_stats(
            rc.simulate_readout(q, *drv_apv, go), t)

        # Mg-relief: gate forced open, at PD and at the antipode
        rec["mgfree_pd_peak"] = sc_stats(rc.simulate_readout(p, *drv, 2), t)[0]
        rec["mgfree_anti_peak"] = sc_stats(rc.simulate_readout(p, *drv_anti, 2), t)[0]
        rec["ctl_anti_peak"] = sc_stats(rc.simulate_readout(p, *drv_anti, go), t)[0]

        # ifenprodil: remove the slow (GluN2B-like) component
        q = p.copy(); q[iw] = 0.0
        pk, auc, th = sc_stats(rc.simulate_readout(q, *drv, go), t)
        rec.update(ifen_peak=pk, ifen_auc=auc, ifen_thalf=th)
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "probe_pharmacology.csv", index=False)

    def rel(a, b):
        return (d[a] / d[b].replace(0, np.nan)).median()

    print(f"=== in-silico pharmacology ({gate} gate, {len(d)} cells; medians) ===")
    print(f"  control SC peak {d.ctl_peak.median():.1f} Hz | anti-PD {d.ctl_anti_peak.median():.1f} Hz")
    print("\n  APV:")
    print(f"    cell-only block  SC peak {d.ctl_peak.median():.1f} -> {d.peak_block100.median():.1f} Hz "
          f"({100*(1-rel('peak_block100','ctl_peak')):.0f}% reduction)  "
          f"[incomplete: cell still inherits the network's SC]")
    print(f"    FULL block (network+cell)  SC peak {d.ctl_peak.median():.1f} -> "
          f"{d.apv_full_peak.median():.1f} Hz "
          f"({100*(1-rel('apv_full_peak','ctl_peak')):.0f}% reduction, "
          f"AUC {100*rel('apv_full_auc','ctl_auc'):.0f}% of control)")
    print("  partial block dose-response (SC peak, % of control):")
    for b in [25, 50, 75, 100]:
        print(f"    {b:3d}% block -> {100*rel(f'peak_block{b}','ctl_peak'):5.1f}%")
    print("\n  Mg-relief (gate forced open) -- THE discriminative test:")
    print(f"    PD      SC peak {d.ctl_peak.median():.1f} -> {d.mgfree_pd_peak.median():.1f} Hz")
    print(f"    anti-PD SC peak {d.ctl_anti_peak.median():.1f} -> {d.mgfree_anti_peak.median():.1f} Hz")
    sel_ctl = d.ctl_anti_peak / d.ctl_peak.replace(0, np.nan)
    sel_mg = d.mgfree_anti_peak / d.mgfree_pd_peak.replace(0, np.nan)
    print(f"    anti/PD selectivity ratio: {sel_ctl.median():.3f} -> {sel_mg.median():.3f}"
          f"  ({int((d.mgfree_anti_peak > 5).sum())}/{len(d)} cells gain anti-PD SC >5 Hz)")
    print("\n  ifenprodil (GluN2B / slow component removed):")
    print(f"    SC peak {d.ctl_peak.median():.1f} -> {d.ifen_peak.median():.1f} Hz "
          f"| AUC {100*rel('ifen_auc','ctl_auc'):.0f}% of control "
          f"| half-decay {d.ctl_thalf.median():.0f} -> {d.ifen_thalf.median():.0f} ms")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fr = [0, 25, 50, 75, 100]
    y = [100.0] + [100 * rel(f"peak_block{b}", "ctl_peak") for b in [25, 50, 75, 100]]
    ax[0].plot(fr, y, "o-", color="crimson")
    ax[0].set_xlabel("% NMDA block"); ax[0].set_ylabel("SC peak (% of control)")
    ax[0].set_title("APV dose-response (prediction)")

    ax[1].scatter(d.ctl_anti_peak, d.mgfree_anti_peak, s=22, alpha=.75, color="navy")
    lim = [0, max(1, d.mgfree_anti_peak.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", alpha=.4); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("anti-PD SC, control (Hz)"); ax[1].set_ylabel("anti-PD SC, Mg-free (Hz)")
    ax[1].set_title("Mg-relief unmasks anti-PD SC?")

    ax[2].scatter(d.ctl_peak, d.ifen_peak, s=22, alpha=.75, color="seagreen")
    lim = [0, d.ctl_peak.max() * 1.05]
    ax[2].plot(lim, lim, "k--", alpha=.4); ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("SC peak, control (Hz)"); ax[2].set_ylabel("SC peak, ifenprodil (Hz)")
    ax[2].set_title("GluN2B block")
    fig.tight_layout(); fig.savefig(HERE / "probe_pharmacology.png", dpi=130)
    print("saved -> probe_pharmacology.csv / .png")


if __name__ == "__main__":
    main()
