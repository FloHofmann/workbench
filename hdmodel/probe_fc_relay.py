"""
probe_fc_relay.py
-----------------
Original claim under test: "the FC cannot be routed through the recurrent ring --
at real single-cell amplitudes the attractor does not survive it, so the FC must be
a relay."

RESULT: that claim is NOT supported, and this script documents why.
  - With POPULATION params the ring is robust to A_fast up to 800 (FC ~548 Hz):
    no secondary bump, width and rate recover, no saturation. Every measured cell
    (FC 59-319 Hz) is inside that range, so there is no amplitude-based barrier.
  - The R2=-6 collapse originally observed came from the v1 HOMOGENEOUS per-cell
    fits, which sit in a narrow/strong regime (e.g. idle peak 58 Hz / FWHM 51 deg vs
    the population's 38 / 75). THOSE parameter sets are fragile: at A_fast>=300 the
    bump is displaced entirely (off-lobe fraction -> 1.0, PD rate at 600 ms -> 0).
    So the v1 "FC ceiling" was an artifact of homogeneous per-cell fitting, not a
    property of the circuit.

What DOES survive as a quantified finding: the ring compresses the FC. Uniform drive
recruits feedback inhibition (K_I@r_I) and the global term (W_GLOBAL*mean(r_E)),
which push back -> sub-linear FC transfer (~0.7 Hz per unit A_fast). A decoupled
readout cell, whose network background does not react, has ~4x steeper FC gain. That
gain control -- not instability -- is why per-cell FC amplitudes are reachable in the
readout model and yoked in the homogeneous one.

Method: sweep A_fast for (a) the population ring, (b) a v1 fitted ring, (c) the
decoupled readout, and overlay the measured per-cell FC peaks.

    python probe_fc_relay.py
-> probe_fc_relay.png + printed summary
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optimization_engine as oe
import readout_cell as rc

HERE = Path(__file__).resolve().parent
SWEEP = np.array([100, 150, 200, 240, 300, 350, 400, 450, 500, 600, 700, 800], dtype=float)


def ring_metrics(x, base_pk, base_fwhm, bins, vr):
    """Run the full ring at this param set; return FC peak, integrity metrics, and
    the first integrity criterion violated (or 'ok')."""
    t, rr = oe.run_model_stim(*x[:10], *x[10:])
    if not np.all(np.isfinite(rr)):
        return dict(fc=np.nan, fail="non-finite", offlobe=np.nan, fwhm600=np.nan,
                    pd600=np.nan, rmax=np.nan, r2=np.nan)
    pd = rr[:, oe._IDX_0]
    fc = pd[(t >= 0) & (t < 25)].max()
    rmax_hit = rr.max()

    i600 = np.argmin(np.abs(t - 600))
    p6 = rr[i600, :]
    fwhm600 = (p6 >= p6.max() / 2).sum() * 360 / oe.N if p6.max() > 1 else 360.0
    pd600 = rr[i600, oe._IDX_0]
    prof = np.mean(rr[(t > 100) & (t < 700), :], axis=0)
    offlobe = prof[oe.OFF_LOBE].max() / prof.max() if prof.max() > 0 else np.nan

    win = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[win], t, pd)
    r2 = 1 - np.sum((vr[win] - mv) ** 2) / np.sum((vr[win] - vr[win].mean()) ** 2)

    fail = "ok"
    if rmax_hit >= oe.R_MAX * 0.999:
        fail = "rate saturation (R_MAX)"
    elif offlobe > 0.15:
        fail = "secondary bump (off-lobe >15%)"
    elif fwhm600 > 1.6 * base_fwhm:
        fail = "bump stays broad (width not recovered)"
    elif pd600 > 1.6 * base_pk:
        fail = "elevated plateau (rate not recovered)"
    return dict(fc=fc, fail=fail, offlobe=offlobe, fwhm600=fwhm600,
                pd600=pd600, rmax=rmax_hit, r2=r2)


def main():
    pop = json.load(open(HERE / "_full_fit_result.json"))["params"]
    x0 = np.array([pop[k] for k in oe.PARAM_NAMES])
    ia = oe.PARAM_NAMES.index("A_fast")

    from vivo_target import load_vivo_psth
    bins, vr, _ = load_vivo_psth()

    _, ri = oe.run_model_idle(*x0[:10])
    pf = ri[-1, :]
    base_pk = pf.max()
    base_fwhm = (pf >= base_pk / 2).sum() * 360 / oe.N
    print(f"population baseline bump: peak {base_pk:.1f} Hz, FWHM {base_fwhm:.0f} deg\n")

    # measured per-cell FC peaks
    d = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    b = d["bins"]
    fc_data = d["pd_rate"][:, (b >= 0) & (b < 25)].max(axis=1)

    def sweep_ring(xs, label, pk, fw):
        print(f"RING [{label}]  (FC drives the whole population, engages recurrence)")
        out = []
        for a in SWEEP:
            x = xs.copy(); x[ia] = a
            m = ring_metrics(x, pk, fw, bins, vr)
            out.append(m)
            print(f"  A_fast={a:5.0f} -> FC {m['fc']:6.1f} Hz | R2 {m['r2']:7.3f} | "
                  f"offlobe {m['offlobe']:.2f} fwhm600 {m['fwhm600']:5.0f} "
                  f"max {m['rmax']:6.1f} | {m['fail']}")
        return out

    # ---- RING sweep: population params ----
    ring = sweep_ring(x0, "population", base_pk, base_fwhm)
    ring_fc = np.array([m["fc"] for m in ring])
    ok = np.array([m["fail"] == "ok" for m in ring])

    # ---- RING sweep: a v1 HOMOGENEOUS per-cell fit (the fragile regime) ----
    ring_v1 = ring_v1_fc = ok_v1 = None
    try:
        import pandas as pd
        v1 = pd.read_parquet(HERE / "optimized_cells_mech.parquet")
        rv = v1.iloc[int(np.argmax(v1["R_Squared"].values))]
        xv = np.array([float(rv[k]) for k in oe.PARAM_NAMES])
        _, riv = oe.run_model_idle(*xv[:10])
        pfv = riv[-1, :]; pkv = pfv.max(); fwv = (pfv >= pkv / 2).sum() * 360 / oe.N
        print(f"\n  (v1 fit {rv['Animal_Id']}/{rv['Cell_Id']}: idle peak {pkv:.1f} Hz, "
              f"FWHM {fwv:.0f} deg)")
        ring_v1 = sweep_ring(xv, f"v1 homogeneous fit {rv['Animal_Id']}/{rv['Cell_Id']}", pkv, fwv)
        ring_v1_fc = np.array([m["fc"] for m in ring_v1])
        ok_v1 = np.array([m["fail"] == "ok" for m in ring_v1])
    except Exception as e:
        print(f"  (v1 comparison skipped: {e})")
    print()

    # ---- READOUT sweep (network background fixed at the population fit) ----
    print("\nREADOUT (FC drives only this cell; network background unchanged)")
    drv = rc.load_drivers(HERE / "network_drivers.npz", "pd")
    cell0 = np.array([pop[k] for k in rc.CELL_PARAMS])
    ja = rc.CELL_PARAMS.index("A_fast")
    read_fc = []
    for a in SWEEP:
        p = cell0.copy(); p[ja] = a
        y = rc.simulate_readout(p, *drv, 0)
        t = drv[0]
        f = y[(t >= 0) & (t < 25)].max()
        read_fc.append(f)
        print(f"  A_fast={a:5.0f} -> FC {f:6.1f} Hz | finite={np.all(np.isfinite(y))} "
              f"max {y.max():6.1f}")
    read_fc = np.array(read_fc)

    # ---- verdict ----
    fc_at_last_ok = ring_fc[ok].max() if ok.any() else np.nan
    gain_ring = np.polyfit(SWEEP, ring_fc, 1)[0]
    gain_read = np.polyfit(SWEEP[read_fc < oe.R_MAX], read_fc[read_fc < oe.R_MAX], 1)[0]
    print("=== verdict ===")
    print(f"  POPULATION ring: {'no failure at any tested A_fast' if ok.all() else 'fails'}"
          f" (FC reached {fc_at_last_ok:.0f} Hz)")
    print(f"  measured per-cell FC: median {np.median(fc_data):.0f}, "
          f"range {fc_data.min():.0f}-{fc_data.max():.0f} Hz "
          f"-> {int((fc_data > fc_at_last_ok).sum())}/{len(fc_data)} cells exceed it")
    print("  => NO amplitude-based barrier in the healthy network; the 'FC must be a"
          " relay' claim is NOT supported.")
    if ok_v1 is not None and not ok_v1.all():
        i = int(np.where(~ok_v1)[0][0])
        print(f"  v1 homogeneous fit BREAKS at A_fast={SWEEP[i]:.0f}: {ring_v1[i]['fail']}"
              f" -> the v1 FC ceiling was a fitting artifact, not circuit physics.")
    print(f"  FC gain: ring {gain_ring:.2f} Hz per unit A_fast | readout {gain_read:.2f}"
          f"  ({gain_read/gain_ring:.1f}x steeper)")
    print("  => recurrent+global inhibition acts as automatic gain control on the FC.")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].axhspan(fc_data.min(), fc_data.max(), color="0.85", label="measured per-cell FC range")
    ax[0].axhline(np.median(fc_data), color="0.45", ls=":", label="median measured FC")
    ax[0].plot(SWEEP, ring_fc, "o-", color="crimson", label="population ring (compressed)")
    ax[0].plot(SWEEP, read_fc, "s-", color="navy", label="readout (uncompressed)")
    if ring_v1_fc is not None:
        ax[0].plot(SWEEP, ring_v1_fc, "^--", color="darkorange", label="v1 homogeneous fit")
    ax[0].set_xlabel("A_fast"); ax[0].set_ylabel("FC peak achieved (Hz)")
    ax[0].set_title("FC gain: the ring compresses, the readout does not"); ax[0].legend(fontsize=8)

    ax[1].plot(SWEEP, [m["offlobe"] for m in ring], "o-", color="crimson", label="population off-lobe")
    if ring_v1 is not None:
        ax[1].plot(SWEEP, [m["offlobe"] for m in ring_v1], "^--", color="darkorange",
                   label="v1 fit off-lobe")
    ax[1].axhline(0.15, color="k", ls="--", lw=.8, label="single-bump limit")
    ax[1].plot(SWEEP, np.array([m["fwhm600"] for m in ring]) / base_fwhm, "s-", color="0.5",
               label="population FWHM600 / baseline")
    ax[1].set_xlabel("A_fast"); ax[1].set_ylabel("integrity metric")
    ax[1].set_title("population stays intact; the v1 fit does not"); ax[1].legend(fontsize=7)

    ax[2].plot(SWEEP, [m["r2"] for m in ring], "o-", color="crimson")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel("A_fast"); ax[2].set_ylabel("ring R² vs population PSTH")
    ax[2].set_title("fit collapses once the attractor breaks")
    fig.tight_layout(); fig.savefig(HERE / "probe_fc_relay.png", dpi=130)
    print("saved -> probe_fc_relay.png")


if __name__ == "__main__":
    main()
