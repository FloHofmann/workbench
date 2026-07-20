"""
probe_ablation_ring.py
----------------------
Ablation of the 32-param POPULATION ring (the model that reproduces the population
PSTH at R2=0.92 and carries the headline mechanistic claim).

Same logic as probe_ablation_cell.py but for the full recurrent network:
  Type M  amplitude/conductance -> 0        (A_fast, A_sc, g_T, g_h, g_a, W_GLOBAL,
                                             I_HD, W_IE, W_EI, J1, I_baseline)
  Type F  shape param -> keep, only refit    (taus, kappas, half-activations, ...)
          [a ring shape param cannot be "shared across cells" -- there is one ring --
           so Type F here just measures whether the fit recovers when it is nudged;
           the meaningful ring question is Type M: is the mechanism needed?]

Four criteria via ring_metrics() (reused from probe_fc_relay.py) + anti-PD SC:
  selectivity  anti-PD SC peak
  shape        R2 vs population PSTH
  recovery     FWHM600/baseline and PD600/baseline
  (+ validity gates from _run_full_fit.score: a gate failure is itself a result,
   e.g. removing I_HD should destroy the baseline bump)

  tier 1  ablate, no refit (instant, upper bound)
  tier 2  warm-start L-BFGS refit of the other 31 params (pattern from _polish_stf.py)

    python probe_ablation_ring.py [--tier2]
-> probe_ablation_ring_tier{1,2}.csv
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import optimization_engine as oe
from vivo_target import load_vivo_psth
from probe_fc_relay import ring_metrics
from _run_full_fit import score

HERE = Path(__file__).resolve().parent
NAMES = list(oe.PARAM_NAMES)

TYPE_M = ["A_fast", "A_sc", "g_T", "g_h", "g_a", "W_GLOBAL", "I_HD", "W_IE", "W_EI",
          "J1", "I_baseline"]
TYPE_F = [p for p in NAMES if p not in TYPE_M]


def evaluate(x, bins, vr, base_pk, base_fwhm):
    m = ring_metrics(x, base_pk, base_fwhm, bins, vr)
    t, rr = oe.run_model_stim(*x[:10], *x[10:])
    sc = (t > 40) & (t < 300)
    anti = rr[:, oe._IDX_180][sc].max() if np.all(np.isfinite(rr)) else np.nan
    return m, anti


def refit(x0, fix_i, bounds, args):
    free = [i for i in range(len(NAMES)) if i not in fix_i]
    fb = [bounds[i] for i in free]

    def L(xf):
        x = np.array(x0, float); x[free] = xf
        for i, v in fix_i.items():
            x[i] = v
        return oe.loss_joint(x, *args)

    r = minimize(L, [x0[i] for i in free], bounds=fb, method="L-BFGS-B",
                 options={"maxiter": 600})
    x = np.array(x0, float); x[free] = r.x
    for i, v in fix_i.items():
        x[i] = v
    return x


def main():
    pop = json.load(open(HERE / "_full_fit_result.json"))["params"]
    x0 = np.array([pop[k] for k in NAMES])
    bins, vr, vs = load_vivo_psth()
    _, ri = oe.run_model_idle(*x0[:10])
    pf = ri[-1, :]; base_pk = pf.max(); base_fwhm = (pf >= base_pk / 2).sum() * 360 / oe.N
    med = {p: float(pop[p]) for p in NAMES}
    tier2 = "--tier2" in sys.argv
    args = (bins, vr, vs)
    bounds = oe.JOINT_BOUNDS

    abl = [("(none) full model", "-", {})] \
        + [(p, "M", {p: 0.0}) for p in TYPE_M] \
        + [(p, "F", {p: med[p]}) for p in TYPE_F]  # F = re-optimize with this nudged/held

    rows = []
    for name, typ, fixes in abl:
        t0 = _time.time()
        fix_i = {NAMES.index(k): v for k, v in fixes.items()}
        x = np.array(x0, float)
        for i, v in fix_i.items():
            x[i] = v
        if tier2 and fixes:
            x = refit(x0, fix_i, bounds, args)
        m, anti = evaluate(x, bins, vr, base_pk, base_fwhm)
        _, valid = score(x, bins, vr, vs)
        rows.append(dict(param=name, type=typ, r2=m["r2"], anti=anti,
                         offlobe=m["offlobe"], fwhm600=m["fwhm600"] / base_fwhm,
                         pd600=m["pd600"] / base_pk, valid=valid))
        if tier2:
            print(f"  {name:20s} [{typ}] R2={m['r2']:7.3f} anti={anti:6.1f} "
                  f"off={m['offlobe']:.2f} fwhm600={m['fwhm600']/base_fwhm:.2f} "
                  f"valid={valid} ({_time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    full = df.iloc[0]
    df["dR2"] = full["r2"] - df["r2"]
    df = df.sort_values("dR2", ascending=False)
    tag = "tier2" if tier2 else "tier1"
    df.to_csv(HERE / f"probe_ablation_ring_{tag}.csv", index=False)
    print(f"\n=== RING ablation {tag} (full R2={full['r2']:.3f}, baseline pk={base_pk:.1f} "
          f"fwhm={base_fwhm:.0f}) ===")
    print(df[["param", "type", "dR2", "r2", "anti", "offlobe", "fwhm600", "pd600", "valid"]]
          .to_string(index=False))
    print(f"saved -> probe_ablation_ring_{tag}.csv")


if __name__ == "__main__":
    main()
