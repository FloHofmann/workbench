"""
compare_gates.py
----------------
Model comparison of SC gate mechanisms on the 77-cell AD population.

WHY THIS EXISTS
    R2 on the training target cannot support a mechanistic claim: 32 free parameters
    against ~17-18 effectively independent time points fits almost any smooth curve.
    "Mg gating is responsible for the SC" therefore requires (a) removing the gate to
    fail, and (b) rival mechanisms to also fail -- judged OUT OF SAMPLE so no variant
    wins on flexibility. See OPEN_ISSUES.md A1/A2 and CLAIMS.md.

THE VARIANTS (identical 32-parameter vector in every mode -- `tau_scg` doubles as the
Mg mode's dendritic time constant -- so none can win by being more flexible):
    rec    low-passed recurrent drive (bump membership). The current model.
    mg     NMDA Mg2+ block on low-passed u_E -> voltage-dependent gate. THE CLAIM.
    none   no gate at all. Necessity control.
    noSC   A_sc suppressed -> pure I_T post-inhibitory rebound.
    tuned  SC arrives already PD-tuned. REFERENCE UPPER BOUND, NOT a mechanism: there
           is no plausible in-vivo connectivity for a PD-aligned salience projection.
           Included because it fits better and must be addressed, not hidden.

PROTOCOL
    Repeated split-half CV over the 77 cells: fit on the mean of one half, score on the
    held-out half's mean. The held-out target is a ~38-cell mean, so it is far less noisy
    than a single cell (which is why leave-one-cell-out was uninformative).
    Anti-PD silence is NOT enforced during fitting -- it is a model assumption, not a data
    constraint, and enforcing it would disqualify `none` before it competes. It is
    reported as an outcome instead.

    python compare_gates.py [--splits 10] [--maxfev 2500]
-> compare_gates.csv + a printed table
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import differential_evolution, minimize

import optimization_engine as oe
from vivo_target import load_target, load_percell
# NOTE: this module deliberately does NOT use _run_full_fit.score -- it needs the graded
# validity violation from evaluate() below, not a flat accept/reject.

HERE = pathlib.Path(__file__).resolve().parent

ALL_MODES = [("rec", oe.GATE_REC), ("mg", oe.GATE_MG), ("none", oe.GATE_NONE),
             ("noSC", oe.GATE_NOSC), ("tuned", oe.GATE_TUNED),
             ("mg_rec", oe.GATE_MG_REC), ("mm_u", oe.GATE_MM_U)]
MODES = list(ALL_MODES)   # overridden by --modes


def r2(pred, targ):
    return 1.0 - np.sum((targ - pred) ** 2) / np.sum((targ - targ.mean()) ** 2)


def bootstrap_ceiling(X, n_boot=2000, seed=0):
    """R2 a PERFECT model would score against the noisy observed mean of these cells."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    boot = np.array([X[rng.integers(0, n, n)].mean(0) for _ in range(n_boot)])
    m = X.mean(0)
    return float(1.0 - np.mean(boot.std(0) ** 2) / m.var())


def evaluate(x, bins, vr, gate_mode):
    """Return (r2, violation). violation == 0 <=> the fit is structurally VALID.

    Mirrors _run_full_fit.score's gate but returns HOW BADLY each constraint is missed
    instead of a flat reject. A flat penalty makes the search space a featureless plateau
    in 32 dimensions -- differential evolution then has no gradient toward the valid
    region and loses to any warm start (observed: DE lost for all 5 modes). Grading the
    penalty gives the optimiser a slope to descend.

    Anti-PD silence is deliberately NOT a constraint here: it is a model assumption, not
    a data constraint, and enforcing it would disqualify the no-gate control (see
    module docstring).
    """
    x = np.asarray(x, dtype=float)
    s1, s2 = x[:10], x[10:]
    _, ri = oe.run_model_idle(*s1)
    pf = ri[-1, :]
    if not np.all(np.isfinite(pf)):
        return None, 1e3
    pk = float(pf.max())
    pre = (bins >= -50) & (bins < 0)
    base_hz = float(np.mean(vr[pre])) if np.any(pre) else 40.0
    # Recovery bound DERIVED FROM THIS TARGET, not hardcoded. The old flat 1.35 cap was
    # tuned on soso; the AD data's own 600 ms rate is ~1.55x its baseline, so that cap
    # would REJECT THE REAL TRACE and handicap every gated variant (OPEN_ISSUES N1).
    late = (bins > 550) & (bins < 650)
    recov_max = 1.35 * (float(np.mean(vr[late])) / base_hz) if np.any(late) else 1.35

    v = 0.0
    v += max(0.0, 0.825 * base_hz - pk) / base_hz + max(0.0, pk - 1.175 * base_hz) / base_hz
    base_fwhm = float((pf >= pk / 2).sum()) * 360.0 / oe.N
    v += max(0.0, 45.0 - base_fwhm) / 45.0 + max(0.0, base_fwhm - 95.0) / 95.0
    v += max(0.0, pf[oe.OFF_LOBE].max() / max(pk, 1e-9) - 0.10)

    t, rr = oe.run_model_stim(*s1, *s2, gate_mode=gate_mode)
    if not np.all(np.isfinite(rr)):
        return None, 1e3
    pd600 = float(rr[np.argmin(np.abs(t - 600)), oe._IDX_0])
    p6 = rr[np.argmin(np.abs(t - 600)), :]
    fwhm600 = float((p6 >= p6.max() / 2).sum()) * 360.0 / oe.N
    v += max(0.0, pd600 / max(pk, 1e-9) - recov_max)
    v += max(0.0, fwhm600 / max(base_fwhm, 1e-9) - 1.35)

    if v > 0.0:
        return None, v

    w = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[w], t, rr[:, oe._IDX_0])
    return float(r2(mv, vr[w])), 0.0


class BestValid:
    """Objective that remembers the best VALID point it saw.

    The objective is a flat penalty outside the validity gate, so a local optimiser can
    wander out and return something worse than its start. Tracking the best valid point
    guarantees a fit is never worse than its warm start (and mirrors _run_full_fit's
    "rank by R2 among valid" rule).
    """

    def __init__(self, bins, target_full, gate_mode):
        self.bins, self.target, self.gate = bins, target_full, gate_mode
        self.best_r2, self.best_x = -np.inf, None

    def __call__(self, x):
        r, viol = evaluate(x, self.bins, self.target, self.gate)
        if r is None:
            # graded: 1e3 dominates any valid score (-R2 <= 1) so validity always wins,
            # but the optimiser still sees a slope back toward the valid region.
            return 1e3 + 10.0 * viol
        if r > self.best_r2:
            self.best_r2, self.best_x = r, np.array(x, dtype=float)
        return -r


def fit_mode(bins, target_full, gate_mode, starts, maxfev):
    """Fit one gate mode from several starts; keep the best VALID result.

    Multi-start matters for FAIRNESS: every mode is seeded from the rec-mode fit, so
    without it `rec` begins at its own optimum while the others begin at parameters
    tuned for a different gate -- and "mg is worse" would only mean "mg wasn't fitted".
    The gates also pass very different amounts of current (rec ~0.8 on the bump,
    none = 1.0), so each needs its own A_sc scale; that is what the starts vary.
    """
    best_x, best_r2 = None, -np.inf
    for x0 in starts:
        obj = BestValid(bins, target_full, gate_mode)
        obj(np.asarray(x0, dtype=float))       # seed the tracker with the start itself
        minimize(obj, x0=np.asarray(x0, dtype=float), bounds=oe.JOINT_BOUNDS,
                 method="Powell", options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4})
        if obj.best_x is not None and obj.best_r2 > best_r2:
            best_x, best_r2 = obj.best_x, obj.best_r2
    if best_x is None:                          # no valid point anywhere for this mode
        return np.asarray(starts[0], dtype=float), -np.inf
    return best_x, best_r2


def fit_mode_de(bins, target_full, gate_mode, maxiter, popsize, seed=0):
    """Independent global fit for one mode -- NO warm start, no rec-shaped seed.

    Removes the confound that every warm start descends from a rec-optimised fit, which
    gives `rec` an advantage on the other 31 parameters. Run for EVERY mode so the budget
    is identical; the caller keeps whichever of {DE, warm-start} found the better valid
    fit. Kept alongside (not instead of) the warm start because the valid recovering
    basin is narrow and plain DE is known to undershoot here (MODEL.md 10).
    """
    obj = BestValid(bins, target_full, gate_mode)
    differential_evolution(obj, oe.JOINT_BOUNDS, maxiter=maxiter, popsize=popsize,
                           seed=seed, tol=1e-6, init="sobol", mutation=(0.5, 1.5),
                           recombination=0.7, polish=False, workers=1, disp=False)
    return obj.best_x, obj.best_r2


def make_starts(x0, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Warm start replicated across A_sc scalings (see fit_mode docstring)."""
    i = oe.PARAM_NAMES.index("A_sc")
    lo, hi = oe.JOINT_BOUNDS[i]
    out = []
    for s in scales:
        y = np.array(x0, dtype=float)
        y[i] = float(np.clip(y[i] * s, lo, hi))
        out.append(y)
    return out


def model_trace(x, bins, w, gate_mode):
    t, rr = oe.run_model_stim(*x[:10], *x[10:], gate_mode=gate_mode)
    return np.interp(bins[w], t, rr[:, oe._IDX_0]), t, rr


def warm_start():
    """Soso fit rescaled into the AD baseline band (its 38 Hz bump fails the AD gate)."""
    x = np.array([json.load(open(HERE / "_full_fit_result.json"))["params"][k]
                  for k in oe.PARAM_NAMES])
    x[oe.PARAM_NAMES.index("I_HD")] *= 0.70
    x[oe.PARAM_NAMES.index("I_baseline")] *= 0.70
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--maxfev", type=int, default=2500)
    ap.add_argument("--independent", action="store_true",
                    help="also run an independent DE fit per mode (removes the "
                         "warm-start-favours-rec confound); keeps the better valid fit")
    ap.add_argument("--de-maxiter", type=int, default=40)
    ap.add_argument("--de-popsize", type=int, default=12)
    ap.add_argument("--modes", type=str, default="",
                    help="comma-separated subset of modes to run (default: all). "
                         "'rec,mg,mg_rec,mm_u' is the 2x2 disambiguation.")
    ap.add_argument("--tag", type=str, default="",
                    help="extra suffix for output files, so runs don't clobber each other")
    ap.add_argument("--de-modes", type=str, default="",
                    help="comma-separated modes to give the independent DE budget to "
                         "(default: all). The confound only matters for the two real "
                         "contenders, so 'rec,mg' buys most of the fairness for ~1/2 the "
                         "compute; none/noSC already lose decisively and a better fit "
                         "could only raise them.")
    args = ap.parse_args()
    suffix = ("_indep" if args.independent else "") + args.tag
    de_modes = {m.strip() for m in args.de_modes.split(",") if m.strip()}

    global MODES
    want = [m.strip() for m in args.modes.split(",") if m.strip()]
    if want:
        by_name = dict(ALL_MODES)
        missing = [m for m in want if m not in by_name]
        if missing:
            raise SystemExit(f"unknown mode(s): {missing}; known: {list(by_name)}")
        MODES = [(m, by_name[m]) for m in want]

    bins, vr, _ = load_target()
    _, per_cell = load_percell()
    n = per_cell.shape[0]
    w = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    x0 = warm_start()

    full_ceiling = bootstrap_ceiling(per_cell[:, w])
    half_ceiling = bootstrap_ceiling(per_cell[: n // 2, w])
    print(f"AD dataset: {n} cells, {int(w.sum())} bins in [{oe.WIN_LO}, {oe.WIN_HI}] ms")
    print(f"noise ceiling: full {n}-cell mean {full_ceiling:.4f} | "
          f"~{n//2}-cell half mean {half_ceiling:.4f}  <- CV reference")
    _b = float(vr[(bins >= -50) & (bins < 0)].mean())
    _r = float(vr[(bins > 550) & (bins < 650)].mean()) / _b
    print(f"baseline {_b:.1f} Hz | data recovery @600ms {_r:.2f}x -> models allowed "
          f"{1.35 * _r:.2f}x (was a flat 1.35x; OPEN_ISSUES N1)\n")

    # ---- full-data fit per mode (also the per-mode warm start for the CV folds) ----
    if args.independent:
        print(f"FULL-DATA FIT (all 77 cells) -- warm-start multi-start AND independent DE "
              f"(maxiter={args.de_maxiter}, popsize={args.de_popsize}) for EVERY mode")
    else:
        print("FULL-DATA FIT (all 77 cells)")
    print(f"{'mode':>6} {'R2':>8} {'SCpeak':>8} {'antiPD':>8} {'PD@600':>8}  {'won by':>8}")
    full_fits = {}
    starts = make_starts(x0)
    for name, gm in MODES:
        xf, rf = fit_mode(bins, vr, gm, starts, args.maxfev)
        won = "warm"
        if args.independent and (not de_modes or name in de_modes):
            xd, rd = fit_mode_de(bins, vr, gm, args.de_maxiter, args.de_popsize)
            if xd is not None and rd > rf:
                xf, rf, won = xd, rd, "DE"
        _, t, rr = model_trace(xf, bins, w, gm)
        pd_, anti = rr[:, oe._IDX_0], rr[:, oe._IDX_180]
        m = (t > 40) & (t < 300)
        full_fits[name] = xf
        print(f"{name:>6} {rf:>8.4f} {pd_[m].max():>8.1f} {anti[m].max():>8.1f} "
              f"{pd_[np.argmin(np.abs(t - 600))]:>8.1f}  {won:>8}", flush=True)
        json.dump({"gate_mode": name,
                   "r2_full": float(rf) if np.isfinite(rf) else None,
                   "params": dict(zip(oe.PARAM_NAMES, xf.tolist()))},
                  open(HERE / f"_fit_ad_{name}{suffix}.json", "w"), indent=2)

    # ---- split-half cross-validation ----
    print(f"\nSPLIT-HALF CV ({args.splits} random splits, train {n//2} / test {n - n//2})")
    rng = np.random.default_rng(0)
    rows = {name: [] for name, _ in MODES}
    for s in range(args.splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        tr_full = np.zeros(bins.shape); tr_full[w] = per_cell[a][:, w].mean(0)
        te = per_cell[b][:, w].mean(0)
        for name, gm in MODES:
            # each mode starts from ITS OWN full-data optimum -> equal footing
            xs, _ = fit_mode(bins, tr_full, gm, [full_fits[name]], args.maxfev)
            mv, _, _ = model_trace(xs, bins, w, gm)
            rows[name].append(r2(mv, te))
        print(f"  split {s}: " + "  ".join(f"{nm} {rows[nm][-1]:+.3f}" for nm, _ in MODES),
              flush=True)

    print(f"\n=== RESULT: held-out R2 vs a ~{n - n//2}-cell mean "
          f"(ceiling {half_ceiling:.3f}) ===")
    print(f"{'mode':>6} {'CV R2':>9} {'SD':>7}   note")
    out = []
    note = {"rec": "MM form  x  recurrent drive   (current model)",
            "mg": "J-S form x  membrane potential (THE CLAIM)",
            "mg_rec": "J-S form x  recurrent drive   (isolates FORM)",
            "mm_u": "MM form  x  membrane potential (isolates DRIVER)",
            "none": "no gate at all (necessity control)",
            "noSC": "no SC input (pure I_T rebound)",
            "tuned": "REFERENCE ONLY - not a mechanism"}
    for name, _ in MODES:
        v = np.array(rows[name])
        print(f"{name:>6} {v.mean():>+9.4f} {v.std():>7.4f}   {note[name]}")
        out.append((name, v.mean(), v.std()))
    np.savetxt(HERE / f"compare_gates{suffix}.csv",
               np.array([[m, f"{a:.6f}", f"{b:.6f}"] for m, a, b in out], dtype=object),
               delimiter=",", fmt="%s", header="gate_mode,cv_r2_mean,cv_r2_sd", comments="")
    print(f"\nsaved -> compare_gates{suffix}.csv and _fit_ad_<mode>{suffix}.json")
    print("Reminder: folds are warm-started from each mode's full-data fit, so absolute "
          "CV R2 is optimistic; the bias applies to every mode alike, so the COMPARISON "
          "stands.")


if __name__ == "__main__":
    main()
