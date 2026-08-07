"""
probe_sc_feasibility.py
-----------------------
Can a BASIC ring attractor -- no I_T, no I_h, no Mg, no NMDA kinetics, no gate -- produce
the observed SC at PD while the antipode stays sub-threshold, using inhibition alone?

That is the supervisor's hypothesis: the SC input is ubiquitously distributed and anti-PD
is silent only because inhibition holds it below threshold ("the SC is gated by
inhibition"). If true, the whole gating apparatus is unnecessary.

THE TEST: maximise R2 against the measured PD trace while holding anti-PD below a
threshold, then sweep that threshold. The question is

    does forcing anti-PD silent cost the ability to reproduce the measured response?

  (b) thr = inf  -- unconstrained reference: the best this model can do at all, and what
      anti-PD does when nothing stops it.
  (a) thr -> 0   -- the supervisor's regime. If R2 holds up as anti-PD is clamped silent,
      inhibition alone suffices and no gate is needed. If R2 collapses, it does not.

R2 on the whole trace, NOT peak SC amplitude: maximising a scalar peak lets the optimiser
win with a response of the wrong shape (an early version hit 157% of the observed SC while
sitting at baseline at 60 ms, where the real SC peaks).

Three inhibition architectures, explorable individually and together: `ring` (tuned
interneuron ring only), `global` (uniform pool only), `both`.

The optimiser uses a GRADED validity penalty. A flat accept/reject makes 17-D space a
featureless plateau and differential evolution then loses to any warm start -- that
already produced a silently-useless run once this session.

    python probe_sc_feasibility.py [--modes ring,global,both] [--de-maxiter 40]
-> probe_sc_feasibility.csv + probe_sc_feasibility.png
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import differential_evolution, minimize

import minimal_ring as mr
from vivo_target import load_target

HERE = pathlib.Path(__file__).resolve().parent
THRESHOLDS = [0.5, 2.0, 5.0, 10.0, 20.0, 40.0]   # anti-PD SC ceiling, Hz


# ── data reference points ────────────────────────────────────────────────────
def data_reference():
    bins, rate, _ = load_target()
    pre = (bins >= -50) & (bins < 0)
    sc = (bins > 40) & (bins < 300)
    base = float(rate[pre].mean())
    late = (bins > 550) & (bins < 650)
    # The DATA's own recovery ratio at ~600 ms is ~1.55 -- the SC has NOT fully decayed.
    # Measure it rather than assuming: a hardcoded 1.35 cap (the full model's gate, tuned
    # on soso) is STRICTER THAN THE DATA and would reject the real trace, biasing the
    # whole test toward "inhibition cannot do it".
    return {
        "bins": bins, "rate": rate,
        "baseline": base,
        "sc_peak": float(rate[sc].max()),
        "fc_peak": float(rate.max()),
        "recovery": float(rate[late].mean() / base),
    }


# ── graded validity ─────────────────────────────────────────────────────────
def violation(m, baseline_hz, anti_thr, recov_max):
    """How badly this parameter set misses the structural requirements. 0 == valid.

    Graded, not boolean, so the optimiser has a slope to descend toward validity.
    """
    if not m["ok"]:
        return 1e3
    v = 0.0
    # 1. idle bump must sit at the data's own baseline rate (+/-17.5%)
    v += max(0.0, 0.825 * baseline_hz - m["base_peak"]) / baseline_hz
    v += max(0.0, m["base_peak"] - 1.175 * baseline_hz) / baseline_hz
    # 2. a real, single, HD-shaped bump
    v += max(0.0, 45.0 - m["base_fwhm"]) / 45.0 + max(0.0, m["base_fwhm"] - 95.0) / 95.0
    # 3. antipode silent at rest (that is what "anti-PD" means)
    v += max(0.0, m["base_anti"] - 1.0) / 5.0
    # 4. the FC MUST reach the antipode -- externally observed, and it proves anti-PD is
    #    not just unconditionally sub-threshold
    v += max(0.0, 5.0 - m["anti_fc"]) / 5.0
    # 5. NO RUNAWAY. R_MAX is a safety clip, so a diverged solution still reads as
    #    "finite" -- the first run exploited exactly this, returning PD SC = 1000.0 Hz
    #    (= R_MAX) with anti-PD 0.0 and calling it valid.
    v += max(0.0, m["max_rate"] - 0.9 * mr.R_MAX) / mr.R_MAX * 10.0
    # 6. RECOVERY: the measured SC is a TRANSIENT -- PD returns to ~baseline by ~600 ms.
    #    Without this, "max PD SC" can be won by latching into a permanently elevated
    #    state. `recov_max` is 1.35x the DATA's own recovery ratio, not an absolute number.
    v += max(0.0, m["pd_600"] / max(m["base_peak"], 1e-9) - recov_max)
    # 7. the constraint under test
    v += max(0.0, m["anti_sc"] - anti_thr) / max(anti_thr, 1.0)
    return v


def run(p, mode):
    t, r = mr.simulate(p, mode)
    return mr.metrics(t, r)


class FitObjective:
    """Maximise R2 against the PD PSTH subject to anti-PD staying below `anti_thr`.

    R2, NOT peak SC amplitude. Maximising a scalar peak lets the optimiser win with a
    response of the wrong SHAPE: an early attempt produced PD SC 140 Hz (157% of the
    observed) with anti-PD at 0 -- but its trace was still at baseline at 60 ms, where the
    real SC peaks, and only rose to a slow late hump after 100 ms. Fitting the whole trace
    is what actually asks the scientific question:

        does forcing anti-PD silent cost the ability to reproduce the measured response?

    `anti_thr = inf` recovers the unconstrained fit.
    """

    def __init__(self, mode, baseline_hz, bins, rate, recov_max, anti_thr=np.inf):
        self.mode, self.baseline, self.recov_max = mode, baseline_hz, recov_max
        self.thr = anti_thr
        self.w = (bins >= -50) & (bins <= 700)
        self.bins, self.target = bins, rate[(bins >= -50) & (bins <= 700)]
        self.best, self.best_x = -np.inf, None

    def __call__(self, x):
        t, r = mr.simulate(x, self.mode)
        m = mr.metrics(t, r)
        v = violation(m, self.baseline, self.thr, self.recov_max)
        if v > 0.0:
            return 1e3 + 10.0 * v
        mv = np.interp(self.bins[self.w], t, r[:, mr.IDX_PD])
        r2 = 1 - np.sum((self.target - mv) ** 2) / np.sum((self.target - self.target.mean()) ** 2)
        if r2 > self.best:
            self.best, self.best_x = r2, np.array(x, float)
        return -r2


def optimise(obj, maxiter, popsize, seed=0, x0=None):
    """DE + Powell polish. `x0` may be a single start or a LIST of candidate starts.

    Without an anchor DE cannot locate the valid region from scratch at a realistic
    budget (observed: 0/6 thresholds found any valid point). Different thresholds are
    reachable from different anchors -- the tight ones from the previous (looser)
    solution, but sometimes only from the far unconstrained fit -- so try every supplied
    candidate, register each, and start DE from whichever scores best.
    """
    cands = [] if x0 is None else ([x0] if np.ndim(x0[0]) == 0 else list(x0))
    best_seed, best_val = None, np.inf
    for c in cands:
        val = obj(np.asarray(c, float))     # registers it if it satisfies validity
        if val < best_val:
            best_seed, best_val = np.asarray(c, float), val
    kw = {"x0": best_seed} if best_seed is not None else {}
    differential_evolution(obj, mr.BOUNDS, maxiter=maxiter, popsize=popsize, seed=seed,
                           tol=1e-6, init="sobol", mutation=(0.5, 1.5), recombination=0.7,
                           polish=False, workers=1, disp=False, **kw)
    start = obj.best_x if obj.best_x is not None else best_seed
    if start is not None:
        minimize(obj, x0=np.asarray(start, float), bounds=mr.BOUNDS, method="Powell",
                 options={"maxfev": 4000, "xtol": 1e-3, "ftol": 1e-4})
    return obj.best_x, obj.best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(mr.INH_MODES))
    ap.add_argument("--de-maxiter", type=int, default=40)
    ap.add_argument("--de-popsize", type=int, default=12)
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    d = data_reference()
    recov_max = 1.35 * d["recovery"]
    print(f"AD data: baseline {d['baseline']:.1f} Hz | FC peak {d['fc_peak']:.1f} | "
          f"SC peak {d['sc_peak']:.1f} Hz  <- the target to reach")
    print(f"data recovery @600ms = {d['recovery']:.2f}x baseline -> model allowed up to "
          f"{recov_max:.2f}x  (the DATA is not back to baseline either)\n")

    # (b) first: it is unconstrained in anti-PD, so it reliably lands in the valid region
    # and its solution anchors the constrained searches below.
    print("(b) FIT to the PD PSTH, anti-PD UNCONSTRAINED")
    print(f"{'mode':>7} {'R2':>8} {'PD SC':>8} {'antiPD SC':>10}")
    fit_rows, anchors, fit_r2 = [], {}, {}
    for mode in modes:
        obj = FitObjective(mode, d["baseline"], d["bins"], d["rate"], recov_max)
        x, r2 = optimise(obj, args.de_maxiter, args.de_popsize)
        if x is None:
            print(f"{mode:>7} {'no valid fit':>8}")
            continue
        anchors[mode] = x
        fit_r2[mode] = r2
        m = run(x, mode)
        print(f"{mode:>7} {r2:>8.4f} {m['pd_sc']:>8.1f} {m['anti_sc']:>10.1f}", flush=True)
        fit_rows.append((mode, r2, m["pd_sc"], m["anti_sc"]))
        json.dump({"mode": mode, "r2": float(r2), "metrics": m,
                   "params": dict(zip(mr.PARAM_NAMES, x.tolist()))},
                  open(HERE / f"_fit_minimal_{mode}.json", "w"), indent=2)

    print("\n(a) FEASIBILITY -- best achievable FIT with anti-PD held below a threshold")
    print("    (R2 of the whole PD trace, so the SC must have the right shape AND timing)")
    print(f"{'mode':>7} {'antiPD<=':>9} {'best R2':>9} {'PD SC':>8} {'antiPD':>7} {'basePk':>7}")
    rows = []
    for mode in modes:
        # Walk the constraint from LOOSE to TIGHT, seeding each run with the previous
        # solution (a short hop) AND the far unconstrained fit.
        fit_anchor = anchors.get(mode)
        prev = fit_anchor
        pool = []          # every valid solution found, for the feasibility frontier
        if fit_anchor is not None:
            fm = run(fit_anchor, mode)
            pool.append((float(fit_r2[mode]), fm["anti_sc"], fm["pd_sc"], fm["base_peak"]))
        for thr in sorted(THRESHOLDS, reverse=True):
            obj = FitObjective(mode, d["baseline"], d["bins"], d["rate"], recov_max, thr)
            cands = [c for c in (prev, fit_anchor) if c is not None]
            x, r2 = optimise(obj, args.de_maxiter, args.de_popsize, x0=(cands or None))
            if x is not None:
                prev = x
                m = run(x, mode)
                pool.append((float(r2), m["anti_sc"], m["pd_sc"], m["base_peak"]))
                json.dump({"mode": mode, "anti_thr": thr, "r2": float(r2), "metrics": m,
                           "params": dict(zip(mr.PARAM_NAMES, x.tolist()))},
                          open(HERE / f"_feas_{mode}_{thr:g}.json", "w"), indent=2)

        # FEASIBILITY FRONTIER: a solution valid at a TIGHT anti-PD threshold is valid at
        # every LOOSER one, so the true "best R2 given anti-PD <= thr" is the best R2 over
        # ALL pooled solutions whose anti_sc <= thr. Removes optimiser noise and guarantees
        # monotonicity (a looser threshold can never yield a worse best).
        for thr in sorted(THRESHOLDS, reverse=True):
            elig = [s for s in pool if s[1] <= thr + 1e-6]
            if not elig:
                print(f"{mode:>7} {thr:>9.1f} {'no valid fit':>9}")
                rows.append((mode, thr, np.nan, np.nan, np.nan, np.nan))
                continue
            r2, anti_sc, pd_sc, base_peak = max(elig, key=lambda s: s[0])
            print(f"{mode:>7} {thr:>9.1f} {r2:>9.4f} {pd_sc:>8.1f} "
                  f"{anti_sc:>7.1f} {base_peak:>7.1f}", flush=True)
            rows.append((mode, thr, r2, pd_sc, anti_sc, base_peak))

    with open(HERE / "probe_sc_feasibility.csv", "w") as f:
        f.write("kind,mode,anti_thr,r2,pd_sc,anti_sc,base_peak\n")
        for mo, thr, r2, pd_, an, bp in rows:
            f.write(f"feasibility,{mo},{thr},{r2},{pd_},{an},{bp}\n")
        f.write("kind,mode,r2,pd_sc,anti_sc\n")
        for mo, r2, pd_, an in fit_rows:
            f.write(f"fit,{mo},{r2},{pd_},{an}\n")

    _plot(rows, fit_rows, d)
    print("\nsaved -> probe_sc_feasibility.csv + .png")


def _plot(rows, fit_rows, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {"ring": "#c0392b", "global": "#2980b9", "both": "#27ae60"}
    for mode in mr.INH_MODES:
        pts = [(t, q) for mo, t, q, _, _, _ in rows if mo == mode and np.isfinite(q)]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=colors[mode], label=f"{mode} inhibition", lw=2)
    ax.axhline(0.9726, color="k", ls="--", lw=1.6, label="noise ceiling (0.973)")
    ax.set_xscale("log")
    ax.set_xlabel("anti-PD SC allowed (Hz, log scale)  -- data says ~0")
    ax.set_ylabel("best achievable fit to the PD trace (R2)")
    ax.set_title("Can inhibition alone gate the SC?\n"
                 "basic attractor: no I_T / I_h / Mg / NMDA / gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "probe_sc_feasibility.png", dpi=150)


if __name__ == "__main__":
    main()
