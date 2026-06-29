"""Throwaway single-stage JOINT fit runner (background). All params at once."""
import json
import pathlib
import numpy as np
from scipy.optimize import differential_evolution, minimize
import optimization_engine as oe
from vivo_target import load_vivo_psth


N_SEEDS = 18  # the 27-D landscape is multimodal -> multistart, keep the best basin


def score(x, bins, vr, vs):
    """Return (R2, valid). Rank candidates by R2 (the loss is NOT R2-aligned: its
    regularizers can let a flat low-R2 trace score a lower total loss), but only
    among STRUCTURALLY VALID fits (single baseline bump, anti-PD silent, the bump
    doesn't blow up) so best-R2 can't pick a degenerate trace."""
    s1, s2 = list(x[:10]), list(x[10:])
    _, ri = oe.run_model_idle(*s1)
    pf = ri[-1, :]; pk = pf.max()
    # Baseline must be a REAL ~40 Hz bump (was 20-90, which let the fit cheat with a
    # weak 28 Hz baseline so a flat elevated plateau counted as "recovered").
    if not np.all(np.isfinite(pf)) or pk < 33 or pk > 47:
        return -1e9, False
    base_fwhm = (pf >= pk / 2).sum() * 360 / oe.N
    if base_fwhm < 45 or base_fwhm > 95 or pf[oe.OFF_LOBE].max() / pk > 0.10:
        return -1e9, False
    t, rr = oe.run_model_stim(*s1, *s2)
    if not np.all(np.isfinite(rr)):
        return -1e9, False
    m = (t > 40) & (t < 700)
    if rr[:, oe._IDX_180][m].max() > 6:                 # anti-PD must be silent
        return -1e9, False
    # RECOVERY (RELATIVE to the idle baseline, not an absolute cutoff): by ~600 ms
    # the SC must have decayed back to ~baseline in BOTH rate and width -- no
    # latching into a broad/elevated 2nd attractor AND no flat plateau well above
    # baseline. Must be a genuine return to the idle bump.
    pd600 = rr[np.argmin(np.abs(t - 600)), oe._IDX_0]
    p6 = rr[np.argmin(np.abs(t - 600)), :]
    fwhm600 = (p6 >= p6.max() / 2).sum() * 360 / oe.N
    if pd600 > 1.35 * pk:                               # rate back to ~baseline
        return -1e9, False
    if fwhm600 > 1.35 * base_fwhm:                      # width back to ~baseline
        return -1e9, False
    pd = rr[:, oe._IDX_0]
    win = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[win], t, pd)
    r2 = 1 - np.sum((vr[win] - mv) ** 2) / np.sum((vr[win] - vr[win].mean()) ** 2)
    return float(r2), True


def main():
    bins, vr, vs = load_vivo_psth()
    args = (bins, vr, vs)

    best_x, best_r2 = None, -1e9
    # Anchor on the previous saved best (if any): polish from it so a run never
    # REGRESSES just because this batch of seeds missed the good basin. We polish
    # from TWO warm starts: the saved params as-is, AND the saved params with
    # W_GLOBAL forced into the monostable band (~8) -- a sweep showed the old fit
    # latched into a broad/elevated 2nd attractor because W_GLOBAL was too weak, so
    # we hand the optimizer a start that already recovers and let it refine.
    prev = pathlib.Path("_full_fit_result.json")
    if prev.exists():
        try:
            x0 = [json.load(open(prev))["params"][k] for k in oe.PARAM_NAMES]
            x0_mono = list(x0); x0_mono[oe.PARAM_NAMES.index("W_GLOBAL")] = 8.0
            for tag, start in [("prev best", x0), ("prev+Wg8 (monostable)", x0_mono)]:
                ra = minimize(oe.loss_joint, x0=start, bounds=oe.JOINT_BOUNDS, args=args,
                              method="L-BFGS-B", options={"maxiter": 600})
                r2, ok = score(ra.x, bins, vr, vs)
                print(f"  anchor ({tag}): R2={r2:.3f} valid={ok}", flush=True)
                if ok and r2 > best_r2:
                    best_x, best_r2 = ra.x, r2
        except (KeyError, ValueError):
            pass  # param set changed (different dims) -> skip anchor

    for seed in range(N_SEEDS):
        r = differential_evolution(
            oe.loss_joint, oe.JOINT_BOUNDS, args=args, seed=seed, maxiter=500,
            popsize=24, tol=1e-6, init="sobol", mutation=(0.7, 1.9),
            recombination=0.5, polish=True, disp=False, workers=-1)
        rp = minimize(oe.loss_joint, x0=r.x, bounds=oe.JOINT_BOUNDS, args=args,
                      method="L-BFGS-B", options={"maxiter": 400})
        r2, ok = score(rp.x, bins, vr, vs)
        print(f"  seed {seed}: loss {rp.fun:.1f}  R2={r2:.3f}  valid={ok}", flush=True)
        if ok and r2 > best_r2:
            best_x, best_r2 = rp.x, r2
    if best_x is None:
        print("NO VALID FIT FOUND -- keeping previous _full_fit_result.json", flush=True)
        return
    p = list(best_x)
    s1, s2 = p[:10], p[10:]
    print("BEST R2 = %.3f" % best_r2, flush=True)
    print(" stage1:", {k: round(v, 2) for k, v in zip(oe.PARAM_NAMES[:10], s1)}, flush=True)
    print(" stage2:", {k: round(v, 2) for k, v in zip(oe.PARAM_NAMES[10:], s2)}, flush=True)

    # baseline check
    _, ri = oe.run_model_idle(*s1)
    pf = ri[-1, :]; pk = pf.max()
    print("BASELINE pk=%.0f FWHM=%.0f sec%%=%.3f antiPD=%.2f"
          % (pk, (pf >= pk / 2).sum() * 360 / oe.N, pf[oe.OFF_LOBE].max() / max(pk, .01), pf[oe._IDX_180]), flush=True)

    t, rr = oe.run_model_stim(*s1, *s2)
    pd = rr[:, oe._IDX_0]; anti = rr[:, oe._IDX_180]
    win = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[win], t, pd)
    ss = np.sum((vr[win] - mv) ** 2); st = np.sum((vr[win] - vr[win].mean()) ** 2)
    r2v = 1 - ss / st

    p0 = list(s2); p0[4] = 0.0; p0[9] = 0.0
    _, r0 = oe.run_model_stim(*s1, *p0); pd0 = r0[:, oe._IDX_0]
    m = (t > 40) & (t < 300)

    def at(x, a=pd):
        return a[np.argmin(np.abs(t - x))]

    fc_win = (t >= s2[2]) & (t < s2[2] + 12)
    print("R2 = %.3f" % r2v, flush=True)
    print("PD FCpeak=%.0f(data~170) @dip=%.1f @40=%.0f @55=%.0f @100=%.0f @200=%.0f"
          % (pd[fc_win].max(), pd[(t > s2[2] + 8) & (t < s2[2] + 18)].mean(), at(40), at(55), at(100), at(200)), flush=True)
    print("SC full=%.1f@%.0fms ablated=%.1f drop=%.0f%% | antiPD SC max=%.1f"
          % (pd[m].max(), t[m][pd[m].argmax()], pd0[m].max(),
             100 * (1 - pd0[m].max() / max(pd[m].max(), .01)), anti[m].max()), flush=True)

    json.dump({"params": dict(zip(oe.PARAM_NAMES, p)), "r2": float(r2v)},
              open("_full_fit_result.json", "w"), indent=2)
    print("saved _full_fit_result.json", flush=True)


if __name__ == "__main__":
    main()
