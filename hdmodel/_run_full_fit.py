"""Throwaway single-stage JOINT fit runner (background). All params at once."""
import json
import pathlib
import numpy as np
from scipy.optimize import differential_evolution, minimize
import optimization_engine as oe
from vivo_target import load_vivo_psth


N_SEEDS = 6  # the 24-D landscape is multimodal -> multistart, keep the best basin


def main():
    bins, vr, vs = load_vivo_psth()
    args = (bins, vr, vs)

    best = None
    # Anchor on the previous saved best (if any): polish from it so a run never
    # REGRESSES just because this batch of seeds missed the good basin.
    prev = pathlib.Path("_full_fit_result.json")
    if prev.exists():
        try:
            x0 = [json.load(open(prev))["params"][k] for k in oe.PARAM_NAMES]
            ra = minimize(oe.loss_joint, x0=x0, bounds=oe.JOINT_BOUNDS, args=args,
                          method="L-BFGS-B", options={"maxiter": 600})
            print(f"  anchor (prev best): loss {ra.fun:.1f}", flush=True)
            best = ra
        except (KeyError, ValueError):
            pass  # param set changed (different dims) -> skip anchor

    for seed in range(N_SEEDS):
        r = differential_evolution(
            oe.loss_joint, oe.JOINT_BOUNDS, args=args, seed=seed, maxiter=400,
            popsize=18, tol=1e-6, init="sobol", mutation=(0.7, 1.9),
            recombination=0.5, polish=True, disp=False, workers=-1)
        rp = minimize(oe.loss_joint, x0=r.x, bounds=oe.JOINT_BOUNDS, args=args,
                      method="L-BFGS-B", options={"maxiter": 400})
        print(f"  seed {seed}: loss {rp.fun:.1f}", flush=True)
        if best is None or rp.fun < best.fun:
            best = rp
    rp = best
    p = list(rp.x)
    s1, s2 = p[:10], p[10:]
    print("JOINT loss", round(rp.fun, 1), flush=True)
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
