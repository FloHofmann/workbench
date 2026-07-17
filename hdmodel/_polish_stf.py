"""Direct-R2 polish of the STF (emergent-SC) fit. Seeds from _full_fit_result.json
(DE's best), gated Nelder-Mead multi-restart. Saves back to _full_fit_result.json.

ponytail: same pattern as _polish_linear.py; separate file so the DE seed source differs.
"""
import json
import numpy as np
from scipy.optimize import minimize
import optimization_engine as oe
from vivo_target import load_vivo_psth
from _run_full_fit import score

bins, vr, vs = load_vivo_psth()
p = json.load(open("_full_fit_result.json"))["params"]
x0 = np.array([p[k] for k in oe.PARAM_NAMES])
lo = np.array([b[0] for b in oe.JOINT_BOUNDS]); hi = np.array([b[1] for b in oe.JOINT_BOUNDS])


def negR2(x):
    r2, ok = score(x, bins, vr, vs)
    return -r2 if ok else 1e6


best, bestr = x0.copy(), score(x0, bins, vr, vs)[0]
print(f"start R2={bestr:.4f}", flush=True)
for trial in range(10):
    rng = np.random.default_rng(trial)
    xs = np.clip(x0 * (1 + 0.05 * rng.standard_normal(len(x0))), lo, hi)
    res = minimize(negR2, xs, method="Nelder-Mead",
                   options={"maxiter": 8000, "xatol": 1e-3, "fatol": 1e-4})
    xr = np.clip(res.x, lo, hi); r2, ok = score(xr, bins, vr, vs)
    print(f"  trial{trial}: R2={r2:.4f} valid={ok}", flush=True)
    if ok and r2 > bestr:
        best, bestr = xr, r2
print(f"BEST STF R2={bestr:.4f}", flush=True)
json.dump({"params": dict(zip(oe.PARAM_NAMES, [float(v) for v in best])), "r2": float(bestr)},
          open("_full_fit_result.json", "w"), indent=2)
print("saved _full_fit_result.json", flush=True)
