"""Direct-R2 polish of the LINEAR-gain fit (run with LINEAR_GAIN=True in engine).
Seeds from _warm_linear.json; gated Nelder-Mead multi-restart. Saves _fit_linear.json."""
import json
import numpy as np
from scipy.optimize import minimize
import optimization_engine as oe
from vivo_target import load_vivo_psth
from _run_full_fit import score

bins, vr, vs = load_vivo_psth()
x0 = np.array(json.load(open("_warm_linear.json")))
lo = np.array([b[0] for b in oe.JOINT_BOUNDS]); hi = np.array([b[1] for b in oe.JOINT_BOUNDS])


def negR2(x):
    r2, ok = score(x, bins, vr, vs)
    return -r2 if ok else 1e6


best, bestr = x0.copy(), score(x0, bins, vr, vs)[0]
print(f"start R2={bestr:.4f}", flush=True)
for trial in range(8):
    rng = np.random.default_rng(trial)
    xs = np.clip(x0 * (1 + 0.03 * rng.standard_normal(len(x0))), lo, hi)
    res = minimize(negR2, xs, method="Nelder-Mead",
                   options={"maxiter": 6000, "xatol": 1e-3, "fatol": 1e-4})
    xr = np.clip(res.x, lo, hi); r2, ok = score(xr, bins, vr, vs)
    tag = "*" if (ok and r2 > bestr) else " "
    print(f"  trial{trial}: R2={r2:.4f} valid={ok} {tag}", flush=True)
    if ok and r2 > bestr:
        best, bestr = xr, r2
print(f"BEST LINEAR R2={bestr:.4f}", flush=True)
json.dump({"params": dict(zip(oe.PARAM_NAMES, [float(v) for v in best])), "r2": float(bestr)},
          open("_fit_linear.json", "w"), indent=2)
print("saved _fit_linear.json", flush=True)
