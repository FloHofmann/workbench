"""
compile_min_results.py
----------------------
Aggregate the minimal-model fits + verify the FOUR criteria against v3 and the data.
Mirrors compile_v3_results.py (reuses the AD baseline lookup) and adds a criteria
summary so the minimal model's cost is stated honestly.

    python compile_min_results.py
-> optimized_cells_min.parquet + .csv, and a printed 4-criteria comparison
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compile_results import _cell_key, _load_baseline_lookup, BASELINE_COLS
import readout_cell_v3 as r3

HERE = Path(__file__).resolve().parent


def main():
    files = sorted((HERE / "results_min").glob("cell_*.json"))
    if not files:
        print("no results_min/"); return
    baseline = _load_baseline_lookup(None)
    rows = []
    for jf in files:
        d = json.load(open(jf))
        row = {"Animal_Id": d["animal_id"], "Cell_Id": d["cell_id"], "R_Squared": d["r2"],
               "SC_pd": d["sc_pd"], "SC_anti": d["sc_anti"], "SC_anti_data": d["sc_anti_data"],
               **d["params"]}
        k = _cell_key(d["animal_id"], d["cell_id"])
        row.update(baseline.get(k, {c: float("nan") for c in BASELINE_COLS}))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["Animal_Id", "Cell_Id"]).reset_index(drop=True)
    df.to_parquet(HERE / "optimized_cells_min.parquet", index=False)
    df.to_csv(HERE / "optimized_cells_min.csv", index=False)

    # recovery + self-consistency need re-simulation
    d = np.load(HERE / "vivo_percell.npz", allow_pickle=True)
    drv = r3.load_drivers(HERE / "network_drivers.npz", "pd")
    t, b = drv[0], d["bins"]
    ids = list(zip([str(a) for a in d["animal_ids"]], [float(c) for c in d["cell_ids"]]))
    Y, recov_m, recov_d = [], [], []
    for _, r in df.iterrows():
        i = ids.index((str(r["Animal_Id"]), float(r["Cell_Id"])))
        x = np.array([float(r[k]) for k in r3.CELL_PARAMS])
        y = r3.simulate_readout_v3(x, *drv, 0); Y.append(y)
        base = y[(t >= -50) & (t < 0)].mean()
        recov_m.append(y[np.argmin(np.abs(t - 600))] / base if base > 1 else np.nan)
        pr = d["pd_rate"][i]
        bd = pr[(b >= -50) & (b < 0)].mean()
        recov_d.append(pr[(b > 550) & (b < 700)].mean() / bd if bd > 1 else np.nan)
    Y = np.vstack(Y)
    mean_model = Y.mean(0); mean_data = d["pd_rate"].mean(0)
    w = (b >= -50) & (b <= 700); mi = np.interp(b[w], t, mean_model)
    sc_r2 = 1 - np.sum((mean_data[w] - mi) ** 2) / np.sum((mean_data[w] - mean_data[w].mean()) ** 2)

    n_free = len([k for k in ["I_HD", "A_fast", "A_sc", "g_a", "g_T", "V_half_T", "k_T"]])
    print(f"MINIMAL MODEL -- {n_free} free params/cell, {df.shape[0]} cells")
    print("  1 shape (R2)        median %.3f  (noise ceiling 0.615; v3=0.692 incl. overfit)"
          % df["R_Squared"].median())
    print("  2 selectivity       anti-PD SC median %.1f Hz | %d/77 leak >5 Hz  (v3: 12/77)"
          % (df["SC_anti"].median(), int((df["SC_anti"] > 5).sum())))
    print("  3 recovery ratio    model %.2f  vs data %.2f  (rate@600 / baseline)"
          % (np.nanmedian(recov_m), np.nanmedian(recov_d)))
    print("  4 heterogeneity     free-param CVs: " + ", ".join(
        f"{k} {df[k].std()/abs(df[k].mean()):.2f}"
        for k in ["I_HD", "A_fast", "A_sc", "g_a", "g_T", "V_half_T", "k_T"]))
    print("  self-consistency    mean of 77 fits vs mean data: R2 %.3f" % sc_r2)
    print("saved -> optimized_cells_min.parquet + .csv")


if __name__ == "__main__":
    main()
