"""
compile_v3_results.py
---------------------
Aggregate the v3 per-cell fits, one table per gate variant, joined with each cell's
AD baseline tuning metrics. Mirrors compile_readout_results.py (v2) and reuses the
same baseline lookup, so v1 / v2 / v3 / DoVM tables all join 1:1 on
(Animal_Id, Cell_Id).

    python compile_v3_results.py            # both gate variants
-> optimized_cells_v3_{gate}.parquet + .csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for workbench.data
from compile_results import _cell_key, _load_baseline_lookup, BASELINE_COLS

HERE = Path(__file__).resolve().parent


def compile_gate(gate="mg", mat_path=None):
    files = sorted((HERE / f"results_v3_{gate}").glob("cell_*.json"))
    if not files:
        print(f"No v3 results for gate '{gate}'")
        return None
    baseline = _load_baseline_lookup(mat_path)

    rows, missing = [], 0
    for jf in files:
        d = json.load(open(jf))
        row = {
            "Animal_Id": d["animal_id"], "Cell_Id": d["cell_id"],
            "R_Squared": d["r2"], "gate": d.get("gate", gate),
            "FC_model": d.get("fc_model"), "FC_data": d.get("fc_data"),
            "SC_pd": d.get("sc_pd"), "SC_anti": d.get("sc_anti"),
            "SC_anti_data": d.get("sc_anti_data"),
            **d["params"],
        }
        key = _cell_key(d["animal_id"], d["cell_id"])
        if key in baseline:
            row.update(baseline[key])
        else:
            missing += 1
            for c in BASELINE_COLS:
                row[c] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["Animal_Id", "Cell_Id"]).reset_index(drop=True)
    stem = HERE / f"optimized_cells_v3_{gate}"
    df.to_parquet(f"{stem}.parquet", engine="pyarrow", index=False)
    df.to_csv(f"{stem}.csv", index=False)
    print(f"{gate:7s}: {df.shape[0]} cells x {df.shape[1]} cols | "
          f"R2 median={df['R_Squared'].median():.3f} | "
          f"anti-PD SC median={df['SC_anti'].median():.1f} Hz "
          f"({int((df['SC_anti'] > 5).sum())} cells >5 Hz) -> {stem.name}.parquet")
    if missing:
        print(f"  warning: {missing} cells had no baseline match")
    return df


if __name__ == "__main__":
    gates = sys.argv[1:] or ["mg", "nogate"]
    for g in gates:
        compile_gate(g)
