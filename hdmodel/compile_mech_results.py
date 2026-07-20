"""
compile_mech_results.py
-----------------------
Aggregate the per-cell MECHANISTIC fits (results_mech/cell_*.json) into one tidy
table, joined with each cell's AD baseline tuning metrics. Mirror of
compile_results.py (which does the same for the phenomenological DoVM fits) --
reuses its baseline-lookup so the two tables join 1:1 on (Animal_Id, Cell_Id).

    python compile_mech_results.py
-> optimized_cells_mech.parquet  +  optimized_cells_mech.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for workbench.data
from compile_results import _cell_key, _load_baseline_lookup, BASELINE_COLS


def compile_mech(results_dir="results_mech",
                 out_stem="optimized_cells_mech",
                 mat_path=None):
    files = sorted(Path(results_dir).glob("cell_*.json"))
    if not files:
        print(f"No JSON files in {results_dir}/")
        return

    print(f"{len(files)} result files. Loading AD baseline metrics...")
    baseline = _load_baseline_lookup(mat_path)

    rows, missing = [], 0
    for jf in files:
        d = json.load(open(jf))
        row = {
            "Animal_Id": d["animal_id"],
            "Cell_Id": d["cell_id"],
            "R_Squared": d["r2"],
            "valid": d.get("valid", None),
            **d["params"],            # all 32 per-cell params
        }
        key = _cell_key(d["animal_id"], d["cell_id"])
        if key in baseline:
            row.update(baseline[key])
        else:
            missing += 1
            for col in BASELINE_COLS:
                row[col] = float("nan")
        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values(["Animal_Id", "Cell_Id"])
          .reset_index(drop=True))
    df.to_parquet(f"{out_stem}.parquet", engine="pyarrow", index=False)
    df.to_csv(f"{out_stem}.csv", index=False)

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    if missing:
        print(f"Warning: {missing} cells had no baseline match")
    print(f"Saved -> {out_stem}.parquet + .csv")


if __name__ == "__main__":
    compile_mech()
