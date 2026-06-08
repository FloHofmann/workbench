import json
from pathlib import Path

import pandas as pd

from workbench.data.load_ad_table import load

BASELINE_COLS = [
    "HDpeakFRsmooth", "HDpeakFR", "HDInx",
    "pValS", "pValR", "nSpk", "HDFR", "HDAngle",
    "vm_deltaBIC", "vm_deltaAIC", "vm_deltaLL",
    "vm_mu1_deg", "vm_mu2_deg", "vm_kappa", "vm_kappa2",
    "vm_pi", "vm_mu_uni_deg", "vm_kappa_uni",
]


def _cell_key(animal_id, cell_id) -> tuple[str, str]:
    """Normalise (Animal_Id, Cell_Id) to (str, str-of-int) for dict lookup."""
    try:
        cid = str(int(float(cell_id)))
    except (ValueError, TypeError):
        cid = str(cell_id)
    return (str(animal_id), cid)


def _load_baseline_lookup(mat_path=None) -> dict:
    """
    Return {(animal_id_str, cell_id_int_str): dict_of_metrics} for every
    Baseline row in the AD combined table.
    """
    kwargs = {"mat_path": mat_path} if mat_path else {}
    ad = load(**kwargs)
    bl = ad.df[ad.df["Condition"] == "Baseline"]

    lookup = {}
    for _, row in bl.iterrows():
        key = _cell_key(row["Animal_Id"], row["Cell_Id"])
        metrics = {}
        for col in BASELINE_COLS:
            if col not in bl.columns:
                metrics[col] = float("nan")
                continue
            val = row[col]
            try:
                metrics[col] = float(val) if pd.notna(val) else float("nan")
            except (TypeError, ValueError):
                metrics[col] = float("nan")
        lookup[key] = metrics

    return lookup


def compile_results(
    results_dir: str = "results",
    output_file: str = "optimized_cells.parquet",
    mat_path=None,
):
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("cell_*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}/")
        return

    print(f"Found {len(json_files)} result files. Loading AD baseline metrics...")
    baseline = _load_baseline_lookup(mat_path)
    print(f"AD table: {len(baseline)} baseline rows loaded.")

    compiled, missing = [], 0

    for json_file in json_files:
        with open(json_file) as fh:
            d = json.load(fh)

        row = {
            "Animal_Id": d["animal_id"],
            "Cell_Id":   d["cell_id"],
            "Loss_100N": d["loss_100N"],
            "Loss_360N": d["loss_360N"],
            "R_Squared": d["r_squared"],
            **d["params"],
        }

        key = _cell_key(d["animal_id"], d["cell_id"])
        if key in baseline:
            row.update(baseline[key])
        else:
            missing += 1
            for col in BASELINE_COLS:
                row[col] = float("nan")

        compiled.append(row)

    df = (
        pd.DataFrame(compiled)
        .sort_values(["Animal_Id", "Cell_Id"])
        .reset_index(drop=True)
    )

    df.to_parquet(output_file, engine="pyarrow", index=False)

    print(f"\nShape  : {df.shape[0]} rows × {df.shape[1]} columns")
    if missing:
        print(f"Warning: {missing} cells had no baseline match in AD table")
    print(f"Saved  → {output_file}")


if __name__ == "__main__":
    compile_results()
