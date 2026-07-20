#!/bin/bash
# Per-cell mechanistic fit as a SLURM array job. One cell per task.
#
#   # set the array bound first (0 .. n_cells-1):
#   sbatch --array=0-$(($(python fit_cell_mech.py --count)-1)) fit_cells_mech.sbatch
#   # then aggregate:
#   python compile_mech_results.py
#
# Prereqs on the node: the repo checkout, its venv, and vivo_percell.npz +
# _full_fit_result.json present in the hdmodel/ dir (fit_cell_mech.py reads only
# these -- no network share / .mat needed).
#
#SBATCH --job-name=mech_percell
#SBATCH --array=0-76             # override on the CLI; = (python fit_cell_mech.py --count) - 1
#SBATCH --cpus-per-task=1        # workers=1 inside; the array IS the parallelism
#SBATCH --time=01:00:00          # per-cell wall (24-dim DE: 22 evoked + 2 drive, structure frozen)
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/mech_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs results_mech

# <<< CLUSTER-SPECIFIC: activate the environment. Fill ONE of these in. >>>
# module load python/3.11
# source /path/to/.venv/bin/activate
# conda activate hdmodel

python fit_cell_mech.py "${SLURM_ARRAY_TASK_ID}"
