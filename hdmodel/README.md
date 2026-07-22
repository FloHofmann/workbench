# hdmodel — mechanistic model of auditory-evoked HD-cell responses

**The centerpiece is the *average* (population) mechanistic model** — a head-direction
ring attractor with thalamic channel dynamics that reproduces the population PSTH of
AD-thalamic HD cells (FC → dip → slow component), fit to R² ≈ **0.95**. Everything else in
this folder is secondary: a per-cell *extension* of that same model, the *probes* that
stress-test it, and *archived* superseded code.

**Start here:** [DATA.md](DATA.md) (what the data *is*, and its limits — **read this first**) ·
[MODEL.md](MODEL.md) (the model, with a notation appendix) ·
[CLAIMS.md](CLAIMS.md) (each claim → evidence → how strongly the evidence supports it) ·
[OPEN_ISSUES.md](OPEN_ISSUES.md) (what to fix/answer before presenting this).

> **⚠️ Read before quoting any result (2026-07-21).** The `soso` (SoundSource) data is
> **head-fixed** — the cell is held at its PD for the whole recording, and the four speakers
> are **sound-source locations, not head directions**. This dataset therefore **cannot test
> direction-selectivity**; the "antipode shows FC only, no SC" constraint comes from a
> **separate dataset not in this repo**. All "anti-PD" quantities computed here from the
> *farthest speaker* are void as data claims (they compare two samples of the same
> condition). Model-internal numbers — R², ablation costs, identifiability — are unaffected.
> Details and the retraction list: [DATA.md](DATA.md).

---

## 1. Average model — THE model (population PSTH)

The ring attractor fit jointly to the population-mean PSTH. This is what to explain to
advisors.

| file | role |
|---|---|
| [MODEL.md](MODEL.md) | full write-up + notation appendix |
| `optimization_engine.py` | the ring model (dynamics, currents, loss) |
| `_run_full_fit.py` | R²-search with validity gate → the reported fit |
| `diagnose_mechanistic.py` | diagnostic plot + ablations |
| `vivo_target.py` | **the data loader** (population PSTH; single source of truth) |
| `_full_fit_result.json` | the 32-parameter fit (R² = 0.9534 vs the speaker-pooled target) |
| `_diagnostic.png` | model vs data figure |
| `mechanistic_model.ipynb` / `.pdf` | exploration notebook / export |
| `_polish_stf.py`, `_polish_bi.log`, `_warm_gsc.json`, `_full_fit_injected.json` | fit-polish helpers/artifacts |

## 2. Per-cell extension (secondary)

The *same* readout dynamics applied to each of 77 cells inside a fixed average network —
heterogeneity, not a new model. See [MODEL_MINIMAL.md](MODEL_MINIMAL.md) /
[MODEL_V3.md](MODEL_V3.md) / [PARAMS.md](PARAMS.md) / [METHODS.md](METHODS.md).

| files | role |
|---|---|
| `network_background.py`, `network_drivers.npz` | record the fixed avg-network drivers (+ exactness test) |
| `readout_cell.py` (v2), `readout_cell_v3.py` | single-cell simulator (v2 / Mg-explicit v3) |
| `fit_cell_readout.py`, `fit_cell_v3.py`, `fit_cell_min.py` | per-cell fitters (v2 / v3 / 7-param minimal) |
| `compile_{readout,v3,min}_results.py` | aggregate fits → `optimized_cells_*.{parquet,csv}` |
| `compile_results.py` | **shared baseline lookup** (imported by v3/min compilers — keep) |
| `extract_percell_targets.py`, `vivo_percell*.npz` | build per-cell PD/anti-PD targets |
| `results_v3_*/`, `results_min/` | per-cell fit outputs |
| `explore_readout_cells.ipynb` | per-cell exploration |
| `DoVM_model_slurm.py` | **shared** binning constants + cell list (imported by extract) |
| `df.csv` | DoVM phenomenological R² per cell (compared against in the explore notebook) |

**v1 per-cell (superseded — retained only as a probe benchmark):** `fit_cell_mech.py`,
`fit_cells_mech.sbatch`, `compile_mech_results.py`, `optimized_cells_mech.{parquet,csv}`,
`results_mech/`, `explore_mechanistic_cells.ipynb`. The homogeneous full-32 per-cell fit
(capped at R²≈0.5); replaced by the readout-cell approach. Kept because
`probe_fc_relay.py` reads `optimized_cells_mech.parquet` to show the v1 fragility.

## 3. Probes (validation experiments)

Standalone scripts, each a script + figure/CSV, that stress one claim. See
[METHODS.md](METHODS.md) for what ablation / curvature / split-half each answer.

| probe | question |
|---|---|
| `probe_fc_relay.*` | is the FC just relayed? (ring compresses FC; robust to large A_fast) |
| `probe_ablation_{cell,ring}.*` | which parameters does the model *need* (delete + refit) |
| `probe_param_effect.*` | what each parameter *does* to the trace (bound sweeps) |
| `probe_identifiability.*` | which parameters the data constrains (sloppiness) |
| `probe_split_half.*` | out-of-sample R² + per-cell noise ceiling (odd/even trials) |
| `probe_gate_identifiability.*` | is the NMDA gate functional / where does it live |
| `probe_pharmacology.*` | in-silico Mg-relief prediction (anti-PD SC when gate forced open) |
| `probe_heterogeneous_ring.*` | does a ring of the fitted cells stay stable |

## 4. `_archive/` (superseded, non-code-path)

Original phenomenological CANN training (`hdmodel.py`, `attractor_tuning.py`,
`hd_model_training.py`, `worker_safe_hdmodel.py`, `model_generation_all_cells.py`), the
DoVM curve-fitting model (`DoVM_model.py`) and old exploration notebooks + editor cruft.
Nothing in §1–§3 imports these (verified). Moved with `git mv`; history preserved.
(`DoVM_model_slurm.py` and `df.csv` stayed in §2 — still referenced.)

---

## Figures — the average-model story

| figure | one-line caption |
|---|---|
| `_diagnostic.png` | model vs in-vivo PSTH: the three phases (FC / dip / SC) and full recovery to baseline by ~700 ms |
| `probe_param_effect_ring.png` | sweep each ring parameter → what that single knob does to the PD trace |
| `probe_fc_relay.png` | FC-amplitude sweep: the ring auto-compresses the FC (recurrent+global inhibition = gain control), robust to large `A_fast` |
| `probe_pharmacology.png` | in-silico Mg-relief: forcing the gate open makes anti-PD express the SC — the discriminative prediction |
| `probe_identifiability.png` | parameter sloppiness spectrum — which parameters the data actually constrains |

## Reproduce the main result

```bash
python _run_full_fit.py          # fit the 32-param ring (R2-search + validity gate)
python diagnose_mechanistic.py   # -> _diagnostic.png + ablation printout
```

Per-cell extension: see [MODEL_MINIMAL.md](MODEL_MINIMAL.md) (`network_background.py` →
`fit_cell_min.py --all` → `compile_min_results.py`).
