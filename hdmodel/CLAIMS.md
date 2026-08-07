# CLAIMS — what the model asserts, and how strongly the evidence backs it

> **Corrected 2026-07-21.** Rows touching direction-selectivity were re-sourced. The `soso`
> data is **head-fixed with the cell always at PD**; its 4 speakers are **sound-source
> locations, not head directions**, so this dataset cannot test direction-selectivity at
> all. See [DATA.md](DATA.md).

The point of this table is to keep "the model fits" separate from "the brain does this."
Read **strength** as the honest ceiling of each statement.

**Strength vocabulary**
- **DATA-SUPPORTED** — visible in *these* recordings.
- **EXTERNAL** — real, but established in a **different dataset not included here**; the
  model was built to respect it. Cite the other dataset, never this analysis.
- **MODEL-SUFFICIENT** — the model *demonstrates the mechanism can produce it*; a
  sufficiency result, **not** proof the brain uses it (equally-good alternatives not excluded).
- **PLAUSIBLE** — fit-consistent + biophysically motivated, not dissociated by data.
- **LITERATURE / ASSUMED** — imported from published work, not established here.
- **NOT NEEDED** — the model runs without it.

| # | claim | mechanism in model | evidence | strength |
|---|---|---|---|---|
| 1 | AD HD cells show a 3-phase auditory response (FC → dip → SC) at PD | — (observation) | population PSTH, head-fixed at PD, pooled over sound-source location | **DATA-SUPPORTED** |
| 2 | At the antipode only the FC appears — no dip, no SC | — (observation) | **a separate dataset, not included in this repo.** The soso data is head-fixed at PD and cannot test this | **EXTERNAL** — *never cite the soso analysis for this* |
| 3 | AD HD tuning sits on a ring / continuous attractor | Mexican-hat recurrent connectivity | Peyrache 2015 (AD in vivo); Zhang 1996; Amari 1977 | **LITERATURE / ASSUMED** |
| 4 | A global untuned SC input can be made direction-selective by the bump-gate | `A_sc` × `sc_gate` (low-passed bump membership) | model only: ablating `A_sc` collapses SC 88→55 Hz; the gate yields ≈0 at the ring antipode while the input stays global. Target behaviour imported from claim 2 | **MODEL-SUFFICIENT** — *but the gate is NOT REQUIRED: see claim 13* |
| 13 | **A gate is not needed at all — global inhibition alone gates the SC** | uniform `W_GLOBAL·mean(r_E)` tracks and cancels the uniform SC at the antipode; `I_HD` keeps PD above threshold | a basic ring with **no** I_T/I_h/Mg/NMDA/gate reaches **R² 0.9477** (97.4% of the 0.9726 ceiling) with anti-PD at **exactly 0.0 Hz** ([MODEL_INHIBITION_ONLY.md](MODEL_INHIBITION_ONLY.md)) | **MODEL-SUFFICIENT** — *supersedes the necessity of claim 4's gate; specifically GLOBAL inhibition — the tuned interneuron ring alone fails (R² −0.24)* |
| 14 | **The fitted ring is not a self-sustaining attractor** | tuning is inherited from the tonic `I_HD` drive, not maintained by recurrence | removing `I_HD` collapses the bump in the minimal model (26.4→2.0 Hz, `J1`=0.26) **and in the original 32-param model** (26.7→5.1 Hz) | **DATA-INDEPENDENT / MODEL FACT** — consistent with AD inheriting tuning from LMN, but does not support calling the ring an attractor that *maintains* the bump |
| 5 | Recovery to baseline needs a **global** (mean-field) adaptation, not per-cell | `g_a·a(t)` subtracted uniformly | ablation `g_a=0` → bump stuck broad (FWHM ~105 vs ~69°); per-cell adaptation makes the bump travel | **MODEL-SUFFICIENT** |
| 6 | SC amplitude vs recovery are in tension in one attractor; decoupling them resolves it | inherited SC amplitude + slow global adaptation | earlier R²≈0.72 latched-broad state; the 700 ms window exposes non-recovery | **MODEL-SUFFICIENT** (insight about this model class) |
| 7 | The SC drive is **NMDA** (voltage-dependent Mg²⁺ block) | NMDA-EPSC + Mg-block as the gate | gate source unidentifiable, GluN2A/2B split not required, **no pharmacology performed** — and out of sample the Mg gate **loses to the recurrent gate**, CV R² 0.429 vs 0.758 (`compare_gates_recovfix.csv`, corrected recovery bound) | **WEAK / NOT SUPPORTED as a mechanism claim** — *the data support "a gated global drive"; the Mg identity is interpretation, and the one test run on it went against it* |
| 8 | I_T (T-type Ca²⁺) shapes the SC onset/rebound | de-inactivation → recurrent-gated rebound | ring ablation ~0.10–0.17 R² (shape, not amplitude) | **PLAUSIBLE** |
| 9 | I_h (HCN) contributes the SC tail | slow HCN activation | near-idle; **cut in v3 at ~0 cost** | **NOT NEEDED** |
| 10 | The FC is a brief, untuned, relayed auditory volley | uniform `A_fast` pulse | reproduces the FC — but the strict "FC *must* be a relay" claim was **REFUTED** (`probe_fc_relay`: the ring tolerates large `A_fast`; the R²=−6 collapse was a v1 artifact) | **PLAUSIBLE**, self-corrected |
| 11 | The dip is emergent feedback inhibition (interneuron lag), not an imposed input | co-tuned interneuron ring lagging by `τ_I` | overshoot → dip; no hand-drawn dip | **MODEL-SUFFICIENT** |
| 12 | Global inhibition `W_GLOBAL` stabilizes the linear ring | inhibition ∝ total activity | ablation → network blows up | **MODEL-SUFFICIENT** |

## Void as data claims (retracted 2026-07-21)
Anything derived from the *farthest speaker* as a stand-in for the antipode: per-cell
anti-PD targets, `SC_anti_data`, the "12/77 leak >5 Hz" comparison **against data**, the
"held-out anti-PD prediction 0.0/0.0", and `probe_pharmacology`'s anti-PD contrast **as a
data result**. These compared two samples of the *same* condition. The corresponding
**model-internal** numbers remain valid as statements about the model.

## Superseded 2026-07-24
Claim 4's emergent bump-gate was the centrepiece. **Claim 13 shows no gate is required at
all**: uniform global inhibition reproduces the response with the antipode exactly silent, at
a *higher* R² than the gated model. The gate is one sufficient mechanism among at least two,
and the simpler one is currently ahead.

**Two things the corrected gate comparison changed** (`compare_gates_recovfix.csv`; the old
run capped recovery at 1.35× baseline while the data sits at 1.55× — see
[OPEN_ISSUES.md](OPEN_ISSUES.md) N1):

- **Claim 7 is downgraded.** The Mg gate loses out of sample to the recurrent gate, 0.429 vs
  0.758. "The Mg gating is responsible for the SC" is the one thing the model comparison was
  built to test, and it did not survive it.
- **Stop saying the gate beats the tuned reference.** `tuned` (a PD-aligned SC input — a
  reference bound, *not* a mechanism, since no plausible connectivity delivers it) gains
  +0.271 under the corrected bound and now ties `rec`, 0.732 vs 0.758, at half the variance.
  The previous margin was largely an artefact of a constraint that would have rejected the
  real data.

## The one sentence to lead with
> *A head-direction ring attractor, driven only by spatially global auditory and slow-drive
> inputs, reproduces the measured FC → dip → SC time course of AD HD cells at their
> preferred direction.* (Strength: **DATA-SUPPORTED** for the time course.)

## The honest counter-sentence (say it before the reviewer does)
> *This dataset is head-fixed with the cell always at PD, so it cannot test
> direction-selectivity — that constraint comes from a separate dataset. The emergent-gating
> mechanism, the NMDA identity of the slow drive, and the roles of I_T/I_h are
> model-internal or literature-based, not dissociated by these recordings; and the fit is to
> a single smoothed population-mean trace.*
