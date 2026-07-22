# DATA — what underlies the model, and what it can/can't support

> **Corrected 2026-07-21 (twice).** Earlier versions (a) treated speaker azimuth as if it
> were the animal's head direction, and (b) attributed the 77-cell EB dataset to the
> population model. Both wrong. The errors are recorded below so they are not repeated.

## There are TWO datasets. Do not mix them.

| | **soso** (SoundSource) | **AD combined** |
|---|---|---|
| file | `soso_comb.parquet` (local copy in `hdmodel/`) | `AD_combined_table.mat` (via `workbench.data.load_ad_table`) |
| cells | **17 rows; 13 after `n_stims > 40`** | **77 HD cells**, animals EB09/12/13/18/20 |
| stimulus | 4 speakers `a/w/e/r` at world azimuths 93/178/272/356°, **pseudorandomized** | sound from a **single fixed location** |
| preparation | **head-fixed** | **head-fixed** |
| used by | [vivo_target.py](vivo_target.py) → **the population / average mechanistic model** (the centerpiece) | [extract_percell_targets.py](extract_percell_targets.py) → the **per-cell** extension; also the earlier **phenomenological (DoVM)** model |

**The centerpiece model's R² is measured against the mean of 13 soso cells.** The 77 EB
cells belong to a *different* dataset and a different part of the project.

## Both preparations are HEAD-FIXED — there is no head-direction axis anywhere

The head angle does not change within a recording, so **the cell sits at its preferred
direction (PD) for the entire recording**. There is no "distance from PD" variable in
either dataset. In soso the four speakers are **sound-source locations, not head
directions**, and there is **no statistically significant response difference across
them** — which is why they are **pooled** (see below).

`HDAngle` in the soso table is `spike_angle_mean` = the cell's **preferred direction** (a
tuning property of the cell), *not* a per-trial heading. `preprocess.py`
(`condition == 'soso'`) builds rasters around every trigger per speaker letter with no
heading conditioning — none is needed in a head-fixed preparation.

## The fit target (average model)

- **Pooled over all four speaker locations**, then averaged across the 13 cells.
- **6 ms bins**, PSTH **Gaussian-smoothed σ = 3 bins (≈18 ms)**; fit window
  **[−50, 700] ms** (stimulus at t = 0).
- Phases: **FC** ~190 Hz onset transient; **dip** ~15–30 ms; **SC** phasic re-excitation
  decaying back to the ~36.5 Hz baseline by ~700 ms.

**Pooling was fixed on 2026-07-21.** Previously `vivo_target.py` kept only the speaker
whose azimuth was closest to the cell's `HDAngle` (`new_rate_sort[:, :, 0]`), discarding
3 of 4 speakers on a criterion that is meaningless in a head-fixed preparation. Effect of
pooling: shape essentially unchanged (corr **0.982**, peak 190.1→192.7 Hz, baseline
37.0→36.5 Hz) but trial noise down ~37% (roughness 15.0→9.4 Hz). The previously reported
fit scores **R² 0.9214 → 0.9534** against the pooled target *with no refit* — i.e. the
discarded quarter-data target contained noise the model was correctly not fitting.

## Direction-selectivity — NOT measurable in either dataset

The model reproduces the observation that **at the antipode only the FC appears — no dip,
no SC**. That observation is **real, but its evidence comes from a separate dataset that
is not in this repo.** Cite it as an **external/prior constraint the model was built to
respect** — never as a result of this analysis.

Both preparations here are head-fixed at one angle, so every trace is a PD trace. Any
"anti-PD" quantity computed from the *farthest speaker* is **invalid as a data claim** — it
compares two samples of the same condition.

## The errors that were here (recorded so they aren't repeated)

1. **Speaker azimuth ≠ head direction.** An earlier version read "speaker farthest from
   `HDAngle`" as "cell at the antipode of the HD bump", and framed the issue as one of
   *resolution* ("only 4 speakers, so the antipode is coarse"). Wrong in kind, not degree:
   sound-source location and bump position are different axes.
2. **Dataset conflation.** The 77 EB cells were wrongly presented as the population
   model's data. They are a different dataset with a single fixed sound location.

**Void as data claims:** per-cell anti-PD targets, `SC_anti_data`, the "12/77 leak >5 Hz"
comparison *against data*, the "held-out anti-PD prediction 0.0/0.0", and
`probe_pharmacology`'s anti-PD contrast *as a data result*. Model-internal numbers (R²,
ablation costs, split-half reliability, identifiability) are unaffected — they never used
the speaker axis.

### Verified 2026-07-21: the per-cell "anti-PD data" was literally zeros

`vivo_percell.npz` was inspected directly: **`anti_rate` is all-zero for all 77 cells**
(`rows with any signal: 0`). AD_combined has one sound location, so `_pd_anti()` in
[extract_percell_targets.py](extract_percell_targets.py) takes its *array path*, which
returns `np.zeros(nbins)` as the anti-PD trace — a schema placeholder, never a
measurement. So the recorded *"held-out anti-PD prediction: 0.0 Hz predicted, 0.0 Hz
measured"* was the model being compared against a **constant zeros array**, and every
`sc_anti_data` in the fit outputs is identically 0.0.

Scope: `anti_rate` feeds **only** the logged `sc_anti_data` metric — **no loss or fit
depends on it**, so all fitted parameters and R² values are unaffected. The "12/77 cells
leak >5 Hz" figure remains valid as a *model-internal* statement (model anti-PD SC vs a
5 Hz threshold); it is void only as a comparison against data.

## Reliability / noise ceiling

- Per-cell **split-half** noise ceiling **R² = 0.615** (in-sample 0.573, out-of-sample
  0.528) — this is from the **77-cell AD dataset**, per-cell, for the **readout** model.
- ⚠️ It is therefore **doubly incomparable** to the ring's R²: different dataset *and*
  different aggregation (single cells vs a 13-cell mean). Never present them on one scale.

**The soso population-mean ceiling (computed 2026-07-21):**

| estimate | value | meaning |
|---|---|---|
| bootstrap variance, 4000 resamples of the 13 cells | **0.889** | R² a *perfect* model would score against the noisy observed mean — **use this one** |
| split-half across cells + Spearman-Brown | 0.972 | correlation-based; ignores amplitude mismatch, so optimistic for R² |

**The fit's R² = 0.9534 is ABOVE the 0.889 ceiling**, i.e. it explains more of the observed
mean than is reproducible — evidence it is fitting noise in a 13-cell average. With 13
cells the mean is loose: FC peak 192.7 Hz **95% CI [166, 218]**, SC peak 101.1 Hz
**[80, 121]**, mean pointwise SD 6.9 Hz. See [OPEN_ISSUES.md](OPEN_ISSUES.md) A1–A3.

## What this data CANNOT tell you

1. **Nothing about head direction.** Head-fixed, one angle, cell always at PD. No tuning
   curve, no antipode, no distance-from-PD effect.
2. **Nothing about sound-source tuning either** — soso responses do not differ
   significantly across the 4 locations (which is why they are pooled).
3. **No intracellular / membrane-potential data.** This is a firing-rate model; `u` is an
   abstract activation, not a recorded voltage. **I_T, I_h and the Mg²⁺ gate are inferred,
   not measured.**
4. **Upstream sources are not observed.** The FC volley and the global slow drive are
   *posited inputs*.
5. **No pharmacology / optogenetics.** Mg-relief, APV etc. are **predictions**, not results.
6. **Sufficiency, not necessity.** A good fit shows the mechanism *can* generate the
   response, not that the brain uses it over an equally-good alternative.
7. **13 cells, one averaged trace, ~17–18 effectively independent time points vs 32 free
   parameters.** This is the tightest constraint on what may be claimed — see
   [OPEN_ISSUES.md](OPEN_ISSUES.md) A1/A2.
