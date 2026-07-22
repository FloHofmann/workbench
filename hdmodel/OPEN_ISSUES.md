# OPEN ISSUES — surviving findings from the first review round

Consolidated 2026-07-21. The first adversarial review was run on a briefing that mis-read the
`soso` data (speaker azimuth ≠ head direction; see [DATA.md](DATA.md)), so the briefing and
report were discarded. **These findings do not depend on that error** — they are code-grounded
(reruns of `optimization_engine` on the reported 32-parameter fit, plus
`probe_ablation_ring_tier{1,2}.csv`) and remain live. Fix or answer these before the model is
presented.

---

## A. Statistical power — the most serious

**A1. The fit is nominally under-determined.** Window [−50, 700] ms at 6 ms bins = 125 bins,
but σ=3-bin Gaussian smoothing (FWHM ≈ 7.1 bins ≈ 42 ms) leaves **~17–18 effectively
independent time points** against **32 free parameters**. One averaged trace cannot carry 32
parameters; the argument has to lean on the ablations, the validity gate and the per-cell
extension, not on R².

**A2. Out-of-sample validation — ✅ LOCO RUN 2026-07-21. Result: the model adds nothing
over a plain average, and the test is underpowered.** Leave-one-cell-out over the 13 soso
cells (refit per fold on the mean of the other 12, warm-started, best-valid tracking):

| median over 13 folds | model | baseline (train mean as predictor) | gap |
|---|---|---|---|
| in-sample R² vs fold train mean | 0.9550 | – | – |
| held-out raw R² vs the left-out cell | +0.134 | +0.191 | **+0.002** (wins 7/13) |
| held-out **shape correlation** | 0.853 | 0.869 | **−0.002** (wins 6/13) |

The 32-parameter mechanistic model predicts a held-out cell **exactly as well as simply
averaging the other 12** — gap ≈ 0, winning on chance-level half the folds. It does not
overfit *catastrophically* (it is not worse), but it buys **zero generalisation benefit**.

⚠️ **Underpowered:** a single cell is so noisy that even the baseline only reaches r≈0.87 /
R²≈0.19, so LOCO cannot resolve the noise-fitting that A3's ceiling analysis already
demonstrates. Raw R² against one cell is also gain-dominated (cells vary hugely in absolute
rate), which is why the shape metric is reported alongside. A better-powered design is
train-on-half / test-on-the-other-half-mean, or moving to the 77-cell dataset (A5).

**A5. The 13-cell soso target is the wrong dataset — the 77-cell AD set is strictly better.**
Verified: binning is **identical** (500 bins, same edges), so the loader is near drop-in.
Same preparation and phenomenon (shape corr between the two population means = 0.861).

| | AD (77 cells) | soso (13 cells) |
|---|---|---|
| baseline | 24.1 Hz | 36.5 Hz |
| FC peak | 161.8 Hz | 192.7 Hz |
| SC peak | 89.0 Hz | 101.1 Hz |
| mean pointwise SD | **2.94 Hz** | 6.89 Hz |
| **bootstrap noise ceiling** | **0.9726** | 0.8907 |

Switching raises the ceiling 0.889 → 0.973, which would put the current R² ≈ 0.95
*legitimately below* ceiling instead of above it. soso's only advantage was the 4-speaker
contrast, which turned out not to be a head-direction axis at all — so it is now strictly
dominated. **Refactor cost is mostly the hardcoded baseline constants:** `_run_full_fit.score`
gates baseline peak to **33–47 Hz** and `loss_stage2` pins recovery to ~40 Hz — both tuned to
soso's 36.5 Hz baseline and both would reject everything at AD's 24.1 Hz. Requires a fresh
refit (the soso warm start may not pass a retuned gate).
**Note this does NOT fix A1** — 32 parameters against ~17–18 effective time points is
unchanged by adding cells; that still argues for cutting parameters.
(The 0.528 out-of-sample figure is per-cell split-half of the **readout** model — transferring
it to the ring is a category error.)

**A3. Population-mean noise ceiling — ✅ COMPUTED 2026-07-21, and the model is ABOVE it.**
Resampling the 13 soso cells (no fitting):

| estimate | value | what it measures |
|---|---|---|
| bootstrap variance (4000 resamples) | **0.889** | the R² a *perfect* model would score against the noisy observed mean — **the correct reference for R²** |
| split-half across cells + Spearman-Brown | 0.972 | a *correlation* reliability; insensitive to amplitude mismatch, so optimistic for R² (R²(halfA→halfB) is only 0.51) |

**The fit sits at R² = 0.9534, i.e. ~0.06 ABOVE the 0.889 ceiling** — it is explaining more
of the observed mean than is reproducible, the signature of fitting noise in a 13-cell
average. (Directly parallel to v3's per-cell 0.692 exceeding its 0.615 ceiling.) With 13
cells the mean is loose: FC peak 192.7 Hz **95% CI [166, 218]**, SC peak 101.1 Hz
**[80, 121]**, mean pointwise SD 6.9 Hz.

Note the per-cell 0.615 remains a *different* number entirely — different dataset *and*
different aggregation (see [DATA.md](DATA.md)); never put them on one axis.

**A4. Trials thrown away — ✅ FIXED 2026-07-21.** `vivo_target.py` used to keep only the
PD-closest speaker (`new_rate_sort[:, :, 0]`), discarding 3 of 4 speakers on a criterion
meaningless in a head-fixed preparation. Now pools all four. Shape unchanged (corr 0.982),
trial noise −37%, and the existing fit rose **R² 0.9214 → 0.9534 without refitting**. A local
L-BFGS polish on the new target made it *worse* (0.9156), confirming C4's point that
`loss_joint` is not R²-aligned; the incumbent parameters were kept.

## B. Mechanism claims that overstate what the model does

**B1. The SC is *not* mechanistically a post-inhibitory rebound.** `sc_delay = 0` in the fit,
so the global SC **input switches on at flash onset, coincident with the FC — not triggered by
the dip**. Its apparent post-dip rise comes only from the gate low-pass (τ_scg ≈ 29 ms) and the
EPSC rise (τ_sc_on ≈ 15 ms). MODEL.md §7/§9 already splits this correctly (**amplitude = the
injected gated input; shape/onset = the I_T/I_h channels**) but the "post-inhibitory rebound"
language overstates causation if read as the amplitude mechanism. Say the split plainly.

**B2. I_T shapes, it does not amplify.** Zero-no-refit `g_T = 0` makes the PD SC peak *larger*
(93.7 → 120.5 Hz) while thinning the flank and narrowing the late bump (FWHM600 69 → 45°).
I_T sets onset/width, not height.

**B3. Direction-selectivity is *enforced*, not reproduced.** `loss_stage2` adds a silence
penalty `mean(anti_pd_rate[40–700 ms]²)·10` and `_run_full_fit.score` rejects any fit with
anti-PD SC max > 6 Hz. There is **no MSE term against any recorded anti-PD trace**. The ring's
anti-PD SC max = 0.0 Hz **by construction** (`sc_gate = rec/(rec+REC_HALF)`, rec ≈ 0 at the
antipode). Combined with the fact that this dataset has no anti-PD condition at all
([DATA.md](DATA.md)), the emergent-tuning result is **purely architectural**.

**B4. The emergent demonstration risks being near-tautological.** `sc_gate` is *defined* as
bump membership, which peaks at PD; multiplying a global input by "how much bump is here" and
reporting a bump-shaped output is substantially built in. This needs an argument for why it is
non-trivial, not just a demonstration.

## C. Fit hygiene

**C1. `A_sc = 0` does more than collapse the SC — it breaks the attractor.** In the
zero-no-refit run, anti-PD rises to **261 Hz**, the bump migrates to −180°, and PD@600 ms → 0.
The tidy "SC 88→55 Hz" headline is only the PD reading *before the bump leaves*. Restate it.

**C2. Parameters railed at bounds.** `tau_a` sits at its **upper** bound (600 ms; bounds
[80, 600]) and the bound was **never relaxed** — the optimizer wanted to go higher, so the true
recovery timescale is unresolved. Also railed: `E_Ca`, `sc_delay`, `J1`, `g_T`. `w_sc` and
`tau_sc_off2` are effectively hand-set. Railed parameters undercut biophysical interpretation.

**C3. Three slow tail constants are confounded.** τ_sc_off ≈ 416, τ_a = 600, τ_h_off ≈ 584 ms
all shape one smooth decay — individually unidentifiable.

**C4. Headline ablation numbers are tier-1 (zero, no refit).** MODEL.md quotes these;
`probe_ablation_ring.py` tier-2 refits the other 31 params. Tier-2 re-minimizes `loss_joint`,
which is **not R²-aligned**, so it drifts R² by ~0.05–0.09 even for a null change — any tier-2
cost below ~0.09 is noise. Label which tier each quoted number is.

| ablation | tier-1 ΔR² (no refit) | tier-2 ΔR² (refit) |
|---|---|---|
| W_IE = W_EI | 1.26 | **6.79** |
| W_GLOBAL | 13.0 | 2.63 |
| I_HD | 2.64 | 1.79 |
| J1 | 2.85 | 0.96 |
| A_fast | 0.51 | 0.58 |
| A_sc | 7.55 | 0.56 |
| g_a | 0.09 | 0.29 |
| g_T | 0.42 | 0.17 |
| g_h (I_h) | 0.0008 | **0.07 — inside the noise floor, i.e. null** |

**C5. Corrected numbers.** Rerun on the reported fit: `A_sc=0` → PD SC peak **93.7 → 54.9 Hz**;
`g_a=0` → FWHM@600 ≈ **123° vs 69°** (MODEL.md's "~105°" is an older/looser window figure).

## D. Provenance / bookkeeping

**D1. The "12/77 cells leak > 5 Hz" figure could not be reproduced** from the scripted
`.py`/CSV pipeline — it appears to originate in a notebook. Provenance unverified. (Also now
void as a *data* claim — see [DATA.md](DATA.md).)

**D2. Documentation drift in `optimization_engine.py`:** stale "20-param" and "supralinear
gain" comments (the code is 32-param and linear); differing `u_E` floors between simulators;
an `r_I` 500 Hz cap that is undocumented; "40 Hz" vs "37 Hz" baseline used inconsistently.
Comment/wording drift only — no math errors found.

**D3. The aggressive ring strip was never fit.** Only I_h(×4) + `tau_sc_off2` + `w_sc` +
`reversal_potential` were dropped ("~24 params"). A ring stripped to {global SC amplitude + one
slow decay + HD tuning + FC} has never been optimized, so its R² cost is **unmeasured**. Note
also that the "7 free" minimal model is the **per-cell readout**, not the ring — its 0.577 is
against per-cell targets from a *different dataset*, and is not comparable to the ring's 0.95.
