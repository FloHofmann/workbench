# Can inhibition alone gate the SC? — YES (supervisor's hypothesis confirmed)

**Verdict: confirmed, with two important qualifications.** A basic ring — no I_T, no I_h, no
Mg block, no NMDA kinetics, **no gate of any kind** — reproduces the measured PD response at
**R² = 0.9477** (noise ceiling 0.9726, so **97.4% of the ceiling**) while the antipode stays
at **exactly 0.0 Hz** throughout the slow component. The SC input is spatially **uniform**.

This tested the supervisor's claim that the SC is ubiquitously distributed and anti-PD is
silent only because inhibition holds it sub-threshold. It survives. It also falsifies the
prior expectation that "the SC cannot reach its amplitude without anti-PD bumping out".

## The result

`probe_sc_feasibility.py` maximises R² against the AD 77-cell PD trace while clamping the
anti-PD SC below a threshold, sweeping that threshold. **R² on the whole trace, not peak
amplitude** — maximising a scalar peak let an earlier version "win" with a response that was
still at baseline at 60 ms, where the real SC peaks.

| anti-PD SC allowed | `ring` | `global` | `both` |
|---|---|---|---|
| unconstrained | −0.070 | 0.901 | 0.977 \* |
| ≤ 40 Hz | 0.275 | 0.952 | 0.854 |
| ≤ 20 Hz | −0.153 | 0.948 | 0.854 |
| ≤ 5 Hz | −0.240 | 0.948 | 0.854 |
| **≤ 0.5 Hz** | **−0.240** | **0.948** | **0.854** |

\* above the 0.9726 ceiling ⇒ fitting noise.

**Clamping anti-PD to silence costs `global` almost nothing** (0.952 → 0.948). No gate needed.

## Qualification 1 — it is GLOBAL inhibition, not the Mexican-hat ring

`ring` (tuned interneuron ring, no global pool) **fails outright** — it cannot fit the trace
even with anti-PD unconstrained (R² = −0.07). Adding the interneuron ring to the global pool
(`both`) *hurts* (0.948 → 0.854). The work is done specifically by **uniform inhibition
proportional to mean activity**.

The mechanism is visible in the equations. At the antipode the tuned terms vanish, leaving

```
u_anti = I_baseline + A_sc·g(t) − W_GLOBAL·mean(r_E)
```

As the uniform SC drives the bump up, `mean(r_E)` rises with it, so `W_GLOBAL·mean(r_E)`
tracks and cancels the SC at the antipode — while at PD the tuned drive `I_HD` keeps the cell
above threshold. That is a **normalisation**, and it is literally "the SC is gated by
inhibition". Biologically this reads as a broadly-projecting inhibitory population (TRN-like),
not local Mexican-hat interneurons.

## Qualification 2 — the winning model is NOT an attractor

The fitted solution has **`J1 = 0.26`** — essentially no recurrent excitation. Removing the
tonic tuned drive collapses the bump (26.4 → 2.0 Hz): the tuning is **inherited from `I_HD`**,
not sustained by recurrence. So this is a tuned feedforward filter with global inhibition, not
a continuous attractor.

**This is not specific to the minimal model.** The same test on the original 32-parameter model
collapses too (bump 26.7 → 5.1 Hz without `I_HD`). It is consistent with the stated biology —
AD *inherits* HD tuning from LMN (MODEL.md §6) — but it does not support describing the ring
as an attractor that *maintains* the bump. See [OPEN_ISSUES.md](OPEN_ISSUES.md).

## Fit quality, honestly

| t (ms) | data | model PD | model anti |
|---|---|---|---|
| −20 | 24.0 | 26.4 | 0.0 |
| 10 (FC) | 161.8 | 139.8 | 69.6 |
| 30 (dip) | 49.8 | 66.9 | 0.0 |
| 50 | 83.8 | 72.7 | 0.0 |
| 70 (SC) | 82.2 | 73.1 | 0.0 |
| 200 | 60.1 | 61.4 | 0.0 |
| 600 | 36.6 | 36.5 | 0.0 |

The model **undershoots the SC peak** (73 vs 89 Hz = 82%) and its **dip is too shallow**
(67 vs 50 Hz). So the amplitude concern was not baseless — it is just not decisive, because
the overall time course is captured. The FC does reach the antipode (73.9 Hz), as the external
dataset requires.

## Caveats
- **In-sample.** R² is against the full 77-cell mean with no cross-validation. It is *below*
  the noise ceiling (unlike `both` at 0.977), which argues against noise-fitting, but a
  split-half CV has not been run.
- 17 free parameters against ~18 effectively independent time points — the same
  over-parameterisation that applies to every model here ([OPEN_ISSUES.md](OPEN_ISSUES.md) A1).
- Anti-PD behaviour is an **external constraint**, not measurable in this dataset
  ([DATA.md](DATA.md)); it enters as a modelling requirement, not as fitted data.

## Consequence for the gated models
The gate-free minimal model (0.948) **beats the full 32-parameter gated model** on the same
data (`rec` gate: 0.819).

I first suspected an artefact: the gate comparison capped recovery at 1.35× baseline while the
AD data sits at **1.55×**, so it would have rejected the real trace. **That was a real bug but
not the explanation.** Re-run with a data-derived bound (2.10× on AD), full-data `rec` scores
**0.8187** vs 0.819 before — unchanged; `mg` and `noSC` return bit-identical results, i.e. the
cap was never binding for them. The gap is real.

The re-run did change one thing: `tuned` — the *reference upper bound, not a mechanism* — gains
+0.271 CV R² and now ties `rec` (0.732 vs 0.758) at half the variance. So "the recurrent gate
beats the tuned bound" was itself partly a constraint artefact. Details and the full table in
[OPEN_ISSUES.md](OPEN_ISSUES.md) N1.

**What most likely explains the gap: the inhibition architecture, not the gate.** The full
model's `none` mode (global SC, no gate) scores ≈0 and leaks 39.3 Hz at the antipode, yet the
minimal model with a global SC and no gate reaches 0.948 with the antipode silent. The
difference is that the minimal winner runs **pure global inhibition** (`W_IE = W_EI = 0`) — and
in the minimal model, adding the interneuron ring back (`both`) drops it to 0.854. The full
model's `none` keeps that ring. Untested: fitting the full architecture with the interneuron
ring switched off. See [OPEN_ISSUES.md](OPEN_ISSUES.md) N3.

## Reproduce
```bash
python minimal_ring.py            # sanity checks incl. the cancellation argument
python probe_sc_feasibility.py --modes ring,global,both
python compare_gates.py --modes rec,mg,none,noSC,tuned --tag _recovfix   # ~90 min
```
Files: `minimal_ring.py` (17 params, njit), `probe_sc_feasibility.py`,
`probe_sc_feasibility.{csv,png}`, `_feas_<mode>_<thr>.json`, `_fit_minimal_<mode>.json`,
`compare_gates_recovfix.csv`.

**Start here to understand the model:**
[`minimal_ring_walkthrough.ipynb`](minimal_ring_walkthrough.ipynb) — the equations, all 17
parameters explained, and ten figures: the fit, PD vs antipode, the whole ring over time, the
population at single time points, the cancellation mechanism, the three inhibition
architectures, the feasibility frontier, the attractor test, and a robustness sweep.

**On robustness** (from the notebook, Fig 10): scaling `W_GLOBAL` around its fitted value
shows the antipode is at **exactly 0 Hz at the fitted gain and at every stronger one** — there
is no upper cliff — and leaks only ~2.8 Hz at 15% weaker. Fit quality is the sensitive
quantity, and only mildly (±10% costs ~0.03 R²). So the cancellation does **not** need fine
tuning; "inhibition strong enough" is a lower bound, not a set point. This answers the natural
objection to any cancellation mechanism.
