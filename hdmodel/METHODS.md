# How we decided which parameters to keep — in plain language

Three different questions get asked about a parameter, and they have three different
answers. Most confusion comes from mixing them up. This document keeps them apart.

| question | method | what a good score means |
|---|---|---|
| Does the model **need** this part? | **Ablation** | removing it makes the fit clearly worse |
| Can we **read** this parameter off one cell's data? | **Curvature** ("sloppiness") | the fit gets sharply worse if you nudge it |
| Does the reading **replicate**? | **Split-half reliability** | odd and even trials give the same value |

A parameter is only cut if it fails the **first** question. The other two explain
*why* a parameter is hard to interpret, but they never justify a cut on their own.

---

## 1. Ablation — "does the model need this part?"

The direct test. Take the mechanism out, refit everything else, and measure how much
worse the model gets. This is the number that belongs in a paper, because it answers
the question a reader actually has.

Two kinds of "taking it out", and we label every parameter as one:

- **Type M — mechanism off.** For a strength (a conductance `g`, an input amplitude
  `A`), setting it to zero removes that piece of biology entirely. "Turn off the
  T-type calcium current." If the fit barely changes, the current wasn't doing much.
- **Type F — freedom removed.** A shape parameter (a time constant, a half-activation
  voltage) can't be set to zero — zero isn't "off", it's nonsense. Instead we fix it
  at one shared value for all cells and refit. This asks a subtler question: does this
  need to be a *free knob*, or can it be a *constant*? A constant is simpler and just
  as communicable.

**The one limitation to state honestly:** after ablating, we refit only *locally*
(nudge the remaining parameters from where they were, don't search the whole space
again). A full global refit might recover a little more. So an ablation cost is an
**upper bound** — "it costs at most this much." For the population ring we can afford
the fuller refit; for the 77 cells we refit a stratified subset. This is a compute
compromise, not a hidden assumption.

**Why ablation is the backbone and the others are not:** a parameter can be
impossible to pin down yet essential. Adaptation (`g_a`) is the worked example — its
split-half reliability is only 0.16 (you can't read it reliably off one cell), but
ablating it destroys the bump's recovery to baseline. Reliability said "unclear";
ablation said "load-bearing". Ablation is right, because the question was necessity.

---

## 2. Curvature — "can we read this parameter off the data?" (the "sloppy" number)

Sit at the best fit. Nudge one parameter a little up, a little down. Watch the
mismatch between model and data.

- Mismatch shoots up → the data **pins** this parameter. You can trust its value.
- Mismatch stays flat → the data **doesn't care**. Any value in a wide range fits
  equally well, so the fitted number is basically arbitrary.

Do this for every parameter and rank them. You can also nudge *combinations* — often
you can raise one parameter and lower another together with almost no effect; that
combination is unconstrained even if each parameter alone looks fine.

**"16.5 decades"** is the headline number from this analysis. It means the flattest
combination of parameters was about 10^16.5 times less constrained than the steepest.
That is enormous — it says most of the model's freedom lives in directions the data
never sees. This is normal for biophysical models (they are famously "sloppy"), but
it is the quantitative reason not to over-interpret individual fitted values.

**Limitations:** it is only valid right around the best fit (a local picture), and its
cleanest statements are about *combinations*, not single parameters. It tells you what
you can read, never what the model needs.

---

## 3. Split-half reliability — "does the reading replicate?"

Each cell was recorded over many repetitions of the sound. Split them into odd and
even repetitions — two noisy but **independent** measurements of the same cell. Fit
the model separately to each half. If a parameter captures something real about the
cell, the value from the odd trials should match the value from the even trials.
Correlate across all 77 cells (a number `r` from −1 to 1).

- `V_half_T`: r = 0.72. The two halves agree — a real, stable cell property.
- `w_sc`: r = 0.02. The halves are unrelated — fitting this to one recording tells you
  nothing about the next. It is noise-chasing.

This is the most intuitive of the three and the easiest to defend to a skeptic,
because it uses nothing but the raw data twice. It agreed almost perfectly with the
curvature ranking, which is reassuring — two independent methods, same verdict.

**Limitation:** like curvature, it measures *readability*, not *necessity*. A reliable
parameter can still be removable (if the model fits fine without it), and an unreliable
one can still be essential (`g_a` again).

---

## What none of the three can tell you

- Whether the model is *correct* — only whether it fits and whether its knobs are
  legible. A wrong model can fit well.
- Whether a per-cell parameter reflects biology or the fitting procedure — that is what
  split-half guards against, but only for the parameters that pass it.
- Anything about parameters we never included. Absence of a mechanism is not tested by
  removing the ones we have.

---

## The model's history, one paragraph each

- **v1 — homogeneous per cell (32 params).** Fit all 32 parameters separately to each
  cell. The hidden cost: this builds a ring of 120 identical copies of the target cell,
  so it erases the very heterogeneity we wanted to study, and the "network" parameters
  become meaningless (they are whatever makes a uniform ring imitate one cell). It also
  capped at R² ≈ 0.5 — but that turned out to be an artifact of the homogeneous fits,
  not the biology (see `probe_fc_relay.py`).

- **v2 — heterogeneous readout cell (24 params).** Keep the network fixed at the
  population fit (the average head-direction activity the model was validated on), and
  make only the one cell we care about heterogeneous. Justified because each cell
  contributes < 1% of the input it receives, so "swap in this cell" and "drive one cell
  with the recorded average input" are the same model — confirmed exact to 14 decimal
  places, and 180× faster. Fit quality rose to R² 0.70 and the FC artifact vanished.

- **v3 — Mg-explicit, leaner (18 params).** Six parameters that failed all the
  readability tests were removed, and the SC gate was rewritten as the actual NMDA
  receptor Mg²⁺ block (Jahr–Stevens), which makes external magnesium an explicit knob
  and turns the key experimental prediction into a real dose–response. Same fit quality
  with a third fewer parameters.

- **minimal — this study.** Keep only the parameters whose *ablation* costs something
  against four criteria (fit, direction-selectivity, recovery, and genuine per-cell
  variation). Everything else becomes a fixed constant or is deleted. The result is the
  model in `MODEL_MINIMAL.md`, where every remaining parameter has a one-sentence reason
  to exist.
