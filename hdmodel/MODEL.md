# A mechanistic model of auditory-evoked responses in thalamic head-direction cells

This document describes the model implemented in [`optimization_engine.py`](optimization_engine.py):
a head-direction (HD) ring attractor with intrinsic thalamic channel dynamics that
reproduces the auditory-evoked firing of HD cells in the anterodorsal (AD) thalamus —
**mechanistically**, with no hand-drawn response functions.

---

## 1. The biological problem

Brief auditory stimuli drive a stereotyped, time-varying response in AD-thalamic HD
cells (population PSTH, preferred-direction cells):

1. **Fast component (FC)** — a sharp ~170–190 Hz transient at stimulus onset.
2. **Brief inhibition (dip)** — firing drops for ~15–30 ms.
3. **Slow component (SC)** — a phasic re-excitation peaking ~50–70 ms post-stimulus
   (~91 Hz) that **decays smoothly back to the ~37 Hz baseline by ~700 ms** (a
   *transient*, not a new sustained plateau).

In **anti-preferred-direction (anti-PD)** cells only the FC appears — the dip and SC
are **direction-selective**. Both evoked drives (FC and SC) arrive as **global, spatially
UNTUNED inputs** — the FC a fast auditory volley, the SC a slower global salience/arousal
drive (their upstream source is left open). Crucially the SC's **direction-selectivity is
NOT in the input**: the ring gates the global SC by bump-membership, so the tuned SC
**output** emerges from the network (Section 7.3). The **dip**, **selectivity**, and
**recovery** are likewise genuine AD-network properties, not imposed on any input.

> **⚠️ Provenance of the anti-PD observation (corrected 2026-07-21).** The "anti-PD shows
> only the FC" constraint above is **real but comes from a separate dataset that is not in
> this repo.** The `soso` data fitted here is **head-fixed with the cell held at its PD for
> the whole recording**, and its four speakers are **sound-source locations, not head
> directions** — so this dataset **cannot test direction-selectivity at all**, and every
> trace in it is a PD trace. Treat the anti-PD behaviour as an **external constraint the
> model was built to respect**, and the emergent bump-gating below (§7.3, §8b, §9) as a
> **model-sufficiency result**, not something validated by the data fitted here. See
> [DATA.md](DATA.md) and [CLAIMS.md](CLAIMS.md).

The substrate is the HD system: HD cells fire as a function of head azimuth and are
thought to be maintained by a **continuous (ring) attractor** — a localized "bump" of
activity on a ring of cells ordered by preferred direction
(Skaggs et al. 1995; Zhang 1996; Knierim & Zhang 2012), with attractor dynamics now
observed directly in AD thalamus in vivo (Peyrache et al. 2015). AD relay cells
inherit their HD tuning from upstream (lateral mammillary nucleus → anterodorsal
thalamus; Taube 2007) and possess the classic intrinsic conductances of thalamic
relay neurons — low-threshold T-type Ca²⁺ (I_T) and hyperpolarization-activated I_h
(Jahnsen & Llinás 1984; McCormick & Pape 1990) — which produce **post-inhibitory
rebound bursts** (Llinás 1988). That rebound is the model's mechanistic candidate for
the SC.

This is a **firing-rate** model (not conductance-based Hodgkin–Huxley): `u` is an
abstract membrane-potential-like variable and `r` a firing rate (Hz). Intrinsic
currents are represented by gating variables that modulate the rate dynamics — a
common level of abstraction that keeps the network tractable while retaining the
qualitative biophysics (cf. Destexhe et al. 1996 for the reduced T-current).

---

## 2. Architecture at a glance

| Component | Mechanism | Produces | Biology |
|---|---|---|---|
| Ring of `N=120` excitatory cells + interneurons | tuned Mexican-hat connectivity | the HD bump | CANN / ring attractor |
| Threshold-linear gain `r = f(u)` | `clip(GAIN_SLOPE·[u]₊, 0, R_MAX)` | firing rate | linear f-I (supralinear tested & dropped) |
| Tonic tuned drive `I_HD` | upstream HD input | baseline tuning + anchoring | LMN/DTN → AD |
| Global FC volley `A_fast` | brief untuned sensory input | **FC** | auditory volley (ubiquitous ⇒ untuned) |
| Feedback inhibition | interneuron ring | **dip** (emergent) | feedforward/feedback inhibition |
| Global SC input `A_sc` × bump-gate | untuned NMDA-EPSC (bi-exp) × `rec_gate` | **tuned SC output** (~91 Hz hump) | global glutamatergic (NMDA) drive gated by the HD bump |
| SC gate memory `tau_scg` | low-passed `rec_E` | bridges the dip ⇒ smooth SC onset | synaptic/membrane memory of bump-membership |
| `I_T` (T-type Ca²⁺) | de-inactivation + activation gates | SC rebound **shaping** (~0.10 R²) | thalamic rebound burst |
| `I_h` (HCN) | slow hyperpolarization-activated current | SC tail **shaping** | thalamic sag/ADP |
| recurrent gating of `I_T` | only bump cells rebound | spatial confinement | rebound needs depolarizing drive |
| Global adaptation `g_a` | mean-field current ∝ recent activity | **width recovery to baseline** | SK / M-current (spike-freq. adaptation) |
| Global inhibition `W_GLOBAL` | ∝ total activity | **stabilizes the linear ring** | broad/TRN inhibition |

---

## 3. Ring connectivity

`N = 120` excitatory cells are placed on a ring at preferred directions
$\theta_i = -\pi + 2\pi i/N$. Connectivity depends only on the angular difference
$\Delta\theta_{ij}=\theta_i-\theta_j$ through two von-Mises-like kernels:

$$
K^E_{ij} = \tfrac{1}{N}\,e^{\kappa_E(\cos\Delta\theta_{ij}-1)},
\qquad
K^I_{ij} = \tfrac{1}{N}\,e^{\kappa_I(\cos\Delta\theta_{ij}-1)} .
$$

Excitation is **narrow** ($\kappa_E$ large) and inhibition **broad** ($\kappa_I<\kappa_E$).
Narrow excitation minus broad inhibition is a **Mexican-hat** profile, the classic
recipe for a single localized bump with lateral competition
(Amari 1977; Ben-Yishai et al. 1995). Building the surround from a *broad inhibitory
kernel* (rather than subtracting a DC term) keeps the inhibition spatially structured,
which is essential: a near-uniform surround ($\kappa_I\!\to\!0$) gives no restoring
force and the bump width becomes unstable.

---

## 4. Gain (threshold-linear) — and why supralinearity was dropped

The firing rate is a **threshold-linear** function of the activation:

$$
r = f(u) = \min\!\big(R_{\max},\; \text{GAIN\_SLOPE}\cdot[u]_+\big),\qquad [u]_+=\max(0,u),
$$

with `GAIN_SLOPE`$=6$ (baseline operating point $u\approx6\!-\!7\to\sim40$ Hz) and
$R_{\max}=1000$ Hz kept only as a safety clip (never reached in normal fits).

**Why not supralinear.** A supralinear (Naka–Rushton / SSN) gain was tried first, to
make the ring **monostable** (winner-take-all): with a threshold-linear gain the ring
is *marginal* and, if the SC were **generated recurrently**, a perturbation could kick
the bump into a stuck broad state (SSN theory: Ahmadian, Rubin & Miller 2013; Rubin,
Van Hooser & Miller 2015). But in the final model the SC is an **inherited input**, not
recurrently generated, and the bump **width** is recovered by the global adaptation
current (Section 8b). Under those conditions the winner-take-all is not only
unnecessary but mildly *harmful* — it distorts the network's linear tracking of the
injected SC. Refitting with a plain linear gain **raised R² from 0.877 to 0.94**, so
supralinearity was removed. Stability against runaway/drift (which the supralinear gain
had also provided) is instead supplied by the global inhibition $W_{\text{glob}}$
(Section 5); ablating $W_{\text{glob}}$ makes the linear ring blow up.

---

## 5. Network dynamics

Excitatory activation $u^E_i$ and interneuron activation $u^I_i$ evolve as leaky
integrators (Euler, $\Delta t=0.2$ ms):

$$
\tau_E \frac{du^E_i}{dt}
= -u^E_i
+ \underbrace{J_1 (K^E r_E)_i}_{\text{recurrent exc.}}
- \underbrace{W_{IE}(K^I r_I)_i}_{\text{tuned inh.}}
- \underbrace{W_{\text{glob}}\,\overline{r_E}}_{\text{global inh.}}
+ I^{\text{ext}}_i + I^{SC}_i + I^T_i + I^h_i
- \underbrace{g_a\,a}_{\text{adaptation}} ,
$$

$$
\tau_I \frac{du^I_i}{dt} = -u^I_i + W_{EI}(K^E r_E)_i ,
\qquad r_I=[u^I]_+ ,
\qquad r_E=f(u^E).
$$

- $J_1$ scales recurrent excitation; $W_{IE}$ the tuned (Mexican-hat) inhibition.
- $\overline{r_E}=\frac1N\sum_j r_{E,j}$ is the mean rate; $W_{\text{glob}}$ is a
  **global inhibition** proportional to total activity. With the linear gain this is
  the term that **stabilizes** the ring against runaway/drift (ablating it makes the
  network blow up) — a broadly-projecting / reticular-thalamic (TRN) feedback
  (Pinault 2004; Crabtree 2018).
- The interneuron ring is co-tuned with the local excitatory population through
  $K^E$, giving feedback inhibition that lags excitation by $\tau_I$ — the lag is what
  lets the post-flash inhibition transiently overshoot and produce the **dip**.

`u_E` is floored (at `reversal_potential` during the stimulus, else $-100$) but **not
clamped from above** — the rate saturates through $f(u)$ instead. Keeping `u_E`
unclamped preserves the *rank order* of cells when rates saturate, which preserves the
bump's location/identity through the strong FC (so the rebound stays PD-selective).

---

## 6. External inputs

**Tonic upstream HD drive.** AD does not generate the HD signal; it inherits a
persistent, head-direction-tuned input from the lateral mammillary / dorsal tegmental
nuclei (Taube 2007). Modeled as a fixed tuned bump at the cue direction (0°) scaled by
a fitted amplitude $I_{HD}$:

$$
I^{\text{ext}}_i = I_{\text{base}} + I_{HD}\,e^{\kappa_{HD}(\cos\theta_i-1)} .
$$

Being *tuned* (peaks at PD, ≈0 at the antipode) it can be strong without driving
off-bump cells — it sets baseline tuning height **and anchors the bump** so the
attractor recovers to the cue after a perturbation.

**Fast component (sensory flash).** During the brief window
$[\,t_0,\,t_0+\Delta_{fc}\,]$ a uniform drive $A_{\text{fast}}$ is added to all cells
($t_0 = $ `stim_delay`). It is deliberately **brief** (a few ms) so it cannot account
for the 30–300 ms SC — the SC must come from intrinsic currents. The flash hits all
cells, so both PD and anti-PD show the FC, matching the data.

**Conduction latency.** `stim_delay` (~5–10 ms) shifts the flash onset so the model FC
aligns with the in-vivo FC, which begins ~10 ms after the speaker fires (sensory
transmission delay).

---

## 7. Intrinsic currents (the SC generators)

After the FC, strong feedback inhibition crashes the cells (the dip). This
hyperpolarization is the trigger for the classic thalamic **post-inhibitory rebound**.

### 7.1 I_T — low-threshold T-type Ca²⁺ current

Two gates: fast (instantaneous) activation $m_T$ and slow inactivation $h_T$:

$$
m_T = \sigma\!\Big(\tfrac{u^E-V^T_{1/2}}{k_T}\Big),\qquad
h_{T,\infty}=\sigma\!\Big(\!-\tfrac{u^E-V^T_{1/2}}{k_T}\Big),\qquad
\tau_{hT}\frac{dh_T}{dt}=h_{T,\infty}-h_T,
$$

$$
I^T = g_T\,m_T\,h_T\,(E_{Ca}-u^E)\,\cdot\,\underbrace{\frac{\text{rec}_E}{\text{rec}_E+\text{REC}_{1/2}}}_{\text{recurrent gate}} ,
$$

where $\sigma(x)=1/(1+e^{-x})$ and $\text{rec}_E=J_1(K^E r_E)$.

**Biology / behaviour.** At rest $h_T\approx0$ (inactivated). The post-flash crash
hyperpolarizes the cell → $h_{T,\infty}\to1$, so $h_T$ **de-inactivates**. As the bump
recovers and $u^E$ climbs back through $V^T_{1/2}$, the fast activation $m_T$ turns on
while $h_T$ is still high — a **transient inward Ca²⁺ current** that drives the rebound;
then $u^E$ rises, $h_{T,\infty}\to0$, $h_T$ decays, and $I_T$ shuts off. This is exactly
the de-inactivation→burst sequence of thalamic T-channels (Jahnsen & Llinás 1984;
Huguenard 1996; Destexhe et al. 1996; Perez-Reyes 2003). The reversal $E_{Ca}$ is set
high-positive so the current is depolarizing whenever the gates open (no positive-
feedback latch on the compressed rate scale).

The **recurrent gate** $\text{rec}_E/(\text{rec}_E+\text{REC}_{1/2})$ is the crucial
spatial term: a real T-type rebound needs the cell to be depolarized to threshold,
which off-bump cells (no recurrent input) never reach. So even though the crash
de-inactivates $h_T$ everywhere, only cells receiving recurrent excitation (the bump)
express the rebound. This confines the SC to the bump and keeps the attractor from
broadening/dissolving from a spatially-global rebound.

### 7.2 I_h — hyperpolarization-activated cation current (HCN)

A single slow activation gate $m_h$, additive (depolarizing):

$$
m_{h,\infty}=\sigma\!\Big(\!-\tfrac{u^E-V^h_{1/2}}{k_h}\Big),\qquad
\tau_h(u)\frac{dm_h}{dt}=m_{h,\infty}-m_h,\qquad
I^h = g_h\,m_h .
$$

$I_h$ activates on hyperpolarization (during the dip) and is depolarizing — the HCN
"sag"/rebound current of thalamic relay cells (McCormick & Pape 1990; Pape 1996;
Robinson & Siegelbaum 2003; Biel et al. 2009). It is modeled **additively** (not
ohmic) to avoid a numerical latch on the abstract membrane scale. Its time constant is
**asymmetric** — fast when charging ($m_{h,\infty}>m_h$, $\tau_{h,\text{on}}$), slow
when decaying ($\tau_{h,\text{off}}$, hundreds of ms) — reflecting the strongly
voltage-dependent activation kinetics of HCN. The fast charge lets it build during the
brief dip; the slow decay produces the **long SC tail**.

### 7.3 Global SC input gated by bump-membership (the SC output *emerges*)

The intrinsic rebound currents above **shape** the SC, but they cannot by themselves
build its ~91 Hz *amplitude* without breaking the attractor (see Section 8b). The SC
amplitude comes from a second **GLOBAL, spatially UNIFORM** input — an **NMDA-like EPSC**
(a rise × bi-exponential decay) after the dip — that is *not* tuned to the cue. Its
direction-selectivity **emerges**: the input is multiplied by the ring's own
**bump-membership gate** $\text{sc\_gate}$ (a network state, not a property of the input):

$$
I^{SC}_i(t) = A_{sc}\,\underbrace{\Big(1-e^{-\Delta t/\tau^{\text{on}}_{sc}}\Big)}_{\text{rise}}
\underbrace{\Big(w\,e^{-\Delta t/\tau^{\text{off}}_{sc}} + (1{-}w)\,e^{-\Delta t/\tau^{\text{off2}}_{sc}}\Big)}_{\text{bi-exp decay (GluN2B slow + GluN2A fast)}}
\,\text{sc\_gate}_i,\qquad \Delta t = t - (t_0 + \text{sc\_delay}),
$$

The two decay components (fitted: slow $\tau^{\text{off}}_{sc}\!\approx\!416$ ms with weight
$w\!\approx\!0.70$, fast $\tau^{\text{off2}}_{sc}\!\approx\!60$ ms) mirror the GluN2B / GluN2A
NMDA EPSC. The second component is **not required** by the fit (a single decay already
fits the smooth tail; R² 0.918→0.921, flat) — it makes the input *literally* an NMDA
EPSC, matching the **SC = NMDA-gated global drive** hypothesis (the voltage-dependent
Mg²⁺-block of the same NMDARs is the natural biophysical basis of the bump-gate below).

active for $t\ge t_0+\text{sc\_delay}$. The gate is the same bump-membership signal that
confines $I_T$, but **low-passed** (memory $\tau_{scg}$) so the brief dip does not slam it
shut:

$$
\tau_{scg}\,\dot{\ell}_i = \text{rec}_i - \ell_i,\qquad
\text{sc\_gate}_i = \frac{\ell_i}{\ell_i + \text{REC}_{1/2}},\qquad \text{rec}_i=J_1(K^E r_E)_i.
$$

**Why it works.** Bump cells ($\text{rec}_i$ high) receive the SC → the PD bump rises to
~91 Hz; the antipode ($\text{rec}_i\approx0 \Rightarrow \ell_i\approx0 \Rightarrow
\text{sc\_gate}\approx0$) receives ≈nothing → **anti-PD stays silent** even though the
*input* is global. The **memory** $\tau_{scg}$ (~30 ms) is essential: without it the gate
crashes during the dip and re-opens abruptly at ~50 ms, dumping the SC in as a sharp
spike; the low-pass bridges the dip so the SC enters smoothly (matching the data's early
rise). Biologically: a global salience/arousal drive that the HD attractor **transforms**
into a tuned output — the SC is an *ADN output*, not a tuned input. Ablating it
($A_{sc}=0$) collapses the SC (88→55 Hz) — the evidence that the gated global input builds
the hump.

---

## 8. Ca-activated disinhibition (removed)

An earlier version added a Ca-activated disinhibition (a per-cell $I_T$-driven
reduction of local inhibition) to hold up the SC tail. An ablation of the final fit
showed it contributes **0.000 R²** (the fitted conductance $g_{\text{dis}}$ collapsed to
~0): once the SC is an inherited input with its own slow decay (plus $I_h$), no extra
disinhibition is needed. It was **removed** — the SC tail is set by the SC drive's
$\tau^{\text{off}}_{sc}$ and $I_h$.

---

## 8b. The amplitude–recovery tension, and global adaptation

A central finding shaped the final design. In a **single** ring attractor, a slow
component that is *fully emergent* (produced by recurrent amplification of the intrinsic
rebound) and tall enough to reach ~91 Hz is **incompatible with recovery**:

- The ~91 Hz SC is ~2.5× baseline. Reaching it by recurrence requires strong
  excitatory/channel gain, which pushes the ring into a **second stable (bistable)
  state** — a broad, elevated bump (~82 Hz, FWHM ~117°) that **never relaxes** once the
  channels drain. (The earlier R²≈0.72 fit was exactly this latched state; a 300 ms fit
  window hid the failure to recover.)
- Mechanisms that *force* recovery act on the same elevated activity and therefore also
  **suppress the peak**: global inhibition $W_{\text{glob}}$ crushes peak and tail
  together; cranking the channel conductances to restore the peak instead
  **destabilizes** the bump (it drifts off the cue or flips to the antipode).

The resolution is to **decouple amplitude from recovery**: (1) **amplitude** is supplied
by the gated global SC input (Section 7.3), which is not recurrently self-amplified and
so does not engage the bistability; (2) **recovery** is supplied by a slow, **global
(mean-field) adaptation** current — the network analogue of an SK / Ca-activated-K⁺ or
M-type spike-frequency-adaptation current (Madison & Nicoll 1984; Stocker 2004; Benda &
Herz 2003; Sanchez-Vives et al. 2000):

$$
\tau_a\frac{da}{dt} = \overline{r_E} - a,\qquad I^{\text{adapt}} = g_a\,a,
$$

subtracted **uniformly** from every cell's $u^E$. Three properties make it the right
tool: it is **slow** ($\tau_a\sim$ hundreds of ms) so $a\approx0$ at the sharp ~55 ms
peak — *the peak is preserved*; it **accrues over the elevated tail** and so pulls the
bump back to baseline and **destabilizes the latched broad state** — *true recovery*;
and it is **global, not per-cell**, which matters because a *per-cell* adaptation makes
the ring bump **travel** (the adapted peak fatigues and neighbours take over). A uniform
pull-down has no preferred direction, so the bump stays put, and it also **re-narrows**
the bump, recovering both rate **and** width. It is the one genuinely load-bearing
recovery mechanism: ablating it ($g_a=0$) leaves the bump stuck broad (FWHM ~105 vs
~69°).

This makes the SC's **direction-tuning fully emergent**: every input (FC and SC) is
spatially **global/untuned**, and the PD-selective SC output is produced entirely by the
network — the bump-membership gate silences the antipode, the intrinsic channels shape the
onset/tail, and adaptation recovers the width. Only the SC's *temporal* profile is an
inherited input (like the FC); its *spatial* structure is the ring's own doing.

---

## 9. How each response feature emerges

- **FC**: the brief uniform flash drives every cell → sharp onset transient in PD and anti-PD.
- **Dip**: the flash recruits feedback inhibition that, lagging by $\tau_I$, transiently overshoots → firing crashes.
- **SC onset/shape**: the crash de-inactivates $I_T$; as the bump recovers, $I_T$ (gated to the bump) fires a phasic rebound, and $I_h$ (slow) shapes the tail.
- **SC amplitude**: the **global** SC input, gated by bump-membership and low-pass-smoothed ($\tau_{scg}$), lifts the PD bump to ~91 Hz (Section 7.3).
- **Direction selectivity**: the SC input is global but the bump-gate ($\approx0$ at the antipode) admits it only on the bump; the $I_T$ rebound is likewise gated; the non-saturating `u_E` preserves the bump through the FC — so PD carries the SC and **anti-PD shows FC only**.
- **Recovery to baseline**: as the global SC input decays, the **global adaptation** $g_a$ pulls the bump back down and re-narrows it, while $W_{\text{glob}}$ + the $I_{HD}$ anchor keep it stable and centered — PD and FWHM return to baseline by ~700 ms (Section 8b).

---

## 10. Fitting

All 32 parameters are fit **jointly** to the population PSTH
([`vivo_target.load_vivo_psth`](vivo_target.py)) by a single objective
[`loss_joint`](optimization_engine.py) = baseline-geometry regularizers
(`loss_stage1`: peak ~40 Hz, FWHM 60–90°, single bump, anti-PD silent) + an
evoked data term (`loss_stage2`: weighted MSE of the PD trace vs. the in-vivo PSTH over
the **[-50, 700] ms** window + structural guards: a dip exists, anti-PD silent through
the SC, late-window single bump, and **two-sided recovery** of both rate and width to
baseline). Extending the window to 700 ms (vs the earlier 300 ms) is what exposes — and
then demands — the SC's decay back to baseline.

A practical caveat: `loss_joint` is **not** R²-aligned (its structural regularizers can
trade data-fit for validity), and the valid *recovering* basin is narrow, so plain
differential evolution (Storn & Price 1997) under-shoots (~0.6). The reported fit is
obtained by a **direct R²-search with a validity gate** ([`_run_full_fit.py`](_run_full_fit.py))
that rejects any non-recovering or degenerate solution (baseline must be a real ~40 Hz
bump; rate and width must return near baseline by 600 ms). **Ablation** verifies the
mechanism split: $A_{sc}=0$ collapses the SC (88→55 Hz); $g_a=0$ leaves the bump stuck
broad; $W_{\text{glob}}=0$ makes it blow up. Best fit: **R² = 0.9534** on the [-50, 700] ms
window — with a **fully untuned SC input** whose tuned, PD-selective output emerges from
the bump-gate. (An earlier version that injected a *spatially tuned* SC reached 0.945;
the ~0.03 cost buys emergent tuning — anti-PD 0, SC ~88 Hz decaying to baseline.)

---

## 11. Parameters

**Geometry / network (Stage 1, fit):** `tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE,
KAPPA_I, W_EI, I_HD, W_GLOBAL`.
**Stimulus / channels (Stage 2, fit):** `A_fast, fc_stim_duration, stim_delay,
reversal_potential, g_T, V_half_T, k_T, tau_hT, E_Ca, g_h, V_half_h, tau_h_on,
tau_h_off, g_a, tau_a, A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg, tau_sc_off2, w_sc`
(the last nine: global adaptation + the global NMDA-EPSC SC input (bi-exponential decay
`tau_sc_off`/`tau_sc_off2`/`w_sc`) + its bump-gate memory `tau_scg`).
**Fixed constants:** `N=120, DT=0.2, GAIN_SLOPE=6, R_MAX=1000 (safety clip),
KAPPA_HD=2, K_H_SLOPE=8, REC_HALF=10`.

---

## 11b. Notation (symbol glossary)

So this document stands on its own for a reader who did not build it. **Units:** this is a
firing-rate model — `u` is an abstract membrane-potential-like *activation* (call it
"u-units", loosely mV-like), `r` is a firing rate in **Hz**, times are **ms**, gains/gates
are dimensionless. It is not a conductance-based (mV/nA) model.

| symbol (code) | meaning | units |
|---|---|---|
| `N` | number of excitatory ring cells (=120) | count |
| $\theta_i$ | preferred direction of cell *i* | rad |
| $\Delta\theta_{ij}$ | angular difference $\theta_i-\theta_j$ | rad |
| $u^E_i$ (`u_E`) | excitatory activation (membrane-like) | u-units |
| $u^I_i$ (`u_I`) | interneuron activation | u-units |
| $r_E=f(u^E)$ | excitatory firing rate | Hz |
| $r_I=[u^I]_+$ | interneuron firing rate | Hz |
| $\overline{r_E}$ | mean excitatory rate over the ring | Hz |
| $f(\cdot)$, `GAIN_SLOPE`, `R_MAX` | threshold-linear gain; slope (=6); safety clip (=1000) | Hz/u-unit, Hz |
| $K^E,K^I$; $\kappa_E,\kappa_I$ (`KAPPA_E/I`) | narrow-E / broad-I connectivity kernels; their concentrations | — |
| `J1` | recurrent excitation gain | — |
| `W_IE` / `W_EI` | tuned (Mexican-hat) inhibition gain / E→I drive gain | — |
| `W_GLOBAL` ($W_\text{glob}$) | global inhibition gain (∝ total activity) | — |
| $\tau_E,\tau_I$ | membrane time constants (E, I) | ms |
| `I_baseline`, `I_HD`, $\kappa_{HD}$ (`KAPPA_HD`) | uniform tonic drive; tuned HD drive amplitude; HD tuning width (=2) | u-units, — |
| `A_fast`, `fc_stim_duration`, `stim_delay` ($t_0$) | FC volley amplitude; FC pulse width; conduction latency / flash onset | u-units, ms, ms |
| `reversal_potential` | $u_E$ floor during the stimulus | u-units |
| $I^T$, `g_T`, $m_T$, $h_T$ | T-type Ca²⁺ current; conductance; activation / inactivation gates | u-units, —, 0–1 |
| `V_half_T` ($V^T_{1/2}$), `k_T`, $\tau_{hT}$, `E_Ca` | I_T half-activation / slope / inactivation τ / Ca reversal (=80) | u-units, u-units, ms, u-units |
| $\text{rec}_E=J_1(K^Er_E)$, `REC_HALF` ($\text{REC}_{1/2}$) | recurrent-excitation signal (gate input); gate half-saturation (=10) | u-units |
| $I^h$, `g_h`, $m_h$, `V_half_h`, `K_H_SLOPE`, $\tau_{h,\text{on/off}}$ | HCN current; conductance; gate; half-activation; slope (=8); charge/decay τ | u-units, —, 0–1, u-units, u-units, ms |
| $a$, `g_a`, $\tau_a$ | global adaptation state (low-pass of $\overline{r_E}$); gain; time constant | Hz, —, ms |
| $I^{SC}_i$, `A_sc` | gated global SC input; its amplitude | u-units |
| `tau_sc_on`, `tau_sc_off`, `tau_sc_off2`, `w_sc` ($w$), `sc_delay` | SC EPSC rise; slow/fast decay; slow-component weight; onset delay | ms, ms, ms, 0–1, ms |
| $\text{sc\_gate}_i$, $\ell_i$, `tau_scg` | bump-membership gate (emergent tuning); its low-passed drive; gate memory τ | 0–1, u-units, ms |
| $\sigma(x)$, `DT` | logistic sigmoid $1/(1+e^{-x})$; Euler step (=0.2) | —, ms |

---

## 12. References

- Amari S (1977). Dynamics of pattern formation in lateral-inhibition type neural fields. *Biol Cybern* 27:77–87.
- Ben-Yishai R, Bar-Or RL, Sompolinsky H (1995). Theory of orientation tuning in visual cortex. *PNAS* 92:3844–3848.
- Skaggs WE, Knierim JJ, Kudrimoti HS, McNaughton BL (1995). A model of the neural basis of the rat's sense of direction. *NIPS* 7:173–180.
- Zhang K (1996). Representation of spatial orientation by the intrinsic dynamics of the head-direction cell ensemble. *J Neurosci* 16:2112–2126.
- Taube JS (1995). Head direction cells recorded in the anterior thalamic nuclei of freely moving rats. *J Neurosci* 15:70–86.
- Taube JS (2007). The head direction signal: origins and sensory-motor integration. *Annu Rev Neurosci* 30:181–207.
- Knierim JJ, Zhang K (2012). Attractor dynamics of spatially correlated neural activity in the limbic system. *Annu Rev Neurosci* 35:267–285.
- Peyrache A, Lacroix MM, Petersen PC, Buzsáki G (2015). Internally organized mechanisms of the head direction sense. *Nat Neurosci* 18:569–575.
- Jahnsen H, Llinás R (1984). Ionic basis for the electroresponsiveness and oscillatory properties of guinea-pig thalamic neurones in vitro. *J Physiol* 349:227–247.
- Llinás RR (1988). The intrinsic electrophysiological properties of mammalian neurons. *Science* 242:1654–1664.
- Huguenard JR (1996). Low-threshold calcium currents in central nervous system neurons. *Annu Rev Physiol* 58:329–348.
- Destexhe A, Contreras D, Sejnowski TJ, Steriade M (1994). A model of spindle rhythmicity in the isolated thalamic reticular nucleus. *J Neurophysiol* 72:803–818. (reduced T-current)
- Perez-Reyes E (2003). Molecular physiology of low-voltage-activated T-type calcium channels. *Physiol Rev* 83:117–161.
- Sherman SM (2001). Tonic and burst firing: dual modes of thalamocortical relay. *Trends Neurosci* 24:122–126.
- McCormick DA, Pape HC (1990). Properties of a hyperpolarization-activated cation current and its role in rhythmic oscillation in thalamic relay neurones. *J Physiol* 431:291–318.
- Pape HC (1996). Queer current and pacemaker: the hyperpolarization-activated cation current in neurons. *Annu Rev Physiol* 58:299–327.
- Robinson RB, Siegelbaum SA (2003). Hyperpolarization-activated cation currents: from molecules to physiological function. *Annu Rev Physiol* 65:453–480.
- Biel M, Wahl-Schott C, Michalakis S, Zong X (2009). Hyperpolarization-activated cation channels: from genes to function. *Physiol Rev* 89:847–885.
- Ahmadian Y, Rubin DB, Miller KD (2013). Analysis of the stabilized supralinear network. *Neural Comput* 25:1994–2037.
- Rubin DB, Van Hooser SD, Miller KD (2015). The stabilized supralinear network: a unifying circuit motif underlying multi-input integration in sensory cortex. *Neuron* 85:402–417.
- Priebe NJ, Ferster D (2008). Inhibition, spike threshold, and stimulus selectivity in primary visual cortex. *Neuron* 57:482–497. (power-law f-I)
- Hansel D, van Vreeswijk C (2002). How noise contributes to contrast invariance of orientation tuning in cat visual cortex. *J Neurosci* 22:5118–5128.
- Naka KI, Rushton WAH (1966). S-potentials from colour units in the retina of fish. *J Physiol* 185:536–555. (Naka–Rushton)
- Albrecht DG, Hamilton DB (1982). Striate cortex of monkey and cat: contrast response function. *J Neurophysiol* 48:217–237.
- Carandini M, Heeger DJ (2012). Normalization as a canonical neural computation. *Nat Rev Neurosci* 13:51–62.
- Pinault D (2004). The thalamic reticular nucleus: structure, function and concept. *Brain Res Rev* 46:1–31.
- Crabtree JW (2018). Functional diversity of thalamic reticular subnetworks. *Front Syst Neurosci* 12:41.
- Hughes SW, Cope DW, Blethyn KL, Crunelli V (2002). Cellular mechanisms of the slow (<1 Hz) oscillation in thalamocortical neurons in vitro. *Neuron* 33:947–958. (Ca-activated cation current / slow ADP)
- Madison DV, Nicoll RA (1984). Control of the repetitive discharge of rat CA1 pyramidal neurones in vitro. *J Physiol* 354:319–331. (Ca-activated K⁺ / spike-frequency adaptation)
- Stocker M (2004). Ca²⁺-activated K⁺ channels: molecular determinants and function of the SK family. *Nat Rev Neurosci* 5:758–770. (SK current)
- Benda J, Herz AVM (2003). A universal model for spike-frequency adaptation. *Neural Comput* 15:2523–2564.
- Sanchez-Vives MV, Nowak LG, McCormick DA (2000). Cellular mechanisms of long-lasting adaptation in visual cortical neurons in vitro. *J Neurosci* 20:4286–4299.
- Storn R, Price K (1997). Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces. *J Glob Optim* 11:341–359.
