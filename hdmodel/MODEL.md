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
are **direction-selective**. A purely phenomenological model (injecting the SC as a
fitted exponential) reproduces the trace but explains nothing. The goal here is for
each component to **emerge** from biophysically-motivated mechanisms.

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
| Supralinear gain `r = f(u)` | Naka–Rushton / SSN | **monostable** sharp bump | power-law f-I curve |
| Tonic tuned drive `I_HD` | upstream HD input | baseline tuning + anchoring | LMN/DTN → AD |
| Brief flash `A_fast` | sensory volley | **FC** | auditory input |
| Feedback inhibition | interneuron ring | **dip** | feedforward/feedback inhibition |
| Tuned SC drive `A_sc` | feedforward double-exponential at PD | **SC amplitude** (the ~91 Hz hump) | upstream phasic re-excitation |
| `I_T` (T-type Ca²⁺) | de-inactivation + activation gates | SC rebound **shaping** | thalamic rebound burst |
| `I_h` (HCN) | slow hyperpolarization-activated current | SC tail **shaping** | thalamic sag/ADP |
| recurrent gating of `I_T` | only bump cells rebound | spatial confinement | rebound needs depolarizing drive |
| Ca-activated disinhibition | rebound suppresses local inhibition | tail support | slow ADP / reduced TRN drive |
| Global adaptation `g_a` | mean-field current ∝ recent activity | **SC recovery to baseline** | SK / M-current (spike-freq. adaptation) |
| Global inhibition `W_GLOBAL` | ∝ total activity | width regulation | broad/TRN inhibition |

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

## 4. Supralinear gain (the key to a well-behaved attractor)

The firing rate is a **supralinear (Naka–Rushton) function** of the activation:

$$
r = f(u) = R_{\max}\,\frac{[u]_+^{\,p}}{\sigma^{p} + [u]_+^{\,p}},
\qquad [u]_+=\max(0,u),
$$

with $R_{\max}=1000$ Hz, $\sigma=35$, $p=2$ (constants `R_MAX, GAIN_SIGMA, GAIN_P`).
Over the operating range (0–200 Hz) the activation stays well below $\sigma$, so the
curve is effectively the **power law** $r\propto[u]_+^{2}$ — the **Stabilized
Supralinear Network (SSN)** regime.

**Why it matters.** With a threshold-*linear* gain ($r=[u]_+$) the ring attractor is
*marginal*: it supports a continuum of bump widths, so the SC rebound permanently
kicks the bump into a stuck broad state. A **supralinear** gain makes the network
**winner-take-all**: high-rate cells dominate the recurrent loop and broad, low-rate
bumps become unstable. The attractor is then **monostable** — a single sharp bump —
and any perturbation (the SC) relaxes back toward baseline width. This is the central
result of SSN theory (Ahmadian, Rubin & Miller 2013; Rubin, Van Hooser & Miller
2015); power-law / expansive nonlinearities are also the empirically-measured cortical
f-I relation (Priebe & Ferster 2008; Hansel & van Vreeswijk 2002), and the
Naka–Rushton form is the standard saturating-supralinear gain (Naka & Rushton 1966;
Albrecht & Hamilton 1982; Carandini & Heeger 2012). $R_{\max}$ is set far above the
firing range so the operating point sits in the convex (supralinear) part rather than
the saturating shoulder — putting it in the saturating region re-introduces the broad
state, and making $p$ too large ($\gtrsim 2.5$) makes the winner-take-all so strong
the SC spawns a spurious bump that drifts off the cue.

---

## 5. Network dynamics

Excitatory activation $u^E_i$ and interneuron activation $u^I_i$ evolve as leaky
integrators (Euler, $\Delta t=0.2$ ms):

$$
\tau_E \frac{du^E_i}{dt}
= -u^E_i
+ \underbrace{J_1 (K^E r_E)_i}_{\text{recurrent exc.}}
- \underbrace{d_i\,W_{IE}(K^I r_I)_i}_{\text{tuned inh.}}
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
- $d_i\in[0.2,1]$ is the **disinhibition** factor (Section 8); $d_i\equiv1$ pre-stimulus.
- $\overline{r_E}=\frac1N\sum_j r_{E,j}$ is the mean rate; $W_{\text{glob}}$ is a
  **global inhibition** proportional to total activity. Because a broad bump has more
  total activity than a sharp one, this term penalizes broad states and helps
  regulate bump width — a broadly-projecting / reticular-thalamic (TRN) feedback
  (Pinault 2004; Crabtree 2018). It complements the supralinear gain.
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

### 7.3 Tuned feedforward SC drive (the SC amplitude)

The intrinsic rebound currents above **shape** the SC, but they cannot by themselves
set its *amplitude* without breaking the attractor (see Section 8b). The SC amplitude is
therefore supplied by a tuned, feedforward transient input — a double-exponential
(fast rise, slow decay) that turns on after the dip and is tuned to the cue direction:

$$
I^{SC}_i(t) = A_{sc}\,\Big(e^{-\Delta t/\tau^{\text{off}}_{sc}} - e^{-\Delta t/\tau^{\text{on}}_{sc}}\Big)\,
e^{\kappa_{HD}(\cos\theta_i-1)},\qquad \Delta t = t - (t_0 + \text{sc\_delay}),
$$

active for $t\ge t_0+\text{sc\_delay}$. Because it is **feedforward**, it sets the SC
amplitude *without* recurrent amplification, so the monostable+adapting network simply
**tracks** it up to ~91 Hz and then follows it back down as it decays — the SC stays
tall *and* transient. Because it is **tuned** ($\propto e^{\kappa_{HD}(\cos\theta-1)}$,
the same shape as $I_{HD}$), it is ≈0 at the antipode, so the SC is **PD-selective**
for free (anti-PD shows only the FC). Biologically this is the phasic re-excitation AD
inherits from upstream HD / sensory structures after the startle, riding on top of the
intrinsic $I_T$/$I_h$ rebound. Ablating it ($A_{sc}=0$) collapses the SC amplitude
toward baseline — the evidence that the drive supplies the hump.

---

## 8. Ca-activated disinhibition (tail support)

A purely depolarizing current cannot, by itself, hold firing above the attractor's
set-point: the E/I loop is homeostatic (more excitation → more feedback inhibition →
cancellation). To produce a *sustained elevated* SC tail the model instead transiently
**reduces inhibition**, raising the bump's set-point. A per-cell gate $s_{\text{dis}}$
is driven by the cell's own $I_T$ burst (a Ca²⁺ proxy), rises slowly (lag
$\tau_{\text{on}}$=`DELAY_TAU`) and decays slowly ($\tau_{\text{dis}}$):

$$
\text{drive}_i=\mathrm{clip}\!\big(I^T_i/200,\,0,\,1\big),\qquad
\frac{ds_{\text{dis},i}}{dt}=\frac{\text{drive}_i-s_{\text{dis},i}}{\tau(\cdot)},\qquad
d_i = \max\!\big(0.2,\ 1-g_{\text{dis}}\,s_{\text{dis},i}\big),
$$

and $d_i$ multiplies the tuned inhibition in the $u^E$ equation. Driving it off $I_T$
makes it automatically **selective** (the antipode never bursts, so it never
disinhibits → stays silent) and **transient** (no latch). This is the thalamic **slow
afterdepolarization / post-burst disinhibition** motif — Ca²⁺ entry during the
low-threshold spike engages slow depolarizing/disinhibitory processes (Ca-activated
nonselective cation currents and reduced reticular drive; Hughes et al. 2002;
Zhu et al. 1999; Pinault 2004). The slow rise also **delays** the disinhibition so it
lifts the tail without inflating the rebound peak.

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
*feedforward* by the tuned SC drive (Section 7.3), which is not recurrently amplified and
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
pull-down has no preferred direction, so the bump stays put, and — combined with the
supralinear gain — it also **re-narrows** the bump, recovering both rate **and** width.
Ablating it ($g_a=0$) makes the bump latch elevated again.

This makes the model **semi-emergent**: the SC *shape* (rebound timing, tail) emerges
from the intrinsic channels and the network, while the SC *amplitude* is a tuned
feedforward input — the deliberate, minimal departure from full emergence required to
reproduce a slow component that is simultaneously **tall, transient, and recovering**.

---

## 9. How each response feature emerges

- **FC**: the brief uniform flash drives every cell → sharp onset transient in PD and anti-PD.
- **Dip**: the flash recruits feedback inhibition that, lagging by $\tau_I$, transiently overshoots → firing crashes.
- **SC onset/shape**: the crash de-inactivates $I_T$; as the bump recovers, $I_T$ (gated to the bump) fires a phasic rebound, and $I_h$ (slow) plus Ca-activated disinhibition shape the tail.
- **SC amplitude**: the tuned feedforward SC drive $A_{sc}$ lifts the PD bump to ~91 Hz without recurrent runaway (Section 7.3).
- **Direction selectivity**: the SC drive and $I_T$ rebound are both tuned/gated to the bump; the non-saturating `u_E` preserves the bump through the FC so PD recovers first; the reformed bump's lateral inhibition keeps the antipode silent — anti-PD shows FC only.
- **Recovery to baseline**: as the SC drive decays, the **global adaptation** $g_a$ pulls the bump back down and destabilizes any elevated state, while the supralinear gain + $I_{HD}$ anchor re-narrow and re-center it — PD and FWHM return to baseline by ~700 ms (Section 8b).

---

## 10. Fitting

All 31 parameters are fit **jointly** to the population PSTH
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
mechanism split: $A_{sc}=0$ collapses the SC amplitude; $g_a=0$ makes the bump latch
elevated (no recovery). Best fit to date: **R² ≈ 0.88** on the [-50, 700] ms window,
with a monostable, PD-selective attractor whose SC is tall, transient, and **recovers**
to baseline in both rate (~92→~40 Hz) and width (FWHM back to ~baseline).

---

## 11. Parameters

**Geometry / network (Stage 1, fit):** `tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE,
KAPPA_I, W_EI, I_HD, W_GLOBAL`.
**Stimulus / channels (Stage 2, fit):** `A_fast, fc_stim_duration, stim_delay,
reversal_potential, g_T, V_half_T, k_T, tau_hT, E_Ca, g_h, V_half_h, tau_h_on,
tau_h_off, g_dis, tau_dis, g_a, tau_a, A_sc, tau_sc_on, tau_sc_off, sc_delay`
(the last six: global adaptation and the tuned feedforward SC drive).
**Fixed constants:** `N=120, DT=0.2, R_MAX=1000, GAIN_SIGMA=35, GAIN_P=2, KAPPA_HD=2,
K_H_SLOPE=8, DELAY_TAU=40, REC_HALF=10`.

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
