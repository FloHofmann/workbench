# Parameter dossier

One row per parameter. **Verdict** is decided by *ablation* (does removing it hurt the
four criteria?); reliability and curvature explain *why* a parameter is legible, they
never decide a cut on their own. See [METHODS.md](METHODS.md) for what each column means.

Columns: **ablation** = R² lost when the parameter is removed and the rest refit
(negative = removing it *improves* the median fit); **type** = M (mechanism → 0) or
F (freedom → shared constant); **reliab** = split-half correlation (does the fitted
value replicate across odd/even trials, −1…1); **curv** = curvature rank (1 = most
constrained by the data). Verdict ∈ {**free**, **fix**, **cut**}.

---

## Readout cell (the per-cell model)

Sorted by ablation cost. The minimal model keeps the **free** rows only.

| symbol | biophysics | what it does to the trace | type | ablation | reliab | curv | verdict | one-sentence reason |
|---|---|---|---|---|---|---|---|---|
| `A_fast` | fast feedforward drive | height of the FC spike | M | **+0.215** | 0.47 | 1 | **free** | Sets the fast-component spike; removing it is the single biggest loss. |
| `A_sc` | NMDA-EPSC amplitude | height of the slow component | M | **+0.143** | 0.36 | 6 | **free** | Sets the slow-component drive; the SC collapses without it. |
| `g_a` | adaptation conductance | pulls the tail back to baseline | M | **+0.108** | 0.16 | 12 | **free** | Load-bearing for recovery even though its value is hard to read per cell — the worked example of "unreadable ≠ unnecessary". |
| `g_T` | T-type Ca conductance | shapes the SC onset/rebound | M | **+0.107** | 0.41 | 2 | **free** | The intrinsic current that sharpens the slow component. |
| `I_HD` | tuned HD drive | sets baseline bump height | M | **+0.051** | 0.41 | 7 | **free** | Each cell's tonic firing level; also anchors direction-selectivity. |
| `V_half_T` | I_T half-activation V | where I_T switches on | F | −0.034 | **0.72** | 8 | **free** | Kept free despite ~0 median cost: it is the *most* replicable parameter, and sharing it lets I_T fire at the antipode of strong cells → the SC leaks to anti-PD (breaks selectivity). |
| `k_T` | I_T activation slope | steepness of I_T onset | F | −0.038 | 0.31 | 3 | **free** | Partners `V_half_T` in placing the I_T activation; sharing it reintroduces the anti-PD leak. |
| `mg_conc` | external [Mg²⁺] | strength of the NMDA gate | F | +0.007 | – | – | **fix** | Shared: it is a bath condition, not a cell property, and freeing it *reopens* the gate at anti-PD (selectivity explodes). |
| `tau_sc_off` | NMDA decay | SC decay time | F | +0.005 | 0.16 | 11 | **fix** | One shared NMDA decay; per-cell values don't replicate (r 0.16). |
| `tau_dend` | dendritic low-pass | gate memory through the dip | F | −0.025 | 0.19 | – | **fix** | Shared dendritic time constant; bridges the inhibitory dip, same for all cells. |
| `tau_sc_on` | NMDA rise | SC onset speed | F | −0.031 | 0.01 | 16 | **fix** | Shared; per-cell value is pure noise (r 0.01). |
| `sc_delay` | SC latency | when the SC starts | F | −0.025 | 0.24 | 15 | **fix** | Shared onset delay. |
| `E_Ca` | Ca reversal | I_T driving force | F | −0.008 | 0.20 | 14 | **fix** | Shared reversal potential. |
| `tau_hT` | I_T inactivation | I_T time course | F | −0.034 | 0.22 | 21 | **fix** | Shared. |
| `tau_E` | membrane τ | overall response speed | F | −0.025 | 0.40 | 10 | **fix** | Reliable but sharing costs nothing for the effect; a single membrane τ suffices. |
| `fc_stim_duration` | FC pulse width | FC spike width | F | −0.004 | 0.52 | 4 | **fix** | Genuinely varies per cell (r 0.52) but sharing *improves* the median fit and isn't part of the effect — free it only if FC width per se is of interest. |
| `stim_delay` | conduction latency | shifts the whole response | F | −0.073 | 0.58 | 5 | **fix** | Real per-cell latency (r 0.58), but sharing improves generalization and it is not "the effect"; the clearest keep-free candidate if per-cell timing matters. |
| `I_baseline` | uniform tonic drive | baseline floor | M | **−0.012** | −0.08 | 9 | **cut** | Removing it *improves* the fit — it is degenerate with `I_HD` (both set tonic drive) and does not replicate (r −0.08). |
| `g_h`, `V_half_h`, `tau_h_on/off` | HCN / I_h | slow depolarizing tail | M | (cut in v3) | ≤0.15 | sloppy | **cut** | I_h sits near-idle; removing all four cost nothing across two independent tests. |
| `tau_sc_off2`, `w_sc` | 2nd NMDA decay (GluN2A/2B split) | bi-exponential tail | F | (cut in v3) | −0.04, 0.02 | 17,13 | **cut** | The subunit-kinetics split does not replicate across halves of the *same* cell; a single decay is a standard NMDA EPSC. |
| `reversal_potential` | (mislabelled) | hyperpolarization floor | F | (cut in v3) | −0.13 | 0 (inert) | **cut** | Curvature sensitivity exactly zero — `u_E` never reached it; replaced by a fixed constant. |

**Minimal readout model = 7 free per cell** (`I_HD, A_fast, A_sc, g_a, g_T, V_half_T, k_T`)
+ 10 shared constants. R² median 0.577 (94% of the 0.615 noise ceiling), selectivity and
recovery identical to v3, all 7 free params vary cell-to-cell.

---

## Population ring (the network model)

Ablation with a fuller refit. **Caveat:** re-minimizing the training loss (which
includes shape regularizers) drifts R² down by ~0.05–0.09 even for a null change, so
Type-F costs below ~0.09 sit in that noise floor and cannot be individually resolved.
The Type-M (mechanism) costs are above it and are the informative rows.

| symbol | biophysics | ablation | verdict | one-sentence reason |
|---|---|---|---|---|
| `W_IE` = `W_EI` | Mexican-hat inhibition (paired) | **+6.79** | **free** | The lateral inhibition that makes a single bump; the two gains act as one and removing either destroys the attractor. |
| `W_GLOBAL` | global inhibition | **+2.63** | **free** | Makes the broad post-stim state unstable so the bump recovers; without it the bump latches wide. |
| `I_HD` | tuned drive | **+1.79** | **free** | Sets and anchors the bump; removing it lets activity spread to the antipode. |
| `J1` | recurrent excitation | **+0.96** | **free** | The self-excitation that sustains the bump. |
| `A_fast` / `A_sc` | FC / SC drives | +0.58 / +0.56 | **free** | The two evoked inputs. |
| `I_baseline` | uniform drive | +0.30 | **free** | In the ring (unlike the single cell) it is not redundant with `I_HD` — it lifts the whole ring and refit can't fully absorb it. |
| `g_a` | adaptation | +0.29 | **free** | Width/rate recovery. |
| `g_T`, `V_half_T`, `E_Ca` | T-type Ca | +0.17 / +0.17 / +0.19 | **free** | The intrinsic current behind the SC. |
| `g_h` (I_h) | HCN | +0.07 ≈ noise floor | **cut** | Indistinguishable from the refit noise floor; near-idle, as in the cell model. |
| `tau_sc_off2`, `w_sc`, `reversal_potential` | 2nd decay, floor | +0.06–0.07 ≈ floor | **cut** | Same cuts as the cell model; no resolvable contribution. |
| all other shape τ / κ | connectivity/kinetics shape | +0.04–0.09 (floor) | **free (kept)** | Not individually resolvable, but a recurrent attractor genuinely needs its connectivity — the ring is near-minimal in its *mechanisms*. |

**Minimal ring ≈ 24 params** (32 − I_h ×4 − `tau_sc_off2` − `w_sc` − `reversal_potential` −
degenerate slack). Unlike the cell, the ring cannot be reduced to a handful: its
connectivity is the mechanism.
