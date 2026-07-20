# Model v3 — Mg-explicit readout cell

**Status:** additive. v1 (homogeneous full-32) and v2 (`readout_cell.py`) are untouched and
still fittable. v3 lives in `readout_cell_v3.py` / `fit_cell_v3.py` and shares the v2 network
background (`network_drivers.npz`) unchanged — the drivers describe the *population ring*, not
the readout cell, so they are model-version independent.

## Lineage

| version | what it is | per-cell params | headline |
|---|---|---|---|
| v1 | all 32 params fitted per cell (homogeneous ring of 120 copies of the target cell) | 32 | R² ~0.53; conceptually incoherent (removes all heterogeneity) |
| v2 | heterogeneous readout cell in an average network | 24 | R² 0.704 in-sample / **0.528 out-of-sample** (ceiling 0.615) |
| v3 | v2 + explicit NMDA Mg block, leaner | **18** | matches/beats v2 with 6 fewer params |

## What changed in v3, and the evidence for each change

Every cut is backed by **two independent methods that agree**: Hessian curvature
(`probe_identifiability.py`) and split-half replication across odd/even trials of the same cell
(`probe_split_half.py`).

1. **Jahr–Stevens Mg²⁺ block replaces the Michaelis gate.**
   `B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(−0.062·V))`
   The Mg²⁺ pore block *is* the NMDA receptor's own voltage gate, so the model's gating is now
   literal NMDA biophysics rather than a proxy. `mg_conc` is an explicit fitted parameter, which
   turns the Mg-relief prediction into a dose–response in units an experimenter manipulates
   (previously it could only be simulated by artificially forcing `sc_gate = 1`).

   **Dendritic, not somatic.** A literal Mg block is instantaneous, and an instantaneous *somatic*
   block slams shut during the inhibitory dip — historically that cost 0.918 → 0.762. It is
   applied instead to a low-passed dendritic voltage (`tau_dend`, formerly `tau_scg`), because
   dendritic NMDA plateau potentials are regenerative and outlast a brief somatic dip by tens to
   ~200 ms. This gives the low-pass a mechanism instead of leaving it a bare memory term.
   Fixed: `V_REST = −65 mV`, `K_V = 1.5 mV` per unit of `u_E`.

2. **Single-exponential SC** — dropped `tau_sc_off2`, `w_sc`.
   Hessian rank 17/24 and 13/24; split-half reliability **−0.04 and 0.02** (i.e. the GluN2B/GluN2A
   split does not replicate across halves of the *same cell*). The second component only ever
   bought 0.918 → 0.921. Removing it deletes an unsupported subunit claim; a single slow decay is
   still a standard NMDA EPSC.

3. **Dropped `reversal_potential`** — Hessian sensitivity exactly **0**, split-half reliability
   **−0.13**. It was only the hyperpolarisation floor clamp and `u_E` never reached it; replaced by
   a fixed `U_FLOOR = −100`.

4. **Dropped I_h** (`g_h`, `V_half_h`, `tau_h_on`, `tau_h_off`) — all in the sloppiest Hessian tier,
   split-half reliability ≤ 0.15 (`tau_h_off` = 0.03). Consistent with the long-standing finding
   that I_h sits near-idle. Whether this costs fit quality is *measured* (v3 vs v2 R²), not assumed.

## What did NOT change, and why

The **gate itself stays**, because it is what converts a spatially *untuned* global SC input into
a *tuned* output — the whole basis of the emergent-tuning model. Removing it destroys
direction-selectivity in 71/77 cells (`probe_pharmacology.py`). `fit_cell_v3.py --gate nogate`
refits every cell with the gate removed as the proper necessity control (PD-only R² can be
compensated by `A_sc`; the held-out anti-PD prediction is what exposes the gate's role).

## Known caveats carried forward

- **Sloppiness.** v2 spanned 16.5 decades between stiffest and softest direction. v3 should be
  re-measured with `probe_identifiability.py`.
- **`I_baseline` is a further cut candidate** — split-half reliability −0.08, and it is degenerate
  with `I_HD` (reliability 0.41) since both set tonic drive. v3 still fits it.
- **`tau_sc_off` reliability is only 0.16**, so even the single SC decay constant is weakly
  constrained per cell. Per-cell SC *kinetics* should not be over-interpreted; per-cell SC
  *amplitude* (`A_sc`, reliability 0.36, Hessian rank 6) is on firmer ground.
- **Conditional estimates.** Every cellular parameter is estimated *given* the population network
  background. They are not drop-in circuit components: rings built from them do not form stable
  bumps (`probe_heterogeneous_ring.py`), though that failure is driven by the *evoked* parameters —
  drive-only heterogeneity (`I_HD` + `I_baseline`) leaves the attractor stable.

## Reproducing

```bash
python network_background.py                 # drivers + exactness test (must pass ~1e-14)
python fit_cell_v3.py --gate mg     --all    # 18-param Mg-block fits, all 77 cells
python fit_cell_v3.py --gate nogate --all    # necessity control
python compile_v3_results.py                 # -> optimized_cells_v3_{gate}.parquet + .csv
```
