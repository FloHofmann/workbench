# Minimal model

The simplest version that still demonstrates the effect. Every free parameter earns its
place by an ablation cost against four criteria (see [PARAMS.md](PARAMS.md)); everything
else is a shared constant or removed. Nothing from v1/v2/v3 is overwritten
(`readout_cell_v3.py` is reused unchanged — the minimal model is that simulator with most
parameters pinned).

## What it is

- **Network:** the population ring at its published fit, recorded once as a fixed
  background (`network_drivers.npz`). Not re-fit per cell.
- **Cell:** one readout neuron with **7 free parameters**, the rest shared constants.

| free (per cell) | what it is |
|---|---|
| `I_HD` | baseline bump height (the cell's tonic firing level) |
| `A_fast` | fast-component spike amplitude |
| `A_sc` | slow-component (NMDA) drive amplitude |
| `g_a` | adaptation strength → recovery to baseline |
| `g_T` | T-type Ca conductance → shapes the SC |
| `V_half_T`, `k_T` | where/how sharply I_T activates (kept free because sharing them leaks the SC to anti-PD) |

Shared constants (10): `mg_conc`, `tau_sc_on`, `tau_sc_off`, `sc_delay`, `tau_dend`,
`fc_stim_duration`, `stim_delay`, `tau_E`, `tau_hT`, `E_Ca`.
Cut: `I_baseline` (degenerate with `I_HD`), I_h (×4), `tau_sc_off2`, `w_sc`,
`reversal_potential`.

## The effect, and what it costs

The central claim is **emergent direction-selectivity**: a spatially *untuned* global SC
input, gated by the NMDA Mg²⁺ block, produces a slow component only at the preferred
direction. The minimal model keeps it intact.

| criterion | minimal (7 free) | v3 (18 free) | data |
|---|---|---|---|
| 1 — direction-selectivity (cells leaking anti-PD SC >5 Hz) | **12/77** | 12/77 | 0/77 |
| 2 — response shape, R² median | **0.577** | 0.692\* | ceiling **0.615** |
| 3 — recovery (rate@600ms / baseline) | 1.40× | – | 1.52× |
| 4 — heterogeneity (all 7 free params vary, CV) | 0.24–1.12 | – | – |
| self-consistency (mean of 77 fits vs mean data, R²) | 0.899 | 0.966 | – |

\* v3's 0.692 is *above* the noise ceiling — i.e. partly fitting trial-to-trial noise
(its out-of-sample R² was 0.528). The minimal model's 0.577 is **94% of the 0.615
ceiling**, the most any model can reach on this data. So going 18 → 7 free parameters
gives up almost no *reproducible* signal while making the model sayable in one breath.

**Honest cost:** the self-consistency drops 0.966 → 0.899 (fewer free knobs reproduce the
population mean slightly less perfectly), and 12/77 cells still show a small anti-PD leak
(unchanged from v3 — this is a model limit, not a cost of minimizing).

## One deliberate judgment call

`stim_delay` (conduction latency) and `fc_stim_duration` (FC width) *do* vary reliably per
cell (split-half r 0.58, 0.52). They are shared here because sharing them improves
generalization and they are not part of the selectivity effect. If per-cell timing is
itself of interest, free them — they are the first two to add back.

## Reproduce

```bash
python network_background.py        # fixed network background + exactness test
python fit_cell_min.py --all        # 7-free fits, 77 cells (~4 min)
python compile_min_results.py       # -> optimized_cells_min.parquet + 4-criteria summary
```

Files: `fit_cell_min.py`, `compile_min_results.py`, `optimized_cells_min.{parquet,csv}`,
`results_min/`. Dynamics: `readout_cell_v3.py` (shared).
