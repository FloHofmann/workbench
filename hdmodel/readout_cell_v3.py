"""
readout_cell_v3.py
------------------
v3 of the heterogeneous readout cell. Self-contained: v2 (readout_cell.py) is left
untouched so both can be fitted and compared.

Changes from v2, each driven by a probe result:

  1. JAHR-STEVENS Mg BLOCK replaces the Michaelis gate.
         B(V) = 1 / (1 + [Mg]/3.57 * exp(-0.062 V))
     The Mg2+ pore block IS the NMDA receptor's own voltage gate, so this makes the
     model's gating literally NMDA biophysics instead of a proxy. Crucially [Mg2+]
     becomes an explicit parameter, so the Mg-relief experiment is a dose-response
     in units an experimenter manipulates -- not an artificial "force gate = 1".

     The block is instantaneous, but an instantaneous SOMATIC block slams shut during
     the inhibitory dip (v2 history: that cost 0.918 -> 0.762). It is applied instead
     to a low-passed DENDRITIC voltage: dendritic NMDA plateau potentials are
     regenerative and outlast a brief somatic dip by tens to ~200 ms. tau_dend (v2's
     tau_scg) is therefore a dendritic low-pass with a mechanism, not a bare memory term.

  2. SINGLE-EXPONENTIAL SC (drop tau_sc_off2, w_sc). Both were sloppy
     (rank 17 and 13 of 24) and the second component only bought 0.918 -> 0.921.
     Removes an unsupported GluN2B/GluN2A subunit claim; a single slow decay is still
     a standard NMDA EPSC.

  3. DROP reversal_potential (Hessian sensitivity exactly 0 -- inert). The
     hyperpolarisation floor is kept as a fixed constant for numerical safety.

  4. DROP I_h (g_h, V_half_h, tau_h_on, tau_h_off): all four in the sloppiest tier,
     consistent with the long-standing finding that I_h sits near-idle. Whether this
     costs fit quality is measured, not assumed (compare v3 vs v2 R2).

24 -> 18 cellular params. Network background (network_drivers.npz) is unchanged and
shared with v2 -- it describes the population ring, not the readout cell.

gate_mode: 0 = Mg block (default), 1 = NO gate (sc_gate == 1, the necessity control).
"""

import numpy as np
from numba import njit

import optimization_engine as oe

DT = oe.DT
R_MAX = oe.R_MAX
GAIN_SLOPE = oe.GAIN_SLOPE
T_STIM = oe.T_STIM

# --- fixed constants (not fitted) ---
V_REST = -65.0        # mV, resting potential the activation variable rides on
K_V = 1.5             # mV per unit of u_E (maps the model's drive variable to mV)
MG_K = 3.57           # Jahr & Stevens (1990) constants
MG_SLOPE = 0.062
U_FLOOR = -100.0      # hyperpolarisation clamp (was the inert `reversal_potential`)

CELL_PARAMS = [
    "tau_E", "I_baseline", "I_HD",
    "A_fast", "fc_stim_duration", "stim_delay",
    "g_T", "V_half_T", "k_T", "tau_hT", "E_Ca",
    "g_a",
    "A_sc", "tau_sc_on", "tau_sc_off", "sc_delay",
    "tau_dend", "mg_conc",
]

_V2B = dict(zip(oe.PARAM_NAMES, oe.JOINT_BOUNDS))
CELL_BOUNDS = [
    _V2B["tau_E"], _V2B["I_baseline"], _V2B["I_HD"],
    _V2B["A_fast"], _V2B["fc_stim_duration"], _V2B["stim_delay"],
    _V2B["g_T"], _V2B["V_half_T"], _V2B["k_T"], _V2B["tau_hT"], _V2B["E_Ca"],
    _V2B["g_a"],
    _V2B["A_sc"], _V2B["tau_sc_on"], _V2B["tau_sc_off"], _V2B["sc_delay"],
    (5.0, 150.0),      # tau_dend  (dendritic low-pass; v2's tau_scg range)
    (0.05, 5.0),       # mg_conc   (mM; physiological ~1.0)
]


@njit(fastmath=True, cache=True)
def simulate_readout_v3(p, time, kE_r, kI_rI, mean_rE, a_net,
                        J1, W_IE, W_GLOBAL, u_E0, hd_shape, gate_mode):
    (tau_E, I_baseline, I_HD,
     A_fast, fc_stim_duration, stim_delay,
     g_T, V_half_T, k_T, tau_hT, E_Ca,
     g_a,
     A_sc, tau_sc_on, tau_sc_off, sc_delay,
     tau_dend, mg_conc) = p

    u_E = u_E0
    h_T = 0.0
    u_dend = u_E0                      # dendritic voltage starts at the somatic value

    I_ext_base = I_baseline + I_HD * hd_shape
    stim_onset = T_STIM + stim_delay
    stim_end = stim_onset + fc_stim_duration
    sc_onset = stim_onset + sc_delay

    out = np.zeros(len(time))
    for k in range(len(time)):
        t = time[k]
        rec_E = J1 * kE_r[k]
        rec_gate = rec_E / (rec_E + oe.REC_HALF)

        # dendritic voltage: low-passed soma (NMDA plateau bridges the somatic dip)
        u_dend += (u_E - u_dend) * (DT / tau_dend)
        if gate_mode == 1:
            sc_gate = 1.0                                   # no gate (control)
        else:
            V = V_REST + K_V * u_dend                       # -> mV
            sc_gate = 1.0 / (1.0 + (mg_conc / MG_K) * np.exp(-MG_SLOPE * V))

        if t >= stim_onset:
            m_T = 1.0 / (1.0 + np.exp(-(u_E - V_half_T) / k_T))
            h_inf = 1.0 / (1.0 + np.exp((u_E - V_half_T) / k_T))
            h_T += (h_inf - h_T) * (DT / tau_hT)
            I_T = g_T * m_T * h_T * (E_Ca - u_E) * rec_gate
        else:
            I_T = 0.0

        I_ext = I_ext_base
        if stim_onset <= t < stim_end:
            I_ext = I_ext_base + A_fast
        if t >= sc_onset:
            dt_sc = t - sc_onset
            g_sc = np.exp(-dt_sc / tau_sc_off) * (1.0 - np.exp(-dt_sc / tau_sc_on))
            I_ext = I_ext + A_sc * g_sc * sc_gate

        I_adapt = 0.0 if t < stim_onset else g_a * a_net[k]

        du_E = (-u_E + rec_E - W_IE * kI_rI[k] - W_GLOBAL * mean_rE[k]
                + I_ext + I_T - I_adapt)
        u_E += du_E * (DT / tau_E)
        if u_E < U_FLOOR:
            u_E = U_FLOOR
        out[k] = min(R_MAX, GAIN_SLOPE * max(0.0, u_E))
    return out


def _weights(mt):
    w = np.where(mt <= 20.0, 5.0, 1.0)
    w = np.where((mt > 25.0) & (mt <= 75.0), 4.0, w)
    w = np.where((mt > 80.0) & (mt <= 200.0), 3.0, w)
    w = np.where((mt > 200.0) & (mt <= 700.0), 2.0, w)
    return w


def loss(p, drv, bins, pd_rate, pd_smooth, gate_mode):
    y = simulate_readout_v3(np.asarray(p, dtype=np.float64), *drv, gate_mode)
    if not np.all(np.isfinite(y)):
        return 1e6
    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mt = bins[m]
    if mt.size < 3:
        return 1e6
    mv = np.interp(mt, drv[0], y)
    target = np.where(mt <= 20.0, pd_rate[m], pd_smooth[m])
    val = np.average((mv - target) ** 2, weights=_weights(mt))
    return float(val) if np.isfinite(val) else 1e6


def r2(p, drv, bins, pd_rate, gate_mode):
    y = simulate_readout_v3(np.asarray(p, dtype=np.float64), *drv, gate_mode)
    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[m], drv[0], y)
    ss = np.sum((pd_rate[m] - mv) ** 2)
    st = np.sum((pd_rate[m] - pd_rate[m].mean()) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")


def seed_from_v2(pop):
    """Warm start: population values for the shared params, sensible defaults for
    the two new ones (physiological [Mg2+] ~1 mM; v2's tau_scg as the dendritic tau)."""
    d = dict(pop)
    d["tau_dend"] = pop.get("tau_scg", 30.0)
    d["mg_conc"] = 1.0
    return np.array([d[k] for k in CELL_PARAMS], dtype=np.float64)


_KNOCKOUTS = [
    ("-SC (A_sc=0)", ["A_sc"]),
    ("-adaptation (g_a=0)", ["g_a"]),
    ("-I_T (g_T=0)", ["g_T"]),
    ("-FC (A_fast=0)", ["A_fast"]),
]


def knockouts(p, drv, gate_mode):
    idx = {k: i for i, k in enumerate(CELL_PARAMS)}
    p = np.asarray(p, dtype=np.float64)
    out = {"full": simulate_readout_v3(p, *drv, gate_mode)}
    for label, keys in _KNOCKOUTS:
        q = p.copy()
        for k in keys:
            q[idx[k]] = 0.0
        out[label] = simulate_readout_v3(q, *drv, gate_mode)
    return out


def load_drivers(path="network_drivers.npz", site="pd"):
    """Shared with v2 -- the drivers describe the population ring, not the cell."""
    d = np.load(path)
    return (d["time"], d[f"kE_r_{site}"], d[f"kI_rI_{site}"], d["mean_rE"], d["a_net"],
            float(d["J1"]), float(d["W_IE"]), float(d["W_GLOBAL"]),
            float(d[f"u_E0_{site}"]), float(d[f"hd_shape_{site}"]))
