"""
readout_cell.py
---------------
Single heterogeneous readout neuron driven by a FIXED average-network background.

The ring stays at the population fit (the "average HD activity" the model was
validated against); only the PD cell is replaced by the cell we fit. Because
K_E's diagonal is 1/N, that cell contributes ~0.8% of the excitation it receives,
so "replace the PD cell" and "drive one cell with recorded network input" are the
same model -- and the decoupled form is ~120x faster.

Three recorded network drivers (from network_background.py) fully specify the
input: (K_E@r_E)_i(t), (K_I@r_I)_i(t), mean(r_E)(t), plus the global adaptation
state a(t). Everything else is local to the cell.

Param split (32 = 8 network + 24 cellular):
  network (frozen to the population fit): KAPPA_E, KAPPA_I, W_EI, tau_I, tau_a
      (shape the recorded presynaptic activity) + J1, W_IE, W_GLOBAL (the synaptic
      input gains -- the cell receives what an average cell receives)
  cellular (fit per cell): the 24 in CELL_PARAMS below.
"""

import numpy as np
from numba import njit

import optimization_engine as oe

DT = oe.DT
REC_HALF = oe.REC_HALF
K_H_SLOPE = oe.K_H_SLOPE
R_MAX = oe.R_MAX
GAIN_SLOPE = oe.GAIN_SLOPE
T_STIM = oe.T_STIM

# 24 cellular params, in the order simulate_readout expects
CELL_PARAMS = [
    "tau_E", "I_baseline", "I_HD", "reversal_potential",
    "A_fast", "fc_stim_duration", "stim_delay",
    "g_T", "V_half_T", "k_T", "tau_hT", "E_Ca",
    "g_h", "V_half_h", "tau_h_on", "tau_h_off",
    "g_a",
    "A_sc", "tau_sc_on", "tau_sc_off", "tau_sc_off2", "w_sc", "sc_delay", "tau_scg",
]
# 8 network params held at the population fit
NET_PARAMS = ["KAPPA_E", "KAPPA_I", "W_EI", "tau_I", "tau_a", "J1", "W_IE", "W_GLOBAL"]

CELL_BOUNDS = [dict(zip(oe.PARAM_NAMES, oe.JOINT_BOUNDS))[k] for k in CELL_PARAMS]


@njit(fastmath=True, cache=True)
def simulate_readout(p, time, kE_r, kI_rI, mean_rE, a_net,
                     J1, W_IE, W_GLOBAL, u_E0, hd_shape, gate_own):
    """Integrate ONE excitatory cell against the recorded network background.

    Mirrors the loop body of oe._simulate_stim for a single index, in the same
    order (forward Euler). `gate_own`: 0 -> SC gate from the network's recurrent
    drive (current mechanism); 1 -> from the cell's OWN depolarization (Mg-block).
    """
    (tau_E, I_baseline, I_HD, reversal_potential,
     A_fast, fc_stim_duration, stim_delay,
     g_T, V_half_T, k_T, tau_hT, E_Ca,
     g_h, V_half_h, tau_h_on, tau_h_off,
     g_a,
     A_sc, tau_sc_on, tau_sc_off, tau_sc_off2, w_sc, sc_delay, tau_scg) = p

    u_E = u_E0
    r_E = min(R_MAX, GAIN_SLOPE * max(0.0, u_E))
    h_T = 0.0
    m_h = 0.0
    sc_lp = 0.0

    I_ext_base = I_baseline + I_HD * hd_shape
    stim_onset = T_STIM + stim_delay
    stim_end = stim_onset + fc_stim_duration
    sc_onset = stim_onset + sc_delay

    out = np.zeros(len(time))
    for k in range(len(time)):
        t = time[k]

        rec_E = J1 * kE_r[k]
        rec_gate = rec_E / (rec_E + REC_HALF)

        # SC gate: 0 = inherited network drive, 1 = the cell's own depolarization
        # (Mg-block), 2 = NO gate (Mg-relief: voltage-independent, always open)
        gate_drive = max(0.0, u_E) if gate_own == 1 else rec_E
        sc_lp += (gate_drive - sc_lp) * (DT / tau_scg)
        sc_gate = 1.0 if gate_own == 2 else sc_lp / (sc_lp + REC_HALF)

        if t >= stim_onset:
            m_T = 1.0 / (1.0 + np.exp(-(u_E - V_half_T) / k_T))
            h_inf = 1.0 / (1.0 + np.exp((u_E - V_half_T) / k_T))
            h_T += (h_inf - h_T) * (DT / tau_hT)
            I_T = g_T * m_T * h_T * (E_Ca - u_E) * rec_gate

            mh_inf = 1.0 / (1.0 + np.exp((u_E - V_half_h) / K_H_SLOPE))
            tau_mh = tau_h_on if mh_inf > m_h else tau_h_off
            m_h += (mh_inf - m_h) * (DT / tau_mh)
            I_h = g_h * m_h
        else:
            I_T = 0.0
            I_h = 0.0

        I_ext = I_ext_base
        if stim_onset <= t < stim_end:
            I_ext = I_ext_base + A_fast
        if t >= sc_onset:
            dt_sc = t - sc_onset
            decay = w_sc * np.exp(-dt_sc / tau_sc_off) + (1.0 - w_sc) * np.exp(-dt_sc / tau_sc_off2)
            g_sc = decay * (1.0 - np.exp(-dt_sc / tau_sc_on))
            I_ext = I_ext + A_sc * g_sc * sc_gate

        I_adapt = 0.0 if t < stim_onset else g_a * a_net[k]

        du_E = (-u_E + rec_E - W_IE * kI_rI[k] - W_GLOBAL * mean_rE[k]
                + I_ext + I_T + I_h - I_adapt)
        u_E += du_E * (DT / tau_E)
        u_E = max(u_E, reversal_potential)
        r_E = min(R_MAX, GAIN_SLOPE * max(0.0, u_E))
        out[k] = r_E

    return out


def _weights(mt):
    """Time-window weighting (FC 5x, rebound 4x, tail 3x, decay 2x) -- same scheme
    as oe.loss_stage2."""
    w = np.where(mt <= 20.0, 5.0, 1.0)
    w = np.where((mt > 25.0) & (mt <= 75.0), 4.0, w)
    w = np.where((mt > 80.0) & (mt <= 200.0), 3.0, w)
    w = np.where((mt > 200.0) & (mt <= 700.0), 2.0, w)
    return w


def loss(p, drv, bins, pd_rate, pd_smooth, gate_own):
    """Weighted MSE to this cell's PD trace. No structural guards -- there is no
    ring to destroy; the network is fixed and clean by construction."""
    y = simulate_readout(np.asarray(p, dtype=np.float64), *drv, gate_own)
    if not np.all(np.isfinite(y)):
        return 1e6
    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mt = bins[m]
    if mt.size < 3:
        return 1e6
    mv = np.interp(mt, drv[0], y)
    target = np.where(mt <= 20.0, pd_rate[m], pd_smooth[m])  # raw FC, smooth tail
    val = np.average((mv - target) ** 2, weights=_weights(mt))
    return float(val) if np.isfinite(val) else 1e6


def r2(p, drv, bins, pd_rate, gate_own):
    y = simulate_readout(np.asarray(p, dtype=np.float64), *drv, gate_own)
    m = (bins >= oe.WIN_LO) & (bins <= oe.WIN_HI)
    mv = np.interp(bins[m], drv[0], y)
    ss = np.sum((pd_rate[m] - mv) ** 2)
    st = np.sum((pd_rate[m] - pd_rate[m].mean()) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")


_KNOCKOUTS = [
    ("-SC (A_sc=0)", ["A_sc"]),
    ("-adaptation (g_a=0)", ["g_a"]),
    ("-channels (g_T=g_h=0)", ["g_T", "g_h"]),
    ("-FC (A_fast=0)", ["A_fast"]),
]


def knockouts(p, drv, gate_own):
    """{label: trace} for the full model + each single-component knockout, so each
    component's contribution to the readout cell reads off directly."""
    idx = {k: i for i, k in enumerate(CELL_PARAMS)}
    p = np.asarray(p, dtype=np.float64)
    out = {"full": simulate_readout(p, *drv, gate_own)}
    for label, keys in _KNOCKOUTS:
        q = p.copy()
        for k in keys:
            q[idx[k]] = 0.0
        out[label] = simulate_readout(q, *drv, gate_own)
    return out


def load_drivers(path="network_drivers.npz", site="pd"):
    """-> (time, kE_r, kI_rI, mean_rE, a_net, J1, W_IE, W_GLOBAL, u_E0, hd_shape)
    ready to splat into simulate_readout. `site`: 'pd' or 'anti'."""
    d = np.load(path)
    return (d["time"], d[f"kE_r_{site}"], d[f"kI_rI_{site}"], d["mean_rE"], d["a_net"],
            float(d["J1"]), float(d["W_IE"]), float(d["W_GLOBAL"]),
            float(d[f"u_E0_{site}"]), float(d[f"hd_shape_{site}"]))
