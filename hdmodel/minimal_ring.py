"""
minimal_ring.py
---------------
The most basic HD ring attractor that can be asked the supervisor's question:

    "The SC input is ubiquitously distributed around the attractor; anti-PD shows no
     SC because it stays sub-threshold. The SC is gated by inhibition."

So this model has NOTHING but a ring, an inhibitory population, and two spatially
UNIFORM inputs. Deliberately absent: I_T, I_h, Mg block, NMDA bi-exponential kinetics,
adaptation, and any bump-membership gate. `sc_gate` is identically 1 everywhere.

Standalone on purpose -- it does not import optimization_engine. The point is a model
you can read in one sitting.

    tau_E du_E/dt = -u_E + J1(K_E@r_E) - W_IE(K_I@r_I) - W_GLOBAL*mean(r_E)
                    + I_baseline + I_HD*HD_SHAPE + FC(t) + SC(t)
    tau_I du_I/dt = -u_I + W_EI(K_E@r_E)
    r_E = clip(GAIN_SLOPE*[u_E]+, 0, R_MAX)      r_I = [u_I]+

HOW A UNIFORM INPUT CAN STILL PRODUCE SELECTIVITY (the actual mechanism):

Subtracting the anti-PD equation from the PD one:

    d(u_PD - u_anti)/dt = -(u_PD - u_anti)
                          + I_HD*dHD + J1*d(K_E@r_E) - W_IE*d(K_I@r_I)

Every spatially uniform term -- I_baseline, W_GLOBAL*mean(r_E), the FC, and the global SC
input -- drops out EXPLICITLY: it enters both equations identically. So a uniform input
contributes nothing to the differential *directly*.

But it does act INDIRECTLY, and this is the crux. The recurrent term J1*(K_E@r_E) depends
on the rates, which the uniform input raises. Because that term is TUNED (much larger on
the bump than at the antipode), lifting everybody raises PD's recurrent drive far more
than anti-PD's -- so the differential GROWS. A uniform SC can therefore buy selectivity
via recurrent amplification. Whether it buys ENOUGH is a quantitative question, which is
what this model exists to answer.

The bar: anti-PD stays silent iff u_anti < 0, i.e. iff the differential exceeds u_PD. With
GAIN_SLOPE=6 the AD baseline (24.1 Hz) sits at u ~ 4.0 and an 89 Hz SC needs u_PD ~ 14.8,
so the differential must roughly QUADRUPLE. I_HD is fixed, so that growth has to come from
recurrence -- pushing toward the regime where a linear-gain ring destabilises. Marginal on
purpose: simulate, don't argue.

`check()` verifies both halves of this numerically; run `python minimal_ring.py`.
"""

import numpy as np
from numba import njit

# ── fixed geometry / scales (NOT fitted) ─────────────────────────────────────
N = 120
DT = 0.2                     # ms, Euler step
GAIN_SLOPE = 6.0             # Hz per unit of u
R_MAX = 1000.0               # safety clip, never reached in normal operation
KAPPA_HD = 2.0               # width of the tonic HD drive (~70-90 deg FWHM)

THETA = -np.pi + 2.0 * np.pi * np.arange(N) / N
_D = THETA[:, None] - THETA[None, :]
COS_D = np.cos(_D)
HD_SHAPE = np.exp(KAPPA_HD * (np.cos(THETA) - 1.0))

IDX_PD = int(np.argmin(np.abs(THETA)))
IDX_ANTI = int(np.argmin(np.abs(THETA - np.pi)))

T_STIM = 0.0                 # stimulus at t = 0
T_START, T_END = -350.0, 750.0   # burn-in from -350; report to +750
TIME = np.arange(T_START, T_END, DT)

PARAM_NAMES = [
    "tau_E", "tau_I", "I_baseline", "J1", "KAPPA_E", "W_IE", "KAPPA_I", "W_EI",
    "I_HD", "W_GLOBAL",
    "A_fast", "fc_stim_duration", "stim_delay",
    "A_sc", "tau_sc_on", "tau_sc_off", "sc_delay",
]

BOUNDS = [
    (5.0, 40.0),      # tau_E
    (2.0, 30.0),      # tau_I
    (0.0, 20.0),      # I_baseline
    (0.1, 40.0),      # J1        (recurrent excitation -- the only way to grow the
                      #            tuned differential, so give it a wide ceiling)
    (2.0, 30.0),      # KAPPA_E   (narrow excitation)
    (0.0, 40.0),      # W_IE
    (0.1, 8.0),       # KAPPA_I   (broad inhibition; small kappa = near-uniform)
    (0.0, 20.0),      # W_EI
    (0.0, 40.0),      # I_HD
    (0.0, 20.0),      # W_GLOBAL
    (1.0, 1500.0),    # A_fast
    (1.0, 30.0),      # fc_stim_duration (ms)
    (0.0, 25.0),      # stim_delay (ms)
    (0.0, 400.0),     # A_sc
    (2.0, 60.0),      # tau_sc_on (ms)
    (20.0, 800.0),    # tau_sc_off (ms)
    (0.0, 80.0),      # sc_delay (ms)
]

# inhibition architectures -- explorable individually and together
INH_RING = "ring"      # interneuron ring only  (W_GLOBAL forced to 0)
INH_GLOBAL = "global"  # uniform pool only      (W_IE = W_EI forced to 0)
INH_BOTH = "both"      # both active
INH_MODES = (INH_RING, INH_GLOBAL, INH_BOTH)


def apply_inh_mode(p, inh_mode):
    """Zero the parameters that a given inhibition architecture switches off."""
    p = np.array(p, dtype=float)
    if inh_mode == INH_RING:
        p[PARAM_NAMES.index("W_GLOBAL")] = 0.0
    elif inh_mode == INH_GLOBAL:
        p[PARAM_NAMES.index("W_IE")] = 0.0
        p[PARAM_NAMES.index("W_EI")] = 0.0
    elif inh_mode != INH_BOTH:
        raise ValueError(f"unknown inh_mode {inh_mode!r}; expected one of {INH_MODES}")
    return p


def _gain(u):
    return np.clip(GAIN_SLOPE * np.maximum(u, 0.0), 0.0, R_MAX)


@njit(cache=True)
def _core(tau_E, tau_I, I_baseline, J1, W_IE, W_EI, I_HD, W_GLOBAL,
          A_fast, fc_dur, stim_delay, A_sc, tau_sc_on, tau_sc_off, sc_delay,
          K_E, K_I, hd_shape, time, u_E, with_fc, with_sc,
          gain_slope, r_max, dt, t_stim):
    """Hot loop, njit'd: 5500 steps of Python-level numpy was ~99 ms per run."""
    n = u_E.shape[0]
    u_I = W_EI * (K_E @ np.minimum(np.maximum(gain_slope * u_E, 0.0), r_max))
    rates = np.zeros((time.shape[0], n))
    us = np.zeros((time.shape[0], n))
    fc_on = t_stim + stim_delay
    fc_off = fc_on + fc_dur
    sc_on = t_stim + stim_delay + sc_delay

    for step in range(time.shape[0]):
        t = time[step]
        r_E = np.minimum(np.maximum(gain_slope * u_E, 0.0), r_max)
        r_I = np.maximum(u_I, 0.0)

        I_ext = I_baseline + I_HD * hd_shape
        if with_fc and fc_on <= t < fc_off:
            I_ext = I_ext + A_fast          # UNIFORM: every cell equally
        if with_sc and t >= sc_on:
            d = t - sc_on
            I_ext = I_ext + A_sc * (1.0 - np.exp(-d / tau_sc_on)) * np.exp(-d / tau_sc_off)
            #       ^ UNIFORM too: the supervisor's hypothesis, no gate anywhere

        du_E = (-u_E + J1 * (K_E @ r_E) - W_IE * (K_I @ r_I)
                - W_GLOBAL * (r_E.sum() / n) + I_ext)
        du_I = -u_I + W_EI * (K_E @ r_E)
        u_E = u_E + du_E * (dt / tau_E)
        u_I = u_I + du_I * (dt / tau_I)

        finite = True
        for i in range(n):
            if not np.isfinite(u_E[i]):
                finite = False
                break
        if not finite:
            for s in range(step, time.shape[0]):
                for i in range(n):
                    rates[s, i] = np.nan
                    us[s, i] = np.nan
            return rates, us

        for i in range(n):
            v = gain_slope * u_E[i]
            if v < 0.0:
                v = 0.0
            elif v > r_max:
                v = r_max
            rates[step, i] = v
            us[step, i] = u_E[i]
    return rates, us


def simulate(p, inh_mode=INH_BOTH, with_fc=True, with_sc=True, time=TIME,
             return_u=False):
    """Run the ring. Returns (time, rates), or (time, rates, u) if return_u.

    `u` is the pre-rectification activation -- needed to test the cancellation argument,
    since the rate is rectified at 0 and clipped at R_MAX and so hides it.
    """
    (tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD, W_GLOBAL,
     A_fast, fc_stim_duration, stim_delay, A_sc, tau_sc_on, tau_sc_off,
     sc_delay) = apply_inh_mode(p, inh_mode)

    K_E = np.ascontiguousarray(np.exp(KAPPA_E * (COS_D - 1.0)) / N)
    K_I = np.ascontiguousarray(np.exp(KAPPA_I * (COS_D - 1.0)) / N)
    u_E = 20.0 * np.exp(KAPPA_E * (np.cos(THETA) - 1.0))   # seed a bump at PD

    rates, us = _core(tau_E, tau_I, I_baseline, J1, W_IE, W_EI, I_HD, W_GLOBAL,
                      A_fast, fc_stim_duration, stim_delay, A_sc, tau_sc_on, tau_sc_off,
                      sc_delay, K_E, K_I, HD_SHAPE, time, u_E, with_fc, with_sc,
                      GAIN_SLOPE, R_MAX, DT, T_STIM)
    return (time, rates, us) if return_u else (time, rates)


# ── measurements ─────────────────────────────────────────────────────────────
def metrics(t, rates):
    """Everything the feasibility test needs, in one pass."""
    pd_, anti = rates[:, IDX_PD], rates[:, IDX_ANTI]
    pre = t < T_STIM
    sc_win = (t > 40.0) & (t < 300.0)
    fc_win = (t >= 0.0) & (t <= 40.0)
    idle = rates[pre][-1] if pre.any() else rates[0]
    pk = float(idle.max())
    fwhm = float((idle >= pk / 2).sum()) * 360.0 / N if pk > 0 else 0.0
    return {
        "ok": bool(np.all(np.isfinite(rates))),
        # max over the whole run: R_MAX is a SAFETY CLIP, so a diverged solution looks
        # "finite" once clipped. Anything at the clip is a runaway, not a fit.
        "max_rate": float(np.nanmax(rates)),
        "base_pd": float(idle[IDX_PD]), "base_anti": float(idle[IDX_ANTI]),
        "base_peak": pk, "base_fwhm": fwhm,
        "pd_sc": float(pd_[sc_win].max()) if sc_win.any() else np.nan,
        "anti_sc": float(anti[sc_win].max()) if sc_win.any() else np.nan,
        "pd_fc": float(pd_[fc_win].max()) if fc_win.any() else np.nan,
        "anti_fc": float(anti[fc_win].max()) if fc_win.any() else np.nan,
        "pd_600": float(pd_[np.argmin(np.abs(t - 600.0))]),
    }


DEFAULTS = dict(zip(PARAM_NAMES, [
    15.0, 8.0, 1.0, 2.0, 12.0, 2.0, 1.0, 4.0, 4.0, 0.5,
    200.0, 8.0, 8.0, 60.0, 15.0, 200.0, 5.0]))


def check():
    """Sanity checks. These validate the ARGUMENT, not just the code."""
    x = [DEFAULTS[k] for k in PARAM_NAMES]
    i_base = PARAM_NAMES.index("I_baseline")

    print("1a. uniform terms cancel EXACTLY when recurrence is off")
    print("    (J1=0, W_IE=0 -> no tuned feedback; +3 to I_baseline must not move u_PD-u_anti)")
    flat = list(x)
    flat[PARAM_NAMES.index("J1")] = 0.0
    flat[PARAM_NAMES.index("W_IE")] = 0.0
    for mode in INH_MODES:
        _, _, u0 = simulate(flat, mode, return_u=True)
        y = list(flat); y[i_base] += 3.0
        _, _, u1 = simulate(y, mode, return_u=True)
        d0 = u0[:, IDX_PD] - u0[:, IDX_ANTI]
        d1 = u1[:, IDX_PD] - u1[:, IDX_ANTI]
        print(f"    {mode:>6}: max |change| = {np.nanmax(np.abs(d1 - d0)):.3e} u  (expect ~0)")

    print("\n1b. with recurrence ON the SAME uniform input DOES move the differential")
    print("    (this is the indirect amplification -- how a uniform SC can buy selectivity)")
    for mode in INH_MODES:
        _, _, u0 = simulate(x, mode, return_u=True)
        y = list(x); y[i_base] += 3.0
        _, _, u1 = simulate(y, mode, return_u=True)
        d0 = u0[:, IDX_PD] - u0[:, IDX_ANTI]
        d1 = u1[:, IDX_PD] - u1[:, IDX_ANTI]
        print(f"    {mode:>6}: differential {np.nanmax(d0):6.2f} -> {np.nanmax(d1):6.2f} u "
              f"(change {np.nanmax(d1) - np.nanmax(d0):+.2f})")

    print("\n2. baseline bump, no stimulus  (target: peak ~24 Hz, anti ~0, FWHM 45-95 deg)")
    for mode in INH_MODES:
        t, r = simulate(x, mode, with_fc=False, with_sc=False)
        m = metrics(t, r)
        print(f"   {mode:>6}: peak {m['base_peak']:6.1f} Hz  FWHM {m['base_fwhm']:5.1f} deg"
              f"  anti {m['base_anti']:5.1f} Hz")

    print("\n3. FC must reach anti-PD  (external constraint: the antipode DOES show the FC)")
    for mode in INH_MODES:
        t, r = simulate(x, mode, with_sc=False)
        m = metrics(t, r)
        print(f"   {mode:>6}: PD FC {m['pd_fc']:6.1f} Hz   anti-PD FC {m['anti_fc']:6.1f} Hz")

    print("\n4. with the uniform SC as well  (data: PD SC ~89 Hz, anti-PD SC ~0)")
    for mode in INH_MODES:
        t, r = simulate(x, mode)
        m = metrics(t, r)
        print(f"   {mode:>6}: PD SC {m['pd_sc']:6.1f} Hz   anti-PD SC {m['anti_sc']:6.1f} Hz")


if __name__ == "__main__":
    check()
