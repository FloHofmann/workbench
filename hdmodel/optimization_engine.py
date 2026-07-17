import numpy as np
from numba import njit

# Constants
N = 120  # 3 degrees per cell
THETA = np.linspace(-np.pi, np.pi, N, endpoint=False)

DT = 0.2  # temporal resolution in ms
T_BURN_IN = -300  # 300 ms for the attractor to settle
T_END = 800
TIME = np.arange(T_BURN_IN, T_END, DT)
T_STIM = 0  # timepoint of the stimulation (aligned with loss masks)

# Create a Matrix of Delta Theta with 1 being identity and -1 being theta - pi
COS_D_THETA = np.cos(THETA[:, None] - THETA[None, :])

# Upstream HD input (LMN/DTN -> AD). AD thalamus does not generate the HD signal;
# it inherits a tonic, head-direction-tuned drive from upstream. Shape is a fixed
# von-Mises bump at PD (0 deg); its amplitude I_HD is FITTED (Stage 1). Because it
# is tuned (peaks at PD, ~0 at the antipode) it can be strong without the +-130 deg
# secondary bumps / anti-PD firing that capped I_baseline -> it supplies bump height
# AND pins the attractor so it recovers to baseline after the stimulus.
KAPPA_HD = 2.0  # fixed HD tuning width (~70-90 deg FWHM)
HD_SHAPE = np.exp(KAPPA_HD * (np.cos(THETA) - 1.0))

# I_h (HCN) activation slope. Fixed (not optimized) to keep the Stage-2 param
# count tractable; only its midpoint V_half_h and gain g_h are free.
K_H_SLOPE = 8.0

# Half-saturation of the recurrent gate on I_T (in units of recurrent excitation
# J1*(K_E@r_E)). The rebound only expresses where a cell receives recurrent drive
# (= it is part of the bump). Off-bump cells de-inactivate (h_T high from the
# crash) but get ~0 recurrent input -> no I_T -> no rebound -> they stay silent,
# so the rebound is spatially CONFINED to the bump and the attractor re-converges
# instead of dissolving/broadening from a global rebound.
REC_HALF = 10.0

# Global (untuned) inhibition proportional to TOTAL excitatory activity is a FITTED
# Stage-1 param (W_GLOBAL). The broad attractor state has MORE total activity than
# the sharp one, so global inhibition destabilizes it -> the ring becomes monostable
# (single sharp bump) and the post-stim rebound relaxes back to baseline width.
# Biophysically a broadly-projecting / TRN feedback inhibition.

# THRESHOLD-LINEAR gain: r = clip(GAIN_SLOPE*[u]+, 0, R_MAX). We tested a supralinear
# (Naka-Rushton / SSN) gain to force a monostable ring, but with the SC delivered as
# an INHERITED input (not recurrently generated) and the bump WIDTH recovered by the
# global adaptation current, the winner-take-all is unnecessary AND slightly hurts the
# THRESHOLD-LINEAR gain: r = clip(GAIN_SLOPE*[u]+, 0, R_MAX). The SC is a GLOBAL
# (untuned) input gated by bump-membership (rec_gate) -- the GATE alone silences the
# antipode (SC input ~0 where rec_E ~0), so no supralinear winner-take-all is needed.
# A supralinear gain was tried for the gating but it makes the low-rate late state
# FRAGILE (the bump collapses to 0 at ~700 ms); the linear gain recovers stably.
R_MAX = 1000.0      # rate clip (safety; evoked range 0-200 Hz)
GAIN_SLOPE = 6.0    # f-I slope; baseline operating point u ~ 6-7 -> ~40 Hz

# Stage-specific (short) time grids. Each eval only needs the window the loss reads.
TIME_STAGE1 = np.arange(T_BURN_IN, 50.0, DT)   # steady state + (-150..-50) variance window
TIME_STAGE2 = np.arange(T_BURN_IN, 750.0, DT)  # burn-in + stim + full RECOVERY window (->700)

# Tuned interneuron ring (replaces the global-uniform u_I + MEXICAN_FRAC hack).
# Excitation kernel exp(KAPPA_E*(cos-1)) is NARROW; inhibition kernel
# exp(KAPPA_I*(cos-1)) is BROAD (KAPPA_I < KAPPA_E). Narrow excite minus broad
# inhibit = a genuine Mexican-hat from real connectivity -> single bump + lateral
# competition, no artificial DC subtraction. Inhibition is no longer spatially
# flat, so anti-PD suppression during the rebound is a real network effect.
STAGE1_BOUNDS = [
    (5.0, 50.0),   # tau_E (excitatory time constant)
    (2.0, 20.0),   # tau_I (interneuron time constant)
    (3.0, 15.0),   # I_baseline (capped LOW: uniform drive lifts ALL cells -> if it
                   #   sets the bump height it also drives distal cells over thresh
                   #   = the +-130 deg secondary bumps. Capping it forces the 40 Hz
                   #   peak to come from RECURRENT J1 (localized) instead, and keeps
                   #   far cells subthreshold so the surround needs only gentle
                   #   inhibition -- which the post-dip rebound also needs.)
    (1.0, 50.0),   # J1 (recurrent excitation amplitude)
    (3.0, 30.0),   # KAPPA_E (excitation width; larger = narrower)
    (2.0, 20.0),   # W_IE (inhibition amp, I->E). One knob serves baseline shape,
                   #   the evoked dip (needs feedback inhibition to crash), the
                   #   rebound shape, AND anti-PD silence -- these conflict, so it
                   #   is optimized JOINTLY with the stim params (loss_joint), where
                   #   the data MSE penalizes the late-spike regime directly.
    (0.9, 1.3),    # KAPPA_I (inhibition width; SMALL = broad). Pinned to the
                   #   SHARPENING band: too broad (~0.5, near-global) gives no
                   #   spatial surround -> the post-stim bump gets stuck WIDE (a
                   #   stable broad attractor, FWHM ~213 deg, never recovers); too
                   #   sharp (>~1.3) breaks anti-PD selectivity. ~1.0 gives a real
                   #   Mexican-hat surround that re-narrows the bump to ~baseline
                   #   while keeping anti-PD ~0 and no +-130 deg secondary bumps.
    (1.0, 15.0),   # W_EI (E->I drive gain)
    (0.0, 25.0),   # I_HD (upstream HD drive amplitude; tuned at PD -> sets bump
                   #   height + pins recovery without secondary bumps)
    (0.0, 12.0),   # W_GLOBAL (global inhibition ~ total activity -> monostable:
                   #   destabilizes the broad post-stim state so the bump recovers)
]

# Stage 2 fits two intrinsic thalamic currents whose interaction *generates* the
# slow component (SC) rebound (no injected SC):
#   I_T (T-type Ca): m_T (fast activation) * h_T (slow inactivation) * (E_Ca - u_E)
#                    -> sharp post-inhibitory rebound (~30-50 ms onset).
#   I_h (HCN):       additive g_h * m_h, m_h slow -> long depolarizing tail.
# FC (A_fast) is kept a BRIEF flash so the input cannot explain the 30-300 ms SC.
# stim_delay = sensory conduction latency: the in-vivo FC starts ~10 ms after the
# speaker fires, so the model's flash (and everything downstream) is shifted by it
# to align with the data; without it the sharp FC spike is mis-registered.
STAGE2_BOUNDS = [
    (100.0, 800.0),   # A_fast (FC input drive). Capped LOW: a huge flash saturates
                      #   all rates -> uniform -> the recurrent bump is erased ->
                      #   the rebound goes global (anti-PD fires). Keeping A_fast
                      #   comparable to recurrent gain preserves PD's advantage
                      #   through the flash -> PD-selective rebound.
    (2.0, 14.0),      # fc_stim_duration. Widened: at 5 ms the FC was a sharp spike
                      #   that crashed by ~13 ms; the data FC rises to a peak at
                      #   ~18 ms. Still <<the 30-300 ms SC, so the input cannot
                      #   explain the SC (emergence argument intact).
    (3.0, 12.0),      # stim_delay (conduction latency, ms; data FC at ~8-10 ms)
    (-50.0, -20.0),   # reversal_potential (hyperpolarization clamp floor)
    # --- I_T (T-type Ca) : sharp post-inhibitory rebound ---
    (5.0, 80.0),      # g_T (T-type conductance)
    (-5.0, 10.0),     # V_half_T (gate midpoint; LOW so I_T activates as u_E
                      #   recovers and DRIVES the rebound, not just boosts it late)
    (3.0, 15.0),      # k_T (gate slope)
    (8.0, 35.0),      # tau_hT (de-inactivation tc; capped FAST so h_T charges
                      #   during the brief dip -> rebound fires right after it, not
                      #   ~40 ms later. Real T-type kinetics are fast.)
    (80.0, 200.0),    # E_Ca (Ca reversal; high + so I_T is depolarizing, no latch)
    # --- I_h (HCN) : slow depolarizing tail (additive, no latch) ---
    (2.0, 60.0),      # g_h (HCN conductance; raised so I_h can lift the tail ~+50)
    (-30.0, 10.0),    # V_half_h (activation midpoint)
    (3.0, 15.0),      # tau_h_on  (FAST HCN activation -> the gate charges ~fully
                      #   during the brief deep dip, not just ~40%)
    (200.0, 900.0),   # tau_h_off (SLOW HCN deactivation -> the elevated SC tail
                      #   persists to ~400 ms, matching the data's slow decay)
    # (Ca-activated disinhibition was removed: ablation showed it contributes 0.000
    #  R^2 -- the fitted g_dis collapsed to ~0. The SC tail is set by the injected SC
    #  drive + I_h, not by disinhibition.)
    # --- slow spike-frequency ADAPTATION (Ca-activated K+ / SK / M-current) ---
    # A hyperpolarizing current a that tracks the cell's own rate r_E with a SLOW
    # time constant tau_a. Because it is slow it is ~0 at the sharp SC peak (~55 ms)
    # -> the peak amplitude is preserved; but it builds over 100-600 ms while the
    # bump is elevated -> it pulls the bump back DOWN to baseline (the SC decay) and
    # DESTABILIZES the persistent elevated/broad attractor (the old pathology: the
    # network latched at PD~82 Hz / FWHM~117 forever). This is the canonical
    # mechanism that makes elevated firing TRANSIENT and the network recover.
    (0.0, 2.0),       # g_a   (adaptation conductance; hyperpolarization ~ g_a * a)
    (80.0, 600.0),    # tau_a (adaptation time constant -> sets the recovery/decay
                      #   timescale of the SC back to baseline, ~hundreds of ms)
    # --- GLOBAL SC input, gated by bump-membership (EMERGENT tuning) ---
    # The SC arrives as a second GLOBAL (spatially UNIFORM) input, double-exponential
    # in time, after the dip -- like the FC but slower. It is NOT spatially tuned. The
    # PD-tuned SC OUTPUT emerges because the input is multiplied by the ring's own
    # bump-membership gate rec_E/(rec_E+REC_HALF) (a network state, not input tuning):
    # bump cells (rec_E high) respond, the antipode (rec_E ~ 0) does not -> anti-PD
    # silent. The supralinear gain amplifies the PD response. Biologically: a global
    # salience/arousal drive that the HD attractor transforms into a tuned SC output.
    (0.0, 300.0),     # A_sc      (global SC input amplitude)
    (5.0, 40.0),      # tau_sc_on (fast rise -> peak ~55 ms)
    (50.0, 500.0),    # tau_sc_off(slow decay -> the tail)
    (0.0, 30.0),      # sc_delay  (SC onset after stim_onset; after the dip)
    (5.0, 150.0),     # tau_scg   (SC gate memory: low-passes rec_E so the gate does
                      #   not slam shut during the brief dip -> smooth SC onset, no
                      #   spike. Antipode rec_E ~0 -> gate stays ~0 -> still silent.)
]

PARAM_NAMES = [
    "tau_E",
    "tau_I",
    "I_baseline",
    "J1",
    "KAPPA_E",
    "W_IE",
    "KAPPA_I",
    "W_EI",
    "I_HD",
    "W_GLOBAL",
    "A_fast",
    "fc_stim_duration",
    "stim_delay",
    "reversal_potential",
    "g_T",
    "V_half_T",
    "k_T",
    "tau_hT",
    "E_Ca",
    "g_h",
    "V_half_h",
    "tau_h_on",
    "tau_h_off",
    "g_a",
    "tau_a",
    "A_sc",
    "tau_sc_on",
    "tau_sc_off",
    "sc_delay",
    "tau_scg",
]


# ---------------------------------------------------------------------------
# Core simulators.
#
# Recurrent excitation is a circular convolution of a cos-tuning kernel with
# r_E. Since the kernel only depends on (theta_i - theta_j), the full weight
# matrix is W_full = J1 * exp(KAPPA*(cos(dtheta) - 1)) / N, built once from the
# precomputed COS_D_THETA.
# ---------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def _gain(u):
    """Threshold-linear f-I curve, r = clip(GAIN_SLOPE*[u]+, 0, R_MAX). Antipode
    silencing comes from the rec_gate on the SC input, not from the gain."""
    return np.minimum(R_MAX, GAIN_SLOPE * np.maximum(0.0, u))


@njit(fastmath=True, cache=True)
def _simulate_idle(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                   W_GLOBAL, time):
    # Tuned connectivity: narrow excitation kernel, broad inhibition kernel.
    # Each interneuron is co-tuned with the local E population (K_E), and feeds
    # back inhibition through a BROAD kernel (K_I, KAPPA_I < KAPPA_E). The net
    # effect (narrow excite - broad inhibit) is a Mexican-hat built from real
    # connectivity -> single bump + lateral competition, no DC subtraction.
    K_E = np.exp(KAPPA_E * (COS_D_THETA - 1.0)) / N  # excitation / E->I drive
    K_I = np.exp(KAPPA_I * (COS_D_THETA - 1.0)) / N  # broad I->E inhibition

    u_E = 40.0 * np.exp(KAPPA_E * (np.cos(THETA) - 1.0))
    r_E = _gain(u_E)
    u_I = W_EI * (K_E @ r_E)
    r_I = np.maximum(0.0, u_I)

    I_ext_base = I_baseline + I_HD * HD_SHAPE  # tonic upstream HD drive (loop-invariant)

    rates = np.zeros((len(time), N))
    for step in range(len(time)):
        du_I = -u_I + W_EI * (K_E @ r_E)
        u_I += du_I * (DT / tau_I)
        r_I = np.maximum(0.0, u_I)

        du_E = (-u_E + J1 * (K_E @ r_E) - W_IE * (K_I @ r_I)
                - W_GLOBAL * (np.sum(r_E) / N) + I_ext_base)
        u_E += du_E * (DT / tau_E)

        # Bound the RATE (output), not the voltage. Clipping u_E at a ceiling
        # would erase the bump's spatial ordering when cells saturate; keeping
        # u_E unclamped preserves PD > off-bump even when both rates hit 500.
        u_E = np.maximum(u_E, -100.0)
        r_E = _gain(u_E)
        rates[step, :] = r_E

    return time, rates


@njit(fastmath=True, cache=True)
def _simulate_stim(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                   W_GLOBAL,
                   A_fast, fc_stim_duration, stim_delay, reversal_potential,
                   g_T, V_half_T, k_T, tau_hT, E_Ca,
                   g_h, V_half_h, tau_h_on, tau_h_off,
                   g_a, tau_a,
                   A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg, time):
    # Tuned connectivity (see _simulate_idle): narrow excite, broad inhibit.
    K_E = np.exp(KAPPA_E * (COS_D_THETA - 1.0)) / N
    K_I = np.exp(KAPPA_I * (COS_D_THETA - 1.0)) / N

    u_E = 40.0 * np.exp(KAPPA_E * (np.cos(THETA) - 1.0))
    r_E = _gain(u_E)
    u_I = W_EI * (K_E @ r_E)
    r_I = np.maximum(0.0, u_I)

    # Intrinsic-current gates. h_T = T-type inactivation (slow, de-inactivated by
    # hyperpolarization), m_h = HCN activation (slow). Both start at 0.
    h_T = np.zeros(N)
    m_h = np.zeros(N)

    # Slow spike-frequency adaptation (Ca-activated K+ / SK / M-current). GLOBAL
    # (mean-field): a single scalar low-pass-filters the network's MEAN rate with
    # tau_a; I_adapt = g_a * a is a uniform hyperpolarization. Global (not per-cell)
    # is deliberate: per-cell adaptation makes the ring bump TRAVEL (the adapted
    # peak cells fatigue, neighbours take over -> drifting bump). A uniform pull-down
    # has no preferred direction so the bump stays put. Slow => ~0 at the sharp SC
    # peak (peak preserved), large over the tail => it RE-NARROWS and lowers the bump
    # back to baseline. This is the load-bearing WIDTH-recovery mechanism (ablating
    # it leaves the bump stuck broad, FWHM ~105 vs ~69 deg).
    a = 0.0

    # SC gate memory: low-passed recurrent drive (per cell). Charges to the baseline
    # bump during burn-in; bridges the brief dip so the SC gate stays open on the bump.
    sc_lp = np.zeros(N)

    I_ext_base = I_baseline + I_HD * HD_SHAPE  # tonic upstream HD drive (loop-invariant)
    # Conduction latency: the flash (and the whole evoked cascade) starts at
    # stim_onset, not T_STIM, so model FC aligns with the in-vivo FC at ~+10 ms.
    stim_onset = T_STIM + stim_delay
    stim_end = stim_onset + fc_stim_duration
    sc_onset = stim_onset + sc_delay  # global SC input starts after the dip

    rates = np.zeros((len(time), N))
    for step in range(len(time)):
        t = time[step]

        # Recurrent excitation each cell receives (= bump membership). Computed
        # once: it both drives u_E and GATES I_T (rec_gate) so the rebound is
        # confined to the bump.
        rec_E = J1 * (K_E @ r_E)
        # Bump-membership gate for I_T (instantaneous): ~1 on the bump, ~0 at antipode.
        rec_gate = rec_E / (rec_E + REC_HALF)
        # SC gate: same idea but on a low-passed rec_E (memory tau_scg), so the brief
        # dip does not slam it shut -> the SC enters smoothly, no spike. Antipode
        # rec_E ~0 -> sc_lp ~0 -> gate ~0 -> still silent.
        sc_lp += (rec_E - sc_lp) * (DT / tau_scg)
        sc_gate = sc_lp / (sc_lp + REC_HALF)

        # Intrinsic currents are GATED OFF (gates frozen at 0, currents 0) until
        # the flash actually arrives (stim_onset). Pre-stim the off-bump cells are
        # tonically suppressed; if their gates charged during the 300 ms burn-in a
        # standing depolarizing current (strongest at the antipode) would invert
        # the attractor to +-180 before the cue (see gotcha 2b). Post-onset the
        # synchronous flash->crash de-inactivates I_T / activates I_h -> phasic.
        if t >= stim_onset:
            # I_T (T-type Ca). m_T: fast activation (instantaneous, ->1 on
            # depolarization). h_T: slow inactivation (->1 when hyperpolarized
            # below V_half_T = de-inactivation by the post-flash crash). Their
            # product is transient: after the crash h_T is high but u_E still
            # low (m_T~0); as the bump recovers through V_half_T, m_T turns on
            # while h_T is still up -> a sharp rebound; then u_E high -> h_T
            # decays -> I_T shuts off. Ohmic with high E_Ca so always depolarizing.
            m_T = 1.0 / (1.0 + np.exp(-(u_E - V_half_T) / k_T))
            h_inf = 1.0 / (1.0 + np.exp((u_E - V_half_T) / k_T))
            h_T += (h_inf - h_T) * (DT / tau_hT)
            # I_T expresses only where recurrent drive exists (rec_gate, computed
            # above), so the rebound stays confined to the bump.
            I_T = g_T * m_T * h_T * (E_Ca - u_E) * rec_gate

            # I_h (HCN). Slow activation by hyperpolarization; additive (NOT
            # ohmic) so it cannot latch in this compressed membrane scale
            # (gotcha 2). tau_h large -> slow depolarizing tail (the SC decay).
            mh_inf = 1.0 / (1.0 + np.exp((u_E - V_half_h) / K_H_SLOPE))
            # Asymmetric HCN kinetics (the real channel's time constant is
            # voltage-dependent): FAST activation when hyperpolarized (mh_inf > m_h
            # = charging) so the gate builds during the brief shallow dip; SLOW
            # deactivation when depolarized (decaying) so the depolarizing tail
            # persists for hundreds of ms -> the long elevated SC decay.
            tau_mh = np.where(mh_inf > m_h, tau_h_on, tau_h_off)
            m_h += (mh_inf - m_h) * (DT / tau_mh)
            I_h = g_h * m_h
        else:
            I_T = np.zeros(N)
            I_h = np.zeros(N)

        I_ext = I_ext_base
        if stim_onset <= t < stim_end:
            I_ext = I_ext_base + A_fast
        # GLOBAL SC input (spatially uniform double-exponential), gated by
        # bump-membership -> tuned SC OUTPUT, antipode silent. The tuning is a
        # network property (rec_gate), NOT spatial tuning of the input.
        if t >= sc_onset:
            dt_sc = t - sc_onset
            g_sc = np.exp(-dt_sc / tau_sc_off) - np.exp(-dt_sc / tau_sc_on)
            I_ext = I_ext + A_sc * g_sc * sc_gate

        # Slow global adaptation (width recovery), gated off pre-stim.
        if t < stim_onset:
            I_adapt = 0.0
        else:
            a += (np.sum(r_E) / N - a) * (DT / tau_a)
            I_adapt = g_a * a

        du_I = -u_I + W_EI * (K_E @ r_E)
        u_I += du_I * (DT / tau_I)
        r_I = np.minimum(500.0, np.maximum(0.0, u_I))

        du_E = (-u_E + rec_E - W_IE * (K_I @ r_I)
                - W_GLOBAL * (np.sum(r_E) / N) + I_ext + I_T + I_h - I_adapt)
        u_E += du_E * (DT / tau_E)

        # Bound the RATE, not the voltage (see _simulate_idle): an unclamped u_E
        # ceiling preserves the bump's spatial ordering through the FC saturation
        # and the crash, so PD stays > off-bump and recovers FIRST -> a fast yet
        # PD-selective rebound. reversal_potential is the hyperpolarization floor.
        u_E = np.maximum(u_E, reversal_potential)
        r_E = _gain(u_E)
        rates[step, :] = r_E

    return time, rates


# ---------------------------------------------------------------------------
# Public wrappers. F_matrix / F_inv_matrix kept in the signature for backward
# compatibility with existing notebook calls, but are no longer used.
# ---------------------------------------------------------------------------
def run_model_idle(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                   W_GLOBAL, F_matrix=None, F_inv_matrix=None):
    return _simulate_idle(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                          W_GLOBAL, TIME)


def run_model_stim(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                   W_GLOBAL,
                   A_fast, fc_stim_duration, stim_delay, reversal_potential,
                   g_T, V_half_T, k_T, tau_hT, E_Ca,
                   g_h, V_half_h, tau_h_on, tau_h_off,
                   g_a, tau_a,
                   A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg,
                   F_matrix=None, F_inv_matrix=None):
    return _simulate_stim(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                          W_GLOBAL,
                          A_fast, fc_stim_duration, stim_delay, reversal_potential,
                          g_T, V_half_T, k_T, tau_hT, E_Ca,
                          g_h, V_half_h, tau_h_on, tau_h_off,
                          g_a, tau_a,
                          A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg, TIME)


# Precompute the index/angle anchors used by the losses.
_IDX_0 = int(np.argmin(np.abs(THETA)))
_IDX_180 = int(np.argmin(np.abs(THETA - np.pi)))
_EXP_THETA = np.exp(1j * THETA)  # for smooth circular center-of-mass
NEAR_PD = np.abs(THETA) <= (np.pi / 4)  # cells within +-45 deg of PD (the rebound population)
OFF_LOBE = np.abs(THETA) > (np.pi / 3)  # cells beyond +-60 deg of 0 (split / secondary-bump territory)


def _circular_center(profile):
    """Smooth, differentiable bump center (radians) via circular mean.

    Replaces argmax-based center penalties, which are integer-valued and
    create flat plateaus that stall differential evolution.
    """
    return np.angle(np.sum(profile * _EXP_THETA))


# implementation of the loss function
def loss_stage1(params):
    tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD, W_GLOBAL = params

    _, rates = _simulate_idle(tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
                              W_GLOBAL, TIME_STAGE1)
    if not np.all(np.isfinite(rates)):
        return 1e6

    steady_state_profile = rates[-1, :]
    peak_fr = np.max(steady_state_profile)

    loss_val = 0.0
    loss_val += (peak_fr - 40.0) ** 2

    if peak_fr > 1.0:
        half_max = peak_fr / 2.0
        active_bins = np.sum(steady_state_profile >= half_max)
        fwhm = active_bins * (360.0 / len(THETA))
    else:
        fwhm = 0.0

    if fwhm < 60.0:
        loss_val += (60.0 - fwhm) ** 2 * 10.0
    elif fwhm > 90.0:
        loss_val += (90.0 - fwhm) ** 2 * 10.0

    loss_val += (steady_state_profile[_IDX_180] - 0.0) ** 2 * 50.0

    # Smooth center penalty (radians off 0) instead of integer argmax distance.
    center = _circular_center(steady_state_profile) if peak_fr > 1.0 else 0.0
    loss_val += (center ** 2) * 500.0

    # Single-bump enforcement: any firing outside the central +-60 deg lobe is a
    # split / secondary bump. Penalize that energy relative to the total.
    if peak_fr > 1.0:
        off_lobe = np.sum(steady_state_profile[OFF_LOBE])
        total = np.sum(steady_state_profile) + 1e-9
        loss_val += (off_lobe / total) ** 2 * 500.0

        # Kill discrete secondary bumps (e.g. the +-130 deg lobes seen at
        # baseline): the strongest off-lobe cell must be a small fraction of the
        # main peak. These arise when the inhibition kernel (KAPPA_I) is too
        # narrow to reach the far field, so I_baseline drives distal cells over
        # threshold. Heavy weight pushes DE toward BROADER inhibition (smaller
        # KAPPA_I) that suppresses the whole surround. The energy-fraction term
        # above tolerates a few small lobes; this targets their peak directly.
        secondary_peak = np.max(steady_state_profile[OFF_LOBE])
        loss_val += (secondary_peak / peak_fr) ** 2 * 800.0

    # TIME_STAGE1 runs to +50 ms; the variance window is fully inside it.
    middle_mask = (TIME_STAGE1 > -150) & (TIME_STAGE1 < -50)
    pd_cell_middle = rates[middle_mask, _IDX_0][::10]
    loss_val += np.var(pd_cell_middle) * 1000.0

    return float(loss_val)


# Stage-2 fit window (ms relative to stim). Stim is at t=0 in both model and data.
# Extended to 700 ms to cover the FULL SC: the in-vivo response is a TRANSIENT that
# peaks ~55 ms (~91 Hz) then decays smoothly back to the ~37 Hz baseline by ~700 ms
# (data: 91->71@200->61@300->46@500->35@700). Fitting only to 300 ms hid the second
# half of that decay, so the optimizer parked at a too-weak W_GLOBAL and the network
# latched into a SECOND stable attractor (PD stuck ~82 Hz, FWHM ~117 forever) that
# never recovered. Fitting the whole glide-back forces a MONOSTABLE network.
WIN_LO, WIN_HI = -50.0, 700.0


def loss_stage2(params, stage1_frozen, bins_plot, vivo_rate, vivo_smooth):
    """Data-driven Stage-2 loss.

    Fits the model PD trace to the REAL population PSTH (vivo_rate / vivo_smooth
    from vivo_target.load_vivo_psth) via a hybrid raw-early / smooth-late MSE.
    The SC OUTPUT emerges from a GLOBAL (untuned) SC input gated by bump-membership
    (rec_gate) and amplified by the supralinear gain; only light *structural*
    regularizers are added (anti-PD silence, single bump, width recovery) so the
    optimizer can't match the trace by destroying the attractor.
    """
    (
        A_fast, fc_stim_duration, stim_delay, reversal_potential,
        g_T, V_half_T, k_T, tau_hT, E_Ca,
        g_h, V_half_h, tau_h_on, tau_h_off,
        g_a, tau_a,
        A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg,
    ) = params

    tau_E = stage1_frozen["tau_E"]
    tau_I = stage1_frozen["tau_I"]
    I_baseline = stage1_frozen["I_baseline"]
    J1 = stage1_frozen["J1"]
    KAPPA_E = stage1_frozen["KAPPA_E"]
    W_IE = stage1_frozen["W_IE"]
    KAPPA_I = stage1_frozen["KAPPA_I"]
    W_EI = stage1_frozen["W_EI"]
    I_HD = stage1_frozen["I_HD"]
    W_GLOBAL = stage1_frozen["W_GLOBAL"]

    t_model, rates = _simulate_stim(
        tau_E, tau_I, I_baseline, J1, KAPPA_E, W_IE, KAPPA_I, W_EI, I_HD,
        W_GLOBAL,
        A_fast, fc_stim_duration, stim_delay, reversal_potential,
        g_T, V_half_T, k_T, tau_hT, E_Ca,
        g_h, V_half_h, tau_h_on, tau_h_off,
        g_a, tau_a,
        A_sc, tau_sc_on, tau_sc_off, sc_delay, tau_scg, TIME_STAGE2,
    )

    if not np.all(np.isfinite(rates)):
        return 1e6

    pd_trace = rates[:, _IDX_0]
    anti_pd_rate = rates[:, _IDX_180]

    # --- CORE: PD trace vs real PSTH (hybrid target, early-weighted) ---
    bins_plot = np.asarray(bins_plot, dtype=np.float64)
    vivo_mask = (bins_plot >= WIN_LO) & (bins_plot <= WIN_HI)
    matched_times = bins_plot[vivo_mask]
    if matched_times.size < 3:
        return 1e6

    model_at_vivo = np.interp(matched_times, t_model, pd_trace)

    # Raw data for the FC startle spike (<=20 ms), smooth data for the SC tail.
    early = matched_times <= 20.0
    target = np.where(early, vivo_rate[vivo_mask], vivo_smooth[vivo_mask])
    # Weights: FC spike 5x; the REBOUND peak (25-75 ms) 4x (fit SC amplitude, no
    # overshoot); the elevated SC tail (80-300 ms) 3x. With the RECURRENT-GATED
    # I_T the rebound (and the I_T-driven disinhibition) is confined to the bump,
    # so the tail no longer requires bump-wide weak inhibition -> we can ask for
    # both the tail AND a sharp recovering attractor (KAPPA_I sharp + FWHM penalty).
    weights = np.where(early, 5.0, 1.0)
    weights = np.where((matched_times > 25.0) & (matched_times <= 75.0), 4.0, weights)
    weights = np.where((matched_times > 80.0) & (matched_times <= 200.0), 3.0, weights)
    # the SC DECAY back to baseline (200-700 ms) is now in the fit -> it forces the
    # rate to come down (a monostable network) instead of latching elevated.
    weights = np.where((matched_times > 200.0) & (matched_times <= 700.0), 2.0, weights)
    weighted_mse = np.average((model_at_vivo - target) ** 2, weights=weights)

    loss_val = weighted_mse

    # --- MECHANISTIC CONSTRAINT: a real but SHALLOW inhibitory dip ---
    # PD must dip clearly below baseline ~15-20 ms post-stim (the inhibitory gap
    # that hyperpolarizes the membrane and de-inactivates I_T -> the rebound
    # prerequisite). But it must NOT crash to 0: zeroing PD erases the recurrent
    # bump, which then has to silently re-form before the rebound can fire -> a
    # late (~80 ms) rebound. A shallow dip keeps the bump alive (the membrane
    # still crashes negative -> I_T de-inactivates while the rate stays ~15-25),
    # so the rebound is both fast (~55 ms) AND PD-selective. The data PSTH floors
    # at ~25 Hz here, so this also matches biology better than forcing 0.
    # Guard only enforces that a dip EXISTS (min < ~25 Hz); the MSE sets its exact
    # depth. Window tied to stim_onset so the FC->dip gap is fixed as stim_delay
    # floats to the true conduction latency.
    stim_onset = T_STIM + stim_delay
    # (i) a dip must EXIST: PD drops below ~30 Hz somewhere in the early window.
    dip_mask = (t_model > stim_onset + 5.0) & (t_model < stim_onset + 22.0)
    if dip_mask.sum() > 0:
        dip_min = np.min(pd_trace[dip_mask])
        loss_val += (np.maximum(0.0, dip_min - 30.0)) ** 2 * 20.0
    # (ii) NB: an earlier "bump-alive floor" (PD >= 15 Hz through the dip) was
    # removed. It forced a SHALLOW dip (PD membrane stays positive), which stops
    # HCN/I_h from ever activating (mh_inf ~ 0 when u > 0) -> no slow tail. The
    # non-saturating-FC fix already preserves the bump's spatial memory through a
    # DEEP brief dip, so PD can crash negative (HCN charges) and still re-form fast
    # and PD-selectively. The dip-exists guard above + the MSE keep the dip sane.

    # --- STRUCTURAL REGULARIZERS ---
    # (a) anti-PD must stay near-silent during the SC window (40-300 ms): the
    #     rebound is a PD-only phenomenon. The channels de-inactivate the antipode
    #     too (it also crashed), so the reformed PD bump must laterally suppress
    #     it -> moderate weight pushes the optimizer to a winner-take-all rebound.
    SC_mask = (t_model > 40.0) & (t_model < 700.0)
    loss_val += np.mean(anti_pd_rate[SC_mask] ** 2) * 10.0

    # (b) RECOVERY to baseline (TWO-SIDED): the in-vivo SC is a transient -- by
    #     ~500-700 ms PD has decayed back to the ~37 Hz baseline. We pin the late
    #     PD level to baseline so the optimizer must pick a MONOSTABLE network (the
    #     broad/elevated post-stim state must be UNSTABLE and relax) instead of
    #     latching into a second stable attractor (the old pathology: PD stuck
    #     ~82 Hz, FWHM ~117 forever). This is now safe to apply two-sided because
    #     W_GLOBAL gives a real way to recover -- enough global inhibition makes the
    #     high-total-activity broad state unstable -- WITHOUT killing the rebound.
    recov_mask = (t_model > 500.0) & (t_model < 700.0)
    if recov_mask.sum() > 0:
        recov_pd = np.mean(pd_trace[recov_mask])
        loss_val += (recov_pd - 40.0) ** 2 * 3.0

    # single-bump topology over the stable late window (keep the attractor intact).
    stable_mask = (t_model > 100.0) & (t_model < 700.0)
    late_profile_mean = np.mean(rates[stable_mask, :], axis=0)
    late_peak = np.max(late_profile_mean)
    loss_val += (np.maximum(0.0, 10.0 - late_peak)) ** 2 * 50.0

    if late_peak > 1.0:
        center = _circular_center(late_profile_mean)
        loss_val += (center ** 2) * 50.0
        off_lobe = np.sum(late_profile_mean[OFF_LOBE])
        total = np.sum(late_profile_mean) + 1e-9
        loss_val += (off_lobe / total) ** 2 * 50.0

        # RECOVERY of bump WIDTH to ~baseline. Measured LATE (400-700 ms) so the
        # transient SC broadening (40-100 ms) is still allowed; only the END state
        # must be narrow again. A REAL weight (was 0.2, one-sided > 100 deg) is now
        # safe: with W_GLOBAL the optimizer can re-narrow by making the broad state
        # unstable instead of by KILLING the rebound (the old worry). Target the
        # idle bump band (~70 deg); penalize anything still broad at the end.
        recov_profile = np.mean(rates[(t_model > 400.0) & (t_model < 700.0), :], axis=0)
        rp_peak = np.max(recov_profile)
        if rp_peak > 1.0:
            recov_fwhm = np.sum(recov_profile >= rp_peak / 2.0) * (360.0 / len(THETA))
            loss_val += (np.maximum(0.0, recov_fwhm - 70.0)) ** 2 * 3.0

    if np.isnan(loss_val) or np.isinf(loss_val):
        return 1e6
    return float(loss_val)


# Joint search space: the 8 attractor/geometry params + the 12 stim/channel params.
JOINT_BOUNDS = STAGE1_BOUNDS + STAGE2_BOUNDS


def loss_joint(params, bins_plot, vivo_rate, vivo_smooth, w_baseline=4.0):
    """Single-stage joint loss = baseline geometry + data-driven evoked fit.

    Optimizes ALL 20 params at once so the shared inhibition (W_IE/KAPPA_I/W_EI)
    is chosen against every constraint together. The two-stage version froze the
    geometry blind to the stim demands, and no frozen W_IE could satisfy baseline
    + dip + smooth rebound + anti-PD silence at once. Here the data MSE in
    loss_stage2 directly penalizes the late-spike rebound regime that strong
    inhibition produces, so the optimizer is pulled to the compromise.

    Reuses loss_stage1 (baseline regularizers, no data) and loss_stage2 (evoked
    MSE + dip + anti-PD + bump survival) unchanged -> one source of truth.
    """
    s1 = params[:10]
    s2 = params[10:]
    base = loss_stage1(s1)
    if base >= 1e6:
        return 1e6
    frozen = dict(zip(PARAM_NAMES[:10], s1))
    evoked = loss_stage2(s2, frozen, bins_plot, vivo_rate, vivo_smooth)
    return float(w_baseline * base + evoked)
