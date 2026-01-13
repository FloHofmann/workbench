import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons


# ----------------------------
# Model: discrete ring attractor for head direction
# ----------------------------
class HDRingAttractor:
    """
    Discrete ring attractor with multiplicative stimulus responses.

    Time units:
      - dt is interpreted as milliseconds
      - tau is milliseconds
      - t_ms is milliseconds

    Dynamics:
      tau dr/dt = -r + phi( gain(theta, t) * base_input(theta, t) )

    where:
      base_input = W @ r + cue_input + velocity_shift
      gain(theta, t) = (1 + global_stim_gain(t)) * (1 + gamma_gain(theta, t))
    """

    def __init__(
        self,
        N=180,
        J0=-0.6,
        J1=1.4,
        tau=10.0,      # ms
        dt=1.0,        # ms
        g_vel=1.0,
        use_relu=True,
        seed_bump_angle=np.pi
    ):
        self.N = int(N)
        self.theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)

        self.dt = float(dt)
        self.tau = float(tau)
        self.g_vel = float(g_vel)

        dtheta = self.theta[:, None] - self.theta[None, :]
        self.W = (J0 + J1 * np.cos(dtheta)) / self.N

        # state
        self.r = 0.05 * np.random.rand(self.N)
        self.r += 0.8 * \
            np.exp(10.0 * np.cos(self.theta - seed_bump_angle))  # seed bump

        # interactive inputs
        self.omega = 0.0
        self.cue_angle = 0.0
        self.cue_amp = 0.0
        self.cue_kappa = 6.0
        self.cue_enabled = True

        self.use_relu = bool(use_relu)

        # simulation clock
        self.t_ms = 0.0

        # ----- Multiplicative GLOBAL stim gain (uniform across ring) -----
        # If active, gain factor is (1 + stim_amp), i.e. stim_amp=0.5 => 1.5x
        self.stim_amp = 0.5
        self._stim_events = []  # list of (t_on_ms, t_off_ms)

        # ----- Multiplicative gamma bump gain (spatially tuned) -----
        # Each event stores stim time and the HD at that moment.
        # Gain factor is (1 + gamma_amp * gamma_pdf(u) * spatial_profile(theta))
        self._gamma_events = []  # list of dicts

    def phi(self, x):
        if self.use_relu:
            return np.maximum(0.0, x)
        return np.log1p(np.exp(x))  # softplus (optional)

    def cue_input(self):
        if (not self.cue_enabled) or (self.cue_amp <= 0.0):
            return np.zeros(self.N)
        return self.cue_amp * np.exp(self.cue_kappa * np.cos(self.theta - self.cue_angle))

    def decode_hd(self):
        z = np.sum(self.r * np.exp(1j * self.theta))
        return float(np.angle(z) % (2 * np.pi))

    # ----------------------------
    # Global stim: multiplicative gain (uniform)
    # ----------------------------
    def schedule_stim(self, delay_ms=9.0, width_ms=2.0):
        """Schedule a global gain pulse for [delay, delay+width] ms after now."""
        t_on = self.t_ms + float(delay_ms)
        t_off = t_on + float(width_ms)
        self._stim_events.append((t_on, t_off))

    def global_stim_gain(self):
        """
        Returns stim_amp if currently within any stim window, else 0.
        This is a *gain increment*; overall factor is (1 + global_stim_gain()).
        """
        if (self.stim_amp <= 0.0) or (not self._stim_events):
            return 0.0

        t = self.t_ms
        active = False
        keep = []

        for (t_on, t_off) in self._stim_events:
            if t <= t_off:
                keep.append((t_on, t_off))
            if (t >= t_on) and (t <= t_off):
                active = True

        self._stim_events = keep
        return self.stim_amp if active else 0.0

    # ----------------------------
    # Gamma bump: multiplicative gain (spatially tuned)
    # ----------------------------
    @staticmethod
    def gamma_pdf(u_ms, alpha, theta_ms):
        """Gamma PDF for u>=0 (ms), 0 for u<0. alpha>0, theta_ms>0."""
        if u_ms < 0.0:
            return 0.0
        denom = (theta_ms ** alpha) * math.gamma(alpha)
        return (u_ms ** (alpha - 1.0)) * math.exp(-u_ms / theta_ms) / denom

    def schedule_gamma_bump(
        self,
        onset_ms=25.0,
        dur_ms=200.0,
        amp=60.0,
        alpha=2.0,
        theta_ms=40.0,
        kappa=8.0
    ):
        """
        Schedule a gamma-shaped *gain* event aligned to the CURRENT decoded HD.
        Overall gain increment at time u is: amp * gamma_pdf(u; alpha, theta_ms) * exp(kappa*cos(theta-hd0))

        Notes:
          - gamma_pdf integrates to 1, so 'amp' behaves like total gain "area" over time.
          - If amp is too large, the network can saturate; reduce amp if needed.
        """
        self._gamma_events.append({
            "t0_ms": self.t_ms,
            "hd0": self.decode_hd(),
            "onset_ms": float(onset_ms),
            "dur_ms": float(dur_ms),
            "amp": float(amp),
            "alpha": float(alpha),
            "theta_ms": float(theta_ms),
            "kappa": float(kappa),
        })

    def gamma_gain_profile(self):
        """
        Returns a vector gain increment over theta (length N) from all active gamma events.
        Overall gamma factor is (1 + gamma_gain_profile()) elementwise.
        """
        if not self._gamma_events:
            return np.zeros(self.N)

        t = self.t_ms
        total = np.zeros(self.N)
        keep = []

        for ev in self._gamma_events:
            t_on = ev["t0_ms"] + ev["onset_ms"]
            t_off = t_on + ev["dur_ms"]

            if t <= t_off:
                keep.append(ev)

            if (t >= t_on) and (t <= t_off):
                u = t - t_on
                g = self.gamma_pdf(u, ev["alpha"], ev["theta_ms"])
                # gain increment over theta
                spatial = np.exp(ev["kappa"] * np.cos(self.theta - ev["hd0"]))
                total += ev["amp"] * g * spatial

        self._gamma_events = keep
        return total

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, n_steps=1):
        for _ in range(int(n_steps)):
            # antisymmetric shift term ~ derivative on ring
            Kr = (np.roll(self.r, -1) - np.roll(self.r, 1)) / 2.0

            base_input = (
                self.W @ self.r
                + self.cue_input()
                + (self.g_vel * self.omega * Kr)
            )

            # multiplicative gains
            g_global = 1.0 + self.global_stim_gain()                 # scalar
            g_gamma_vec = 1.0 + self.gamma_gain_profile()            # vector (N,)
            # vector (N,)
            gain_vec = g_global * g_gamma_vec

            x = gain_vec * base_input
            dr = (-self.r + self.phi(x)) / self.tau
            self.r = self.r + self.dt * dr

            self.t_ms += self.dt


# ----------------------------
# Interactive UI (matplotlib widgets)
# ----------------------------
def run_interactive():
    model = HDRingAttractor(N=180, dt=1.0, tau=10.0)

    running = {"on": True}

    # Slow-mo: how many simulated ms per UI frame?
    speed_ms_per_frame = {"val": 2.0}

    def compute_steps_per_frame():
        return max(1, int(round(speed_ms_per_frame["val"] / model.dt)))

    # Create figure
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 2])

    ax_ring = fig.add_subplot(gs[0, 0])
    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")

    (line_ring,) = ax_ring.plot(model.theta, model.r, lw=2)
    ax_ring.set_xlim(0, 2*np.pi)
    ax_ring.set_xlabel("preferred direction θ (rad)")
    ax_ring.set_ylabel("activity r")
    ax_ring.set_title("HD Ring Attractor Activity (180 neurons)")

    ax_polar.set_title("Decoded HD & External Cue")
    decoded_marker = ax_polar.plot(
        [model.decode_hd()], [1.0], marker="o", markersize=10)[0]
    cue_marker = ax_polar.plot(
        [model.cue_angle], [1.0], marker="x", markersize=10)[0]
    ax_polar.set_rlim(0, 1.2)
    ax_polar.set_rticks([])

    txt = ax_ring.text(
        0.02, 0.95, "", transform=ax_ring.transAxes, va="top", ha="left")

    # Sliders area (dummy)
    ax_sliders = fig.add_subplot(gs[1, :])
    ax_sliders.axis("off")

    # Slider axes (stacked)
    y0 = 0.235
    dy = 0.04

    s_ax_speed = fig.add_axes([0.10, y0 + 1*dy, 0.60, 0.03])
    s_ax_stimamp = fig.add_axes([0.10, y0 + 0*dy, 0.60, 0.03])

    s_ax_omega = fig.add_axes([0.10, y0 - 1*dy, 0.60, 0.03])
    s_ax_cueang = fig.add_axes([0.10, y0 - 2*dy, 0.60, 0.03])
    s_ax_cueamp = fig.add_axes([0.10, y0 - 3*dy, 0.60, 0.03])
    s_ax_ckappa = fig.add_axes([0.10, y0 - 4*dy, 0.60, 0.03])

    # Gamma gain controls (keep them compact but editable)
    s_ax_gamp = fig.add_axes([0.10, y0 - 5*dy, 0.60, 0.03])
    s_ax_onset = fig.add_axes([0.10, y0 - 6*dy, 0.60, 0.03])
    s_ax_dur = fig.add_axes([0.10, y0 - 7*dy, 0.60, 0.03])
    s_ax_alpha = fig.add_axes([0.10, y0 - 8*dy, 0.60, 0.03])
    s_ax_theta = fig.add_axes([0.10, y0 - 9*dy, 0.60, 0.03])
    s_ax_gkappa = fig.add_axes([0.10, y0 - 10*dy, 0.60, 0.03])

    # Sliders
    s_speed = Slider(s_ax_speed, "slow-mo: ms/frame", 0.2, 20.0, valinit=2.0)
    s_stimamp = Slider(s_ax_stimamp, "stim gain amp (global)",
                       0.0, 2.0, valinit=model.stim_amp)

    s_omega = Slider(s_ax_omega, "ω (turn rate)", -0.05,
                     0.05, valinit=0.0, valstep=0.001)
    s_cueang = Slider(s_ax_cueang, "cue angle", 0.0, 2*np.pi, valinit=0.0)
    s_cueamp = Slider(s_ax_cueamp, "cue strength", 0.0, 1.0, valinit=0.0)
    s_ckappa = Slider(s_ax_ckappa, "cue sharpness κ",
                      0.5, 12.0, valinit=model.cue_kappa)

    # Gamma defaults: alpha=2; peak ~ (alpha-1)*theta_ms, so with alpha=2 peak ~ theta_ms after onset
    s_gamp = Slider(s_ax_gamp, "gamma gain amp", 0.0, 400.0, valinit=60.0)
    s_onset = Slider(s_ax_onset, "gamma onset (ms)", 0.0, 150.0, valinit=25.0)
    s_dur = Slider(s_ax_dur, "gamma duration (ms)", 20.0, 800.0, valinit=200.0)
    s_alpha = Slider(s_ax_alpha, "gamma α", 1.1, 8.0, valinit=2.0)
    s_theta = Slider(s_ax_theta, "gamma θ (ms)", 5.0, 300.0, valinit=40.0)
    s_gkappa = Slider(s_ax_gkappa, "gamma spatial κ", 0.5, 20.0, valinit=8.0)

    # Buttons & checkbox
    b_ax_toggle = fig.add_axes([0.75, 0.18, 0.10, 0.06])
    b_ax_reset = fig.add_axes([0.87, 0.18, 0.10, 0.06])
    b_ax_stim = fig.add_axes([0.75, 0.26, 0.22, 0.06])

    b_toggle = Button(b_ax_toggle, "Pause")
    b_reset = Button(b_ax_reset, "Reset")
    b_stim = Button(b_ax_stim, "Stim")

    c_ax = fig.add_axes([0.75, 0.06, 0.22, 0.10])
    c = CheckButtons(c_ax, ["cue enabled"], [True])

    def on_slider(_):
        speed_ms_per_frame["val"] = s_speed.val

        model.stim_amp = s_stimamp.val

        model.omega = s_omega.val
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_ckappa.val

    def on_toggle(_):
        running["on"] = not running["on"]
        b_toggle.label.set_text("Pause" if running["on"] else "Run")

    def on_reset(_):
        seed = model.cue_angle if model.cue_amp > 0 else np.pi
        dt = model.dt
        tau = model.tau

        # re-init
        model.__init__(N=model.N, dt=dt, tau=tau)

        # restore UI settings
        speed_ms_per_frame["val"] = s_speed.val
        model.stim_amp = s_stimamp.val

        model.omega = s_omega.val
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_ckappa.val

        # clear pending events
        model._stim_events = []
        model._gamma_events = []

        # reseed bump
        model.r = 0.05 * np.random.rand(model.N)
        model.r += 0.8 * np.exp(10.0 * np.cos(model.theta - seed))

    def on_check(label):
        if label == "cue enabled":
            model.cue_enabled = not model.cue_enabled

    def on_stim(_):
        # Global multiplicative gain pulse at 9–11 ms post click
        model.schedule_stim(delay_ms=9.0, width_ms=2.0)

        # Gamma multiplicative gain bump aligned to HD at stim time
        model.schedule_gamma_bump(
            onset_ms=s_onset.val,
            dur_ms=s_dur.val,
            amp=s_gamp.val,
            alpha=s_alpha.val,
            theta_ms=s_theta.val,
            kappa=s_gkappa.val
        )

    # bind callbacks
    for s in [s_speed, s_stimamp, s_omega, s_cueang, s_cueamp, s_ckappa,
              s_gamp, s_onset, s_dur, s_alpha, s_theta, s_gkappa]:
        s.on_changed(on_slider)

    b_toggle.on_clicked(on_toggle)
    b_reset.on_clicked(on_reset)
    b_stim.on_clicked(on_stim)

    c.on_clicked(on_check)

    # Animation loop using timer
    timer = fig.canvas.new_timer(interval=30)  # ms UI refresh

    def update(*_):
        if running["on"]:
            model.step(n_steps=compute_steps_per_frame())

        # Update plots
        line_ring.set_ydata(model.r)

        hd = model.decode_hd()
        decoded_marker.set_data([hd], [1.0])
        cue_marker.set_data([model.cue_angle], [1.0])

        cue_state = "on" if model.cue_enabled else "off"

        txt.set_text(
            f"t = {
                model.t_ms:7.1f} ms | slow-mo = {speed_ms_per_frame['val']:.2f} ms/frame\n"
            f"decoded HD: {hd:.3f} rad ({np.degrees(hd):.1f}°)\n"
            f"ω: {model.omega:.4f} | cue: {cue_state} (A={model.cue_amp:.2f}, κ={
                model.cue_kappa:.1f})\n"
            f"global stim gain amp: {model.stim_amp:.2f} | stim events: {
                len(model._stim_events)} | gamma events: {len(model._gamma_events)}"
        )

        fig.canvas.draw_idle()

    timer.add_callback(update)
    timer.start()
    plt.show()


if __name__ == "__main__":
    run_interactive()
