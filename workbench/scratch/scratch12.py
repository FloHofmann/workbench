import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import math

# ----------------------------
# Model: discrete ring attractor for head direction
# additive responses for the stimulations
# ----------------------------


class HDRingAttractor:
    def __init__(self, N=180, J0=-0.6, J1=1.4, tau=10.0, dt=1.0,
                 g_vel=1.0, use_relu=True, seed_bump_angle=np.pi):
        """
        Units:
          - dt is in ms
          - tau is in ms
          - internal clock t_ms is in ms
        """
        self.N = N
        self.theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.dt = float(dt)     # ms
        self.tau = float(tau)   # ms
        self.g_vel = float(g_vel)

        dtheta = self.theta[:, None] - self.theta[None, :]
        self.W = (J0 + J1 * np.cos(dtheta)) / N

        # state
        self.r = 0.05 * np.random.rand(N)
        self.r += 0.8 * \
            np.exp(10.0 * np.cos(self.theta - seed_bump_angle))  # seed bump

        # inputs (controlled interactively)
        self.omega = 0.0                 # angular velocity drive (model units)
        self.cue_angle = 0.0             # external direction cue angle
        self.cue_amp = 0.0               # strength of cue input
        self.cue_kappa = 6.0             # sharpness of cue
        self.cue_enabled = True

        self.use_relu = use_relu

        # --- Simulation clock ---
        self.t_ms = 0.0

        # --- Global stim (entire ring) ---
        self.stim_amp = 0.5
        self._stim_events = []           # list of (t_on_ms, t_off_ms)

        # --- Post-stim gamma-bump events (HD-aligned) ---
        # Each event: dict with keys: t0_ms, hd0, onset_ms, dur_ms, amp, alpha, theta_ms, kappa
        self._gamma_events = []

    def phi(self, x):
        if self.use_relu:
            return np.maximum(0.0, x)
        return np.log1p(np.exp(x))

    def cue_input(self):
        if (not self.cue_enabled) or (self.cue_amp <= 0):
            return np.zeros(self.N)
        return self.cue_amp * np.exp(self.cue_kappa * np.cos(self.theta - self.cue_angle))

    def stim_input(self):
        """Spatially uniform stimulus across the ring."""
        if (self.stim_amp <= 0) or (not self._stim_events):
            return np.zeros(self.N)

        t = self.t_ms
        active = False
        keep = []
        for (t_on, t_off) in self._stim_events:
            if t <= t_off:
                keep.append((t_on, t_off))
            if (t >= t_on) and (t <= t_off):
                active = True
        self._stim_events = keep

        return (self.stim_amp * np.ones(self.N)) if active else np.zeros(self.N)

    @staticmethod
    def gamma_pdf(u_ms, alpha, theta_ms):
        """
        Gamma PDF for u>=0, with shape alpha and scale theta_ms.
        Returns 0 for u<0. Uses continuous gamma formula.
        """
        if u_ms < 0:
            return 0.0
        # pdf = u^(a-1) * exp(-u/theta) / (theta^a * Gamma(a))
        # alpha can be non-integer; use math.gamma
        denom = (theta_ms ** alpha) * math.gamma(alpha)
        return (u_ms ** (alpha - 1.0)) * math.exp(-u_ms / theta_ms) / denom

    def gamma_bump_input(self):
        """
        Sum of HD-aligned gamma-shaped inputs triggered by stim:
          - timecourse: gamma(alpha, theta_ms), starting at onset_ms
          - spatial profile: von Mises around hd0 at stim time
        """
        if not self._gamma_events:
            return np.zeros(self.N)

        t = self.t_ms
        total = np.zeros(self.N)
        keep = []

        for ev in self._gamma_events:
            t0 = ev["t0_ms"]
            onset = ev["onset_ms"]
            dur = ev["dur_ms"]
            hd0 = ev["hd0"]

            t_on = t0 + onset
            t_off = t_on + dur

            # keep if not finished
            if t <= t_off:
                keep.append(ev)

            # active?
            if (t >= t_on) and (t <= t_off):
                u = t - t_on  # time since onset (ms)
                g = self.gamma_pdf(u, ev["alpha"], ev["theta_ms"])
                # Scale gamma to a handy magnitude by multiplying with amp
                # (since pdf integrates to 1, amp roughly sets total "area" of the drive).
                amp_t = ev["amp"] * g

                spatial = np.exp(ev["kappa"] * np.cos(self.theta - hd0))
                total += amp_t * spatial

        self._gamma_events = keep
        return total

    def schedule_stim(self, delay_ms=9.0, width_ms=2.0):
        """Schedule global pulse for [delay, delay+width] ms after now."""
        t_on = self.t_ms + float(delay_ms)
        t_off = t_on + float(width_ms)
        self._stim_events.append((t_on, t_off))

    def schedule_gamma_bump(self, onset_ms=25.0, dur_ms=200.0,
                            amp=60.0, alpha=2.0, theta_ms=40.0, kappa=8.0):
        """
        Schedule a gamma-like bump aligned to the CURRENT decoded HD (hd0).
        onset_ms: when the gamma timecourse starts after stim time (ms)
        dur_ms: duration window to apply it (ms)
        amp: overall strength scaling (you will likely tweak this)
        alpha, theta_ms: gamma parameters
        kappa: spatial sharpness (higher = narrower)
        """
        ev = {
            "t0_ms": self.t_ms,
            "hd0": self.decode_hd(),
            "onset_ms": float(onset_ms),
            "dur_ms": float(dur_ms),
            "amp": float(amp),
            "alpha": float(alpha),
            "theta_ms": float(theta_ms),
            "kappa": float(kappa),
        }
        self._gamma_events.append(ev)

    def decode_hd(self):
        z = np.sum(self.r * np.exp(1j * self.theta))
        return float(np.angle(z) % (2*np.pi))

    def step(self, n_steps=1):
        for _ in range(n_steps):
            Kr = (np.roll(self.r, -1) - np.roll(self.r, 1)) / 2.0

            x = (
                self.W @ self.r
                + self.cue_input()
                + self.stim_input()
                + self.gamma_bump_input()
                + (self.g_vel * self.omega * Kr)
            )
            dr = (-self.r + self.phi(x)) / self.tau
            self.r = self.r + self.dt * dr
            self.t_ms += self.dt


# ----------------------------
# Interactive UI (matplotlib widgets)
# ----------------------------
def run_interactive():
    model = HDRingAttractor(N=180, dt=1.0, tau=10.0)

    running = {"on": True}

    # --- Slow-mo control: simulated ms per visual frame ---
    # steps_per_frame = round(speed_ms_per_frame / dt)
    # default: 2 ms per frame (slow enough to see)
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

    # Slider axes
    # row y positions (top to bottom)
    y0 = 0.255
    dy = 0.045

    s_ax_speed = fig.add_axes([0.10, y0,         0.60, 0.03])
    s_ax_omega = fig.add_axes([0.10, y0 - dy,    0.60, 0.03])
    s_ax_cueang = fig.add_axes([0.10, y0 - 2*dy,  0.60, 0.03])
    s_ax_cueamp = fig.add_axes([0.10, y0 - 3*dy,  0.60, 0.03])
    s_ax_kappa = fig.add_axes([0.10, y0 - 4*dy,  0.60, 0.03])

    # Stim params
    s_ax_stimamp = fig.add_axes([0.10, y0 - 5*dy, 0.60, 0.03])

    # Gamma-bump params
    s_ax_gamp = fig.add_axes([0.10, y0 - 6*dy, 0.60, 0.03])
    s_ax_onset = fig.add_axes([0.10, y0 - 7*dy, 0.60, 0.03])
    s_ax_dur = fig.add_axes([0.10, y0 - 8*dy, 0.60, 0.03])
    s_ax_alpha = fig.add_axes([0.10, y0 - 9*dy, 0.60, 0.03])
    s_ax_theta = fig.add_axes([0.10, y0 - 10*dy, 0.60, 0.03])
    s_ax_gkappa = fig.add_axes([0.10, y0 - 11*dy, 0.60, 0.03])

    # Sliders
    s_speed = Slider(s_ax_speed,  "slow-mo: ms/frame", 0.2, 20.0, valinit=2.0)
    s_omega = Slider(s_ax_omega,  "ω (turn rate)", -0.05,
                     0.05, valinit=0.0, valstep=0.001)
    s_cueang = Slider(s_ax_cueang, "cue angle", 0.0, 2*np.pi, valinit=0.0)
    s_cueamp = Slider(s_ax_cueamp, "cue strength", 0.0, 1.0, valinit=0.0)
    s_kappa = Slider(s_ax_kappa,  "cue sharpness κ",
                     0.5, 12.0, valinit=model.cue_kappa)

    s_stimamp = Slider(s_ax_stimamp, "stim amp (global)",
                       0.0, 2.0, valinit=model.stim_amp)

    # Gamma-bump default guesses:
    # alpha=2; theta_ms controls peak time ~ (alpha-1)*theta_ms for alpha>1.
    # If you want peak around ~50ms after onset, theta_ms ~ 50 for alpha=2.
    s_gamp = Slider(s_ax_gamp,   "gamma bump amp", 0.0, 400.0, valinit=60.0)
    s_onset = Slider(s_ax_onset,  "gamma onset (ms)", 0.0, 100.0, valinit=25.0)
    s_dur = Slider(s_ax_dur,    "gamma duration (ms)",
                   20.0, 600.0, valinit=200.0)
    s_alpha = Slider(s_ax_alpha,  "gamma α", 1.1, 8.0, valinit=2.0)
    s_theta = Slider(s_ax_theta,  "gamma θ (ms)", 5.0, 200.0, valinit=40.0)
    s_gkappa = Slider(s_ax_gkappa, "gamma spatial κ", 0.5, 20.0, valinit=8.0)

    # Buttons & checkbox
    b_ax_toggle = fig.add_axes([0.75, 0.18, 0.10, 0.06])
    b_ax_reset = fig.add_axes([0.87, 0.18, 0.10, 0.06])
    b_ax_stim = fig.add_axes([0.75, 0.26, 0.22, 0.06])

    b_toggle = Button(b_ax_toggle, "Pause")
    b_reset = Button(b_ax_reset,  "Reset")
    b_stim = Button(b_ax_stim,   "Stim")

    c_ax = fig.add_axes([0.75, 0.06, 0.22, 0.10])
    c = CheckButtons(c_ax, ["cue enabled"], [True])

    # Callbacks
    def on_slider(_):
        speed_ms_per_frame["val"] = s_speed.val
        model.omega = s_omega.val
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_kappa.val
        model.stim_amp = s_stimamp.val

    def on_toggle(_):
        running["on"] = not running["on"]
        b_toggle.label.set_text("Pause" if running["on"] else "Run")

    def on_reset(_):
        seed = model.cue_angle if model.cue_amp > 0 else np.pi
        model.__init__(N=model.N, dt=model.dt,
                       tau=model.tau)  # re-init defaults

        # restore UI-controlled params
        speed_ms_per_frame["val"] = s_speed.val
        model.omega = s_omega.val
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_kappa.val
        model.stim_amp = s_stimamp.val

        # clear pending events
        model._stim_events = []
        model._gamma_events = []

        # re-seed bump near seed
        model.r = 0.05 * np.random.rand(model.N)
        model.r += 0.8 * np.exp(10.0 * np.cos(model.theta - seed))

    def on_check(label):
        if label == "cue enabled":
            model.cue_enabled = not model.cue_enabled

    def on_stim(_):
        # Global "stim" still scheduled at 9–11 ms post-click (you can adjust here)
        model.schedule_stim(delay_ms=9.0, width_ms=2.0)

        # Gamma-like HD-aligned bump: 25 ms after click, lasts ~200 ms
        model.schedule_gamma_bump(
            onset_ms=s_onset.val,
            dur_ms=s_dur.val,
            amp=s_gamp.val,
            alpha=s_alpha.val,
            theta_ms=s_theta.val,
            kappa=s_gkappa.val
        )

    # bind callbacks
    for s in [s_speed, s_omega, s_cueang, s_cueamp, s_kappa,
              s_stimamp, s_gamp, s_onset, s_dur, s_alpha, s_theta, s_gkappa]:
        s.on_changed(on_slider)

    b_toggle.on_clicked(on_toggle)
    b_reset.on_clicked(on_reset)
    b_stim.on_clicked(on_stim)
    c.on_clicked(on_check)

    # Animation loop
    # UI refresh; slow-mo handled via steps_per_frame
    timer = fig.canvas.new_timer(interval=30)

    def update(*_):
        if running["on"]:
            model.step(n_steps=compute_steps_per_frame())

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
            f"stim amp(global): {model.stim_amp:.2f} | stim events: {
                len(model._stim_events)} | gamma events: {len(model._gamma_events)}"
        )

        fig.canvas.draw_idle()

    timer.add_callback(update)
    timer.start()
    plt.show()


if __name__ == "__main__":
    run_interactive()
