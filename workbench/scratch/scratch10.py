import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# ----------------------------
# Model: discrete ring attractor for head direction
# ----------------------------


class HDRingAttractor:
    def __init__(self, N=180, J0=-0.6, J1=1.4, tau=10.0, dt=0.5,
                 g_vel=1.0, use_relu=True, seed_bump_angle=np.pi):
        self.N = N
        self.theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.dt = dt
        self.tau = tau
        self.g_vel = g_vel

        dtheta = self.theta[:, None] - self.theta[None, :]
        self.W = (J0 + J1 * np.cos(dtheta)) / N

        # state
        self.r = 0.05 * np.random.rand(N)
        self.r += 0.8 * \
            np.exp(10.0 * np.cos(self.theta - seed_bump_angle))  # seed bump

        # inputs (controlled interactively)
        self.omega = 0.0                 # angular velocity drive
        self.cue_angle = 0.0             # external direction cue angle
        self.cue_amp = 0.0               # strength of cue input
        self.cue_kappa = 6.0             # sharpness of cue
        self.cue_enabled = True

        self.use_relu = use_relu

    def phi(self, x):
        if self.use_relu:
            return np.maximum(0.0, x)
        # softplus alternative (smooth ReLU) if you ever want it:
        return np.log1p(np.exp(x))

    def cue_input(self):
        if (not self.cue_enabled) or (self.cue_amp <= 0):
            return np.zeros(self.N)
        # von Mises-shaped cue centered at cue_angle
        return self.cue_amp * np.exp(self.cue_kappa * np.cos(self.theta - self.cue_angle))

    def decode_hd(self):
        z = np.sum(self.r * np.exp(1j * self.theta))
        return float(np.angle(z) % (2*np.pi))

    def step(self, n_steps=1):
        for _ in range(n_steps):
            # antisymmetric shift term ~ derivative on ring
            Kr = (np.roll(self.r, -1) - np.roll(self.r, 1)) / 2.0

            x = self.W @ self.r + self.cue_input() + (self.g_vel * self.omega * Kr)
            dr = (-self.r + self.phi(x)) / self.tau
            self.r = self.r + self.dt * dr


# ----------------------------
# Interactive UI (matplotlib widgets)
# ----------------------------
def run_interactive():
    model = HDRingAttractor(N=180)

    # Simulation control
    running = {"on": True}
    steps_per_frame = 3  # increase for faster dynamics, decrease for smoother UI

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 2])

    ax_ring = fig.add_subplot(gs[0, 0])
    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")

    # Main ring activity plot
    (line_ring,) = ax_ring.plot(model.theta, model.r, lw=2)
    ax_ring.set_xlim(0, 2*np.pi)
    ax_ring.set_xlabel("preferred direction θ (rad)")
    ax_ring.set_ylabel("activity r")
    ax_ring.set_title("HD Ring Attractor Activity (180 neurons)")

    # Polar readout: decoded HD + cue angle
    ax_polar.set_title("Decoded HD & External Cue")
    decoded_marker = ax_polar.plot(
        [model.decode_hd()], [1.0], marker="o", markersize=10)[0]
    cue_marker = ax_polar.plot(
        [model.cue_angle], [1.0], marker="x", markersize=10)[0]
    ax_polar.set_rlim(0, 1.2)
    ax_polar.set_rticks([])

    # Text readout
    txt = ax_ring.text(
        0.02, 0.95, "", transform=ax_ring.transAxes, va="top", ha="left"
    )

    # Sliders area
    ax_sliders = fig.add_subplot(gs[1, :])
    ax_sliders.axis("off")

    # Slider axes (manually positioned inside bottom area)
    s_ax_omega = fig.add_axes([0.10, 0.18, 0.60, 0.03])
    s_ax_cueang = fig.add_axes([0.10, 0.13, 0.60, 0.03])
    s_ax_cueamp = fig.add_axes([0.10, 0.08, 0.60, 0.03])
    s_ax_kappa = fig.add_axes([0.10, 0.03, 0.60, 0.03])

    s_omega = Slider(s_ax_omega, "ω (turn rate)", -0.05,
                     0.05, valinit=0.0, valstep=0.001)
    s_cueang = Slider(s_ax_cueang, "cue angle", 0.0, 2*np.pi, valinit=0.0)
    s_cueamp = Slider(s_ax_cueamp, "cue strength", 0.0, 1.0, valinit=0.0)
    s_kappa = Slider(s_ax_kappa, "cue sharpness κ",
                     0.5, 12.0, valinit=model.cue_kappa)

    # Buttons & checkbox
    b_ax_toggle = fig.add_axes([0.75, 0.15, 0.10, 0.06])
    b_ax_reset = fig.add_axes([0.87, 0.15, 0.10, 0.06])
    b_toggle = Button(b_ax_toggle, "Pause")
    b_reset = Button(b_ax_reset, "Reset")

    c_ax = fig.add_axes([0.75, 0.03, 0.22, 0.10])
    c = CheckButtons(c_ax, ["cue enabled"], [True])

    # Callbacks
    def on_slider(_):
        model.omega = s_omega.val
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_kappa.val

    def on_toggle(_):
        running["on"] = not running["on"]
        b_toggle.label.set_text("Pause" if running["on"] else "Run")

    def on_reset(_):
        # re-seed bump close to current cue if cue strength > 0, else pi
        seed = model.cue_angle if model.cue_amp > 0 else np.pi
        model.__init__(N=model.N)  # re-init defaults
        model.cue_angle = s_cueang.val
        model.cue_amp = s_cueamp.val
        model.cue_kappa = s_kappa.val
        model.omega = s_omega.val

        # re-seed bump near seed
        model.r = 0.05 * np.random.rand(model.N)
        model.r += 0.8 * np.exp(10.0 * np.cos(model.theta - seed))

    def on_check(label):
        if label == "cue enabled":
            model.cue_enabled = not model.cue_enabled

    s_omega.on_changed(on_slider)
    s_cueang.on_changed(on_slider)
    s_cueamp.on_changed(on_slider)
    s_kappa.on_changed(on_slider)
    b_toggle.on_clicked(on_toggle)
    b_reset.on_clicked(on_reset)
    c.on_clicked(on_check)

    # Animation loop using timer (no FuncAnimation headaches)
    timer = fig.canvas.new_timer(interval=30)  # ms

    def update(*_):
        if running["on"]:
            model.step(n_steps=steps_per_frame)

        # Update plots
        line_ring.set_ydata(model.r)

        hd = model.decode_hd()
        decoded_marker.set_data([hd], [1.0])
        cue_marker.set_data([model.cue_angle], [1.0])

        txt.set_text(
            f"decoded HD: {hd:.3f} rad ({np.degrees(hd):.1f}°)\n"
            f"ω: {model.omega:.4f}  | cue: {
                'on' if model.cue_enabled else 'off'} "
            f"(A={model.cue_amp:.2f}, κ={model.cue_kappa:.1f})"
        )

        fig.canvas.draw_idle()

    timer.add_callback(update)
    timer.start()

    plt.show()


if __name__ == "__main__":
    run_interactive()
