"""Live visualization windows. Each is its own matplotlib figure (= its own
OS window on your laptop), updated by the launcher/dashboard loop.

Every window reads only from bus.latest + engine state — pure observers,
zero influence on the simulation (same discipline as RViz).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import deque

ROBOT_SHAPE = np.array([[0.25, 0.0], [-0.15, 0.13], [-0.15, -0.13]])
EST_COLORS = {'ekf': 'tab:blue', 'ukf': 'tab:orange',
              'factor_graph': 'tab:green'}


def _robot_patch(ax, color='tab:red', z=5):
    p = plt.Polygon([[0, 0]], closed=True, fc=color, ec='k',
                    zorder=z, alpha=0.9)
    ax.add_patch(p)
    return p


def _place(patch, x, y, psi, scale=1.0):
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    patch.set_xy(ROBOT_SHAPE * scale @ R.T + [x, y])


class MapWindow:
    """World, path, robot, trail — the main 'Gazebo view'."""
    title = 'Robot / World'

    def __init__(self, engine):
        self.e = engine
        self.fig, self.ax = plt.subplots(num=self.title, figsize=(7, 5))
        w = engine.world
        if w.has_obstacles:
            from planners.planners import RES
            self.ax.imshow(w.map.grid.T, origin='lower', cmap='Greys',
                           extent=[0, w.map.nx * RES, 0, w.map.ny * RES],
                           alpha=.55, zorder=0)
            self.ax.plot(*w.goal, 'r*', ms=16, zorder=4, label='goal')
        self.path_line, = self.ax.plot([], [], 'k--', lw=1, alpha=.6,
                                       label='path')
        self.trail, = self.ax.plot([], [], '-', color='tab:blue', lw=1.4,
                                   alpha=.8, label='driven')
        self.robot = _robot_patch(self.ax)
        self.hud = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                va='top', fontsize=9, family='monospace',
                                bbox=dict(boxstyle='round', fc='w', alpha=.85))
        self.ax.set(xlim=w.bounds[:2], ylim=w.bounds[2:], aspect='equal')
        self.ax.legend(loc='lower right', fontsize=7)
        self.tx, self.ty = deque(maxlen=1200), deque(maxlen=1200)

    def update(self):
        st = self.e.bus.latest.get('/sim/state')
        if st is None:
            return
        x = st['x']
        self.tx.append(x[0]); self.ty.append(x[1])
        self.trail.set_data(self.tx, self.ty)
        _place(self.robot, x[0], x[1], x[2])
        if self.e.path is not None:
            self.path_line.set_data(self.e.path[:, 0], self.e.path[:, 1])
        m = self.e.bus.latest.get('/metrics', {})
        self.hud.set_text(
            f"t={st['t']:6.1f}s  v={x[3]:.2f} m/s\n"
            f"ctrl={self.e.cfg['controller']}  "
            f"e_ct={m.get('e_ct', 0):+.3f} m\n"
            f"planner={self.e.cfg['planner']}  "
            f"est={self.e.cfg['estimator']}")


class LocalizationWindow:
    """Ground truth vs every estimator, with 2-sigma covariance ellipse."""
    title = 'Localization'

    def __init__(self, engine):
        self.e = engine
        self.fig, self.ax = plt.subplots(num=self.title, figsize=(6, 4.5))
        w = engine.world
        self.gt = _robot_patch(self.ax, 'k', z=6)
        self.est_pts = {n: self.ax.plot([], [], 'o', ms=5, color=c,
                                        label=n.upper())[0]
                        for n, c in EST_COLORS.items()}
        self.ellipse = Ellipse((0, 0), 0.1, 0.1, fill=False,
                               color='tab:blue', lw=1.2, zorder=7)
        self.ax.add_patch(self.ellipse)
        self.hud = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                va='top', fontsize=9, family='monospace',
                                bbox=dict(boxstyle='round', fc='w', alpha=.85))
        self.ax.set(xlim=w.bounds[:2], ylim=w.bounds[2:], aspect='equal',
                    title='GT (black) vs estimators')
        self.ax.legend(loc='lower right', fontsize=7)

    def update(self):
        st = self.e.bus.latest.get('/sim/state')
        est = self.e.bus.latest.get('/estimate')
        if st is None or est is None:
            return
        x = st['x']
        _place(self.gt, x[0], x[1], x[2])
        for n, pt in self.est_pts.items():
            d = est['all'].get(n)
            pt.set_data([d['x']], [d['y']]) if d else pt.set_data([], [])
        act = est['all'][est['active']]
        cov = act['cov']
        if cov is not None:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-12)
            self.ellipse.set_center((act['x'], act['y']))
            self.ellipse.width, self.ellipse.height = 4 * np.sqrt(vals)
            self.ellipse.angle = float(np.degrees(
                np.arctan2(vecs[1, -1], vecs[0, -1])))
        err = np.hypot(act['x'] - x[0], act['y'] - x[1])
        self.hud.set_text(f"active={est['active']}\n|err|={err:.3f} m")


class ControllerWindow:
    """Steering/throttle commands + cross-track error, scrolling."""
    title = 'Controller'

    def __init__(self, engine, window_s=15.0):
        self.e = engine
        n = int(window_s / engine.cfg['dt'])
        self.buf = {k: deque(maxlen=n) for k in ('t', 'a', 'dd', 'ect')}
        self.fig, axs = plt.subplots(3, 1, num=self.title, figsize=(6, 5),
                                     sharex=True)
        self.l_a, = axs[0].plot([], [], color='tab:green')
        axs[0].set_ylabel('a [m/s²]'); axs[0].set_ylim(-3.2, 2.2)
        self.l_d, = axs[1].plot([], [], color='tab:orange')
        axs[1].set_ylabel('δ̇ [rad/s]'); axs[1].set_ylim(-1.1, 1.1)
        self.l_e, = axs[2].plot([], [], color='tab:red')
        axs[2].set_ylabel('e_ct [m]'); axs[2].set_ylim(-0.5, 0.5)
        axs[2].set_xlabel('t [s]')
        for a in axs:
            a.grid(alpha=.3)
        self.axs = axs
        self.fig.tight_layout()

    def update(self):
        c = self.e.bus.latest.get('/cmd')
        m = self.e.bus.latest.get('/metrics')
        if c is None or m is None:
            return
        self.buf['t'].append(m['t'])
        self.buf['a'].append(c['a'])
        self.buf['dd'].append(c['ddelta'])
        self.buf['ect'].append(m['e_ct'])
        t = list(self.buf['t'])
        self.l_a.set_data(t, self.buf['a'])
        self.l_d.set_data(t, self.buf['dd'])
        self.l_e.set_data(t, self.buf['ect'])
        if t:
            for a in self.axs:
                a.set_xlim(max(0, t[-1] - 15), max(15, t[-1]))


class MetricsWindow:
    """Speed, heading error, solve time, estimation error — scrolling."""
    title = 'Performance'

    def __init__(self, engine, window_s=15.0):
        self.e = engine
        n = int(window_s / engine.cfg['dt'])
        self.buf = {k: deque(maxlen=n) for k in
                    ('t', 'v', 'epsi', 'solve', 'est')}
        self.fig, axs = plt.subplots(2, 2, num=self.title, figsize=(7, 5))
        cfgs = [('v [m/s]', 'v', (0, 2.2), 'tab:blue'),
                ('e_psi [rad]', 'epsi', (-0.6, 0.6), 'tab:purple'),
                ('solve [ms]', 'solve', (0, 25), 'tab:green'),
                ('est err [m]', 'est', (0, 0.3), 'tab:red')]
        self.lines = {}
        for ax, (lab, key, ylim, col) in zip(axs.flat, cfgs):
            self.lines[key], = ax.plot([], [], color=col)
            ax.set_title(lab, fontsize=9)
            ax.set_ylim(*ylim)
            ax.grid(alpha=.3)
        self.axs = axs
        self.fig.tight_layout()

    def update(self):
        m = self.e.bus.latest.get('/metrics')
        if m is None:
            return
        self.buf['t'].append(m['t'])
        self.buf['v'].append(m['v'])
        self.buf['epsi'].append(m['e_psi'])
        self.buf['solve'].append(m['solve_ms'])
        self.buf['est'].append(m['est_err'])
        t = list(self.buf['t'])
        for key, line in self.lines.items():
            line.set_data(t, self.buf[key])
        if t:
            for ax in self.axs.flat:
                ax.set_xlim(max(0, t[-1] - 15), max(15, t[-1]))


ALL_WINDOWS = [MapWindow, LocalizationWindow, ControllerWindow, MetricsWindow]


class LocalPlannerWindow:
    """DWA candidate trajectories: grey = sampled, green = chosen (Nav2-style)."""
    title = 'Local Planner (DWA)'

    def __init__(self, engine):
        self.e = engine
        self.fig, self.ax = plt.subplots(num=self.title, figsize=(6, 4.5))
        w = engine.world
        if w.has_obstacles:
            from planners.planners import RES
            self.ax.imshow(w.map.grid.T, origin='lower', cmap='Greys',
                           extent=[0, w.map.nx * RES, 0, w.map.ny * RES],
                           alpha=.5)
        self.cand_lines = [self.ax.plot([], [], '-', color='grey', lw=0.8,
                                        alpha=0.5)[0] for _ in range(20)]
        self.best_line, = self.ax.plot([], [], '-', color='tab:green', lw=2.5)
        self.robot = _robot_patch(self.ax)
        self.ax.set(xlim=w.bounds[:2], ylim=w.bounds[2:], aspect='equal',
                    title='candidates (grey) vs chosen (green)')

    def update(self):
        st = self.e.bus.latest.get('/sim/state')
        lp = self.e.bus.latest.get('/local_plan')
        if st is None:
            return
        x = st['x']
        _place(self.robot, x[0], x[1], x[2])
        if lp is None:
            return
        for line, item in zip(self.cand_lines,
                              lp['candidates'] + [None] * 20):
            if item is None:
                line.set_data([], [])
            else:
                traj, _c = item
                line.set_data(traj[:, 0], traj[:, 1])
        if lp['best'] is not None:
            self.best_line.set_data(lp['best'][:, 0], lp['best'][:, 1])
