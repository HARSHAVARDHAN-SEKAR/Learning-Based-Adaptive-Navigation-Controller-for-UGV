"""Runtime Dashboard — the control panel of the laboratory.

    python3 runtime/dashboard.py
    python3 runtime/dashboard.py --world obstacles

One window of selectors + buttons drives the whole node graph; the live
windows (map, localization, controller, performance, DWA candidates)
open alongside. Switch controller / planner / estimator WHILE the robot
is driving and watch the behavior change.

Buttons: START/PAUSE toggles the sim - RESET rebuilds the engine with the
current selections - REPLAN re-runs the planner (obstacle world) -
SAVE writes logs/run_<stamp>/ and its PDF report.

Needs an interactive matplotlib backend (sudo apt install python3-tk).
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button

from runtime.engine import Engine, load_config

CONTROLLERS = ['pid', 'pure_pursuit', 'stanley', 'mpc', 'adaptive_mpc', 'dwa']
PLANNERS = ['astar', 'theta_star', 'hybrid_astar', 'rrt_star', 'mppi']
ESTIMATORS = ['ekf', 'ukf', 'factor_graph']
VIZ_HZ = 12


class Dashboard:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.engine = None
        self.windows = []
        self._build_panel()
        self._reset(None)

    # ---------------- panel ----------------
    def _build_panel(self):
        self.fig = plt.figure('Dashboard', figsize=(3.6, 7.2))
        self.fig.text(0.5, 0.975, 'NAV LAB', ha='center', fontsize=13,
                      fontweight='bold')

        def radio(y, h, title, options, active):
            ax = self.fig.add_axes([0.1, y, 0.8, h])
            ax.set_title(title, fontsize=9, loc='left')
            rb = RadioButtons(ax, options, active=options.index(active))
            for lbl in rb.labels:
                lbl.set_fontsize(8)
            return rb

        self.rb_ctrl = radio(0.66, 0.26, 'Controller', CONTROLLERS,
                             self.cfg['controller'])
        self.rb_plan = radio(0.44, 0.19, 'Planner', PLANNERS,
                             self.cfg['planner'])
        self.rb_est = radio(0.30, 0.11, 'Estimator', ESTIMATORS,
                            self.cfg['estimator'])
        self.rb_ctrl.on_clicked(self._on_ctrl)
        self.rb_plan.on_clicked(self._on_plan)
        self.rb_est.on_clicked(self._on_est)

        def button(y, label, cb, color='0.9'):
            ax = self.fig.add_axes([0.1, y, 0.8, 0.05])
            b = Button(ax, label, color=color, hovercolor='0.75')
            b.on_clicked(cb)
            return b

        self.b_start = button(0.22, 'START / PAUSE', self._toggle, '#a8e6a1')
        self.b_reset = button(0.155, 'RESET', self._reset, '#f6d186')
        self.b_replan = button(0.09, 'REPLAN', self._replan)
        self.b_save = button(0.025, 'SAVE RUN + REPORT', self._save,
                             '#a1c6e6')
        self.status = self.fig.text(0.5, 0.005, 'ready', ha='center',
                                    fontsize=8, color='0.3')

    # ---------------- callbacks ----------------
    def _on_ctrl(self, label):
        self.engine.set_controller(label)
        self._maybe_dwa_window()

    def _on_plan(self, label):
        self.engine.set_planner(label)

    def _on_est(self, label):
        self.engine.set_estimator(label)

    def _toggle(self, _):
        self.running = not self.running
        self.status.set_text('running' if self.running else 'paused')

    def _reset(self, _):
        self.cfg['controller'] = self.rb_ctrl.value_selected \
            if self.engine else self.cfg['controller']
        self.cfg['planner'] = self.rb_plan.value_selected \
            if self.engine else self.cfg['planner']
        self.cfg['estimator'] = self.rb_est.value_selected \
            if self.engine else self.cfg['estimator']
        self.running = False
        for w in self.windows:
            plt.close(w.fig)
        self.engine = Engine(dict(self.cfg))
        from visualization.windows import ALL_WINDOWS
        self.windows = [W(self.engine) for W in ALL_WINDOWS]
        self._maybe_dwa_window()
        self.status.set_text('reset — press START')

    def _maybe_dwa_window(self):
        from visualization.windows import LocalPlannerWindow
        have = any(isinstance(w, LocalPlannerWindow) for w in self.windows)
        if self.engine.cfg['controller'] == 'dwa' and not have:
            self.windows.append(LocalPlannerWindow(self.engine))

    def _replan(self, _):
        self.engine.replan()
        self.status.set_text('replanned')

    def _save(self, _):
        run_dir = self.engine.save_run()
        name = os.path.basename(run_dir)
        self.status.set_text(f'saved {name}')
        os.system(f'{sys.executable} runtime/report.py {name} &')

    # ---------------- main loop ----------------
    def loop(self):
        plt.ion()
        plt.show(block=False)
        dt = self.cfg['dt']
        ticks_per_frame = max(1, int((1.0 / VIZ_HZ) / dt))
        t_wall = time.perf_counter()
        t_sim0 = 0.0
        while plt.fignum_exists(self.fig.number):
            if self.running and not self.engine.done:
                for _ in range(ticks_per_frame):
                    self.engine.tick()
                for w in self.windows:
                    if plt.fignum_exists(w.fig.number):
                        w.update()
                        w.fig.canvas.draw_idle()
                # pace to real time
                lag = (self.engine.sim.t - t_sim0) - \
                      (time.perf_counter() - t_wall)
                if lag > 0:
                    time.sleep(min(lag, 0.1))
            else:
                t_wall = time.perf_counter()
                t_sim0 = self.engine.sim.t
            plt.pause(0.02)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--world', choices=['track', 'obstacles'])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.world:
        cfg['world'] = args.world
    Dashboard(cfg).loop()
