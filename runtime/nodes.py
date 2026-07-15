"""Runtime nodes. Each mirrors a future ROS2 node 1:1.

Topics:
    /sim/state    {t, x(5), omega}                ground truth (viz/log only)
    /sensors      {v_enc, omega_gyro, gps|None}
    /estimate     {x, y, psi, cov}
    /path         {points Nx2, planner}
    /cmd          {a, ddelta}
    /metrics      {e_ct, e_psi, v, solve_ms, ...}
    /status       {goal_reached, paused, ...}
"""
import time
import numpy as np

from simulation.vehicle import step, L
from simulation.sensors import SensorSim
from theory.estimators import EKF, UKF, FactorGraph
from controllers.geometric import pid, pure_pursuit, stanley
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC

ESTIMATORS = {'ekf': EKF, 'ukf': UKF, 'factor_graph': FactorGraph}
ADAPTIVE_PARAMS = (1.89, 1.35)          # CEM-learned (benchmarks/run_full_flow.py)


class SimNode:
    """Vehicle dynamics @ sim rate. Consumes /cmd, publishes /sim/state."""

    def __init__(self, bus, world, dt):
        self.bus, self.dt = bus, dt
        self.x = np.array([world.start[0], world.start[1],
                           world.start_psi, 0.0, 0.0])
        self.u = np.zeros(2)
        self.t = 0.0
        bus.subscribe('/cmd', self._on_cmd)

    def _on_cmd(self, msg):
        self.u = np.array([msg['a'], msg['ddelta']])

    def tick(self):
        omega = self.x[3] / L * np.tan(self.x[4])
        self.x = step(self.x, self.u, self.dt)
        self.t += self.dt
        self.bus.publish('/sim/state',
                         {'t': self.t, 'x': self.x.copy(), 'omega': omega})


class SensorNode:
    """Wraps SensorSim: encoder+gyro every tick, GPS at 5 Hz."""

    def __init__(self, bus, dt, seed=0):
        self.bus = bus
        self.sim = SensorSim(dt, seed=seed)
        self.k = 0
        bus.subscribe('/sim/state', self._on_state)

    def _on_state(self, msg):
        v_enc, om, gps = self.sim.measure(msg['x'], msg['omega'], self.k)
        self.k += 1
        self.bus.publish('/sensors',
                         {'t': msg['t'], 'v_enc': v_enc,
                          'omega_gyro': om, 'gps': gps})


class LocalizationNode:
    """Runs the selected estimator (and optionally shadows for comparison)."""

    def __init__(self, bus, x0, dt, estimator='ekf', shadow=False):
        self.bus, self.dt = bus, dt
        self.active = estimator
        names = list(ESTIMATORS) if shadow else [estimator]
        self.filters = {n: ESTIMATORS[n](np.asarray(x0, float)) for n in names}
        bus.subscribe('/sensors', self._on_sensors)

    def _on_sensors(self, msg):
        out = {}
        for n, f in self.filters.items():
            f.predict(msg['v_enc'], msg['omega_gyro'], self.dt)
            if msg['gps'] is not None:
                f.update_gps(msg['gps'])
            cov = getattr(f, 'P', None)
            if cov is None and hasattr(f, 'kf'):
                cov = f.kf.P
            out[n] = {'x': float(f.x[0]), 'y': float(f.x[1]),
                      'psi': float(f.x[2]),
                      'cov': None if cov is None else np.asarray(cov)[:2, :2]}
        est = out[self.active]
        est = {**est, 't': msg['t'], 'all': out, 'active': self.active}
        self.bus.publish('/estimate', est)


class PlannerNode:
    """Plans on demand (start / replan button / planner change)."""

    def __init__(self, bus, world, planner='hybrid_astar'):
        self.bus, self.world, self.planner = bus, world, planner
        self.last_plan_ms = 0.0

    def plan(self):
        t0 = time.perf_counter()
        path = self.world.plan(self.planner)
        self.last_plan_ms = (time.perf_counter() - t0) * 1e3
        self.bus.publish('/path', {'points': path, 'planner': self.planner,
                                   'plan_ms': self.last_plan_ms})
        return path


class ControllerNode:
    """Runs the selected controller on the ESTIMATED pose (realistic mode).
    Publishes to /cmd_raw; the SafetyNode owns /cmd."""

    GEOMETRIC = {'pid': pid, 'pure_pursuit': pure_pursuit, 'stanley': stanley}

    def __init__(self, bus, controller='mpc', use_ground_truth=False,
                 world=None, ai=None, out_topic='/cmd_raw'):
        self.bus = bus
        self.name = controller
        self.use_gt = use_ground_truth
        self.world = world
        self.ai = ai                             # ResidualPolicy or None
        self.out_topic = out_topic
        self.path = None
        self._impl = None
        self.solve_ms = 0.0
        self.est = None
        self.gt = None
        bus.subscribe('/path', self._on_path)
        bus.subscribe('/estimate', self._on_est)
        bus.subscribe('/sim/state', self._on_gt)

    def set_controller(self, name):
        self.name = name
        self._impl = None                        # rebuild on next tick

    def _on_path(self, msg):
        self.path = msg['points']
        self._impl = None                        # new path -> fresh warm start

    def _on_est(self, msg):
        self.est = msg

    def _on_gt(self, msg):
        self.gt = msg

    def _build(self):
        closed = getattr(self.world, 'closed_path', True)
        if self.name == 'mpc':
            self._impl = MPC()
            self._impl.closed = closed
        elif self.name == 'adaptive_mpc':
            self._impl = AdaptiveMPC(ADAPTIVE_PARAMS, self.path)
            self._impl.mpc.closed = closed
        elif self.name == 'dwa':
            from control.dwa import DWA
            self._impl = DWA(self.world)
        else:
            self._impl = self.GEOMETRIC[self.name]

    def tick(self):
        if self.path is None or self.gt is None or \
                (self.est is None and not self.use_gt):
            return
        if self._impl is None:
            self._build()
        gt_x = self.gt['x']
        if self.use_gt:
            x_ctrl = gt_x
        else:
            e = self.est
            x_ctrl = np.array([e['x'], e['y'], e['psi'], gt_x[3], gt_x[4]])
        t0 = time.perf_counter()
        if self.name == 'mpc':
            u, _ = self._impl.solve(x_ctrl, self.path)
        elif self.name == 'adaptive_mpc':
            u, _ = self._impl.solve(x_ctrl)
        else:
            u = self._impl(x_ctrl, self.path)
        if self.name == 'dwa':                    # candidates for the viz
            self.bus.publish('/local_plan', {
                'candidates': [(t, c) for t, c, ok in
                               self._impl.candidates if ok][:20],
                'best': self._impl.best_traj})
        if self.ai is not None:
            obs = np.array([x_ctrl[3], x_ctrl[4],
                            *self.bus.latest.get('/metrics',
                                                 {'e_ct': 0, 'e_psi': 0}
                                                 ).values()][:8])
            u = self.ai.correct(np.asarray(u, float), obs)
        self.solve_ms = (time.perf_counter() - t0) * 1e3
        self.bus.publish(self.out_topic,
                         {'t': self.gt['t'], 'a': float(u[0]),
                          'ddelta': float(u[1]),
                          'solve_ms': self.solve_ms,
                          'controller': self.name})


class MetricsNode:
    """True cross-track/heading error vs the current path, from ground truth."""

    def __init__(self, bus):
        self.bus = bus
        self.path = None
        bus.subscribe('/path', lambda m: setattr(self, 'path', m['points']))
        bus.subscribe('/sim/state', self._on_state)

    def _on_state(self, msg):
        if self.path is None:
            return
        x = msg['x']
        p = self.path
        d = np.linalg.norm(p - x[:2], axis=1)
        i0 = int(np.argmin(d))
        i1 = min(i0 + 1, len(p) - 1)
        tg = p[i1] - p[max(i0 - 1, 0)]
        nrm = max(np.linalg.norm(tg), 1e-9)
        tg = tg / nrm
        e = x[:2] - p[i0]
        e_ct = float(-e[0] * tg[1] + e[1] * tg[0])
        psi_p = float(np.arctan2(tg[1], tg[0]))
        e_psi = float(np.arctan2(np.sin(psi_p - x[2]), np.cos(psi_p - x[2])))
        cmd = self.bus.latest.get('/cmd', {})
        est = self.bus.latest.get('/estimate')
        est_err = (float(np.hypot(est['x'] - x[0], est['y'] - x[1]))
                   if est else 0.0)
        self.bus.publish('/metrics', {
            't': msg['t'], 'e_ct': e_ct, 'e_psi': e_psi, 'v': float(x[3]),
            'delta': float(x[4]), 'solve_ms': cmd.get('solve_ms', 0.0),
            'est_err': est_err})


class LoggerNode:
    """Buffers every topic; flush() writes logs/run_XXX/*.csv + meta.json."""

    def __init__(self, bus):
        self.rows = {k: [] for k in
                     ('trajectory', 'sensors', 'estimate', 'metrics', 'cmd')}
        bus.subscribe('/sim/state', lambda m: self.rows['trajectory'].append(
            [m['t'], *m['x']]))
        bus.subscribe('/sensors', lambda m: self.rows['sensors'].append(
            [m['t'], m['v_enc'], m['omega_gyro'],
             *(m['gps'] if m['gps'] is not None else [np.nan, np.nan])]))
        bus.subscribe('/estimate', lambda m: self.rows['estimate'].append(
            [m['t'], m['x'], m['y'], m['psi']]))
        bus.subscribe('/metrics', lambda m: self.rows['metrics'].append(
            [m['t'], m['e_ct'], m['e_psi'], m['v'], m['delta'],
             m['solve_ms'], m['est_err']]))
        bus.subscribe('/cmd', lambda m: self.rows['cmd'].append(
            [m['t'], m['a'], m['ddelta'], m['solve_ms']]))

    HEADERS = {
        'trajectory': 't,x,y,psi,v,delta',
        'sensors': 't,v_enc,omega_gyro,gps_x,gps_y',
        'estimate': 't,x,y,psi',
        'metrics': 't,e_ct,e_psi,v,delta,solve_ms,est_err',
        'cmd': 't,a,ddelta,solve_ms',
    }

    def flush(self, run_dir, meta):
        import os, json
        os.makedirs(run_dir, exist_ok=True)
        for name, rows in self.rows.items():
            with open(os.path.join(run_dir, f'{name}.csv'), 'w') as f:
                f.write(self.HEADERS[name] + '\n')
                for r in rows:
                    f.write(','.join(f'{v:.6f}' for v in r) + '\n')
        path = meta.pop('_path', None)
        if path is not None:
            np.savetxt(os.path.join(run_dir, 'path.csv'), path,
                       delimiter=',', header='x,y', comments='')
        with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        return run_dir
