"""DWA-style local planner for the bicycle model (Nav2 DWB-inspired).

Samples (target speed, target steering) pairs, rolls each out ~1.2 s with
first-order actuator convergence, scores by path-following + heading +
speed + obstacle clearance, picks the best, and exposes ALL candidate
trajectories so the visualization can show accepted/rejected samples —
exactly how Nav2's DWB debugging view works.
"""
import numpy as np
from simulation.vehicle import L

V_SAMPLES = np.linspace(0.3, 1.8, 6)
D_SAMPLES = np.linspace(-0.45, 0.45, 9)
H_STEPS, H_DT = 12, 0.1
TAU_V, TAU_D = 0.3, 0.15
W_PATH, W_HEAD, W_SPEED, W_OBS = 3.0, 1.0, 0.4, 2.0


class DWA:
    def __init__(self, world):
        self.world = world
        self.candidates = []          # [(traj Nx2, cost, ok)] for viz
        self.best_traj = None

    def _rollout(self, x0, v_t, d_t):
        x = x0.copy()
        traj = np.zeros((H_STEPS, 2))
        min_clear = np.inf
        ok = True
        for i in range(H_STEPS):
            x[3] += (v_t - x[3]) * (H_DT / TAU_V)
            x[4] += (d_t - x[4]) * (H_DT / TAU_D)
            x[0] += x[3] * np.cos(x[2]) * H_DT
            x[1] += x[3] * np.sin(x[2]) * H_DT
            x[2] += x[3] / L * np.tan(x[4]) * H_DT
            traj[i] = x[:2]
            if self.world.has_obstacles:
                if self.world.map.occ(x[0], x[1]):
                    ok = False
                    break
                # coarse clearance probe
                for r, ang in ((0.2, 0.0), (0.2, np.pi / 2), (0.2, -np.pi / 2)):
                    px = x[0] + r * np.cos(x[2] + ang)
                    py = x[1] + r * np.sin(x[2] + ang)
                    if self.world.map.occ(px, py):
                        min_clear = min(min_clear, 0.0)
        return traj[:i + 1], ok, min_clear

    def __call__(self, x, path):
        # lookahead target ~1.5 m along path from nearest point
        d = np.linalg.norm(path - x[:2], axis=1)
        i0 = int(np.argmin(d))
        cap = getattr(self.world, 'core_len', len(path)) - 1
        ig = min(i0 + 60, cap)
        target = path[ig]
        tang = path[min(ig + 1, len(path) - 1)] - path[max(ig - 1, 0)]
        psi_t = np.arctan2(tang[1], tang[0])

        best, best_cost = None, np.inf
        self.candidates = []
        for v_t in V_SAMPLES:
            for d_t in D_SAMPLES:
                traj, ok, clear = self._rollout(x, v_t, d_t)
                if not ok or len(traj) == 0:
                    self.candidates.append((traj, np.inf, False))
                    continue
                end = traj[-1]
                head_end = np.arctan2(traj[-1][1] - traj[-2][1],
                                      traj[-1][0] - traj[-2][0]) \
                    if len(traj) > 1 else x[2]
                c = (W_PATH * np.linalg.norm(end - target)
                     + W_HEAD * abs(np.arctan2(np.sin(psi_t - head_end),
                                               np.cos(psi_t - head_end)))
                     + W_SPEED * (V_SAMPLES[-1] - v_t)
                     + (W_OBS if clear <= 0.0 else 0.0))
                self.candidates.append((traj, c, True))
                if c < best_cost:
                    best_cost, best = c, (v_t, d_t, traj)
        if best is None:                          # everything collides: stop
            self.best_traj = None
            return np.array([-3.0, -x[4] / TAU_D])
        v_t, d_t, self.best_traj = best
        a = np.clip((v_t - x[3]) / TAU_V, -3.0, 2.0)
        dd = np.clip((d_t - x[4]) / TAU_D, -1.0, 1.0)
        return np.array([a, dd])
