"""Learning layer: adaptive MPC trained with the Cross-Entropy Method.

The learned policy schedules MPC reference speed from lookahead path
curvature:   v_ref(kappa) = v_base / (1 + k_curv * |kappa|)

CEM is a derivative-free policy-search method — same objective structure
as the PPO/SAC meta-policy (Deliverable 2), small enough to train in
minutes on CPU. PPO/SAC with the full 58-dim observation is the Isaac Sim
phase; this stage proves the ADAPTATION HYPOTHESIS end-to-end:
a learned speed schedule beats any fixed speed on the error/progress front.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from theory.vehicle_model import step
from controllers.mpc_controller import MPC

DT = 0.05


def path_curvature(path):
    """Discrete curvature at each path point."""
    d1 = np.gradient(path, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-9
    return num / den


class AdaptiveMPC:
    """MPC whose v_ref is scheduled by lookahead curvature (learned params)."""

    def __init__(self, params, path):
        self.v_base, self.k_curv = params
        self.mpc = MPC(v_ref=self.v_base)
        self.kappa = path_curvature(path)
        self.path = path

    def solve(self, x):
        # max curvature over ~0.7 m of lookahead (circular indexing)
        n = len(self.kappa)
        idxs = (self.mpc.prog_idx + np.arange(26)) % n
        k_ahead = np.max(self.kappa[idxs])
        self.mpc.v_ref = np.clip(
            self.v_base / (1.0 + self.k_curv * k_ahead), 0.4, 2.0)
        return self.mpc.solve(x, self.path)


def episode(params, path, T=20.0, x0=None):
    """Run one episode; return objective (lower = better) and metrics."""
    steps = int(T / DT)
    x = np.array([path[0, 0], path[0, 1] - 0.3, np.pi / 4, 0.0, 0.0]) \
        if x0 is None else x0.copy()
    ctrl = AdaptiveMPC(params, path)
    ect, v_hist, prog0 = [], [], None
    for _ in range(steps):
        u, _ = ctrl.solve(x)
        x = step(x, u, DT)
        d = np.linalg.norm(path - x[:2], axis=1)
        i0 = int(np.argmin(d))
        i1 = min(i0 + 1, len(path) - 1)
        tg = path[i1] - path[max(i0 - 1, 0)]
        tg = tg / max(np.linalg.norm(tg), 1e-9)
        e = x[:2] - path[i0]
        ect.append(-e[0] * tg[1] + e[1] * tg[0])
        v_hist.append(x[3])
    ect = np.array(ect[10:])
    v = np.array(v_hist)
    acc = np.gradient(v, DT)
    jerk = np.gradient(acc, DT)[10:]
    rms_e = float(np.sqrt(np.mean(ect ** 2)))
    rms_j = float(np.sqrt(np.mean(jerk ** 2)))
    progress = float(np.sum(v) * DT)               # distance travelled
    # objective: maximize progress, minimize error and jerk
    J = -progress + 40.0 * rms_e + 1.0 * rms_j
    return J, dict(rms_ect=rms_e, rms_jerk=rms_j, progress=progress)


def train_cem(path, iters=4, pop=8, elite=3, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([1.5, 2.0])                      # [v_base, k_curv]
    sig = np.array([0.35, 1.5])
    history = []
    for it in range(iters):
        cand = np.clip(mu + sig * rng.standard_normal((pop, 2)),
                       [0.6, 0.0], [2.0, 8.0])
        scores = []
        for c in cand:
            J, _ = episode(c, path)
            scores.append(J)
        order = np.argsort(scores)
        elites = cand[order[:elite]]
        mu, sig = elites.mean(0), elites.std(0) + 0.02
        history.append((it, float(np.min(scores)), mu.copy()))
        print(f'  CEM iter {it}: best J={np.min(scores):.3f}  '
              f'mu=[v_base={mu[0]:.2f}, k_curv={mu[1]:.2f}]', flush=True)
    return mu, history
