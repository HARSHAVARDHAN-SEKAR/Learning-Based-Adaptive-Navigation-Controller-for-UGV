"""Verification suite: python3 tests/test_all_modules.py (run from repo root)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from theory.vehicle_model import step
from theory.estimators import EKF, UKF, FactorGraph
from planners.planners import GridMap, astar, hybrid_astar, rrt_star, mppi
from controllers.geometric import pid, pure_pursuit, stanley
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC
from rl.ppo_training import MetaMPCEnv

def main():
    x = step(np.zeros(5), np.array([1.0, 0.2]), 0.05); assert x.shape == (5,)
    for E in (EKF, UKF, FactorGraph):
        e = E(np.zeros(5)); e.predict(1.0, 0.1, 0.05)
        e.update_gps(np.array([0.05, 0.0])); assert np.all(np.isfinite(e.x))
    m = GridMap(seed=1)
    for fn in (lambda: astar(m), lambda: astar(m, True), lambda: hybrid_astar(m),
               lambda: rrt_star(m, iters=800), lambda: mppi(m, K=150, max_steps=200)):
        p = fn(); assert p is not None and len(p) > 2
    t = np.linspace(0, 2 * np.pi, 800)
    path = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
    xs = np.array([0, -0.3, np.pi / 4, 0.5, 0.0])
    for c in (pid, pure_pursuit, stanley):
        assert np.all(np.isfinite(c(xs, path)))
    mpc = MPC(); u, _ = mpc.solve(xs, path); assert np.all(np.isfinite(u))
    a = AdaptiveMPC((1.89, 1.35), path); u, _ = a.solve(xs)
    assert np.all(np.isfinite(u))
    env = MetaMPCEnv(); obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    obs, r, *_ = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs)) and np.isfinite(r)
    print('ALL 6 MODULES PASS')

if __name__ == '__main__':
    main()
