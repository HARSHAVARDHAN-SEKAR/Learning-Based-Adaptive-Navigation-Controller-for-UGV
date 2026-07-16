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

    # 7. runtime laboratory: engine + safety + DWA reach goal headless
    from runtime.engine import Engine, load_config
    cfg = load_config(); cfg.update(world='obstacles', controller='dwa')
    e = Engine(cfg)
    for _ in range(2400):
        e.tick()
        if e.done:
            break
    assert e.done, 'runtime lab: DWA failed to reach goal'
    cfg2 = load_config()                      # track world, MPC + EKF
    e2 = Engine(cfg2)
    for _ in range(100):
        e2.tick()
    m = e2.bus.latest['/metrics']
    assert abs(m['e_ct']) < 0.5 and np.isfinite(m['est_err'])

    # 8. RL residual env: obs/action API + reward finiteness
    from rl.train_residual import ResidualEnv
    renv = ResidualEnv('stanley', 'track')
    obs, _ = renv.reset()
    assert obs.shape == (8,)
    obs, r, *_ = renv.step(renv.action_space.sample())
    assert np.all(np.isfinite(obs)) and np.isfinite(r)

    print('ALL 8 MODULES PASS')

if __name__ == '__main__':
    main()
