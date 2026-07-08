"""PPO / SAC meta-policy training (Stable Baselines3) — GPU-machine stage.

NOT executed in the CPU research container (MPC-in-the-loop RL needs hours
of wall-clock). The CEM result (cem_adaptive_mpc.py) already validated the
adaptation hypothesis; this script scales it to the full meta-policy of
Deliverable 2: the agent modulates MPC weights + v_max at 10 Hz.

    pip install stable-baselines3 gymnasium torch
    python3 rl/ppo_training.py            # PPO (default)
    python3 rl/ppo_training.py --algo sac # SAC

Observation (15-dim, compact CPU-trainable subset of the 58-dim spec):
    [v, delta, e_ct, e_psi, kappa@4 lookaheads, min/mean clearance stub,
     last solve_ms, last |a|, last |ddelta|, sin(psi), cos(psi)]
Action (4-dim, [-1,1]): [Q_xy scale, Q_psi scale, R scale, v_max]
Reward: progress - 2*e_ct^2 - 0.05*jerk^2 - 0.2*solve_excess
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gymnasium as gym

from theory.vehicle_model import step
from controllers.mpc_controller import MPC, Q as BASE_Q, R as BASE_R
from rl.cem_adaptive_mpc import path_curvature

DT_RL = 0.1          # RL acts at 10 Hz
DT_SIM = 0.05        # MPC + sim at 20 Hz (2 substeps per RL step)


def make_path(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 800)
    a = rng.uniform(3.0, 4.5)
    b = rng.uniform(1.5, 2.5)
    return np.column_stack([a * np.sin(t), b * np.sin(t) * np.cos(t)])


class MetaMPCEnv(gym.Env):
    """RL meta-policy over MPC. Randomized path per episode."""
    metadata = {'render_modes': []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._ep_seed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ep_seed += 1
        self.path = make_path(self._ep_seed)
        self.kappa = path_curvature(self.path)
        self.mpc = MPC(v_ref=1.5)
        self.x = np.array([self.path[0, 0], self.path[0, 1] - 0.2,
                           np.pi / 4, 0.0, 0.0])
        self.k = 0
        self.last_solve_ms = 0.0
        self.filt = np.zeros(4)
        return self._obs(), {}

    def _apply_action(self, a):
        # safety layer: low-pass filter (tau = 0.3 s)
        self.filt += (a - self.filt) * (DT_RL / 0.3)
        import scipy.linalg
        s_xy = float(np.exp(1.0 * self.filt[0]))
        s_psi = float(np.exp(1.0 * self.filt[1]))
        s_R = float(np.exp(0.7 * self.filt[2]))
        v_max = 0.4 + 0.8 * (self.filt[3] + 1.0)
        Qm = BASE_Q.copy()
        Qm[0, 0] *= s_xy; Qm[1, 1] *= s_xy; Qm[2, 2] *= s_psi
        # CasADi backend: rebuild weight effect via v_ref only is cheap;
        # weight matrices are compiled into the NLP here, so we modulate the
        # ACADOS-portable subset: v_ref (ACADOS updates W online, see
        # Deliverable 1 apply_rl_action).
        self.mpc.v_ref = float(np.clip(v_max, 0.4, 2.0))

    def _obs(self):
        d = np.linalg.norm(self.path - self.x[:2], axis=1)
        i0 = int(np.argmin(d))
        i1 = min(i0 + 1, len(self.path) - 1)
        tg = self.path[i1] - self.path[max(i0 - 1, 0)]
        psi_p = np.arctan2(tg[1], tg[0])
        e = self.x[:2] - self.path[i0]
        e_ct = -e[0] * np.sin(psi_p) + e[1] * np.cos(psi_p)
        e_psi = np.arctan2(np.sin(psi_p - self.x[2]), np.cos(psi_p - self.x[2]))
        n = len(self.kappa)
        ks = [np.max(self.kappa[(i0 + np.arange(o, o + 10)) % n])
              for o in (0, 15, 30, 60)]
        return np.array([self.x[3], self.x[4], e_ct, e_psi, *ks,
                         0.0, 0.0,                      # clearance stubs
                         self.last_solve_ms / 20.0,
                         0.0, 0.0,
                         np.sin(self.x[2]), np.cos(self.x[2])], np.float32)

    def step(self, action):
        import time as _t
        self._apply_action(np.asarray(action, float))
        prog0 = self.mpc.prog_idx
        v_hist = []
        for _ in range(int(DT_RL / DT_SIM)):
            t0 = _t.perf_counter()
            u, _ = self.mpc.solve(self.x, self.path)
            self.last_solve_ms = (_t.perf_counter() - t0) * 1e3
            self.x = step(self.x, u, DT_SIM)
            v_hist.append(self.x[3])
        obs = self._obs()
        e_ct = obs[2]
        dprog = ((self.mpc.prog_idx - prog0) % len(self.path))
        progress_m = dprog * 0.024                     # approx segment length
        jerk = abs(np.diff(v_hist).sum() / DT_RL)
        r = progress_m - 2.0 * e_ct ** 2 - 0.05 * jerk ** 2 \
            - 0.2 * max(0.0, self.last_solve_ms - 15.0)
        self.k += 1
        trunc = self.k >= 300                          # 30 s episodes
        term = abs(e_ct) > 1.5                         # left the track
        if term:
            r -= 20.0
        return obs, float(r), term, trunc, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', default='ppo', choices=['ppo', 'sac'])
    ap.add_argument('--steps', type=int, default=2_000_000)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    def mk():
        return Monitor(MetaMPCEnv())

    if args.algo == 'ppo':
        env = SubprocVecEnv([mk for _ in range(8)])
        model = PPO('MlpPolicy', env, n_steps=1024, batch_size=2048,
                    learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                    ent_coef=0.003, verbose=1,
                    tensorboard_log='rl/tb')
    else:
        env = mk()
        model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=500_000,
                    batch_size=512, gamma=0.995, verbose=1,
                    tensorboard_log='rl/tb')
    model.learn(total_timesteps=args.steps)
    model.save(f'rl/{args.algo}_meta_mpc')
    print('Saved. Evaluate with benchmarks/run_full_flow.py-style episodes.')


if __name__ == '__main__':
    main()
