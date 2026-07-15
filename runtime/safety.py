"""Safety layer + AI residual layer.

SafetyNode sits between the controller and the vehicle:
    /cmd_raw -> [rate clamp] -> [collision rollout] -> /cmd  (+ /safety)
An RL policy NEVER commands the robot directly — its correction passes
through the same filter (the design pattern real systems use).

ResidualPolicy: u = u_controller + clip(policy(obs)) if a trained SB3
model exists at rl/ppo_residual.zip / rl/sac_residual.zip, else identity
(with a one-time warning). Training script: rl/train_residual.py (GPU).
"""
import os
import numpy as np
from simulation.vehicle import L

A_LIM = (-3.0, 2.0)
DD_LIM = (-1.0, 1.0)
ESTOP_HORIZON_S = 0.5
ESTOP_DT = 0.1


class SafetyNode:
    def __init__(self, bus, world, sim):
        self.bus, self.world, self.sim = bus, world, sim
        self.estop = False
        bus.subscribe('/cmd_raw', self._on_cmd)

    def _ttc(self, x, u):
        """Time-to-collision along constant-curvature rollout, or None."""
        if not self.world.has_obstacles or x[3] < 0.2:
            return None
        s = x.copy()
        s[4] = np.clip(s[4] + u[1] * ESTOP_DT, -0.5, 0.5)
        for i in range(int(ESTOP_HORIZON_S / ESTOP_DT)):
            s[3] = np.clip(s[3] + u[0] * ESTOP_DT, 0.0, 2.0)
            s[0] += s[3] * np.cos(s[2]) * ESTOP_DT
            s[1] += s[3] * np.sin(s[2]) * ESTOP_DT
            s[2] += s[3] / L * np.tan(s[4]) * ESTOP_DT
            if self.world.map.occ(s[0], s[1]):
                return (i + 1) * ESTOP_DT
        return None

    def _on_cmd(self, msg):
        a = float(np.clip(msg['a'], *A_LIM))
        dd = float(np.clip(msg['ddelta'], *DD_LIM))
        ttc = self._ttc(self.sim.x, np.array([a, dd]))
        self.estop = ttc is not None and ttc < 0.25
        soft = ttc is not None and not self.estop
        if self.estop:                       # imminent: full brake
            a, dd = A_LIM[0], float(np.clip(-self.sim.x[4] / 0.15, *DD_LIM))
        elif soft:                           # distant: decelerate, keep steering
            a = min(a, -0.8)
        self.bus.publish('/safety', {'estop': self.estop, 'soft': soft,
                                     'ttc': ttc,
                                     'clamped': (a != msg['a']
                                                 or dd != msg['ddelta'])})
        self.bus.publish('/cmd', {**msg, 'a': a, 'ddelta': dd})


class ResidualPolicy:
    """AI layer: additive correction on top of any controller."""

    LIMITS = np.array([0.5, 0.3])           # max |residual| on [a, ddelta]

    def __init__(self, algo='none'):
        self.algo = algo
        self.model = None
        self._warned = False
        if algo in ('ppo_residual', 'sac_residual'):
            path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'rl',
                f"{algo.split('_')[0]}_residual.zip")
            if os.path.exists(path):
                from stable_baselines3 import PPO, SAC
                cls = PPO if algo.startswith('ppo') else SAC
                self.model = cls.load(path, device='cpu')

    def correct(self, u, obs):
        if self.algo == 'none' or self.model is None:
            if self.algo != 'none' and not self._warned:
                print(f'[AI] no trained model for {self.algo} — '
                      f'running base controller (train with '
                      f'rl/train_residual.py on GPU)')
                self._warned = True
            return u
        act, _ = self.model.predict(obs.astype(np.float32),
                                    deterministic=True)
        return u + np.clip(act, -1, 1) * self.LIMITS
