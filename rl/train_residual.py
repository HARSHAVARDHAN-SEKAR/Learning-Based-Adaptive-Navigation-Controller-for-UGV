"""RL residual training — GPU stage.

Trains a small correction policy ON TOP OF a base controller (Stanley or
MPC), never replacing it. Matches runtime/safety.py's ResidualPolicy
exactly: same 8-dim observation (rl/residual_obs.py), same action scale
(±[0.5 m/s^2, 0.3 rad/s]), so a model trained here drops straight into
the lab with `--ai ppo_residual`.

NOT run in this container — training needs real wall-clock and ideally a
GPU. The env below IS smoke-tested (see bottom of this docstring for the
command); training itself is yours to run.

    pip install -r requirements-rl.txt
    python3 rl/train_residual.py --base stanley --algo ppo --world obstacles
    python3 rl/train_residual.py --base mpc --algo sac --world track

Saves to rl/{algo}_residual.zip — exactly where safety.py's
ResidualPolicy looks for it. Also runs a before/after evaluation
(base controller alone vs base+residual) and prints a comparison table,
matching the format of benchmarks/run_full_flow.py so the numbers are
directly comparable to your existing results.

Smoke test (env sanity only, no training, runs in seconds):
    python3 -c "
    from rl.train_residual import ResidualEnv
    e = ResidualEnv('stanley', 'track')
    obs, _ = e.reset()
    obs, r, term, trunc, _ = e.step(e.action_space.sample())
    print('OK', obs.shape, r)"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym

from runtime.engine import Engine, load_config
from rl.residual_obs import residual_observation, OBS_DIM

ACTION_LIMITS = np.array([0.5, 0.3])   # must match runtime/safety.py ResidualPolicy.LIMITS


class ResidualEnv(gym.Env):
    """Wraps the real Engine: RL outputs a correction added to the base
    controller's command, filtered through the SAME safety layer the
    deployed lab uses — the policy trains under the exact constraints
    it will run under, not an idealized version of them."""

    def __init__(self, base_controller='stanley', world='track', seed=0):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                 (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.base = base_controller
        self.world = world
        self._seed = seed
        self.engine = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # A planner (esp. Hybrid A* on tightly-cluttered random maps) can
        # legitimately find no path for some map seeds. During training
        # this must NEVER crash the run -- just skip to the next seed.
        # Bounded retry count so a systematic config error still surfaces
        # as an error instead of spinning forever.
        for _ in range(20):
            self._seed += 1
            cfg = load_config()
            cfg.update(world=self.world, controller=self.base,
                      seed=self._seed, map_seed=self._seed + 1)
            try:
                self.engine = Engine(cfg)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(
                '20 consecutive map seeds failed to plan -- this is a '
                'real config problem (e.g. too many obstacles), not bad luck')
        self._pending_residual = np.zeros(2)
        self._patch_controller()
        self.k = 0
        # goal-progress baseline (obstacle world only -- see step() below
        # for why raw velocity alone is not a safe reward signal here)
        self._prev_dist = (
            float(np.linalg.norm(self.engine.sim.x[:2] - self.engine.world.goal))
            if self.world == 'obstacles' else None)
        return self._obs(), {}

    def _patch_controller(self):
        """Intercept the controller node's output so this env's action
        becomes the residual, instead of routing through ResidualPolicy
        (which would need a saved model — during training there isn't
        one yet)."""
        ctrl = self.engine.ctrl
        orig_tick = ctrl.tick

        def patched_tick():
            ctrl.ai = None                      # disable the normal path
            orig_tick()
            if self.engine.bus.latest.get(ctrl.out_topic):
                msg = self.engine.bus.latest[ctrl.out_topic]
                a = msg['a'] + self._pending_residual[0] * ACTION_LIMITS[0]
                dd = msg['ddelta'] + self._pending_residual[1] * ACTION_LIMITS[1]
                self.engine.bus.publish(ctrl.out_topic,
                                        {**msg, 'a': a, 'ddelta': dd})
        ctrl.tick = patched_tick

    def _obs(self):
        gt = self.engine.bus.latest.get('/sim/state')
        m = self.engine.bus.latest.get('/metrics', {})
        if gt is None:
            return np.zeros(OBS_DIM, np.float32)
        return residual_observation(gt['x'], m)

    def step(self, action):
        self._pending_residual = np.clip(np.asarray(action, float), -1, 1)
        self.engine.tick()
        self.k += 1
        m = self.engine.bus.latest.get('/metrics', {})
        e_ct = m.get('e_ct', 0.0)
        v = m.get('v', 0.0)

        if self.world == 'obstacles':
            # Raw velocity is NOT a safe reward here: a policy can rack up
            # huge cumulative reward by driving fast in tight, easy-to-
            # track loops or oscillations that never advance toward the
            # goal -- exactly the failure this replaced (0% success with
            # BETTER cross-track error than the base controller, because
            # the policy was camping on an easy segment instead of
            # covering the harder ground near the goal). Reward actual
            # advancement instead: distance-to-goal reduction each tick.
            dist = float(np.linalg.norm(
                self.engine.sim.x[:2] - self.engine.world.goal))
            progress = (self._prev_dist - dist
                       if self._prev_dist is not None else 0.0)
            self._prev_dist = dist
            r = (4.0 * progress - 3.0 * e_ct ** 2
                 - 0.02 * float(np.sum(action ** 2)) - 0.05)  # time pressure
        else:
            # Closed-loop track has no single goal to measure progress
            # against; speed around the loop is a reasonable proxy here.
            r = 1.0 * v - 3.0 * e_ct ** 2 - 0.02 * float(np.sum(action ** 2))

        max_t = 40.0 if self.world == 'track' else 120.0
        term = bool(self.engine.done) or abs(e_ct) > 1.0
        trunc = self.engine.sim.t >= max_t
        if term and self.world == 'obstacles' and self.engine.done:
            r += 20.0                            # bonus for reaching goal
        return self._obs(), float(r), term, trunc, {}


def evaluate(base, world, model=None, episodes=5):
    """Compare base controller alone vs base+residual. Same metric
    definitions as benchmarks/run_full_flow.py for direct comparability."""
    results = {'base': [], 'residual': []}
    for use_model, key in ((False, 'base'), (model is not None, 'residual')):
        if key == 'residual' and model is None:
            continue
        for ep in range(episodes):
            cfg = load_config()
            cfg.update(world=world, controller=base, seed=ep, map_seed=ep + 1)
            try:
                eng = Engine(cfg)
            except RuntimeError:
                results[key].append({'rms_ect': np.nan, 'success': False,
                                     'time_s': 0.0})
                continue
            ects = []
            max_t = 40.0 if world == 'track' else 120.0
            while eng.sim.t < max_t and not eng.done:
                if use_model:
                    gt = eng.bus.latest.get('/sim/state')
                    m = eng.bus.latest.get('/metrics', {})
                    if gt is not None:
                        obs = residual_observation(gt['x'], m)
                        act, _ = model.predict(obs, deterministic=True)
                        eng.ctrl.ai = None
                        eng.tick()
                        msg = eng.bus.latest[eng.ctrl.out_topic]
                        a = msg['a'] + act[0] * ACTION_LIMITS[0]
                        dd = msg['ddelta'] + act[1] * ACTION_LIMITS[1]
                        eng.bus.publish(eng.ctrl.out_topic,
                                        {**msg, 'a': a, 'ddelta': dd})
                    else:
                        eng.tick()
                else:
                    eng.tick()
                mm = eng.bus.latest.get('/metrics')
                if mm and mm['t'] > 0.5:
                    ects.append(mm['e_ct'])
            results[key].append({
                'rms_ect': float(np.sqrt(np.mean(np.square(ects)))) if ects else np.nan,
                'success': eng.done or world == 'track',
                'time_s': eng.sim.t})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='stanley',
                    choices=['pid', 'pure_pursuit', 'stanley', 'mpc', 'rpp'])
    ap.add_argument('--algo', default='ppo', choices=['ppo', 'sac'])
    ap.add_argument('--world', default='track', choices=['track', 'obstacles'])
    ap.add_argument('--steps', type=int, default=300_000)
    ap.add_argument('--eval-episodes', type=int, default=5)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(ResidualEnv(args.base, args.world))
    if args.algo == 'ppo':
        model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=1024,
                    batch_size=256, gamma=0.995, ent_coef=0.003, verbose=1,
                    tensorboard_log='rl/tb')
    else:
        model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=200_000,
                    batch_size=256, gamma=0.995, verbose=1,
                    tensorboard_log='rl/tb')

    model.learn(total_timesteps=args.steps)
    out = os.path.join(os.path.dirname(__file__), f'{args.algo}_residual.zip')
    model.save(out)
    print(f'Saved: {out}')

    print(f'\nEvaluating base={args.base} vs base+residual, '
          f'{args.eval_episodes} episodes each...')
    res = evaluate(args.base, args.world, model=model,
                   episodes=args.eval_episodes)
    for key in ('base', 'residual'):
        r = res[key]
        if not r:
            continue
        rms = np.nanmean([x['rms_ect'] for x in r])
        succ = 100 * np.mean([x['success'] for x in r])
        t = np.mean([x['time_s'] for x in r])
        print(f'  {key:10s}  RMS e_ct={rms:.4f} m  success={succ:.0f}%  '
              f'time={t:.1f}s')


if __name__ == '__main__':
    main()
