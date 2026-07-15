"""Runtime engine: assembles the node graph and steps it deterministically.

Order per tick (mirrors the ROS2 data flow, but deterministic):
    sim -> sensors -> localization -> controller -> metrics -> logger
Planner runs on demand (session start, replan requests, planner change).
"""
import os
import json
import datetime
import numpy as np

from runtime.bus import Bus
from runtime.nodes import (SimNode, SensorNode, LocalizationNode,
                           PlannerNode, ControllerNode, MetricsNode,
                           LoggerNode)
from simulation.world import make_world

DEFAULT_CFG = {
    'world': 'track',            # track | obstacles
    'planner': 'hybrid_astar',   # astar | theta_star | hybrid_astar | rrt_star | mppi
    'controller': 'mpc',         # pid | pure_pursuit | stanley | mpc | adaptive_mpc
    'estimator': 'ekf',          # ekf | ukf | factor_graph
    'shadow_estimators': True,   # run all three for the localization window
    'use_ground_truth': False,
    'ai': 'none',             # none | ppo_residual | sac_residual   # feed controller GT instead of estimate
    'dt': 0.05,
    'seed': 0,
    'map_seed': 1,
    'n_obstacles': 8,
}


def load_config(path=None):
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bus = Bus()
        self.world = make_world(cfg['world'],
                                {'map_seed': cfg['map_seed'],
                                 'n_obstacles': cfg['n_obstacles']})
        dt = cfg['dt']
        self.sim = SimNode(self.bus, self.world, dt)
        self.sensors = SensorNode(self.bus, dt, seed=cfg['seed'])
        self.loc = LocalizationNode(self.bus, self.sim.x, dt,
                                    estimator=cfg['estimator'],
                                    shadow=cfg['shadow_estimators'])
        self.planner = PlannerNode(self.bus, self.world, cfg['planner'])
        from runtime.safety import SafetyNode, ResidualPolicy
        ai = ResidualPolicy(cfg.get('ai', 'none'))
        self.ctrl = ControllerNode(self.bus, cfg['controller'],
                                   use_ground_truth=cfg['use_ground_truth'],
                                   world=self.world, ai=ai)
        self.safety = SafetyNode(self.bus, self.world, self.sim)
        self.metrics = MetricsNode(self.bus)
        self.logger = LoggerNode(self.bus)
        self.done = False
        self.path = self.planner.plan()

    def tick(self):
        """One deterministic simulation step."""
        if self.done:
            return
        self.sim.tick()                      # also fires sensors/loc/metrics/log
        self.ctrl.tick()
        if self.world.goal_reached(self.sim.x):
            self.bus.publish('/cmd', {'t': self.sim.t, 'a': -3.0,
                                      'ddelta': 0.0, 'solve_ms': 0.0,
                                      'controller': 'brake'})
            if self.sim.x[3] < 0.05:
                self.done = True

    # -------- live re-configuration (dashboard hooks) --------
    def set_controller(self, name):
        self.cfg['controller'] = name
        self.ctrl.set_controller(name)

    def set_planner(self, name):
        self.cfg['planner'] = name
        self.planner.planner = name
        self.replan()

    def set_estimator(self, name):
        self.cfg['estimator'] = name
        self.loc.active = name

    def replan(self):
        if self.world.name == 'obstacles':
            self.path = self.planner.plan()

    # -------- session persistence --------
    def save_run(self, tag=''):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'logs', f'run_{stamp}{tag}')
        meta = {**self.cfg, 'sim_time_s': self.sim.t,
                'goal_reached': self.done, '_path': self.path}
        return self.logger.flush(run_dir, meta)
