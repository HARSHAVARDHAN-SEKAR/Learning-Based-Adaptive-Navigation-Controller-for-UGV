"""Worlds for the navigation laboratory.

TrackWorld    — closed figure-eight, no obstacles (controller studies)
ObstacleWorld — grid map with obstacles, start->goal, path comes from a
                planner (planning studies + full-stack runs)
"""
import numpy as np
from planners.planners import GridMap, astar, hybrid_astar, rrt_star, mppi, RES


def resample(path, spacing=0.025):
    """Densify a coarse planner path to fixed arclength spacing.
    The MPC reference generator advances point-by-point, so coarse paths
    (Hybrid A* ~0.3 m spacing) make its reference race ahead ~10x too
    fast — every consumer gets a dense path instead."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 2)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, path[:, 0]),
                            np.interp(si, s, path[:, 1])])

PLANNER_FNS = {
    'astar':        lambda m, s: astar(m),
    'theta_star':   lambda m, s: astar(m, theta_variant=True),
    'hybrid_astar': lambda m, s: hybrid_astar(m),
    'rrt_star':     lambda m, s: rrt_star(m, seed=s),
    'mppi':         lambda m, s: mppi(m, seed=s),
}


class TrackWorld:
    """Closed figure-eight track. No planning needed; the path IS the plan."""
    name = 'track'
    closed_path = True
    has_obstacles = False

    def __init__(self, cfg=None):
        t = np.linspace(0, 2 * np.pi, 800)
        self.path = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
        self.start = np.array([self.path[0, 0], self.path[0, 1] - 0.3])
        self.start_psi = np.pi / 4
        self.goal = None
        self.bounds = (-5.5, 5.5, -3.0, 3.0)

    def plan(self, planner_name, seed=0):
        return self.path                    # fixed track

    def goal_reached(self, x):
        return False                        # loops forever


class ObstacleWorld:
    """Cluttered grid world; a planner produces the path start -> goal."""
    name = 'obstacles'
    closed_path = False
    has_obstacles = True

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.map = GridMap(seed=cfg.get('map_seed', 1),
                           n_obs=cfg.get('n_obstacles', 8))
        self.start = self.map.start.copy()
        self.goal = self.map.goal.copy()
        self.start_psi = float(np.arctan2(*(self.goal - self.start)[::-1]))
        self.path = None
        self.bounds = (0, self.map.nx * RES, 0, self.map.ny * RES)

    def plan(self, planner_name, seed=0):
        p = PLANNER_FNS[planner_name](self.map, seed)
        if p is None:
            raise RuntimeError(f'{planner_name} failed to find a path')
        # extend past the goal along the final tangent so the MPC's
        # lookahead reference doesn't wrap to the path start near the end
        tang = p[-1] - p[-2]
        tang = tang / max(np.linalg.norm(tang), 1e-9)
        ext = p[-1] + tang * np.linspace(0.1, 2.0, 20)[:, None]
        full = resample(np.vstack([p, ext]))
        self.core_len = int(np.argmin(
            np.linalg.norm(full - self.goal, axis=1))) + 1
        self.path = full
        return self.path

    def goal_reached(self, x):
        # 0.4 m, not 0.3 m: at 2 m/s and 20 Hz control, a controller can
        # travel ~10 cm between ticks, so a tight capture radius lets a
        # fast approach narrowly overshoot it, drive past the goal, and
        # follow the path all the way to its artificial tail extension --
        # parking 2 m past the real goal with no way back. Found via a
        # reproducible adaptive_mpc stall: missed capture by 0.028 m.
        return bool(np.linalg.norm(x[:2] - self.goal) < 0.4)


def make_world(name, cfg=None):
    return {'track': TrackWorld, 'obstacles': ObstacleWorld}[name](cfg)
