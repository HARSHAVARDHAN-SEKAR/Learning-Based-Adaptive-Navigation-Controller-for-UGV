"""Regulated Pure Pursuit (RPP) — Nav2's default local planner.

Pure Pursuit chases a fixed lookahead point regardless of context. RPP
adds three regulations Nav2 actually ships with:
  1. Curvature regulation: shrink speed when the lookahead arc is tight
     (proportional to sqrt(radius) — the same heuristic Nav2 uses).
  2. Proximity regulation: shrink speed near obstacles (uses the same
     world.map.occ() probe DWA uses, so it's a fair comparison).
  3. Approach regulation: shrink speed near the goal so it doesn't
     overshoot and oscillate at the end of a path.
Falls back to plain Pure Pursuit behavior when none of these apply —
so on the open figure-eight track it should track pure_pursuit.py
closely, and only diverge in the obstacle world.
"""
import numpy as np

L = 0.32
V_MAX = 1.8
LD0, KV = 0.5, 0.5
CURV_GAIN = 1.2          # higher = brakes harder for tight turns
PROX_RADIUS = 0.8        # start regulating speed within this clearance
APPROACH_RADIUS = 1.0    # start regulating speed within this goal distance


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _clearance(x, world):
    if not getattr(world, 'has_obstacles', False):
        return np.inf
    best = np.inf
    for ang in np.linspace(-np.pi / 2, np.pi / 2, 7):
        for r in (0.2, 0.4, 0.6, 0.8):
            px = x[0] + r * np.cos(x[2] + ang)
            py = x[1] + r * np.sin(x[2] + ang)
            if world.map.occ(px, py):
                best = min(best, r)
                break
    return best


def regulated_pure_pursuit(x, path, world=None, v_ref=V_MAX):
    Ld = LD0 + KV * x[3]
    d = np.linalg.norm(path - x[:2], axis=1)
    i0 = int(np.argmin(d))
    s, ig = 0.0, i0
    while ig < len(path) - 1 and s < Ld:
        s += np.linalg.norm(path[ig + 1] - path[ig])
        ig += 1
    goal_pt = path[ig]

    alpha = _wrap(np.arctan2(goal_pt[1] - x[1], goal_pt[0] - x[0]) - x[2])
    curvature = abs(2.0 * np.sin(alpha) / max(Ld, 1e-3))
    delta_des = np.clip(np.arctan(L * curvature * np.sign(alpha)
                                  if alpha != 0 else 0.0), -0.5, 0.5)

    # --- regulation 1: curvature ---
    v_curv = v_ref / (1.0 + CURV_GAIN * curvature)

    # --- regulation 2: proximity to obstacles ---
    v_prox = v_ref
    if world is not None:
        clr = _clearance(x, world)
        if clr < PROX_RADIUS:
            v_prox = v_ref * np.clip(clr / PROX_RADIUS, 0.15, 1.0)

    # --- regulation 3: approach to goal ---
    v_app = v_ref
    if world is not None and getattr(world, 'goal', None) is not None:
        dgoal = np.linalg.norm(x[:2] - world.goal)
        if dgoal < APPROACH_RADIUS:
            v_app = v_ref * np.clip(dgoal / APPROACH_RADIUS, 0.2, 1.0)

    v_target = min(v_curv, v_prox, v_app)
    a = np.clip(1.0 * (v_target - x[3]), -3.0, 2.0)
    dd = np.clip((delta_des - x[4]) / 0.15, -1.0, 1.0)
    return np.array([a, dd])
