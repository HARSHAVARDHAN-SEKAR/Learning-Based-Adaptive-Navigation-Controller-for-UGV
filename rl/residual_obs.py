"""Shared residual-policy observation builder.

Used by BOTH runtime/nodes.py (inference) and rl/train_residual.py
(training) — a single source of truth so the trained policy's input
layout can never silently drift from what the runtime actually feeds it.
The old inline construction in ControllerNode sliced dict.values() with
no guaranteed key order across Python versions/refactors; this replaces
it with named fields.

8-dim, fixed order: [v, delta, e_ct, e_psi, solve_ms/20, est_err,
sin(psi), cos(psi)]
"""
import numpy as np

OBS_DIM = 8


def residual_observation(x_ctrl, metrics):
    """x_ctrl: [X, Y, psi, v, delta]. metrics: dict from /metrics topic
    (or {} before the first tick — all fields default to 0)."""
    v, delta, psi = x_ctrl[3], x_ctrl[4], x_ctrl[2]
    e_ct = metrics.get('e_ct', 0.0)
    e_psi = metrics.get('e_psi', 0.0)
    solve_ms = metrics.get('solve_ms', 0.0)
    est_err = metrics.get('est_err', 0.0)
    return np.array([v, delta, e_ct, e_psi, solve_ms / 20.0, est_err,
                     np.sin(psi), np.cos(psi)], dtype=np.float32)
