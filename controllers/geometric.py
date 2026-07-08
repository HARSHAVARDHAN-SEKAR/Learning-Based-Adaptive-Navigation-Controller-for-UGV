"""Classical geometric path-tracking controllers.

Both output rate-level commands u = [a, ddelta] through the same interface
as the MPC — this makes the smoothness benchmark fair across controllers.
"""
import numpy as np

L = 0.32


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _rate_level(delta_des, v_ref, x, kp_v=1.0, tau_delta=0.15):
    """Convert desired steering angle + speed into [a, ddelta]."""
    a = np.clip(kp_v * (v_ref - x[3]), -3.0, 2.0)
    ddelta = np.clip((delta_des - x[4]) / tau_delta, -1.0, 1.0)
    return np.array([a, ddelta])


def pure_pursuit(x, path, v_ref=1.5, Ld0=0.5, kv=0.5):
    Ld = Ld0 + kv * x[3]                       # adaptive lookahead
    d = np.linalg.norm(path - x[:2], axis=1)
    i0 = int(np.argmin(d))

    # walk forward Ld along the path
    s, ig = 0.0, i0
    while ig < len(path) - 1 and s < Ld:
        s += np.linalg.norm(path[ig + 1] - path[ig])
        ig += 1
    goal = path[ig]

    alpha = _wrap(np.arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2])
    delta_des = np.clip(np.arctan(2 * L * np.sin(alpha) / max(Ld, 1e-3)),
                        -0.5, 0.5)
    return _rate_level(delta_des, v_ref, x)


def stanley(x, path, v_ref=1.5, k=1.2, ks=0.3):
    # front axle position
    f = x[:2] + L * np.array([np.cos(x[2]), np.sin(x[2])])
    d = np.linalg.norm(path - f, axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(path) - 1)
    tang = path[i1] - path[max(i0 - 1, 0)]
    psi_path = np.arctan2(tang[1], tang[0])

    # signed cross-track error (positive = robot left of path)
    e = f - path[i0]
    e_ct = -e[0] * np.sin(psi_path) + e[1] * np.cos(psi_path)

    psi_e = _wrap(psi_path - x[2])
    delta_des = np.clip(psi_e + np.arctan(k * (-e_ct) / (ks + x[3])),
                        -0.5, 0.5)
    return _rate_level(delta_des, v_ref, x)


def pid(x, path, v_ref=1.5, kp_ct=1.5, kd_ct=0.4, kp_psi=1.8, _state={'e_prev': 0.0}):
    """PID on cross-track error + heading P — the simplest baseline."""
    d = np.linalg.norm(path - x[:2], axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(path) - 1)
    tang = path[i1] - path[max(i0 - 1, 0)]
    psi_path = np.arctan2(tang[1], tang[0])
    e = x[:2] - path[i0]
    e_ct = -e[0] * np.sin(psi_path) + e[1] * np.cos(psi_path)
    de = (e_ct - _state['e_prev']) / 0.05
    _state['e_prev'] = e_ct
    psi_e = _wrap(psi_path - x[2])
    delta_des = np.clip(kp_psi * psi_e - kp_ct * e_ct - kd_ct * de, -0.5, 0.5)
    return _rate_level(delta_des, v_ref, x)
