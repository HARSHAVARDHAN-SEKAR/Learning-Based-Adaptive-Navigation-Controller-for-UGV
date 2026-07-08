"""Kinematic bicycle model, RK4 integration.

State  x = [X, Y, psi, v, delta]
Input  u = [a, ddelta]  (acceleration, steering rate)

Identical states/inputs to the ACADOS model (Deliverable 1) so results
transfer 1:1 from this Python research layer to the ROS2 deployment.
"""
import numpy as np

L = 0.32          # wheelbase [m]
V_MAX = 2.0
DELTA_MAX = 0.5


def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    X, Y, psi, v, delta = x
    return np.array([
        v * np.cos(psi),
        v * np.sin(psi),
        v / L * np.tan(delta),
        u[0],
        u[1],
    ])


def step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """RK4 integration + physical limits (mirrors ACADOS box constraints)."""
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[3] = np.clip(xn[3], 0.0, V_MAX)
    xn[4] = np.clip(xn[4], -DELTA_MAX, DELTA_MAX)
    return xn
