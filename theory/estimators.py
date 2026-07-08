"""State estimation: EKF, UKF, and sliding-window factor graph.

All three fuse: wheel encoder (v), gyro (omega), GPS (x, y @ 5 Hz).
State: [x, y, psi]  (v taken directly from encoder as control input).
The factor graph is a scipy least-squares pose-graph — the same math
GTSAM runs, kept dependency-light for the research layer.
"""
import numpy as np
from scipy.optimize import least_squares
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------- EKF
class EKF:
    def __init__(self, x0, sig_gps=0.15):
        self.x = np.array(x0[:3], float)
        self.P = np.diag([0.1, 0.1, 0.05])
        self.Qc = np.diag([0.02, 0.02, 0.01]) ** 2
        self.Rg = np.eye(2) * sig_gps ** 2

    def predict(self, v, omega, dt):
        x, y, psi = self.x
        self.x = np.array([x + v * np.cos(psi) * dt,
                           y + v * np.sin(psi) * dt,
                           _wrap(psi + omega * dt)])
        F = np.array([[1, 0, -v * np.sin(psi) * dt],
                      [0, 1,  v * np.cos(psi) * dt],
                      [0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Qc * dt

    def update_gps(self, z):
        H = np.array([[1., 0, 0], [0, 1., 0]])
        y = z - self.x[:2]
        S = H @ self.P @ H.T + self.Rg
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P


# ---------------------------------------------------------------- UKF
class UKF:
    def __init__(self, x0, sig_gps=0.15):
        pts = MerweScaledSigmaPoints(3, alpha=0.1, beta=2.0, kappa=0.0,
                                     subtract=self._residual_x)
        self.kf = UnscentedKalmanFilter(
            dim_x=3, dim_z=2, dt=0.05,
            fx=self._fx, hx=lambda x: x[:2], points=pts,
            residual_x=self._residual_x, x_mean_fn=self._state_mean)
        self.kf.x = np.array(x0[:3], float)
        self.kf.P = np.diag([0.1, 0.1, 0.05])
        self.kf.Q = np.diag([0.02, 0.02, 0.01]) ** 2 * 0.05
        self.kf.R = np.eye(2) * sig_gps ** 2
        self._u = (0.0, 0.0)

    @staticmethod
    def _residual_x(a, b):
        r = a - b
        r[2] = _wrap(r[2])
        return r

    @staticmethod
    def _state_mean(sigmas, Wm):
        """Weighted mean with CIRCULAR mean for the heading state —
        linear averaging of angles breaks at the ±pi wrap."""
        x = np.zeros(3)
        x[0] = np.dot(Wm, sigmas[:, 0])
        x[1] = np.dot(Wm, sigmas[:, 1])
        x[2] = np.arctan2(np.dot(Wm, np.sin(sigmas[:, 2])),
                          np.dot(Wm, np.cos(sigmas[:, 2])))
        return x

    def _fx(self, x, dt):
        v, om = self._u
        return np.array([x[0] + v * np.cos(x[2]) * dt,
                         x[1] + v * np.sin(x[2]) * dt,
                         _wrap(x[2] + om * dt)])

    def predict(self, v, omega, dt):
        self._u = (v, omega)
        self.kf.predict(dt=dt)

    def update_gps(self, z):
        self.kf.update(z)
        self.kf.x[2] = _wrap(self.kf.x[2])

    @property
    def x(self):
        return self.kf.x


# ------------------------------------------------- Sliding-window factor graph
class FactorGraph:
    """Sliding-window smoother: odometry factors + GPS factors,
    solved with Levenberg-Marquardt at every GPS update."""

    def __init__(self, x0, window=30, sig_gps=0.15):
        self.window = window
        self.poses = [np.array(x0[:3], float)]     # estimates
        self.odom = []                              # (dx, dy, dpsi) body-frame
        self.gps = {0: None}                        # idx -> measurement
        self.sig_odom = np.array([0.02, 0.02, 0.01])
        self.sig_gps = sig_gps
        self.x = np.array(x0[:3], float)

    def predict(self, v, omega, dt):
        d = np.array([v * dt, 0.0, omega * dt])    # body-frame increment
        self.odom.append(d)
        p = self.poses[-1]
        c, s = np.cos(p[2]), np.sin(p[2])
        self.poses.append(np.array([p[0] + c * d[0] - s * d[1],
                                    p[1] + s * d[0] + c * d[1],
                                    _wrap(p[2] + d[2])]))
        self.x = self.poses[-1]

    def update_gps(self, z):
        self.gps[len(self.poses) - 1] = z.copy()
        self._optimize()
        self.x = self.poses[-1]

    def _optimize(self):
        n = len(self.poses)
        lo = max(0, n - self.window)
        idxs = list(range(lo, n))
        x0 = np.concatenate([self.poses[i] for i in idxs])

        odom = self.odom
        gps = [(i - lo, self.gps[i]) for i in idxs if self.gps.get(i) is not None]
        anchor = self.poses[lo].copy()

        def residuals(w):
            P = w.reshape(-1, 3)
            res = [(P[0] - anchor) / 0.1]                        # anchor prior on window start
            for j in range(len(idxs) - 1):
                gi = idxs[j]
                c, s = np.cos(P[j, 2]), np.sin(P[j, 2])
                pred = np.array([P[j, 0] + c * odom[gi][0] - s * odom[gi][1],
                                 P[j, 1] + s * odom[gi][0] + c * odom[gi][1],
                                 P[j, 2] + odom[gi][2]])
                e = P[j + 1] - pred
                e[2] = _wrap(e[2])
                res.append(e / self.sig_odom)
            for j, z in gps:
                res.append((P[j, :2] - z) / self.sig_gps)
            return np.concatenate(res)

        sol = least_squares(residuals, x0, method='lm', max_nfev=60)
        P = sol.x.reshape(-1, 3)
        for j, i in enumerate(idxs):
            self.poses[i] = P[j]
