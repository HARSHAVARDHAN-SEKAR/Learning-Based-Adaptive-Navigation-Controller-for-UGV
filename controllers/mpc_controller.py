"""NMPC via CasADi multiple shooting (ipopt).

Python prototype of the exact OCP later solved by ACADOS in ROS2:
same states, inputs, weights, constraints — only the solver differs.
The validation gate (Deliverable 5): ACADOS closed-loop must match this
trajectory to < 2 cm RMS before deployment proceeds.
"""
import numpy as np
import casadi as ca

L = 0.32
N = 30            # horizon steps
TS = 0.05         # [s]
NX, NU = 5, 2
Q = np.diag([10.0, 10.0, 5.0, 1.0, 0.1])
R = np.diag([0.5, 2.0])
RD = np.diag([0.1, 1.0])          # Δu penalty — matches ACADOS Rd


class MPC:
    def __init__(self, v_ref=1.5):
        self.v_ref = v_ref
        self._build()
        self.w_prev = None
        self.prog_idx = 0          # persistent path progress (no lobe jumps)
        self.closed = True         # False for open start->goal paths

    def _build(self):
        X = ca.SX.sym('X', NX, N + 1)
        U = ca.SX.sym('U', NU, N)
        P = ca.SX.sym('P', NX + NX * (N + 1))     # x0 + state reference

        def f(x, u):
            return ca.vertcat(x[3] * ca.cos(x[2]),
                              x[3] * ca.sin(x[2]),
                              x[3] / L * ca.tan(x[4]),
                              u[0], u[1])

        J = 0
        g = [X[:, 0] - P[:NX]]
        for k in range(N):
            xref = P[NX + k * NX: NX + (k + 1) * NX]
            e = X[:, k] - xref
            e[2] = ca.atan2(ca.sin(e[2]), ca.cos(e[2]))   # wrap heading
            J += e.T @ Q @ e + U[:, k].T @ R @ U[:, k]
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                J += du.T @ RD @ du
            # RK4 multiple-shooting defect
            k1 = f(X[:, k], U[:, k])
            k2 = f(X[:, k] + TS / 2 * k1, U[:, k])
            k3 = f(X[:, k] + TS / 2 * k2, U[:, k])
            k4 = f(X[:, k] + TS * k3, U[:, k])
            g.append(X[:, k + 1] - (X[:, k] + TS / 6 * (k1 + 2*k2 + 2*k3 + k4)))
        xrefN = P[NX + N * NX:]
        eN = X[:, N] - xrefN
        eN[2] = ca.atan2(ca.sin(eN[2]), ca.cos(eN[2]))
        J += eN.T @ (10 * Q) @ eN

        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        prob = {'f': J, 'x': w, 'g': ca.vertcat(*g), 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.max_iter': 100, 'ipopt.tol': 1e-6,
                'ipopt.warm_start_init_point': 'yes'}
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        lbx_st = np.tile([-np.inf, -np.inf, -np.inf, 0.0, -0.5], N + 1)
        ubx_st = np.tile([np.inf, np.inf, np.inf, 2.0, 0.5], N + 1)
        self.lbx = np.concatenate([lbx_st, np.tile([-3.0, -1.0], N)])
        self.ubx = np.concatenate([ubx_st, np.tile([2.0, 1.0], N)])
        self.lbg = np.zeros(NX * (N + 1))
        self.ubg = np.zeros(NX * (N + 1))

    def _reference(self, x0, path):
        """Resample path at v_ref*Ts spacing from current progress point.

        Windowed nearest search around the previous progress index (a
        global search jumps between lobes at the figure-eight crossing).
        All indexing is CIRCULAR: on closed paths a clamping index makes
        the robot stop after one lap — a bug that silently inflates the
        error metric because a stopped on-path robot has zero error.
        """
        n = len(path)
        W = 60                                    # search window [pts]
        if self.closed:
            offs = (self.prog_idx + np.arange(-5, W)) % n
        else:
            offs = np.clip(self.prog_idx + np.arange(-5, W), 0, n - 1)
        d = np.linalg.norm(path[offs] - x0[:2], axis=1)
        idx = int(offs[int(np.argmin(d))])
        self.prog_idx = idx

        xr = np.zeros((NX, N + 1))
        for k in range(N + 1):
            nxt = (idx + 1) % n if self.closed else min(idx + 1, n - 1)
            tang = path[nxt] - path[idx]
            if np.linalg.norm(tang) < 1e-9:
                tang = path[idx] - path[idx - 1]
            xr[:, k] = [path[idx, 0], path[idx, 1],
                        np.arctan2(tang[1], tang[0]),
                        self.v_ref if (self.closed or idx < n - 2) else 0.0,
                        0.0]
            s, target = 0.0, self.v_ref * TS
            while s < target:
                nxt = (idx + 1) % n if self.closed else min(idx + 1, n - 1)
                if not self.closed and nxt == idx:
                    break                        # hold at open-path end
                s += np.linalg.norm(path[nxt] - path[idx])
                idx = nxt
        return xr

    def solve(self, x0, path):
        xr = self._reference(x0, path)
        p = np.concatenate([x0, xr.flatten(order='F')])
        if self.w_prev is None:
            w0 = np.concatenate([np.tile(x0, N + 1), np.zeros(NU * N)])
        else:
            w0 = self.w_prev
        sol = self.solver(x0=w0, p=p, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg)
        w = np.array(sol['x']).flatten()
        Xopt = w[:NX * (N + 1)].reshape(NX, N + 1, order='F')
        Uopt = w[NX * (N + 1):].reshape(NU, N, order='F')
        # shift for warm start
        Xs = np.hstack([Xopt[:, 1:], Xopt[:, -1:]])
        Us = np.hstack([Uopt[:, 1:], Uopt[:, -1:]])
        self.w_prev = np.concatenate([Xs.flatten(order='F'),
                                      Us.flatten(order='F')])
        return Uopt[:, 0], Xopt
