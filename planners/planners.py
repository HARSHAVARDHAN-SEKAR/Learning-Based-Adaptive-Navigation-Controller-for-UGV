"""Planner benchmark implementations.

Grid world in meters (RES m/cell). All planners return an Nx2 path in
world coordinates or None on failure.
"""
import heapq
import numpy as np

RES = 0.1


class GridMap:
    def __init__(self, w_m=6.0, h_m=4.0, seed=0, n_obs=8):
        self.nx, self.ny = int(w_m / RES), int(h_m / RES)
        self.grid = np.zeros((self.nx, self.ny), bool)
        rng = np.random.default_rng(seed)
        for _ in range(n_obs):
            cx = rng.integers(8, self.nx - 8)
            cy = rng.integers(4, self.ny - 4)
            w = rng.integers(3, 9); h = rng.integers(3, 9)
            self.grid[cx:cx + w, cy:cy + h] = True
        # keep start/goal corridors free
        self.grid[:8, :8] = False
        self.grid[-8:, -8:] = False
        self.start = np.array([0.4, 0.4])
        self.goal = np.array([w_m - 0.4, h_m - 0.4])

    def occ(self, x, y):
        i, j = int(x / RES), int(y / RES)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return True
        return self.grid[i, j]

    def line_free(self, p, q):
        n = max(2, int(np.linalg.norm(np.asarray(q) - np.asarray(p)) / (RES / 2)))
        for t in np.linspace(0, 1, n):
            r = (1 - t) * np.asarray(p) + t * np.asarray(q)
            if self.occ(r[0], r[1]):
                return False
        return True


def _cell(p):
    return int(p[0] / RES), int(p[1] / RES)


NBRS = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def astar(m: GridMap, theta_variant=False):
    s, g = _cell(m.start), _cell(m.goal)
    openq = [(0.0, s)]
    gsc = {s: 0.0}
    par = {s: s}
    closed = set()
    while openq:
        _, c = heapq.heappop(openq)
        if c in closed:
            continue
        closed.add(c)
        if c == g:
            path = [c]
            while path[-1] != s:
                path.append(par[path[-1]])
            pts = np.array([[(i + .5) * RES, (j + .5) * RES] for i, j in path[::-1]])
            return pts
        for dx, dy, cost in NBRS:
            n = (c[0] + dx, c[1] + dy)
            if n in closed or m.occ((n[0] + .5) * RES, (n[1] + .5) * RES):
                continue
            # Theta*: any-angle shortcut through the parent if line-of-sight
            if theta_variant and m.line_free(
                    ((par[c][0] + .5) * RES, (par[c][1] + .5) * RES),
                    ((n[0] + .5) * RES, (n[1] + .5) * RES)):
                cand_par, cand_g = par[c], gsc[par[c]] + np.hypot(
                    n[0] - par[c][0], n[1] - par[c][1]) * RES
            else:
                cand_par, cand_g = c, gsc[c] + cost * RES
            if cand_g < gsc.get(n, np.inf):
                gsc[n] = cand_g
                par[n] = cand_par
                h = np.hypot(g[0] - n[0], g[1] - n[1]) * RES
                heapq.heappush(openq, (cand_g + h, n))
    return None


def hybrid_astar(m: GridMap, L=0.32, step=0.3, n_head=16):
    """Kinematic A* with arc motion primitives (forward only)."""
    deltas = [-0.4, -0.2, 0.0, 0.2, 0.4]
    s = (m.start[0], m.start[1], np.arctan2(*(m.goal - m.start)[::-1]))
    gd = lambda st: np.hypot(st[0] - m.goal[0], st[1] - m.goal[1])

    def key(st):
        return (int(st[0] / RES), int(st[1] / RES),
                int(((st[2] + np.pi) / (2 * np.pi)) * n_head) % n_head)

    openq = [(gd(s), 0.0, s)]
    gsc = {key(s): 0.0}
    par = {key(s): (None, s)}
    it = 0
    while openq and it < 60000:
        it += 1
        _, gc, st = heapq.heappop(openq)
        if gd(st) < 0.3:
            path = [st]
            k = key(st)
            while True:
                pk, pst = par[k]
                if pk is None:
                    break
                path.append(pst)
                k = pk
            return np.array([[p[0], p[1]] for p in path[::-1]])
        for d in deltas:
            x, y, psi = st
            # integrate arc in 3 substeps, collision-check each
            ok = True
            for _ in range(3):
                x += (step / 3) * np.cos(psi)
                y += (step / 3) * np.sin(psi)
                psi += (step / 3) / L * np.tan(d)
                if m.occ(x, y):
                    ok = False
                    break
            if not ok:
                continue
            nst = (x, y, np.arctan2(np.sin(psi), np.cos(psi)))
            nk = key(nst)
            ng = gc + step + 0.1 * abs(d)          # slight steering penalty
            if ng < gsc.get(nk, np.inf):
                gsc[nk] = ng
                par[nk] = (key(st), st)
                heapq.heappush(openq, (ng + gd(nst), ng, nst))
    return None


def rrt_star(m: GridMap, iters=3000, step=0.35, radius=0.9, seed=0):
    rng = np.random.default_rng(seed)
    nodes = [m.start.copy()]
    par = [0]
    cost = [0.0]
    goal_idx = None
    for _ in range(iters):
        q = m.goal if rng.random() < 0.1 else rng.uniform(
            [0, 0], [m.nx * RES, m.ny * RES])
        d = np.linalg.norm(np.array(nodes) - q, axis=1)
        i_near = int(np.argmin(d))
        v = q - nodes[i_near]
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            continue
        new = nodes[i_near] + v / nrm * min(step, nrm)
        if not m.line_free(nodes[i_near], new):
            continue
        # choose best parent in radius
        dn = np.linalg.norm(np.array(nodes) - new, axis=1)
        near = np.where(dn < radius)[0]
        best, bc = i_near, cost[i_near] + np.linalg.norm(new - nodes[i_near])
        for i in near:
            c = cost[i] + dn[i]
            if c < bc and m.line_free(nodes[i], new):
                best, bc = int(i), c
        nodes.append(new)
        par.append(best)
        cost.append(bc)
        ni = len(nodes) - 1
        # rewire
        for i in near:
            c = bc + dn[i]
            if c < cost[i] and m.line_free(new, nodes[i]):
                par[i], cost[i] = ni, c
        if np.linalg.norm(new - m.goal) < 0.3 and m.line_free(new, m.goal):
            if goal_idx is None or bc < cost[goal_idx]:
                goal_idx = ni
    if goal_idx is None:
        return None
    path = [m.goal]
    i = goal_idx
    while i != 0:
        path.append(nodes[i])
        i = par[i]
    path.append(m.start)
    return np.array(path[::-1])


def mppi(m: GridMap, K=400, H=30, dt=0.12, lam=0.5, seed=0, max_steps=400):
    """MPPI as receding-horizon planner on a unicycle; returns executed path."""
    rng = np.random.default_rng(seed)
    x = np.array([m.start[0], m.start[1],
                  np.arctan2(*(m.goal - m.start)[::-1])])
    U = np.zeros((H, 2))
    U[:, 0] = 0.8                                    # nominal forward speed
    traj = [x[:2].copy()]

    def rollout_cost(Useq):
        s = x.copy()
        c = 0.0
        for h in range(H):
            v = np.clip(Useq[h, 0], 0.0, 1.2)
            w = np.clip(Useq[h, 1], -2.0, 2.0)
            s = s + dt * np.array([v * np.cos(s[2]), v * np.sin(s[2]), w])
            if m.occ(s[0], s[1]):
                c += 500.0
            c += 2.0 * np.linalg.norm(s[:2] - m.goal)
        return c

    for _ in range(max_steps):
        eps = rng.normal(0, [0.25, 0.8], (K, H, 2))
        costs = np.array([rollout_cost(U + eps[k]) for k in range(K)])
        w = np.exp(-(costs - costs.min()) / lam)
        w /= w.sum()
        U = U + np.einsum('k,khu->hu', w, eps)
        v = np.clip(U[0, 0], 0.0, 1.2)
        om = np.clip(U[0, 1], -2.0, 2.0)
        x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), om])
        if m.occ(x[0], x[1]):
            return None                              # crashed
        traj.append(x[:2].copy())
        U = np.vstack([U[1:], U[-1:]])
        if np.linalg.norm(x[:2] - m.goal) < 0.3:
            return np.array(traj)
    return None
