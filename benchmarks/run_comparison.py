"""Phase-1 controller benchmark on a figure-eight path.

Produces the frozen research table:
    Method | RMS e_ct | Max e_ct | RMS jerk | Steering var | Compute (ms)
and publication plots in benchmarks/plots/.

The ROS2 harness (Deliverable 3) reproduces these exact metric definitions
so Python-research and deployed results are directly comparable.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from theory.vehicle_model import step
from controllers.geometric import pure_pursuit, stanley
from controllers.mpc_controller import MPC

np.random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)

# ---- Reference path: figure-eight (lemniscate of Gerono) ----
t = np.linspace(0, 2 * np.pi, 800)
PATH = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])

DT, T_END = 0.05, 40.0
STEPS = int(T_END / DT)
X_INIT = np.array([PATH[0, 0], PATH[0, 1] - 0.3, np.pi / 4, 0.0, 0.0])


def cross_track(x, path):
    d = np.linalg.norm(path - x[:2], axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(path) - 1)
    tg = path[i1] - path[max(i0 - 1, 0)]
    tg = tg / max(np.linalg.norm(tg), 1e-9)
    e = x[:2] - path[i0]
    return -e[0] * tg[1] + e[1] * tg[0]


def run(controller_name):
    x = X_INIT.copy()
    mpc = MPC() if controller_name == 'MPC' else None
    log = {'x': np.zeros((5, STEPS)), 'u': np.zeros((2, STEPS)),
           'ect': np.zeros(STEPS), 'ms': np.zeros(STEPS)}
    for k in range(STEPS):
        t0 = time.perf_counter()
        if controller_name == 'PurePursuit':
            u = pure_pursuit(x, PATH)
        elif controller_name == 'Stanley':
            u = stanley(x, PATH)
        else:
            u, _ = mpc.solve(x, PATH)
        log['ms'][k] = (time.perf_counter() - t0) * 1000
        x = step(x, u, DT)
        log['x'][:, k] = x
        log['u'][:, k] = u
        log['ect'][k] = cross_track(x, PATH)
    return log


def metrics(log):
    settle = int(0.5 / DT)                        # exclude first 0.5 s
    ect = log['ect'][settle:]
    v = log['x'][3]
    acc = np.gradient(v, DT)
    jerk = np.gradient(acc, DT)[settle:]
    return {
        'RMS_ect_m': float(np.sqrt(np.mean(ect ** 2))),
        'Max_ect_m': float(np.max(np.abs(ect))),
        'RMS_jerk_ms3': float(np.sqrt(np.mean(jerk ** 2))),
        'SteerRateVar': float(np.var(log['u'][1, settle:])),
        'Mean_ms': float(np.mean(log['ms'])),
        'p99_ms': float(np.percentile(log['ms'], 99)),
    }


if __name__ == '__main__':
    names = ['PurePursuit', 'Stanley', 'MPC']
    logs, rows = {}, []
    for n in names:
        print(f'Running {n} ...', flush=True)
        logs[n] = run(n)
        rows.append({'Method': n, **metrics(logs[n])})

    df = pd.DataFrame(rows).set_index('Method')
    print('\n=== RESEARCH TABLE (frozen format) ===')
    print(df.round(4).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'))

    # ---- Publication plots ----
    cols = {'PurePursuit': 'tab:blue', 'Stanley': 'tab:orange', 'MPC': 'tab:green'}
    tv = np.arange(STEPS) * DT
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(PATH[:, 0], PATH[:, 1], 'k--', lw=1, label='Reference')
    for n in names:
        ax[0, 0].plot(logs[n]['x'][0], logs[n]['x'][1], color=cols[n], label=n)
    ax[0, 0].set(title='Trajectory Tracking — Figure-Eight',
                 xlabel='X [m]', ylabel='Y [m]', aspect='equal')
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    for n in names:
        ax[0, 1].plot(tv, logs[n]['ect'], color=cols[n], label=n)
    ax[0, 1].set(title='Cross-Track Error', xlabel='t [s]', ylabel='$e_{ct}$ [m]')
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    for n in names:
        ax[1, 0].plot(tv, logs[n]['x'][3], color=cols[n], label=n)
    ax[1, 0].set(title='Velocity Profile', xlabel='t [s]', ylabel='v [m/s]')
    ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

    for n in names:
        ax[1, 1].plot(tv, logs[n]['u'][1], color=cols[n], label=n, alpha=.8)
    ax[1, 1].set(title='Steering Rate (smoothness proxy)',
                 xlabel='t [s]', ylabel=r'$\dot\delta$ [rad/s]')
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'controller_comparison.png'), dpi=150)
    fig.savefig(os.path.join(OUT, 'controller_comparison.pdf'))
    print(f'\nPlots saved to {OUT}')
