"""Stage 3+4 benchmark: PID, Pure Pursuit, Stanley, MPC, Adaptive-MPC (learned).

Trains the CEM speed-schedule policy, then evaluates all five controllers
on the full 40 s figure-eight with identical metrics, and renders the
final summary dashboard combining all pipeline stages.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from theory.vehicle_model import step
from controllers.geometric import pure_pursuit, stanley, pid
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC, train_cem

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

t = np.linspace(0, 2 * np.pi, 800)
PATH = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
DT, T_END = 0.05, 40.0
STEPS = int(T_END / DT)
X0 = np.array([PATH[0, 0], PATH[0, 1] - 0.3, np.pi / 4, 0.0, 0.0])


def cross_track(x):
    d = np.linalg.norm(PATH - x[:2], axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(PATH) - 1)
    tg = PATH[i1] - PATH[max(i0 - 1, 0)]
    tg = tg / max(np.linalg.norm(tg), 1e-9)
    e = x[:2] - PATH[i0]
    return -e[0] * tg[1] + e[1] * tg[0]


def run(name, adaptive_params=None):
    x = X0.copy()
    mpc = MPC() if name == 'MPC' else None
    ada = AdaptiveMPC(adaptive_params, PATH) if name == 'Adaptive-MPC' else None
    log = {'x': np.zeros((5, STEPS)), 'u': np.zeros((2, STEPS)),
           'ect': np.zeros(STEPS), 'ms': np.zeros(STEPS),
           'vref': np.zeros(STEPS)}
    for k in range(STEPS):
        t0 = time.perf_counter()
        if name == 'PID':
            u = pid(x, PATH)
        elif name == 'PurePursuit':
            u = pure_pursuit(x, PATH)
        elif name == 'Stanley':
            u = stanley(x, PATH)
        elif name == 'MPC':
            u, _ = mpc.solve(x, PATH)
        else:
            u, _ = ada.solve(x)
            log['vref'][k] = ada.mpc.v_ref
        log['ms'][k] = (time.perf_counter() - t0) * 1000
        x = step(x, u, DT)
        log['x'][:, k] = x
        log['u'][:, k] = u
        log['ect'][k] = cross_track(x)
    return log


def metrics(log):
    s = int(0.5 / DT)
    ect = log['ect'][s:]
    v = log['x'][3]
    jerk = np.gradient(np.gradient(v, DT), DT)[s:]
    return {'RMS_ect_m': float(np.sqrt(np.mean(ect ** 2))),
            'RMS_jerk': float(np.sqrt(np.mean(jerk ** 2))),
            'SteerVar': float(np.var(log['u'][1, s:])),
            'MeanSpeed': float(np.mean(v[s:])),
            'Progress_m': float(np.sum(v) * DT),
            'p99_ms': float(np.percentile(log['ms'], 99))}


if __name__ == '__main__':
    print('=== Training adaptive speed schedule (CEM) ===', flush=True)
    t0 = time.time()
    params, hist = train_cem(PATH)
    print(f'Trained in {time.time()-t0:.0f}s -> '
          f'v_base={params[0]:.2f} m/s, k_curv={params[1]:.2f}\n')

    names = ['PID', 'PurePursuit', 'Stanley', 'MPC', 'Adaptive-MPC']
    logs, rows = {}, []
    for n in names:
        print(f'Evaluating {n} ...', flush=True)
        logs[n] = run(n, adaptive_params=params if n == 'Adaptive-MPC' else None)
        rows.append({'Method': n, **metrics(logs[n])})

    df = pd.DataFrame(rows).set_index('Method')
    print('\n=== FULL CONTROLLER BENCHMARK ===')
    print(df.round(4).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), 'controller_results_full.csv'))

    # ---------------- summary dashboard ----------------
    cols = dict(zip(names, ['tab:red', 'tab:blue', 'tab:orange',
                            'tab:green', 'tab:purple']))
    tv = np.arange(STEPS) * DT
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].plot(PATH[:, 0], PATH[:, 1], 'k--', lw=1)
    for n in names:
        ax[0, 0].plot(logs[n]['x'][0], logs[n]['x'][1], color=cols[n],
                      label=n, lw=1.4)
    ax[0, 0].set(title='Trajectories', aspect='equal')
    ax[0, 0].legend(fontsize=7)

    for n in names:
        ax[0, 1].plot(tv, logs[n]['ect'], color=cols[n], lw=1)
    ax[0, 1].set(title='Cross-Track Error [m]', xlabel='t [s]')
    ax[0, 1].grid(alpha=.3)

    for n in names:
        ax[0, 2].plot(tv, logs[n]['x'][3], color=cols[n], lw=1.2)
    ax[0, 2].plot(tv, logs['Adaptive-MPC']['vref'], 'k:', lw=1,
                  label='learned v_ref')
    ax[0, 2].set(title='Velocity — learned schedule slows in corners',
                 xlabel='t [s]')
    ax[0, 2].legend(fontsize=7); ax[0, 2].grid(alpha=.3)

    ax[1, 0].bar(names, df['RMS_ect_m'], color=[cols[n] for n in names])
    ax[1, 0].set(title='RMS Cross-Track Error [m]')
    ax[1, 0].tick_params(axis='x', labelsize=7); ax[1, 0].grid(alpha=.3, axis='y')

    ax[1, 1].bar(names, df['SteerVar'], color=[cols[n] for n in names])
    ax[1, 1].set(title='Steering-Rate Variance (smoothness)')
    ax[1, 1].tick_params(axis='x', labelsize=7); ax[1, 1].grid(alpha=.3, axis='y')

    ax[1, 2].bar(names, df['p99_ms'], color=[cols[n] for n in names])
    ax[1, 2].set_yscale('log')
    ax[1, 2].set(title='Compute p99 [ms, log]')
    ax[1, 2].tick_params(axis='x', labelsize=7); ax[1, 2].grid(alpha=.3, axis='y')

    fig.suptitle('Learning-Based Adaptive Navigation — Full Controller Benchmark',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'full_controller_dashboard.png'), dpi=150)
    print('\nDashboard saved.')
