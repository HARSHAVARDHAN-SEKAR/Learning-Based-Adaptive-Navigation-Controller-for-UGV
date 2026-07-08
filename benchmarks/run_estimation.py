"""Stage 1 benchmark: EKF vs UKF vs Factor Graph.

Ground truth: bicycle model driven by Stanley around the figure-eight.
Sensors: encoder + gyro (100%) and GPS (5 Hz), from simulation/sensors.py.
Metrics: RMS position error, RMS heading error, mean update time.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from theory.vehicle_model import step, L
from controllers.geometric import stanley
from simulation.sensors import SensorSim
from theory.estimators import EKF, UKF, FactorGraph, _wrap

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)

t = np.linspace(0, 2 * np.pi, 800)
PATH = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
DT, STEPS = 0.05, 800
SEEDS = 5


def run_seed(seed):
    x = np.array([0.0, -0.3, np.pi / 4, 0.0, 0.0])
    sens = SensorSim(DT, seed=seed)
    est = {'EKF': EKF(x), 'UKF': UKF(x), 'FactorGraph': FactorGraph(x)}
    err = {n: {'pos': [], 'psi': [], 'ms': []} for n in est}
    truth = np.zeros((3, STEPS))

    for k in range(STEPS):
        u = stanley(x, PATH)
        omega_true = x[3] / L * np.tan(x[4])
        x = step(x, u, DT)
        truth[:, k] = x[:3]
        v_enc, om_gyro, gps = sens.measure(x, omega_true, k)

        for n, e in est.items():
            t0 = time.perf_counter()
            e.predict(v_enc, om_gyro, DT)
            if gps is not None:
                e.update_gps(gps)
            err[n]['ms'].append((time.perf_counter() - t0) * 1000)
            err[n]['pos'].append(np.linalg.norm(e.x[:2] - x[:2]))
            err[n]['psi'].append(abs(_wrap(e.x[2] - x[2])))
    return err, truth


if __name__ == '__main__':
    agg = {n: {'pos': [], 'psi': [], 'ms': []} for n in ['EKF', 'UKF', 'FactorGraph']}
    last = None
    for s in range(SEEDS):
        err, truth = run_seed(s)
        last = err
        for n in agg:
            agg[n]['pos'].append(np.sqrt(np.mean(np.array(err[n]['pos']) ** 2)))
            agg[n]['psi'].append(np.sqrt(np.mean(np.array(err[n]['psi']) ** 2)))
            agg[n]['ms'].append(np.mean(err[n]['ms']))

    rows = []
    for n in agg:
        rows.append({'Estimator': n,
                     'RMS_pos_m': np.mean(agg[n]['pos']),
                     'std': np.std(agg[n]['pos']),
                     'RMS_psi_rad': np.mean(agg[n]['psi']),
                     'Mean_ms': np.mean(agg[n]['ms'])})
    df = pd.DataFrame(rows).set_index('Estimator')
    print('\n=== STATE ESTIMATION BENCHMARK (5 seeds) ===')
    print(df.round(4).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), 'estimation_results.csv'))

    # plot: error over time (last seed) + bar summary
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    tv = np.arange(STEPS) * DT
    for n, c in zip(agg, ['tab:blue', 'tab:orange', 'tab:green']):
        ax[0].plot(tv, last[n]['pos'], color=c, label=n, alpha=.8)
    ax[0].set(title='Position Error Over Time (seed 4)',
              xlabel='t [s]', ylabel='|error| [m]')
    ax[0].legend(); ax[0].grid(alpha=.3)

    names = list(agg)
    m = [np.mean(agg[n]['pos']) for n in names]
    sd = [np.std(agg[n]['pos']) for n in names]
    ax[1].bar(names, m, yerr=sd, capsize=5,
              color=['tab:blue', 'tab:orange', 'tab:green'])
    ax[1].set(title='RMS Position Error (mean ± std, 5 seeds)', ylabel='[m]')
    ax[1].grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'estimation_comparison.png'), dpi=150)
    print('Plot saved.')
