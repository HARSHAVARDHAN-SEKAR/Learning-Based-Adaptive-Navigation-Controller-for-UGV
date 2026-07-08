"""Stage 5: ROBUSTNESS — closed-loop estimation + control integration test.

Most benchmark papers evaluate controllers on ground-truth state. Here we
run each controller on the EKF ESTIMATE computed from noisy sensors
(GPS 5 Hz sigma=0.15 m, gyro with bias walk, encoder noise) and measure the
TRUE tracking error. This quantifies each controller's sensitivity to
realistic state-estimation error — the condition it will actually face
on hardware. 3 noise seeds per controller.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from theory.vehicle_model import step, L
from theory.estimators import EKF
from simulation.sensors import SensorSim
from controllers.geometric import stanley, pure_pursuit
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)

t = np.linspace(0, 2 * np.pi, 800)
PATH = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
DT, STEPS = 0.05, 800
ADAPTIVE_PARAMS = (1.89, 1.35)          # learned by CEM in run_full_flow.py
SEEDS = 3


def cross_track(x_true):
    d = np.linalg.norm(PATH - x_true[:2], axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(PATH) - 1)
    tg = PATH[i1] - PATH[max(i0 - 1, 0)]
    tg = tg / max(np.linalg.norm(tg), 1e-9)
    e = x_true[:2] - PATH[i0]
    return -e[0] * tg[1] + e[1] * tg[0]


def run(name, seed, use_estimator):
    x = np.array([PATH[0, 0], PATH[0, 1] - 0.3, np.pi / 4, 0.0, 0.0])
    sens = SensorSim(DT, seed=seed)
    ekf = EKF(x)
    mpc = MPC() if name == 'MPC' else None
    ada = AdaptiveMPC(ADAPTIVE_PARAMS, PATH) if name == 'Adaptive-MPC' else None
    ect = np.zeros(STEPS)
    for k in range(STEPS):
        if use_estimator:
            x_ctrl = np.array([ekf.x[0], ekf.x[1], ekf.x[2], x[3], x[4]])
        else:
            x_ctrl = x
        if name == 'PurePursuit':
            u = pure_pursuit(x_ctrl, PATH)
        elif name == 'Stanley':
            u = stanley(x_ctrl, PATH)
        elif name == 'MPC':
            u, _ = mpc.solve(x_ctrl, PATH)
        else:
            u, _ = ada.solve(x_ctrl)
        omega_true = x[3] / L * np.tan(x[4])
        x = step(x, u, DT)
        v_enc, om, gps = sens.measure(x, omega_true, k)
        ekf.predict(v_enc, om, DT)
        if gps is not None:
            ekf.update_gps(gps)
        ect[k] = cross_track(x)
    s = int(0.5 / DT)
    return float(np.sqrt(np.mean(ect[s:] ** 2))), ect


if __name__ == '__main__':
    names = ['PurePursuit', 'Stanley', 'MPC', 'Adaptive-MPC']
    rows, traces = [], {}
    for n in names:
        gt, _ = run(n, 0, use_estimator=False)
        noisy = []
        for s in range(SEEDS):
            r, tr = run(n, s, use_estimator=True)
            noisy.append(r)
            if s == 0:
                traces[n] = tr
        rows.append({'Method': n,
                     'GT_state_RMS_m': gt,
                     'EKF_state_RMS_m': float(np.mean(noisy)),
                     'std': float(np.std(noisy)),
                     'Degradation_x': float(np.mean(noisy)) / gt})
        print(f'{n}: GT {gt:.4f} -> EKF {np.mean(noisy):.4f} m', flush=True)

    df = pd.DataFrame(rows).set_index('Method')
    print('\n=== ROBUSTNESS: control on EKF estimate vs ground truth ===')
    print(df.round(4).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), 'robustness_results.csv'))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xpos = np.arange(len(df))
    w = 0.35
    ax[0].bar(xpos - w / 2, df['GT_state_RMS_m'], w, label='Ground-truth state',
              color='tab:gray')
    ax[0].bar(xpos + w / 2, df['EKF_state_RMS_m'], w,
              yerr=df['std'], capsize=4, label='EKF estimate (noisy sensors)',
              color='tab:red')
    ax[0].set_xticks(xpos, df.index, fontsize=8)
    ax[0].set(title='Tracking Error: Perfect vs Estimated State',
              ylabel='RMS cross-track [m]')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')

    tv = np.arange(STEPS) * DT
    for n, c in zip(names, ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']):
        ax[1].plot(tv, traces[n], color=c, label=n, lw=0.9)
    ax[1].set(title='True Cross-Track Error, EKF-in-the-loop (seed 0)',
              xlabel='t [s]', ylabel='$e_{ct}$ [m]')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'robustness_comparison.png'), dpi=150)
    print('Plot saved.')
