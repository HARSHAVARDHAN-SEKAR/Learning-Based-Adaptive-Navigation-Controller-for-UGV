"""Live, real-time animated simulation — watch the robot actually drive.

Unlike the static benchmark plots, this renders the robot moving along
the figure-eight frame-by-frame, at real playback speed (or faster/
slower), with live readouts of speed and cross-track error. Also doubles
as your video-evidence generator: --save writes an .mp4 or .gif directly,
no screen-recording software needed.

Usage:
    python3 benchmarks/live_simulation.py                        # MPC, live window
    python3 benchmarks/live_simulation.py --controller stanley
    python3 benchmarks/live_simulation.py --controller adaptive_mpc
    python3 benchmarks/live_simulation.py --speed 2.0             # 2x playback
    python3 benchmarks/live_simulation.py --save demo.gif         # headless, no window
    python3 benchmarks/live_simulation.py --save demo.mp4         # needs ffmpeg installed

Requires a display (X11/Wayland) for the live window. On a fresh Ubuntu
install with no GUI packages this may need: sudo apt install python3-tk
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--controller', default='mpc',
                choices=['pid', 'pure_pursuit', 'stanley', 'mpc', 'adaptive_mpc'])
ap.add_argument('--speed', type=float, default=1.0, help='playback speed multiplier')
ap.add_argument('--save', default=None, help='output path: demo.gif or demo.mp4')
ap.add_argument('--duration', type=float, default=40.0, help='sim seconds to run')
args = ap.parse_args()

import matplotlib
if args.save:
    matplotlib.use('Agg')          # headless — no window needed to save a file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

from theory.vehicle_model import step
from controllers.geometric import pid, pure_pursuit, stanley
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC

DT = 0.05
t = np.linspace(0, 2 * np.pi, 800)
PATH = np.column_stack([4 * np.sin(t), 2 * np.sin(t) * np.cos(t)])
STEPS = int(args.duration / DT)
X0 = np.array([PATH[0, 0], PATH[0, 1] - 0.3, np.pi / 4, 0.0, 0.0])
ADAPTIVE_PARAMS = (1.89, 1.35)          # CEM-learned, from run_full_flow.py

mpc = MPC() if args.controller == 'mpc' else None
ada = AdaptiveMPC(ADAPTIVE_PARAMS, PATH) if args.controller == 'adaptive_mpc' else None


def cross_track(x):
    d = np.linalg.norm(PATH - x[:2], axis=1)
    i0 = int(np.argmin(d))
    i1 = min(i0 + 1, len(PATH) - 1)
    tg = PATH[i1] - PATH[max(i0 - 1, 0)]
    tg = tg / max(np.linalg.norm(tg), 1e-9)
    e = x[:2] - PATH[i0]
    return -e[0] * tg[1] + e[1] * tg[0]


def get_u(x):
    if args.controller == 'pid':
        return pid(x, PATH)
    if args.controller == 'pure_pursuit':
        return pure_pursuit(x, PATH)
    if args.controller == 'stanley':
        return stanley(x, PATH)
    if args.controller == 'mpc':
        u, _ = mpc.solve(x, PATH)
        return u
    u, _ = ada.solve(x)
    return u


# ---------------- figure setup ----------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(PATH[:, 0], PATH[:, 1], 'k--', lw=1, alpha=0.5, label='Reference path')
trail, = ax.plot([], [], '-', color='tab:blue', lw=1.5, alpha=0.7)
robot_body = plt.Polygon([[0, 0]], closed=True, fc='tab:red', ec='k', zorder=5)
ax.add_patch(robot_body)
ax.set_xlim(-5.5, 5.5); ax.set_ylim(-3, 3); ax.set_aspect('equal')
ax.set_title(f'Live Simulation — {args.controller.upper()}')
info = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=10,
               family='monospace',
               bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.legend(loc='lower right', fontsize=8)

# triangle robot shape, in the robot's own local frame (nose points +X)
ROBOT_SHAPE = np.array([[0.25, 0.0], [-0.15, 0.13], [-0.15, -0.13]])

state = {'x': X0.copy(), 'trail_x': [], 'trail_y': []}


def init():
    trail.set_data([], [])
    return trail, robot_body, info


def frame(i):
    x = state['x']
    u = get_u(x)
    x = step(x, u, DT)
    state['x'] = x
    state['trail_x'].append(x[0])
    state['trail_y'].append(x[1])
    trail.set_data(state['trail_x'], state['trail_y'])

    c, s = np.cos(x[2]), np.sin(x[2])
    R = np.array([[c, -s], [s, c]])
    pts = ROBOT_SHAPE @ R.T + x[:2]
    robot_body.set_xy(pts)

    ect = cross_track(x)
    info.set_text(f'controller: {args.controller}\n'
                  f'v = {x[3]:.2f} m/s\n'
                  f'e_ct = {ect:+.3f} m\n'
                  f't = {i * DT:5.1f} s')
    return trail, robot_body, info


interval_ms = max(1, int(1000 * DT / args.speed))
ani = animation.FuncAnimation(fig, frame, frames=STEPS, init_func=init,
                              interval=interval_ms, blit=True, repeat=False)

if args.save:
    print(f'Rendering {STEPS} frames to {args.save} ...')
    if args.save.endswith('.gif'):
        ani.save(args.save, writer='pillow', fps=int(1 / DT * args.speed))
    else:
        ani.save(args.save, writer='ffmpeg', fps=int(1 / DT * args.speed))
    print(f'Saved: {args.save}')
else:
    plt.show()
