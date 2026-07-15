"""Replay a logged session exactly as it happened (like ros2 bag play).

    python3 runtime/replay.py run_20260715_101530
    python3 runtime/replay.py run_20260715_101530 --speed 2
    python3 runtime/replay.py run_20260715_101530 --save videos/run.gif
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run', help='run directory name under logs/')
ap.add_argument('--speed', type=float, default=1.0)
ap.add_argument('--save', default=None, help='videos/out.gif or .mp4')
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'logs', args.run)
if not os.path.isdir(RUN):
    sys.exit(f'no such run: {RUN}')

import matplotlib
if args.save:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

traj = np.genfromtxt(os.path.join(RUN, 'trajectory.csv'),
                     delimiter=',', names=True)
est = np.genfromtxt(os.path.join(RUN, 'estimate.csv'),
                    delimiter=',', names=True)
path = np.genfromtxt(os.path.join(RUN, 'path.csv'),
                     delimiter=',', names=True)
meta = json.load(open(os.path.join(RUN, 'meta.json')))

ROBOT = np.array([[0.25, 0], [-0.15, 0.13], [-0.15, -0.13]])
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(path['x'], path['y'], 'k--', lw=1, alpha=.5, label='path')
trail, = ax.plot([], [], color='tab:blue', lw=1.4, label='ground truth')
est_trail, = ax.plot([], [], color='tab:orange', lw=1, alpha=.7,
                     label=f"estimate ({meta.get('estimator', '?')})")
body = plt.Polygon([[0, 0]], closed=True, fc='tab:red', ec='k', zorder=5)
ax.add_patch(body)
pad = 0.5
ax.set_xlim(path['x'].min() - pad, path['x'].max() + pad)
ax.set_ylim(path['y'].min() - pad, path['y'].max() + pad)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=8)
hud = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9,
              family='monospace',
              bbox=dict(boxstyle='round', fc='w', alpha=.85))
ax.set_title(f"REPLAY {args.run} — {meta.get('controller')} / "
             f"{meta.get('planner')} / {meta.get('estimator')}")

n = len(traj['t'])
stride = max(1, int(args.speed))


def frame(i):
    j = min(i * stride, n - 1)
    trail.set_data(traj['x'][:j], traj['y'][:j])
    k = min(j, len(est['t']) - 1)
    est_trail.set_data(est['x'][:k], est['y'][:k])
    c, s = np.cos(traj['psi'][j]), np.sin(traj['psi'][j])
    R = np.array([[c, -s], [s, c]])
    body.set_xy(ROBOT @ R.T + [traj['x'][j], traj['y'][j]])
    hud.set_text(f"t={traj['t'][j]:6.1f}s  v={traj['v'][j]:.2f} m/s")
    return trail, est_trail, body, hud


dt = meta.get('dt', 0.05)
ani = animation.FuncAnimation(fig, frame, frames=n // stride,
                              interval=int(1000 * dt), blit=True,
                              repeat=False)
if args.save:
    out = args.save if os.path.isabs(args.save) \
        else os.path.join(ROOT, args.save)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fps = int(1 / dt)
    writer = 'pillow' if out.endswith('.gif') else 'ffmpeg'
    print(f'rendering {n // stride} frames -> {out}')
    ani.save(out, writer=writer, fps=fps)
    print('saved')
else:
    plt.show()
