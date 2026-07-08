"""Stage 2 benchmark: A* vs Theta* vs Hybrid A* vs RRT* vs MPPI.

3 maps x 3 seeds (sampling planners). Metrics: plan time, path length,
smoothness (sum |heading change|), success rate.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from planners.planners import GridMap, astar, hybrid_astar, rrt_star, mppi, RES

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)


def path_metrics(p):
    seg = np.diff(p, axis=0)
    length = float(np.sum(np.linalg.norm(seg, axis=1)))
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    dh = np.abs(np.arctan2(np.sin(np.diff(hd)), np.cos(np.diff(hd))))
    return length, float(np.sum(dh))


PLANNERS = {
    'A*':        lambda m, s: astar(m),
    'Theta*':    lambda m, s: astar(m, theta_variant=True),
    'Hybrid A*': lambda m, s: hybrid_astar(m),
    'RRT*':      lambda m, s: rrt_star(m, seed=s),
    'MPPI':      lambda m, s: mppi(m, seed=s),
}
MAPS = [GridMap(seed=i, n_obs=8) for i in (1, 2, 3)]
SEEDS = 3

if __name__ == '__main__':
    rows, keep = [], {}
    for name, fn in PLANNERS.items():
        times, lens, smooth, succ = [], [], [], 0
        n_runs = 0
        for mi, m in enumerate(MAPS):
            for s in range(SEEDS):
                n_runs += 1
                t0 = time.perf_counter()
                p = fn(m, s)
                dt_ms = (time.perf_counter() - t0) * 1000
                if p is not None:
                    succ += 1
                    times.append(dt_ms)
                    l, sm = path_metrics(p)
                    lens.append(l)
                    smooth.append(sm)
                    if mi == 0 and s == 0:
                        keep[name] = p
        rows.append({'Planner': name,
                     'Time_ms': np.mean(times) if times else np.nan,
                     'Length_m': np.mean(lens) if lens else np.nan,
                     'Smoothness_rad': np.mean(smooth) if smooth else np.nan,
                     'Success_%': 100.0 * succ / n_runs})
        print(f'{name}: done', flush=True)

    df = pd.DataFrame(rows).set_index('Planner')
    print('\n=== PLANNER BENCHMARK (3 maps x 3 seeds) ===')
    print(df.round(3).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), 'planner_results.csv'))

    # ---- plots: map overlay + metric bars ----
    fig = plt.figure(figsize=(13, 5))
    ax0 = fig.add_subplot(1, 2, 1)
    m = MAPS[0]
    ax0.imshow(m.grid.T, origin='lower', cmap='Greys',
               extent=[0, m.nx * RES, 0, m.ny * RES], alpha=.6)
    cols = plt.cm.tab10(np.linspace(0, 1, len(PLANNERS)))
    for (name, p), c in zip(keep.items(), cols):
        ax0.plot(p[:, 0], p[:, 1], color=c, label=name, lw=2)
    ax0.plot(*m.start, 'go', ms=10, label='start')
    ax0.plot(*m.goal, 'r*', ms=14, label='goal')
    ax0.set(title='Planned Paths — Map 1', xlabel='X [m]', ylabel='Y [m]')
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.bar(df.index, df['Time_ms'], color=cols)
    ax1.set_yscale('log')
    ax1.set(title='Planning Time [ms, log]')
    ax1.grid(alpha=.3, axis='y')

    ax2 = fig.add_subplot(2, 2, 4)
    w = 0.35
    xpos = np.arange(len(df))
    ax2.bar(xpos - w / 2, df['Length_m'], w, label='Length [m]')
    ax2.bar(xpos + w / 2, df['Smoothness_rad'], w, label='Smoothness [rad]')
    ax2.set_xticks(xpos, df.index, fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'planner_comparison.png'), dpi=150)
    print('Plot saved.')
