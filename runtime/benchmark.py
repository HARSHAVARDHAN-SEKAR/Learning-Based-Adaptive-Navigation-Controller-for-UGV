"""Benchmark Manager — multi-episode experiments from one command.

    python3 runtime/benchmark.py --world obstacles --episodes 5
    python3 runtime/benchmark.py --controllers mpc dwa stanley --episodes 10

Each episode uses a different map/noise seed. Progress prints live;
results land in results/<stamp>/summary.csv and a comparison PNG.
"""
import sys, os, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from runtime.engine import Engine, load_config

ALL = ['pid', 'pure_pursuit', 'stanley', 'mpc', 'adaptive_mpc', 'dwa']


def episode(cfg, seed):
    cfg = dict(cfg, seed=seed, map_seed=seed + 1)
    e = Engine(cfg)
    ects = []
    max_t = 120.0 if cfg['world'] == 'obstacles' else 40.0
    while e.sim.t < max_t and not e.done:
        e.tick()
        m = e.bus.latest.get('/metrics')
        if m and m['t'] > 0.5:
            ects.append(m['e_ct'])
    return {'success': e.done or cfg['world'] == 'track',
            'time_s': e.sim.t,
            'rms_ect': float(np.sqrt(np.mean(np.square(ects))))
            if ects else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='obstacles',
                    choices=['track', 'obstacles'])
    ap.add_argument('--controllers', nargs='+', default=ALL, choices=ALL)
    ap.add_argument('--episodes', type=int, default=5)
    args = ap.parse_args()

    base = load_config()
    base['world'] = args.world
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', f'bench_{stamp}')
    os.makedirs(out, exist_ok=True)

    rows = []
    total = len(args.controllers) * args.episodes
    k = 0
    for ctrl in args.controllers:
        res = []
        for ep in range(args.episodes):
            k += 1
            print(f'[{k}/{total}] {ctrl} episode {ep + 1} ...', flush=True)
            res.append(episode(dict(base, controller=ctrl), seed=ep))
        rows.append({
            'controller': ctrl,
            'success_%': 100.0 * np.mean([r['success'] for r in res]),
            'mean_time_s': float(np.mean([r['time_s'] for r in res])),
            'rms_ect_m': float(np.nanmean([r['rms_ect'] for r in res])),
            'rms_ect_std': float(np.nanstd([r['rms_ect'] for r in res])),
        })

    import csv
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'controller':14s} {'succ%':>6s} {'time':>7s} {'rms_ect':>9s}")
    for r in rows:
        print(f"{r['controller']:14s} {r['success_%']:6.0f} "
              f"{r['mean_time_s']:7.1f} {r['rms_ect_m']:9.4f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    names = [r['controller'] for r in rows]
    axs[0].bar(names, [r['rms_ect_m'] for r in rows],
               yerr=[r['rms_ect_std'] for r in rows], capsize=4)
    axs[0].set_title(f'RMS cross-track [m] — {args.world}, '
                     f'{args.episodes} episodes')
    axs[1].bar(names, [r['success_%'] for r in rows], color='tab:green')
    axs[1].set_title('Success [%]'); axs[1].set_ylim(0, 105)
    for a in axs:
        a.tick_params(axis='x', labelsize=8)
        a.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'comparison.png'), dpi=140)
    print(f'\nresults: {out}')


if __name__ == '__main__':
    main()
