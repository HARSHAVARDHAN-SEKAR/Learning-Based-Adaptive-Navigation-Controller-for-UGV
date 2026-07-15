"""Session launcher — run the laboratory with live windows or headless.

    python3 runtime/launcher.py                                  # GUI, defaults
    python3 runtime/launcher.py --world obstacles --controller dwa
    python3 runtime/launcher.py --headless --duration 30         # no display
    python3 runtime/launcher.py --config configs/default.json

Every session auto-saves logs/run_<stamp>/ on exit. Use runtime/replay.py
to play a session back and runtime/report.py to build its PDF.
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.engine import Engine, load_config

VIZ_HZ = 15          # window refresh; sim runs dt-accurate regardless


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--world', choices=['track', 'obstacles'])
    ap.add_argument('--controller',
                    choices=['pid', 'pure_pursuit', 'stanley', 'mpc',
                             'adaptive_mpc', 'dwa'])
    ap.add_argument('--planner',
                    choices=['astar', 'theta_star', 'hybrid_astar',
                             'rrt_star', 'mppi'])
    ap.add_argument('--estimator', choices=['ekf', 'ukf', 'factor_graph'])
    ap.add_argument('--ai', choices=['none', 'ppo_residual', 'sac_residual'])
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--speed', type=float, default=1.0,
                    help='playback speed (GUI mode)')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    return ap.parse_args()


def build_engine(args):
    cfg = load_config(args.config)
    for k in ('world', 'controller', 'planner', 'estimator', 'ai'):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return Engine(cfg)


def run_headless(engine, duration):
    steps = int(duration / engine.cfg['dt'])
    for i in range(steps):
        engine.tick()
        if engine.done:
            break
    m = engine.bus.latest.get('/metrics', {})
    print(f"done: t={engine.sim.t:.1f}s goal={engine.done} "
          f"e_ct={m.get('e_ct', 0):+.3f}")


def run_gui(engine, duration, speed):
    import matplotlib.pyplot as plt
    from visualization.windows import ALL_WINDOWS, LocalPlannerWindow
    windows = [W(engine) for W in ALL_WINDOWS]
    if engine.cfg['controller'] == 'dwa':
        windows.append(LocalPlannerWindow(engine))
    plt.ion()
    plt.show(block=False)
    dt = engine.cfg['dt']
    ticks_per_frame = max(1, int((1.0 / VIZ_HZ) / dt * speed))
    steps = int(duration / dt)
    t_wall = time.perf_counter()
    for i in range(steps):
        engine.tick()
        if engine.done:
            break
        if i % ticks_per_frame == 0:
            for w in windows:
                w.update()
                w.fig.canvas.draw_idle()
            plt.pause(0.001)
            # real-time pacing
            target = engine.sim.t / speed
            lag = target - (time.perf_counter() - t_wall)
            if lag > 0:
                time.sleep(min(lag, 0.05))
    for w in windows:
        w.update()
    plt.ioff()
    print('Session ended — close windows to exit.')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    eng = build_engine(args)
    try:
        if args.headless:
            run_headless(eng, args.duration)
        else:
            run_gui(eng, args.duration, args.speed)
    finally:
        if not args.no_save:
            run_dir = eng.save_run()
            print(f'Session saved: {run_dir}')
            print(f'Replay: python3 runtime/replay.py {os.path.basename(run_dir)}')
            print(f'Report: python3 runtime/report.py {os.path.basename(run_dir)}')
