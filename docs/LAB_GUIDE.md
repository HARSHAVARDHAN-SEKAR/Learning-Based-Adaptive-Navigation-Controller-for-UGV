# Navigation Laboratory — User Guide (V2)

The lab is a ROS2-style node runtime: sim → sensors → localization →
planner → controller → **safety filter** → vehicle, all observable in
live windows, all swappable at runtime, every session logged and
replayable. Migrating to real ROS2 later = replacing `runtime/bus.py`
with rclpy; node logic is already 1:1.

## Quick start (on your laptop, venv activated)

```bash
sudo apt install python3-tk        # once — interactive matplotlib backend

make lab                           # dashboard + live windows, track world
make lab-obstacles                 # obstacle world: plan -> drive -> goal
```

The **Dashboard** window has radio selectors (controller / planner /
estimator) and buttons (START/PAUSE, RESET, REPLAN, SAVE RUN + REPORT).
Switch the controller *while the robot drives* and watch the behavior
change. Selecting `dwa` opens the Local Planner window showing candidate
trajectories (grey) vs the chosen one (green), Nav2-style.

Live windows: Robot/World - Localization (GT vs EKF/UKF/FG + covariance
ellipse) - Controller (a, δ̇, e_ct scrolling) - Performance (v, e_psi,
solve ms, estimation error) - Local Planner (DWA only).

## Headless / CLI

```bash
python3 runtime/launcher.py --headless --world obstacles --controller dwa --duration 60
python3 runtime/launcher.py --world track --controller adaptive_mpc     # GUI, no dashboard
```

## Sessions: log -> replay -> report

Every session auto-saves `logs/run_<stamp>/` (trajectory, sensors,
estimate, cmd, metrics CSVs + path + meta.json — a ros2-bag equivalent).

```bash
python3 runtime/replay.py run_20260715_115639                  # play back
python3 runtime/replay.py run_20260715_115639 --save videos/demo.gif
python3 runtime/report.py run_20260715_115639                  # PDF, no re-sim
```

## Benchmark manager

```bash
make lab-bench                                                  # 5 eps, all controllers
python3 runtime/benchmark.py --controllers mpc dwa --episodes 10
```
Each episode randomizes the map and sensor noise; results land in
`results/bench_<stamp>/` with summary.csv + comparison.png.

## Safety layer (always on)

`/cmd_raw -> SafetyNode -> /cmd`: input clamping, constant-curvature
collision rollout, graduated response (soft deceleration if collision
0.25–0.5 s ahead, full brake if imminent). Any AI correction passes
through it — RL never commands the robot directly.

## AI layer

- `adaptive_mpc` controller = the CEM-trained speed schedule (works today)
- `--ai ppo_residual` / `sac_residual` = additive residual policy; loads
  `rl/ppo_residual.zip` if you've trained one (GPU stage), otherwise runs
  the base controller with a one-time notice. Residuals are clipped to
  ±[0.5 m/s², 0.3 rad/s] and still pass the safety filter.

## What was deliberately NOT included (V3+ roadmap)

Particle filter/AMCL, TEB, costmap layers, LiDAR simulation, more vehicle
models, live RL-training window. Reason: five verified layers beat
twenty-five half-working ones. Each has a natural slot: new estimators →
`localization/`, new local planners → `control/` (follow `dwa.py`), new
vehicles → `simulation/vehicle.py` + a controller adapter.

## Bugs found while building V2 (all fixed, all tested)

1. Safety rollout held steering *rate* constant → predicted phantom
   collisions on every arc → deadlock. Fix: constant-curvature rollout.
2. DWA targeted the path *extension* beyond the goal → parked 0.37 m
   short. Fix: lookahead capped at the true goal index.
3. MPC reference generator assumed dense paths (0.024 m spacing); Hybrid
   A* outputs ~0.3 m spacing → reference raced at ~6 m/s and wrapped.
   Fix: all planner paths resampled to 0.025 m + explicit open-path mode
   (clamp, hold-at-end, terminal v_ref = 0) in the MPC.
