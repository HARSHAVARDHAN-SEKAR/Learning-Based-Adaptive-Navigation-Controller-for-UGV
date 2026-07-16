# Learning-Based Adaptive Navigation Controller for UGVs

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10|3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)](ros2_ws/)

> Push the badges live by connecting this repo to GitHub Actions — the
> workflow file is already in `.github/workflows/ci.yml` and passes
> locally (`make test`).

## Evidence Status

| Artifact | Status | Where |
|---|---|---|
| Source code, 7 modules | DONE — tested, all pass | `tests/test_all_modules.py` |
| Benchmark results (4 studies) | DONE — reproduced from clean checkout | `benchmarks/*.csv` |
| Comparison charts (5) | DONE — regenerated, committed | `benchmarks/plots/` |
| Technical report (PDF) | DONE — auto-generated from results | `paper_report.pdf` |
| Dockerfile (research layer) | DONE — builds + self-tests | `Dockerfile` |
| Dockerfile (ROS2 layer) | PARTIAL — written, not build-tested (needs ROS2 host/CI runner) | `ros2_ws/Dockerfile` |
| CI workflow | PARTIAL — passes locally, not yet run on GitHub | `.github/workflows/ci.yml` |
| ROS2 launch + node | PARTIAL — written, not run against a live ROS2 graph | `launch/`, `ros2_ws/` |
| **Navigation Laboratory (V2)** | **DONE — GUI verified on independent hardware, 8+ live sessions logged** | `runtime/`, `docs/LAB_GUIDE.md` |
| Demo video | TODO — script ready, not recorded | `docs/VIDEO_SCRIPT.md` |
| PPO/SAC training | TODO — env smoke-tested, not trained (GPU stage) | `rl/ppo_training.py` |

This table is here on purpose — a repo that quietly implies everything
works is less credible than one that states exactly what's proven and
what isn't.



Full open-source research pipeline: **sensor simulation → state estimation →
planner benchmark → controller benchmark → learning layer → ROS2 deployment.**
Every result below was produced by running the code in this repo (fixed seeds).

## Pipeline & Verified Results

### Stage 1 — State Estimation (5 seeds, GPS 5 Hz + IMU + encoder)

| Estimator | RMS pos [m] | RMS heading [rad] | Compute [ms] |
|---|---|---|---|
| EKF | 0.0876 ± 0.004 | 0.0355 | 0.04 |
| UKF | 0.0874 ± 0.004 | 0.0354 | 0.17 |
| Factor Graph (sliding window) | 0.1101 ± 0.011 | 0.0646 | 14.2 |

*Findings:* UKF ≈ EKF once the sigma-point mean uses a **circular mean for
heading** (a bug we hit and fixed — linear angle averaging breaks at ±π).
The smoother doesn't pay off in this GPS-rich scenario; its advantage
appears with loop closures / delayed measurements (future SLAM stage).

### Stage 2 — Planner Benchmark (3 maps × 3 seeds)

| Planner | Time [ms] | Length [m] | Smoothness [rad] | Success |
|---|---|---|---|---|
| A* | 4.4 | 6.50 | 11.78 | 100% |
| Theta* | 91.0 | 6.17 | 1.11 | 100% |
| Hybrid A* | 18.2 | 6.10 | **0.83** | 100% |
| RRT* | 1762 | 6.28 | 1.34 | 100% |
| MPPI | 4991 | 6.28 | 7.40 | 100% |

*Findings:* Hybrid A* dominates for car-like robots — near-shortest,
smoothest, kinematically feasible, 5× faster than any-angle Theta*.

### Stage 3+4 — Controllers + Learning (figure-eight, 40 s)

| Method | RMS e_ct [m] | Steer var | Mean speed [m/s] | p99 [ms] |
|---|---|---|---|---|
| PID | 0.2262 | 0.193 | 1.48 | 0.07 |
| Pure Pursuit | 0.1879 | 0.049 | 1.48 | 0.22 |
| Stanley | 0.0269 | 0.054 | 1.48 | 0.07 |
| MPC (CasADi/ipopt) | 0.0187 | 0.023 | 1.78 | 13.0 |
| **Adaptive-MPC (learned)** | **0.0165** | **0.014** | 1.38 | 12.5 |

The learning layer (Cross-Entropy Method, 88 s training on CPU) learns a
curvature-scheduled speed policy `v_ref = v_base/(1 + k_curv·|κ|)` →
converged to v_base = 1.89 m/s, k_curv = 1.35. Result: **12% lower
tracking error and 40% lower steering variance than fixed MPC**, trading
average speed — the adaptation hypothesis confirmed end-to-end. CEM is
the CPU-friendly stand-in for the PPO/SAC meta-policy (see docs
Deliverable 2), which trains on the full 58-dim observation in Isaac Sim.

### Bugs found & fixed during this build (paper Discussion material)

1. **MPC reference lobe-jumping** at the figure-eight self-intersection
   (global nearest-point search) → windowed progress tracking.
2. **UKF heading divergence** from linear angle averaging in the
   sigma-point mean → circular mean.
3. **Closed-path index clamping**: MPC silently stopped after one lap,
   which *inflated* its error metric (a stopped on-path robot has zero
   error) → circular indexing. This one would have invalidated the
   benchmark.

## Repo Layout

```
learning_navigation/
├── theory/            vehicle_model.py, estimators.py (EKF/UKF/FactorGraph)
├── simulation/         sensors.py, vehicle.py, world.py (track + obstacle worlds)
├── planners/            planners.py (A*, Theta*, Hybrid A*, RRT*, MPPI) — V1 core
├── planning/             thin re-exports of planners/ for the V2 node layout
├── controllers/          geometric.py (PID/PP/Stanley), mpc_controller.py — V1 core
├── control/              thin re-exports + dwa.py (local planner, V2-only)
├── localization/         thin re-exports of theory/estimators.py for V2 nodes
├── rl/                   cem_adaptive_mpc.py (learning layer), ppo_training.py,
│                         sac_training.py (SB3 scripts, GPU stage)
├── runtime/               bus.py, nodes.py, engine.py, safety.py, dashboard.py,
│                         launcher.py, replay.py, report.py, benchmark.py — V2 lab
├── visualization/         windows.py (5 live matplotlib windows)
├── configs/               default.json
├── benchmarks/            run_estimation.py, run_planners.py, run_full_flow.py,
│                         run_robustness.py, make_report.py, make_diagram.py,
│                         live_simulation.py — V1 batch pipeline, *.csv, plots/
├── logs/                  session recordings (gitignored, regenerated by the lab)
├── results/               benchmark manager output (gitignored)
├── reports/               per-session PDFs (gitignored)
├── videos/                replay exports (gitignored)
├── launch/                navigation_demo.launch.py (ROS2)
├── ros2_ws/               controller_node.py, Dockerfile (ROS2 deployment)
├── docs/                  LAB_GUIDE.md, ROS2_DEMO.md, VIDEO_SCRIPT.md
└── tests/                 test_all_modules.py (7 modules incl. the V2 runtime)
```

## Navigation Laboratory (V2 — interactive runtime)

```bash
sudo apt install python3-tk   # once
make lab                      # dashboard + live windows (track world)
make lab-obstacles            # plan -> drive -> goal with safety layer
make lab-bench                # multi-episode benchmark manager
```
Node-based runtime (ROS2-style topics), 6 swappable controllers incl. a
DWA local planner with Nav2-style candidate visualization, 3 estimators
switchable live, a graduated safety filter, session logging with replay
(`runtime/replay.py`) and per-run PDF reports (`runtime/report.py`).
Full guide: `docs/LAB_GUIDE.md`.

## Run Everything

```bash
pip install -r requirements.txt
make bench      # all 4 benchmark studies, ~10 min total
make report     # + PDF report + architecture diagram
```

Or with Docker (no local Python setup needed):
```bash
make docker-build
make docker-run
```

## Repository Evidence

- **CI:** `.github/workflows/ci.yml` — runs the module test suite and two
  smoke tests on every push, on Python 3.10 and 3.11
- **Docker:** `Dockerfile` builds the research layer and runs the full
  test suite *as part of the build* — a broken build means broken code
- **Reproducibility:** `Makefile` — `make bench` is the entire pipeline,
  one command, same results every time (fixed seeds throughout)
- **ROS2:** `launch/navigation_demo.launch.py` + `ros2_ws/Dockerfile` —
  see `docs/ROS2_DEMO.md` for the exact bring-up sequence and the one
  known integration step left to do (package.xml for the controller node)
- **Video:** `docs/VIDEO_SCRIPT.md` — shot-by-shot script for a 2:30 demo
  video, written to lead with the robustness finding


## What Runs Where

- **This repo (CPU, verified):** everything above.
- **Your machine next:** Gazebo/Isaac sensor plugins replace
  `simulation/sensors.py`; PPO/SAC (Stable Baselines3) replaces CEM using
  the spec in Deliverable 2; `controller_node.py` goes into your ros2_ws.

### Stage 5 — Robustness: EKF in the Control Loop (3 seeds)

| Method | GT-state RMS [m] | EKF-state RMS [m] | Degradation |
|---|---|---|---|
| Pure Pursuit | 0.188 | 0.157 | 0.84x (improves!) |
| Stanley | 0.027 | 0.066 | 2.4x |
| MPC | 0.019 | 0.066 | 3.5x |
| Adaptive-MPC | 0.017 | 0.064 | 3.9x |

*Key finding:* all precision controllers collapse to a common ~0.065 m
error floor set by estimation accuracy; Pure Pursuit improves because its
lookahead low-pass filters estimation noise. **Localization, not control,
is the binding constraint below ~0.07 m** — this motivates the estimator-
health signals in the RL observation (Deliverable 2).

### Stage 9 — Independent Live Verification (V2 Navigation Laboratory)

Every result above was produced by me, in one environment. The obstacle
world in the interactive lab was then run **independently, live, on
different hardware**, through the actual GUI (not headless) — 8 sessions,
6 different controller/planner/estimator combinations, each auto-logged
and auto-reported:

| Planner | Controller | Estimator | Goal? | RMS e_ct [m] | Duration [s] |
|---|---|---|---|---|---|
| Hybrid A* | **DWA** | EKF | True | **0.072** | **7.5** |
| A* | Pure Pursuit | EKF | True | 0.097 | 11.4 |
| Theta* | Stanley | Factor Graph | True | 0.080 | 11.5 |
| Hybrid A* | PID | UKF | True | 0.282 | 12.5 |
| MPPI | Pure Pursuit | EKF | **False** | 0.081–0.097 | 12–19 |

Two things this run confirmed independently of the original benchmark:
**PID is worst** (0.28 m, matching Stage 3's fixed-track result) and
**DWA is the strongest all-rounder** in cluttered environments (fastest,
smoothest, zero safety-filter interventions needed — see
`docs/LAB_GUIDE.md`).

It also caught a **new bug the batch pipeline never would have found**:
`planner=mppi` failed to reach the goal in both live trials. Root cause:
the MPPI planner's internal model is an unconstrained unicycle, so it
plans turns tighter than the bicycle-model vehicle (±0.5 rad steering
limit) can physically execute — a kinematic-feasibility mismatch between
planner and vehicle. This only surfaces with a live GUI and a real
obstacle map; it's now tracked as a known issue (fix: cap MPPI's angular
rate to the vehicle's turning limit before using it as a global planner).

Runs were reproducible even through the GUI: two separate live sessions
with identical configuration produced bit-identical results (RMS 0.0973,
duration 11.35 s, to 4 decimal places) — the fixed-seed determinism holds
end-to-end, not just in headless batch mode.

## Report & Tests

- `paper_report.pdf` — 4-page research report with all tables and figures
- `tests/test_all_modules.py` — verification of all 7 modules, incl. the
  V2 runtime laboratory (passes)
- `rl/ppo_training.py`, `rl/sac_training.py` — SB3 meta-policy training
  (Gymnasium env smoke-tested here; training is the GPU stage)

