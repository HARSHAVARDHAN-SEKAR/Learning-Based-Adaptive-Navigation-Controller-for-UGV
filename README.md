# Learning-Based Adaptive Navigation Controller for UGVs

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
├── simulation/        sensors.py (noisy GPS/IMU/encoder — Gazebo stand-in)
├── planners/          planners.py (A*, Theta*, Hybrid A*, RRT*, MPPI)
├── controllers/       geometric.py (PID/PP/Stanley), mpc_controller.py
├── rl/                cem_adaptive_mpc.py (learning layer)
├── benchmarks/        run_estimation.py, run_planners.py, run_full_flow.py
│                      *.csv results, plots/
└── ros2_ws/src/navigation_controller/controller_node.py (deployment)
```

## Run Everything

```bash
pip install numpy scipy matplotlib pandas casadi filterpy
python3 benchmarks/run_estimation.py     # Stage 1  (~1 min)
python3 benchmarks/run_planners.py       # Stage 2  (~1 min)
python3 benchmarks/run_full_flow.py      # Stage 3+4, trains CEM (~5 min)
```

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

## Report & Tests

- `paper_report.pdf` — 4-page research report with all tables and figures
- `tests/test_all_modules.py` — verification of all 6 modules (passes)
- `rl/ppo_training.py`, `rl/sac_training.py` — SB3 meta-policy training
  (Gymnasium env smoke-tested here; training is the GPU stage)
