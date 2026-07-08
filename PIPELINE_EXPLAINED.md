# Full Pipeline Explained + Improvement Roadmap

## Part A — How the Flow Works, Stage by Stage

### Stage 1 — Simulation Layer (why it exists)
Everything downstream needs a *plant* and *sensors*. The plant is a kinematic
bicycle — 5 states [X, Y, psi, v, delta], 2 inputs [a, delta-dot] — integrated
with RK4 at 50 ms. Rate-level inputs matter: real actuators can't jump in
steering angle, and using rates means every controller is judged on the same
physical interface. The sensor sim wraps ground truth with the noise real
hardware produces: GPS at 5 Hz with 0.15 m sigma, a gyro whose bias random-walks
(the property that makes dead-reckoning drift), and encoder speed noise.
In deployment this whole layer is replaced by Gazebo/Isaac plugins publishing
the same quantities on ROS2 topics — that substitutability is the design.

### Stage 2 — State Estimation (why before planning)
The robot never knows where it is; it fuses. Prediction step: propagate pose
with encoder + gyro (fast, drifts). Correction step: pull toward GPS
(slow, absolute). EKF linearizes the motion model with a Jacobian; UKF
propagates sigma points through the exact nonlinear model; the factor graph
re-optimizes a 30-pose sliding window each GPS tick (the GTSAM idea in
40 lines of scipy). Measured result: EKF and UKF tie at ~0.088 m — the
system is only mildly nonlinear, so the UKF's extra machinery buys nothing
*once its heading bug is fixed* (linear averaging of angles across the ±pi
seam corrupts the sigma-point mean; the fix is a circular mean). The
smoother trails at 0.110 m because with GPS every 200 ms and no loop
closures, there is nothing for smoothing to exploit. That negative result
is publishable: it tells a reader when NOT to pay the 300x compute cost.

### Stage 3 — Global Planning (what the numbers mean)
Given the fused pose and a map, produce a geometric route. A* searches the
grid — fastest (4 ms) but its 8-connected moves produce 11.8 rad of
accumulated turning, terrible for a real vehicle. Theta* fixes that with
any-angle line-of-sight shortcuts (1.1 rad) but pays 91 ms in collision
checks. Hybrid A* searches in (x, y, heading) using steering-arc motion
primitives, so every edge is *drivable by the bicycle model* — smoothest
(0.83 rad), near-shortest, 18 ms. RRT* trades determinism for
high-dimensional scalability (1.8 s here — overkill for 2D grids, essential
for manipulators). MPPI is really a sampling controller run as a planner:
its 5 s is execution time, and its 7.4 rad reflects that it reacts rather
than plans. Conclusion the data supports: Hybrid A* for car-like UGVs.

### Stage 4 — Control Layer (the core benchmark)
The planner's path is geometry; the controller turns it into wheel commands
20 times per second. PID on cross-track error is the floor (0.226 m —
it fights the curvature it can't anticipate). Pure Pursuit chases a
lookahead point — geometric, robust, but it corner-cuts (0.188 m).
Stanley adds heading alignment at the front axle and lands at 0.027 m for
0.07 ms of compute — the best classical value. NMPC solves a 30-step
optimal-control problem every cycle: predict 1.5 s ahead with the model,
minimize tracking error + control effort + control-rate, subject to actuator
limits — 0.019 m, at 13 ms. Three benchmark-integrity bugs were found here,
and the worst was silent: on the closed figure-eight the progress index
clamped at the path end, the MPC stopped after one lap, and *its error
score improved* because a parked on-path robot has zero cross-track error.
Fixed with circular indexing. A benchmark that can reward stopping is the
kind of thing reviewers (rightly) reject papers over.

### Stage 5 — Learning Layer (the thesis of the project)
Fixed MPC weights are tuned for one regime. The hypothesis: a learned
policy that adapts the controller online beats any fixed tuning. Proof at
minimum viable scale: a 2-parameter speed schedule
v_ref = v_base / (1 + k_curv·|kappa|), trained by Cross-Entropy Method
(sample parameter candidates, run episodes, refit a Gaussian to the
elites — derivative-free policy search) in 88 s of CPU. Learned
v_base = 1.89 m/s, k_curv = 1.35 — i.e., sprint the straights, brake for
curvature. Versus fixed MPC: 12% lower tracking error, 40% lower
steering-rate variance. The PPO/SAC scripts and Gymnasium environment
(smoke-tested) scale the same idea to a 15-dim observation and 4-dim
action that also modulates Q/R weights — the GPU stage.

### Stage 6 — Robustness Gate (the unique contribution)
Standard benchmarks feed controllers ground-truth state — hardware never
has it. Re-running every controller on the *EKF estimate* from noisy
sensors: MPC degrades 0.019→0.066 m, Stanley 0.027→0.066 m, Adaptive
0.017→0.064 m — all colliding with a ~0.065 m floor that is the
estimator's own error. Pure Pursuit *improves* (0.188→0.157) because its
lookahead geometry low-pass-filters estimation noise. The insight:
below ~0.07 m, money spent on control is wasted; spend it on
localization — or make the controller estimation-aware (see improvement
#1 below, which is your second paper).

### Stage 7 — ROS2 Deployment
`controller_node.py` subscribes /odometry/filtered + /plan, runs any of the
five controllers, publishes /cmd_vel at 20 Hz, exposes controller choice
and learned parameters as ROS parameters. The research layer imports
unchanged — one implementation from theory to robot.

---

## Part B — Improvement Roadmap (ranked by research value)

### Tier 1 — Direct paper material

1. **Estimation-aware adaptive control** (follows from the floor finding).
   Feed the EKF covariance trace into the adaptation policy: when
   localization degrades, soften position weights / lower v_max
   automatically. Nobody's LinkedIn project has this; it's a genuine RA-L
   angle. Cheap first version: v_max = f(trace(P)) rule, then learned.
2. **Full PPO/SAC meta-policy** on GPU using the shipped env: 5 seeds,
   PPO-vs-SAC sample-efficiency curves, ablation of the solver-health
   observation dims. This upgrades CEM's 2 params to contextual adaptation.
3. **ACADOS RTI port** with the <2 cm validation gate: takes p99 from
   13 ms to <5 ms and makes the Jetson deployment claim real.
4. **Statistical rigor pass**: 20+ seeds per condition, Welch's t-test +
   Cohen's d, std shading on every plot, results regenerated by one
   `make bench` — reviewers check this first.

### Tier 2 — Depth per layer

5. **Estimation**: GPS-outage segments (where the factor graph should
   finally win — complete that story); IMU preintegration; adaptive R from
   innovation statistics; LiDAR scan-matching odometry as a fourth source.
6. **Planning**: Reeds-Shepp analytic expansion + gradient path smoothing
   for Hybrid A*; informed-RRT* sampling; GPU-batched MPPI (PyTorch) to fix
   its 5 s; dynamic obstacles with a time dimension.
7. **Control**: dynamic bicycle with tire slip (Pacejka) + friction
   randomization — where adaptation should shine hardest; obstacle soft
   constraints in the OCP (formulation already in Deliverable 1); LQR as
   the missing classical baseline; actuator-delay compensation.
8. **Learning safety**: replace the low-pass action filter with a control
   barrier function (CBF) safety filter — provable constraint satisfaction
   under any policy output; strong reviewer appeal.

### Tier 3 — Engineering credibility

9. **Gazebo worlds + sim-to-sim**: port the four scenario suites, quantify
   the Python-sim → Gazebo gap honestly.
10. **CI pipeline**: GitHub Actions running tests/test_all_modules.py +
    one smoke episode per controller on every commit.
11. **Foxglove/RViz dashboards** replaying MCAP logs — the GIF factory for
    the LinkedIn series.
12. **Docker image** pinning every dependency; results Parquet in git-lfs.

### Suggested order
(3) ACADOS → (9) Gazebo → (2) PPO/SAC → (1) estimation-aware control →
(4) stats pass → paper. Items 5–8 slot in wherever a reviewer or interview
pushes.
