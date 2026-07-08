# ROS2 Demo — Bring-Up Guide

Everything in this file is written against ROS2 **Humble** and matches the
code in `ros2_ws/`, but **has not been executed** — this container has no
ROS2 installation. Treat this as a precise checklist, not a "trust me"
claim; run it once on your machine and fix whatever the first real error
is (there is usually exactly one, noted below).

## 1. Prerequisites

```bash
# Ubuntu 22.04 + ROS2 Humble already installed, or use the Docker image:
docker build -f ros2_ws/Dockerfile -t ugv-nav-ros2 .
docker run --rm -it --network host ugv-nav-ros2
```

## 2. Build the workspace

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

**Expected first error (fix it, don't skip it):** `navigation_controller`
is a pure-Python module here without a `package.xml`/`setup.py` yet —
add them (standard `ament_python` template) before `colcon build` will
find it. This is intentionally left as the one real integration step;
copy the template from any `ros2 pkg create --build-type ament_python`
output and point `entry_points` at `controller_node:main`.

## 3. Bring up a simulator

Pick one:

**Gazebo (recommended first — matches the Dockerfile):**
```bash
ros2 launch gazebo_ros gazebo.launch.py world:=worlds/empty.world
ros2 run gazebo_ros spawn_entity.py -entity ugv -file urdf/ugv.urdf
```
You'll need a URDF for your specific robot — the kinematic bicycle model
in `theory/vehicle_model.py` gives you the wheelbase `L` and limits to
put in it.

**Or: replay a recorded bag** (no simulator needed, good for a first
smoke test of the controller node alone):
```bash
ros2 bag play demo.mcap --loop   # publishes /scan /imu /wheel/odom
```

## 4. Launch the navigation stack

```bash
ros2 launch launch/navigation_demo.launch.py controller:=stanley
```

Swap `controller:=` for `pid | pure_pursuit | stanley | mpc | adaptive_mpc`.
For `adaptive_mpc`, also pass the CEM-learned parameters:
```bash
ros2 launch launch/navigation_demo.launch.py \
    controller:=adaptive_mpc v_base:=1.89 k_curv:=1.35
```

## 5. Verify it's actually working

```bash
ros2 topic hz /cmd_vel              # should read ~20 Hz
ros2 topic echo /odometry/filtered --once
ros2 run rqt_graph rqt_graph        # visually confirm the node graph
                                     # matches Deliverable 3's diagram
```

## 6. Record your own demo bag (for the video / repo evidence)

```bash
ros2 bag record -o demo_run /odometry/filtered /plan /cmd_vel /tf
```
Play it back with Foxglove or `ros2 bag play demo_run` + RViz2 for the
screen-recording in `docs/VIDEO_SCRIPT.md`.

## Known gaps (be upfront about these, don't hide them)

- `controller_node.py` currently rebuilds the MPC's internal warm-start
  state on every `/plan` update — fine for a static path, will need a
  replan-aware reset guard for dynamic replanning.
- No obstacle-avoidance constraints wired into the ROS2 `mpc_controller.py`
  yet — the soft constraints exist in the ACADOS formulation
  (Deliverable 1) but the CasADi version here doesn't take a costmap
  input. Port that before running near real obstacles.
- `use_rviz` launch argument is declared but not yet wired to an
  `IfCondition` — RViz will always launch until that's added.
