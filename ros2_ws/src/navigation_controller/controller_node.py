"""ROS2 Humble node wrapping the research-layer controllers.

Deployment stage of the pipeline. NOT run in this container (no ROS2
here) — drop into your ros2_ws, `colcon build`, and it subscribes to
/odometry/filtered + /plan and publishes /cmd_vel at 20 Hz.

    ros2 run navigation_controller controller_node --ros-args \
        -p controller:=adaptive_mpc -p v_base:=1.89 -p k_curv:=1.35
"""
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist

# research layer imports (add learning_navigation to PYTHONPATH)
from controllers.geometric import pure_pursuit, stanley, pid
from controllers.mpc_controller import MPC
from rl.cem_adaptive_mpc import AdaptiveMPC


def yaw_from_quat(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControllerNode(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        self.declare_parameter('controller', 'mpc')
        self.declare_parameter('v_base', 1.5)
        self.declare_parameter('k_curv', 1.35)
        self.mode = self.get_parameter('controller').value

        self.x = None            # [X, Y, psi, v, delta]
        self.path = None
        self.mpc = None

        self.create_subscription(Odometry, '/odometry/filtered',
                                 self.odom_cb, 10)
        self.create_subscription(Path, '/plan', self.path_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.control_cb)   # 20 Hz
        self.get_logger().info(f'Controller: {self.mode}')

    def odom_cb(self, msg):
        p = msg.pose.pose
        v = msg.twist.twist.linear.x
        delta = self.x[4] if self.x is not None else 0.0
        self.x = np.array([p.position.x, p.position.y,
                           yaw_from_quat(p.orientation), v, delta])

    def path_cb(self, msg):
        self.path = np.array([[ps.pose.position.x, ps.pose.position.y]
                              for ps in msg.poses])
        if self.mode == 'mpc':
            self.mpc = MPC()
        elif self.mode == 'adaptive_mpc':
            params = (self.get_parameter('v_base').value,
                      self.get_parameter('k_curv').value)
            self.mpc = AdaptiveMPC(params, self.path)

    def control_cb(self):
        if self.x is None or self.path is None:
            return
        if self.mode == 'pure_pursuit':
            u = pure_pursuit(self.x, self.path)
        elif self.mode == 'stanley':
            u = stanley(self.x, self.path)
        elif self.mode == 'pid':
            u = pid(self.x, self.path)
        elif self.mode == 'adaptive_mpc':
            u, _ = self.mpc.solve(self.x)
        else:
            u, _ = self.mpc.solve(self.x, self.path)

        # integrate rate-level command to velocity interface
        dt = 0.05
        self.x[4] = float(np.clip(self.x[4] + u[1] * dt, -0.5, 0.5))
        v_cmd = float(np.clip(self.x[3] + u[0] * dt, 0.0, 2.0))
        msg = Twist()
        msg.linear.x = v_cmd
        msg.angular.z = v_cmd / 0.32 * np.tan(self.x[4])
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ControllerNode())


if __name__ == '__main__':
    main()
