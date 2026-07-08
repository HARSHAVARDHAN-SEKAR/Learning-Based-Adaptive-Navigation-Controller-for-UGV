"""Sensor simulation layer.

Stands in for the Gazebo/Isaac sensor plugins in this research phase:
generates realistic noisy GPS, IMU (gyro with bias walk), and wheel
encoder measurements from ground-truth vehicle states. The estimation
benchmark consumes these exactly as the ROS2 stack later consumes
/scan, /imu, /odom.
"""
import numpy as np


class SensorSim:
    def __init__(self, dt, seed=0):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.gyro_bias = 0.0
        # noise parameters (typical low-cost hardware)
        self.sig_gps = 0.15          # GPS position sigma [m]
        self.sig_enc = 0.03          # encoder speed sigma [m/s]
        self.sig_gyro = 0.01         # gyro sigma [rad/s]
        self.bias_walk = 0.0005      # gyro bias random walk [rad/s/sqrt(s)]
        self.gps_every = 4           # GPS at 5 Hz if dt = 0.05

    def measure(self, x_true, omega_true, k):
        """Return (v_enc, omega_gyro, gps or None) at step k."""
        self.gyro_bias += self.bias_walk * np.sqrt(self.dt) * self.rng.standard_normal()
        v_enc = x_true[3] + self.sig_enc * self.rng.standard_normal()
        omega = omega_true + self.gyro_bias + self.sig_gyro * self.rng.standard_normal()
        gps = None
        if k % self.gps_every == 0:
            gps = x_true[:2] + self.sig_gps * self.rng.standard_normal(2)
        return v_enc, omega, gps
