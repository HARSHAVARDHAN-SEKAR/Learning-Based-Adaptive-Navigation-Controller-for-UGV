"""ROS2 Humble launch file — navigation demo.

Brings up: state estimator (robot_localization EKF), the controller_node
from the research layer, and RViz2 for visualization.

    ros2 launch launch/navigation_demo.launch.py controller:=stanley
    ros2 launch launch/navigation_demo.launch.py controller:=adaptive_mpc v_base:=1.89 k_curv:=1.35

NOTE: requires a running simulator (Gazebo) or bag playback publishing
/scan, /imu, /wheel/odom on the expected topics — see docs/ROS2_DEMO.md
for the full bring-up sequence including how to record and replay a demo.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controller_arg = DeclareLaunchArgument(
        'controller', default_value='mpc',
        description='pid | pure_pursuit | stanley | mpc | adaptive_mpc')
    v_base_arg = DeclareLaunchArgument('v_base', default_value='1.89')
    k_curv_arg = DeclareLaunchArgument('k_curv', default_value='1.35')
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[{
            'frequency': 20.0,
            'sensor_timeout': 0.2,
            'two_d_mode': True,
            'odom0': '/wheel/odom',
            'odom0_config': [True, True, False, False, False, True,
                             False, False, False, False, False, False,
                             False, False, False],
            'imu0': '/imu',
            'imu0_config': [False, False, False, False, False, True,
                            False, False, False, False, False, True,
                            False, False, False],
        }],
        remappings=[('odometry/filtered', '/odometry/filtered')],
    )

    controller_node = Node(
        package='navigation_controller',
        executable='controller_node',
        name='navigation_controller',
        output='screen',
        parameters=[{
            'controller': LaunchConfiguration('controller'),
            'v_base': LaunchConfiguration('v_base'),
            'k_curv': LaunchConfiguration('k_curv'),
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', 'launch/nav_demo.rviz'],
        condition=None,  # gate on use_rviz via IfCondition in a full build
        output='screen',
    )

    return LaunchDescription([
        controller_arg, v_base_arg, k_curv_arg, use_rviz_arg,
        ekf_node, controller_node, rviz_node,
    ])
