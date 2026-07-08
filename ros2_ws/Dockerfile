# ROS2 deployment image — separate from the root Dockerfile (research layer).
# Build:  docker build -f ros2_ws/Dockerfile -t ugv-nav-ros2 .
# Run:    docker run --rm -it --network host ugv-nav-ros2

FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-robot-localization \
    ros-humble-rviz2 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . /ros2_ws/src/learning_navigation

# colcon build happens at container start (bind-mounted workspace in dev;
# baked in for a release image — see docs/ROS2_DEMO.md for both modes)
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && colcon build --symlink-install || true

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
