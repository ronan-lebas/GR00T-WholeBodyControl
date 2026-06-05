# Quest ROS2 → ZMQ relay image.
#
# Build context must be the repo root so we can COPY from both
# third_party/ROS-TCP-Endpoint/ and gear_sonic_deploy/docker/quest_relay/.
#
# Preferred entry point — the launcher handles build, run, port mapping,
# log streaming, and teardown for you:
#   python gear_sonic_deploy/docker/quest_relay/run_quest_relay.py
#
# Manual equivalent (from repo root), if you'd rather drive docker yourself:
#   docker build -t quest-relay -f gear_sonic_deploy/docker/quest_relay/Dockerfile .
#   docker run --rm -p 10000:10000 -p 5559:5559 quest-relay

FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Build deps + CycloneDDS (fixes rootless Docker multicast; same as third_party/ROS-TCP-Endpoint/Dockerfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Build vr_haptic_msgs and ros_tcp_endpoint from the local third_party copy
WORKDIR /ros2_ws
COPY third_party/ROS-TCP-Endpoint/ src/ros_tcp_endpoint/
# vr_haptic_msgs lives inside the repo; move it to a sibling package so colcon
# doesn't see it as a nested package of ros_tcp_endpoint.
RUN mv src/ros_tcp_endpoint/vr_haptic_msgs src/vr_haptic_msgs
RUN . /opt/ros/humble/setup.sh \
    && colcon build --packages-select vr_haptic_msgs ros_tcp_endpoint \
    && rm -rf build log

# CycloneDDS unicast config (no multicast — required for slirp4netns / rootless Docker)
COPY third_party/ROS-TCP-Endpoint/cyclonedds.xml /cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///cyclonedds.xml

RUN pip3 install --no-cache-dir pyzmq msgpack

COPY gear_sonic_deploy/docker/quest_relay/relay.py /relay.py
COPY gear_sonic_deploy/docker/quest_relay/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 10000 = ROS TCP Connector (Quest Unity app connects here)
# 5559  = ZMQ relay output (host quest_manager connects here)
EXPOSE 10000
EXPOSE 5559

ENTRYPOINT ["/entrypoint.sh"]
