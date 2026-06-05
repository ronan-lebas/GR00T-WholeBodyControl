# Quest ROS1 → ZMQ relay image.
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

FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Build tooling + message deps. ROS1 uses a ROS master + TCPROS (no DDS), so no
# CycloneDDS machinery is needed. geometry_msgs/tf2_msgs are required because the
# relay subscribes to the head pose and /tf.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-geometry-msgs \
    ros-noetic-tf2-msgs \
    && rm -rf /var/lib/apt/lists/*

# Build vr_haptic_msgs and ros_tcp_endpoint in a catkin workspace.
WORKDIR /catkin_ws
COPY third_party/ROS-TCP-Endpoint/ src/ros_tcp_endpoint/
# The copy of vr_haptic_msgs vendored inside ROS-TCP-Endpoint ships ament (ROS2)
# build files that won't build under catkin. Drop it and reassemble a catkin
# package from the .msg files plus the catkin build files in quest_relay/.
RUN rm -rf src/ros_tcp_endpoint/vr_haptic_msgs
COPY third_party/ROS-TCP-Endpoint/vr_haptic_msgs/msg/ src/vr_haptic_msgs/msg/
COPY gear_sonic_deploy/docker/quest_relay/vr_haptic_msgs_ros1/package.xml src/vr_haptic_msgs/package.xml
COPY gear_sonic_deploy/docker/quest_relay/vr_haptic_msgs_ros1/CMakeLists.txt src/vr_haptic_msgs/CMakeLists.txt
# Container launch variant without the adb_reverse node (Quest connects over TCP).
COPY gear_sonic_deploy/docker/quest_relay/endpoint_no_adb.launch src/ros_tcp_endpoint/launch/endpoint_no_adb.launch

RUN . /opt/ros/noetic/setup.sh \
    && catkin_make \
    && rm -rf build

RUN pip3 install --no-cache-dir pyzmq msgpack

COPY gear_sonic_deploy/docker/quest_relay/relay.py /relay.py
COPY gear_sonic_deploy/docker/quest_relay/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 10000 = ROS TCP Connector (Quest Unity app connects here)
# 5559  = ZMQ relay output (host quest_manager connects here)
EXPOSE 10000
EXPOSE 5559

ENTRYPOINT ["/entrypoint.sh"]
