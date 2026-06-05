#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Start ros_tcp_endpoint in background (bridges Quest Unity TCP → ROS2 topics)
ros2 launch ros_tcp_endpoint endpoint.py &
ROS_ENDPOINT_PID=$!

# Give the endpoint a moment to initialize before the relay starts subscribing
sleep 2

echo "[quest-relay] ros_tcp_endpoint started (PID $ROS_ENDPOINT_PID)"
echo "[quest-relay] Starting ZMQ relay..."

# -u = unbuffered stdout/stderr so relay logs stream live under `docker run`
# (a pipe, not a TTY) instead of being flushed all at once on exit.
exec python3 -u /relay.py "$@"
