"""
Quest ROS2 → ZMQ relay.

Subscribes to Meta Quest ROS2 topics published via ROS TCP Connector and
republishes the data as a single msgpack blob over a ZMQ PUB socket.

The host-side QuestReader (in quest_manager_thread_server.py) connects as
a ZMQ SUB to consume this data without needing any ROS installation.

ZMQ message format: two-part multipart
  part[0] = b"quest_data"
  part[1] = msgpack.packb({
      "head_pos":          [x, y, z],        # float64, FLU frame
      "head_quat":         [w, x, y, z],     # float64, scalar-first
      "left_wrist_pos":    [x, y, z],        # float64, FLU frame
      "left_wrist_quat":   [w, x, y, z],     # float64, scalar-first
      "right_wrist_pos":   [x, y, z],        # float64
      "right_wrist_quat":  [w, x, y, z],     # float64, scalar-first
      "left_landmarks":    [[x,y,z], ...],   # float64, 21 MANO joints
      "right_landmarks":   [[x,y,z], ...],   # float64, 21 MANO joints
      "left_tracked":      bool,
      "right_tracked":     bool,
      "timestamp":         float,            # time.time()
  })

Subscribed topics (all configurable via CLI):
  /quest/pose/headset        geometry_msgs/PoseStamped      head pose
  /tf                        tf2_msgs/TFMessage             wrist poses
  /quest/hand_pose/left      vr_haptic_msgs/ManoLandmarks   left hand (21 MANO joints)
  /quest/hand_pose/right     vr_haptic_msgs/ManoLandmarks   right hand (21 MANO joints)
"""

import argparse
import threading
import time

import msgpack
import rclpy
import zmq
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from vr_haptic_msgs.msg import ManoLandmarks

_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]
_ZERO_POS = [0.0, 0.0, 0.0]
_ZERO_LANDMARKS = [[0.0, 0.0, 0.0]] * 21


class QuestRelayNode(Node):
    """ROS2 node that subscribes to Quest topics and exposes a thread-safe snapshot."""

    def __init__(self, head_topic, left_hand_topic, right_hand_topic):
        super().__init__("quest_relay")

        self._lock = threading.Lock()
        self._state = {
            "head_pos": list(_ZERO_POS),
            "head_quat": list(_IDENTITY_QUAT),
            "left_wrist_pos": list(_ZERO_POS),
            "left_wrist_quat": list(_IDENTITY_QUAT),
            "right_wrist_pos": list(_ZERO_POS),
            "right_wrist_quat": list(_IDENTITY_QUAT),
            "left_landmarks": list(_ZERO_LANDMARKS),
            "right_landmarks": list(_ZERO_LANDMARKS),
            "left_tracked": False,
            "right_tracked": False,
            "timestamp": 0.0,
        }

        self.create_subscription(PoseStamped, head_topic, self._on_headset, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 10)
        self.create_subscription(ManoLandmarks, left_hand_topic, self._on_left_hand, 10)
        self.create_subscription(ManoLandmarks, right_hand_topic, self._on_right_hand, 10)

        self.get_logger().info(
            f"Subscribed to: {head_topic}, /tf, {left_hand_topic}, {right_hand_topic}"
        )

    def _on_headset(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        with self._lock:
            self._state["head_pos"] = [p.x, p.y, p.z]
            # geometry_msgs quaternion is scalar-last [x,y,z,w]; convert to scalar-first [w,x,y,z]
            self._state["head_quat"] = [o.w, o.x, o.y, o.z]
            self._state["timestamp"] = time.time()

    def _on_tf(self, msg: TFMessage) -> None:
        updates = {}
        for t in msg.transforms:
            child = t.child_frame_id
            if child not in ("hand_left", "hand_right"):
                continue
            p = t.transform.translation
            o = t.transform.rotation
            pos = [p.x, p.y, p.z]
            if abs(o.x) + abs(o.y) + abs(o.z) + abs(o.w) < 1e-6:
                quat = list(_IDENTITY_QUAT)  # undetected hand → identity
            else:
                quat = [o.w, o.x, o.y, o.z]  # scalar-first
            if child == "hand_left":
                updates["left_wrist_pos"] = pos
                updates["left_wrist_quat"] = quat
            else:
                updates["right_wrist_pos"] = pos
                updates["right_wrist_quat"] = quat
        if updates:
            with self._lock:
                self._state.update(updates)
                self._state["timestamp"] = time.time()

    def _on_left_hand(self, msg: ManoLandmarks) -> None:
        landmarks = _mano_to_list(msg)
        with self._lock:
            self._state["left_landmarks"] = landmarks
            self._state["left_tracked"] = len(msg.landmarks) >= 21
            self._state["timestamp"] = time.time()

    def _on_right_hand(self, msg: ManoLandmarks) -> None:
        landmarks = _mano_to_list(msg)
        with self._lock:
            self._state["right_landmarks"] = landmarks
            self._state["right_tracked"] = len(msg.landmarks) >= 21
            self._state["timestamp"] = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


def _mano_to_list(msg: ManoLandmarks) -> list:
    """Convert ManoLandmarks to list of 21 [x,y,z] triples, padded if needed."""
    pts = msg.landmarks
    n = min(len(pts), 21)
    result = [[float(p.x), float(p.y), float(p.z)] for p in pts[:n]]
    result += [[0.0, 0.0, 0.0]] * (21 - n)
    return result


def main():
    parser = argparse.ArgumentParser(description="Quest ROS2 → ZMQ relay")
    parser.add_argument("--zmq-port", type=int, default=5559, help="ZMQ PUB port (default: 5559)")
    parser.add_argument(
        "--head-topic",
        default="/quest/pose/headset",
        help="ROS2 head pose topic (default: /quest/pose/headset)",
    )
    parser.add_argument(
        "--left-hand-topic",
        default="/quest/hand_pose/left",
        help="ROS2 left ManoLandmarks topic (default: /quest/hand_pose/left)",
    )
    parser.add_argument(
        "--right-hand-topic",
        default="/quest/hand_pose/right",
        help="ROS2 right ManoLandmarks topic (default: /quest/hand_pose/right)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=90.0,
        help="ZMQ publish rate in Hz (default: 90, matches Quest frame rate)",
    )
    args = parser.parse_args()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://0.0.0.0:{args.zmq_port}")
    print(f"[relay] ZMQ PUB bound to port {args.zmq_port}")

    rclpy.init(args=None)
    node = QuestRelayNode(
        head_topic=args.head_topic,
        left_hand_topic=args.left_hand_topic,
        right_hand_topic=args.right_hand_topic,
    )

    def _spin() -> None:
        # Swallow the RCLError raised when main() shuts down the context on exit.
        try:
            rclpy.spin(node)
        except Exception:
            pass

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()
    print("[relay] ROS2 spinning. Waiting for Quest data...")

    period = 1.0 / max(1.0, args.hz)
    topic_bytes = b"quest_data"
    last_log = time.time()
    published = 0

    try:
        while True:
            t_start = time.time()

            snap = node.snapshot()
            pub.send_multipart([topic_bytes, msgpack.packb(snap, use_bin_type=True)])
            published += 1

            now = time.time()
            if now - last_log >= 5.0:
                print(
                    f"[relay] {published / (now - last_log):.1f} Hz | "
                    f"head={'ok' if snap['timestamp'] > 0.0 else 'waiting'} | "
                    f"L={'ok' if snap['left_tracked'] else '-'} "
                    f"R={'ok' if snap['right_tracked'] else '-'}"
                )
                last_log = now
                published = 0

            elapsed = time.time() - t_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\n[relay] Shutting down...")
    finally:
        rclpy.shutdown()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
