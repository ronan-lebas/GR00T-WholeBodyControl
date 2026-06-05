"""
Meta Quest + BrainCo hand teleoperation server.

Reads head and wrist poses from Quest ROS2 topics, retargets MANO-21 finger landmarks
to BrainCo 6-DOF motor commands, and streams 3-point VR teleoperation data to the
C++ WBC policy via ZMQ — the same wire format as pico_manager_thread_server.py.

Controller buttons are replaced by local keyboard input.

Usage:
    python quest_manager_thread_server.py

Keyboard controls:
    s  - Start policy + enter VR_3PT mode (calibrates on first press)
    r  - Recalibrate VR 3-pt tracking (operator should be in rest pose)
    f  - Toggle finger retargeting on/off
    c  - Toggle data collection
    x  - Toggle data abort
    q  - Stop policy and exit
"""

# Defer annotation evaluation so ROS message types used only in type hints
# (e.g. PoseStamped, TFMessage) don't need to be importable in ZMQ-relay mode,
# which runs without rclpy installed.
from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from enum import Enum, IntEnum
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as sRot

# Gear-sonic ZMQ utilities
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# Brainco pure-numpy retargeter (no dex_retargeting dependency)
_BRAINCO_PKG = Path(__file__).resolve().parents[2] / "third_party" / "brainco-retargeting" / "brainco_retargeting"
sys.path.insert(0, str(_BRAINCO_PKG))
import np_retargeting  # noqa: E402 (path manipulation above)

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    _HAS_ROS = True
except ImportError:
    print("Warning: rclpy not available. QuestReader will not receive live data.")
    _HAS_ROS = False
    Node = object

try:
    from vr_haptic_msgs.msg import ManoLandmarks
    _HAS_MANO_MSG = True
except ImportError:
    print("Warning: vr_haptic_msgs not available. Hand landmark topics will not be subscribed.")
    _HAS_MANO_MSG = False
    ManoLandmarks = None

# Robot model for FK-based calibration
try:
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
    _HAS_ROBOT_MODEL = True
except ImportError:
    print("Warning: Robot model not available. Calibration will use fixed fallback offsets.")
    instantiate_g1_robot_model = None
    get_g1_key_frame_poses = None
    _HAS_ROBOT_MODEL = False

# msgpack — used by both FeedbackReader and the ZMQ relay reader
try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    print("Warning: msgpack not available. ZMQ relay mode and feedback recalibration disabled.")
    _HAS_MSGPACK = False

# ZMQ feedback reader (for recalibration with measured robot joints)
try:
    from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
    _HAS_ZMQ_POLLER = True
except ImportError:
    print("Warning: ZMQPoller not available. Feedback-based recalibration disabled.")
    _HAS_ZMQ_POLLER = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIRE_HAND_DOF = 7  # always 7 on the wire; BrainCo uses 6 + 1 padding

# On resume from pause, ease from the frozen pose to the live operator pose over
# this many seconds (instead of snapping) to avoid an abrupt jump.
RESUME_RAMP_SEC = 1.0

# Joint max limits (radians) matching np_retargeting._JOINT_LIMITS
_JOINT_LIMITS = {
    "thumb_metacarpal": 1.5184,
    "thumb_proximal": 1.0472,
    "index_proximal": 1.4661,
    "middle_proximal": 1.4661,
    "ring_proximal": 1.4661,
    "pinky_proximal": 1.4661,
}
_JOINT_ORDER = [
    "thumb_metacarpal",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
]


class StreamMode(Enum):
    OFF = 0
    PLANNER_VR_3PT = 5


class LocomotionMode(IntEnum):
    IDLE = 0


# ---------------------------------------------------------------------------
# Keyboard helpers (same pattern as mock_quest_streamer.py)
# ---------------------------------------------------------------------------


def _is_data() -> bool:
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


# ---------------------------------------------------------------------------
# Hand retargeting
# ---------------------------------------------------------------------------


def retarget_hand(landmarks: np.ndarray, side: str) -> list[float]:
    """MANO-21 landmarks (21, 3) → 7-element BrainCo wire list.

    MANO and MediaPipe-21 share the same joint ordering, so np_retargeting
    works directly on Quest hand_pose data.
    Returns 6 normalized [0, 1] values + 1 padding zero.
    """
    angles = np_retargeting.retarget(landmarks, side)
    normalized = [
        float(np.clip(angles[f"{side}_{k}_joint"] / _JOINT_LIMITS[k], 0.0, 1.0))
        for k in _JOINT_ORDER
    ]
    return normalized + [0.0]


# ---------------------------------------------------------------------------
# QuestReader — background ROS2 subscriber thread
# ---------------------------------------------------------------------------

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
_ZERO_POS = np.zeros(3)
_ZERO_LANDMARKS = np.zeros((21, 3))


class QuestReader:
    """
    Subscribes to Quest data and maintains a thread-safe latest-data dict.

    Two modes:
      rclpy mode  (default): subscribes directly to ROS2 topics. Requires rclpy.
      ZMQ relay mode: connects to the quest-relay Docker container ZMQ PUB socket.
                      Set zmq_relay_host to enable. No ROS installation needed.

    Published Quest data is in ROS FLU convention (X-forward, Y-left, Z-up);
    no axis swapping is applied.

    _latest keys:
        head_pos            np.ndarray (3,)  — position in vr_origin frame
        head_quat           np.ndarray (4,)  — quaternion [w,x,y,z] scalar-first
        left_wrist_pos      np.ndarray (3,)
        left_wrist_quat     np.ndarray (4,)
        right_wrist_pos     np.ndarray (3,)
        right_wrist_quat    np.ndarray (4,)
        left_landmarks      np.ndarray (21,3)  — MANO joints
        right_landmarks     np.ndarray (21,3)
        timestamp           float  — time.time() of last update
    """

    def __init__(
        self,
        head_topic: str = "/quest/pose/headset",
        left_hand_topic: str = "/quest/hand_pose/left",
        right_hand_topic: str = "/quest/hand_pose/right",
        zmq_relay_host: str | None = None,
        zmq_relay_port: int = 5559,
    ):
        self._lock = threading.Lock()
        self._latest: dict = {
            "head_pos": _ZERO_POS.copy(),
            "head_quat": _IDENTITY_QUAT_WXYZ.copy(),
            "left_wrist_pos": _ZERO_POS.copy(),
            "left_wrist_quat": _IDENTITY_QUAT_WXYZ.copy(),
            "right_wrist_pos": _ZERO_POS.copy(),
            "right_wrist_quat": _IDENTITY_QUAT_WXYZ.copy(),
            "left_landmarks": _ZERO_LANDMARKS.copy(),
            "right_landmarks": _ZERO_LANDMARKS.copy(),
            "timestamp": 0.0,
        }

        self._head_topic = head_topic
        self._left_hand_topic = left_hand_topic
        self._right_hand_topic = right_hand_topic
        self._zmq_relay_host = zmq_relay_host
        self._zmq_relay_port = zmq_relay_port
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def _use_zmq_relay(self) -> bool:
        return self._zmq_relay_host is not None

    def start(self) -> None:
        if self._use_zmq_relay:
            if not _HAS_MSGPACK:
                print("[QuestReader] msgpack not available — cannot use ZMQ relay mode.")
                return
            self._thread = threading.Thread(target=self._spin_zmq, daemon=True)
            self._thread.start()
            print(f"[QuestReader] ZMQ relay mode: connecting to {self._zmq_relay_host}:{self._zmq_relay_port}")
        else:
            if not _HAS_ROS:
                print("[QuestReader] rclpy not available — reader will not receive live data.")
                return
            rclpy.init(args=None)
            self._node = _QuestNode(
                head_topic=self._head_topic,
                left_hand_topic=self._left_hand_topic,
                right_hand_topic=self._right_hand_topic,
                on_update=self._on_update,
            )
            self._thread = threading.Thread(target=self._spin_rclpy, daemon=True)
            self._thread.start()
            print("[QuestReader] ROS2 node started.")

    def _spin_rclpy(self) -> None:
        try:
            rclpy.spin(self._node)
        except Exception as e:
            print(f"[QuestReader] spin error: {e}")

    def _spin_zmq(self) -> None:
        """Receive quest_data messages from the relay container and update _latest."""
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{self._zmq_relay_host}:{self._zmq_relay_port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"quest_data")
        sub.setsockopt(zmq.RCVTIMEO, 100)  # 100 ms receive timeout
        print(f"[QuestReader] Connected to ZMQ relay at {self._zmq_relay_host}:{self._zmq_relay_port}")
        while not self._stop_event.is_set():
            try:
                parts = sub.recv_multipart()
                if len(parts) < 2:
                    continue
                data = msgpack.unpackb(parts[1], raw=False)
                self._on_update(_relay_dict_to_numpy(data))
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[QuestReader] ZMQ relay error: {e}")
        sub.close()
        ctx.term()

    def stop(self) -> None:
        self._stop_event.set()
        if not self._use_zmq_relay and _HAS_ROS:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _on_update(self, data: dict) -> None:
        with self._lock:
            self._latest.update(data)
            self._latest["timestamp"] = time.time()

    def get_latest(self) -> dict:
        with self._lock:
            return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._latest.items()}

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._latest["timestamp"] > 0.0


class _QuestNode(Node):
    """Internal rclpy Node — callbacks update the reader via on_update."""

    def __init__(self, head_topic, left_hand_topic, right_hand_topic, on_update):
        super().__init__("quest_manager_reader")
        self._on_update = on_update

        self.create_subscription(PoseStamped, head_topic, self._on_headset, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 10)
        if _HAS_MANO_MSG:
            self.create_subscription(ManoLandmarks, left_hand_topic, self._on_left_hand, 10)
            self.create_subscription(ManoLandmarks, right_hand_topic, self._on_right_hand, 10)
        else:
            print("[QuestReader] vr_haptic_msgs unavailable — finger tracking disabled.")

    def _on_headset(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        self._on_update({
            "head_pos": np.array([p.x, p.y, p.z]),
            "head_quat": np.array([o.w, o.x, o.y, o.z]),  # scalar-first
        })

    def _on_tf(self, msg: TFMessage) -> None:
        update = {}
        for t in msg.transforms:
            child = t.child_frame_id
            if child not in ("hand_left", "hand_right"):
                continue
            p = t.transform.translation
            o = t.transform.rotation
            pos = np.array([p.x, p.y, p.z])
            # TF quaternion is [x,y,z,w]; convert to scalar-first [w,x,y,z]
            quat = np.array([o.w, o.x, o.y, o.z])
            if np.allclose(quat, 0.0):
                quat = _IDENTITY_QUAT_WXYZ.copy()
            if child == "hand_left":
                update["left_wrist_pos"] = pos
                update["left_wrist_quat"] = quat
            else:
                update["right_wrist_pos"] = pos
                update["right_wrist_quat"] = quat
        if update:
            self._on_update(update)

    def _on_left_hand(self, msg) -> None:
        self._on_update({"left_landmarks": _landmarks_to_np(msg)})

    def _on_right_hand(self, msg) -> None:
        self._on_update({"right_landmarks": _landmarks_to_np(msg)})


def _landmarks_to_np(msg) -> np.ndarray:
    """Convert ManoLandmarks message to (21, 3) float64 array."""
    pts = msg.landmarks
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)


def _relay_dict_to_numpy(data: dict) -> dict:
    """Convert relay msgpack dict (Python lists) to numpy arrays for QuestReader._latest."""
    return {
        "head_pos": np.array(data["head_pos"], dtype=np.float64),
        "head_quat": np.array(data["head_quat"], dtype=np.float64),
        "left_wrist_pos": np.array(data["left_wrist_pos"], dtype=np.float64),
        "left_wrist_quat": np.array(data["left_wrist_quat"], dtype=np.float64),
        "right_wrist_pos": np.array(data["right_wrist_pos"], dtype=np.float64),
        "right_wrist_quat": np.array(data["right_wrist_quat"], dtype=np.float64),
        "left_landmarks": np.array(data["left_landmarks"], dtype=np.float64),
        "right_landmarks": np.array(data["right_landmarks"], dtype=np.float64),
        "timestamp": float(data.get("timestamp", time.time())),
    }


# ---------------------------------------------------------------------------
# QuestThreePointPose — calibration + 3-pt pose extraction
# ---------------------------------------------------------------------------

# Same kinematic chain constants as pico_manager_thread_server.py:ThreePointPose
_TORSO_LINK_OFFSET_Z = 0.05
_NECK_LINK_LENGTH = 0.35


class QuestThreePointPose:
    """
    Converts Quest wrist + head poses into the calibrated 3-point VR pose format
    expected by the C++ WBC planner.

    Output shape: (3, 7) — rows are [L-wrist, R-wrist, Head].
    Each row: [x, y, z, qw, qx, qy, qz] in robot frame, scalar-first quaternion.

    Calibration aligns Quest vr_origin coordinates to robot root frame using the
    same offset approach as ThreePointPose in pico_manager_thread_server.py:
      1. Capture head quaternion → inv(head_quat) rotates everything to head-neutral
      2. Compute wrist position/orientation offsets vs G1 FK reference at rest pose
    """

    def __init__(self):
        self._calib_head_quat_inv: np.ndarray | None = None  # scalar-first [w,x,y,z]
        self._calib_lwrist_pos_offset: np.ndarray | None = None
        self._calib_rwrist_pos_offset: np.ndarray | None = None
        self._calib_lwrist_rot_offset: sRot | None = None
        self._calib_rwrist_rot_offset: sRot | None = None

        self._override_robot_q: np.ndarray | None = None  # measured joints for recalibration FK

        self._robot_model = None
        if _HAS_ROBOT_MODEL and instantiate_g1_robot_model is not None:
            try:
                self._robot_model = instantiate_g1_robot_model()
                print("[QuestThreePointPose] Robot model loaded for FK calibration.")
            except Exception as e:
                print(f"[QuestThreePointPose] Warning: could not load robot model: {e}")

    @property
    def is_calibrated(self) -> bool:
        return self._calib_head_quat_inv is not None

    def calibrate_now(self, latest: dict) -> bool:
        """Capture calibration using current Quest frame.

        Operator should be in rest / T-pose when calling this.
        Returns True on success.
        """
        try:
            self._capture_calibration(latest)
            print("[QuestThreePointPose] Calibration captured.")
            return True
        except Exception as e:
            print(f"[QuestThreePointPose] Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reset(self) -> None:
        """Clear all calibration state."""
        self._calib_head_quat_inv = None
        self._calib_lwrist_pos_offset = None
        self._calib_rwrist_pos_offset = None
        self._calib_lwrist_rot_offset = None
        self._calib_rwrist_rot_offset = None
        self._override_robot_q = None
        print("[QuestThreePointPose] Calibration reset.")

    def reset_with_measured_q(self, body_q: np.ndarray) -> None:
        """Schedule recalibration using measured robot joints (29 DOFs) as FK reference.

        Next calibrate_now() call will compute wrist offsets against FK of those joints
        rather than the default rest pose — avoids jumps when re-entering VR_3PT mode.
        """
        self._override_robot_q = body_q.copy()
        print("[QuestThreePointPose] Next calibration will use measured robot joints as FK reference.")

    def _capture_calibration(self, latest: dict) -> None:
        head_quat = latest["head_quat"].copy()  # [w,x,y,z] scalar-first
        lw_pos = latest["left_wrist_pos"].copy()
        rw_pos = latest["right_wrist_pos"].copy()
        lw_quat = latest["left_wrist_quat"].copy()
        rw_quat = latest["right_wrist_quat"].copy()

        head_rot = sRot.from_quat(head_quat, scalar_first=True)
        head_rot_inv = head_rot.inv()
        self._calib_head_quat_inv = head_rot_inv.as_quat(scalar_first=True)

        # Rotate wrist positions/orientations into head-neutral frame
        lw_pos_corr = head_rot_inv.apply(lw_pos)
        rw_pos_corr = head_rot_inv.apply(rw_pos)
        lw_rot_corr = head_rot_inv * sRot.from_quat(lw_quat, scalar_first=True)
        rw_rot_corr = head_rot_inv * sRot.from_quat(rw_quat, scalar_first=True)

        # G1 FK reference positions — use override joints if set (measured robot pose)
        if self._robot_model is not None and get_g1_key_frame_poses is not None:
            fk_q = None
            if self._override_robot_q is not None:
                fk_q = self._robot_model.get_configuration_from_actuated_joints(
                    body_actuated_joint_values=self._override_robot_q[:29]
                )
                self._override_robot_q = None  # consumed
            g1_poses = get_g1_key_frame_poses(self._robot_model, q=fk_q)
            g1_lw_pos = g1_poses["left_wrist"]["position"]
            g1_rw_pos = g1_poses["right_wrist"]["position"]
            g1_lw_rot = sRot.from_quat(g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True)
            g1_rw_rot = sRot.from_quat(g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True)
        else:
            # Fallback: approximate G1 rest wrist positions in robot frame
            g1_lw_pos = np.array([0.25, 0.20, 0.15])
            g1_rw_pos = np.array([0.25, -0.20, 0.15])
            g1_lw_rot = sRot.from_quat([1, 0, 0, 0], scalar_first=True)
            g1_rw_rot = sRot.from_quat([1, 0, 0, 0], scalar_first=True)

        self._calib_lwrist_pos_offset = lw_pos_corr - g1_lw_pos
        self._calib_rwrist_pos_offset = rw_pos_corr - g1_rw_pos
        self._calib_lwrist_rot_offset = g1_lw_rot * lw_rot_corr.inv()
        self._calib_rwrist_rot_offset = g1_rw_rot * rw_rot_corr.inv()

        print(
            f"[QuestThreePointPose] Calibration captured:\n"
            f"  L-wrist pos offset: {self._calib_lwrist_pos_offset}\n"
            f"  R-wrist pos offset: {self._calib_rwrist_pos_offset}"
        )

    def process_quest_pose(self, latest: dict) -> np.ndarray:
        """Return calibrated 3-pt pose: (3, 7) array [L-wrist, R-wrist, Head].

        Each row: [x, y, z, qw, qx, qy, qz] in robot frame, scalar-first quat.
        If not yet calibrated, returns a safe default (arms forward at chest height).
        """
        if not self.is_calibrated:
            return _default_vr3pt_pose()

        return self._apply_calibration(latest)

    def _apply_calibration(self, latest: dict) -> np.ndarray:
        head_quat = latest["head_quat"]
        lw_pos = latest["left_wrist_pos"]
        rw_pos = latest["right_wrist_pos"]
        lw_quat = latest["left_wrist_quat"]
        rw_quat = latest["right_wrist_quat"]

        calib_inv = sRot.from_quat(self._calib_head_quat_inv, scalar_first=True)

        # L-wrist position: rotate into head-neutral, subtract offset
        lw_pos_calib = calib_inv.apply(lw_pos) - self._calib_lwrist_pos_offset
        rw_pos_calib = calib_inv.apply(rw_pos) - self._calib_rwrist_pos_offset

        # L-wrist orientation: rot_offset * (head_inv * current)
        lw_rot_calib = self._calib_lwrist_rot_offset * (
            calib_inv * sRot.from_quat(lw_quat, scalar_first=True)
        )
        rw_rot_calib = self._calib_rwrist_rot_offset * (
            calib_inv * sRot.from_quat(rw_quat, scalar_first=True)
        )

        # Head orientation: calib_inv * current head orientation
        head_rot_calib = calib_inv * sRot.from_quat(head_quat, scalar_first=True)

        # Head position via kinematic chain (same as pico's ThreePointPose)
        head_z = head_rot_calib.apply([0, 0, 1])
        head_pos_calib = np.array([0, 0, _TORSO_LINK_OFFSET_Z]) + _NECK_LINK_LENGTH * head_z

        result = np.zeros((3, 7), dtype=np.float32)
        # Row 0: L-wrist
        result[0, :3] = lw_pos_calib
        result[0, 3:] = lw_rot_calib.as_quat(scalar_first=True)
        # Row 1: R-wrist
        result[1, :3] = rw_pos_calib
        result[1, 3:] = rw_rot_calib.as_quat(scalar_first=True)
        # Row 2: Head
        result[2, :3] = head_pos_calib
        result[2, 3:] = head_rot_calib.as_quat(scalar_first=True)

        return result


def _default_vr3pt_pose() -> np.ndarray:
    """Safe default pose when not yet calibrated."""
    result = np.zeros((3, 7), dtype=np.float32)
    result[0] = [0.3,  0.2, 0.3, 1.0, 0.0, 0.0, 0.0]  # L-wrist
    result[1] = [0.3, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]  # R-wrist
    result[2] = [0.0,  0.0, 0.4, 1.0, 0.0, 0.0, 0.0]  # Head
    return result


def _nlerp_quat(qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized lerp between scalar-first [w,x,y,z] quats (good enough for short ramps)."""
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb  # take the shorter arc
    q = (1.0 - alpha) * qa + alpha * qb
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else qa.copy()


def _blend_vr3pt_pose(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two (3, 7) poses: linear on positions, nlerp on quaternions."""
    out = np.zeros((3, 7), dtype=np.float32)
    out[:, :3] = (1.0 - alpha) * pose_a[:, :3] + alpha * pose_b[:, :3]
    for i in range(3):
        out[i, 3:] = _nlerp_quat(pose_a[i, 3:], pose_b[i, 3:], alpha)
    return out


# ===========================================================================
# === BEGIN TEMP DEBUG STUB — synthetic Quest motion (delete this block) =====
# Generates fake head/wrist/finger motion so the teleop pipeline can be tested
# without a live Quest streaming real data. Remove this block, the helpers
# below, and the reader.get_latest() wrap in run_quest_manager() to revert.
# ===========================================================================
def _debug_make_landmarks(curl: float) -> np.ndarray:
    """Synthetic MANO-21 landmarks; curl in [0, 1] flexes all fingers (0=open, 1=fist)."""
    seg = 0.035  # segment length (meters); only relative geometry matters
    lm = np.zeros((21, 3), dtype=np.float64)
    lm[0] = [0.0, 0.0, 0.0]  # wrist
    beta = curl * 1.4  # per-joint bend angle (rad)
    # index(5), middle(9), ring(13), pinky(17): MCP, PIP, DIP, TIP chains curling in -z
    for mcp, x in {5: 0.03, 9: 0.01, 13: -0.01, 17: -0.03}.items():
        p_mcp = np.array([x, 0.06, 0.0])
        p_pip = p_mcp + seg * np.array([0.0, 1.0, 0.0])
        p_dip = p_pip + seg * np.array([0.0, np.cos(beta), -np.sin(beta)])
        p_tip = p_dip + seg * np.array([0.0, np.cos(2 * beta), -np.sin(2 * beta)])
        lm[mcp], lm[mcp + 1], lm[mcp + 2], lm[mcp + 3] = p_mcp, p_pip, p_dip, p_tip
    # thumb: CMC(1), MCP(2), IP(3), TIP(4) splayed out then curling toward palm
    d = np.array([0.6, 0.6, 0.0])
    d = d / np.linalg.norm(d)
    tb = curl * 0.9

    def _rotz(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

    t_cmc = np.array([0.03, 0.02, 0.0])
    t_mcp = t_cmc + seg * d
    t_ip = t_mcp + seg * _rotz(d, tb)
    t_tip = t_ip + seg * _rotz(d, 2 * tb)
    lm[1], lm[2], lm[3], lm[4] = t_cmc, t_mcp, t_ip, t_tip
    return lm


def _debug_rewrite_quest_data(latest: dict) -> dict:
    """Overwrite head/wrist/landmark fields with smooth synthetic motion."""
    t = time.time()
    # Head: gentle yaw oscillation, fixed position.
    latest["head_pos"] = np.array([0.0, 0.0, 0.0])
    latest["head_quat"] = sRot.from_euler("z", 0.35 * np.sin(0.4 * t)).as_quat(scalar_first=True)
    # Wrists: oscillate position (relative motion drives the arms after calibration)
    # plus a slow pitch so orientation tracking is visible.
    latest["left_wrist_pos"] = np.array(
        [0.30 + 0.08 * np.sin(0.5 * t), 0.20, 0.20 + 0.08 * np.sin(0.7 * t)]
    )
    latest["right_wrist_pos"] = np.array(
        [0.30 + 0.08 * np.sin(0.5 * t + np.pi), -0.20, 0.20 + 0.08 * np.cos(0.7 * t)]
    )
    latest["left_wrist_quat"] = sRot.from_euler("y", 0.3 * np.sin(0.8 * t)).as_quat(scalar_first=True)
    latest["right_wrist_quat"] = sRot.from_euler("y", -0.3 * np.sin(0.8 * t)).as_quat(scalar_first=True)
    # Fingers: open/close all fingers together.
    curl = 0.5 * (1.0 + np.sin(1.5 * t))
    latest["left_landmarks"] = _debug_make_landmarks(curl)
    latest["right_landmarks"] = _debug_make_landmarks(curl)
    return latest
# ===========================================================================
# === END TEMP DEBUG STUB ===================================================
# ===========================================================================


# ---------------------------------------------------------------------------
# Feedback reader (for recalibration with measured robot joints)
# ---------------------------------------------------------------------------

class FeedbackReader:
    """Reads g1_debug ZMQ feedback. Used to recalibrate with measured robot joints."""

    def __init__(self, zmq_host: str = "localhost", zmq_port: int = 5557):
        self._available = _HAS_ZMQ_POLLER and _HAS_MSGPACK
        if self._available:
            self._poller = ZMQPoller(host=zmq_host, port=zmq_port, topic="g1_debug")
        self.full_body_q: np.ndarray | None = None

    def poll(self) -> bool:
        """Poll once. Returns True if fresh data was received."""
        if not self._available:
            return False
        data = self._poller.get_data()
        if data is None:
            print("[FeedbackReader] No feedback data received.")
            return False
        unpacked = msgpack.unpackb(data, raw=False)
        if "body_q_measured" in unpacked:
            self.full_body_q = np.array(unpacked["body_q_measured"], dtype=np.float64)
            return True
        print("[FeedbackReader] body_q_measured not in feedback data.")
        return False


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------

def run_quest_manager(
    port: int = 5556,
    head_topic: str = "/quest/pose/headset",
    left_hand_topic: str = "/quest/hand_pose/left",
    right_hand_topic: str = "/quest/hand_pose/right",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    zmq_relay_host: str | None = None,
    zmq_relay_port: int = 5559,
) -> None:
    """Start the Quest teleoperation manager.

    When zmq_relay_host is set, reads Quest data from the relay Docker container
    over ZMQ instead of subscribing to ROS2 topics directly (no rclpy needed).
    """
    # --- ZMQ setup ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f"[Manager] ZMQ PUB socket bound to port {port}")

    # --- Quest reader ---
    reader = QuestReader(
        head_topic=head_topic,
        left_hand_topic=left_hand_topic,
        right_hand_topic=right_hand_topic,
        zmq_relay_host=zmq_relay_host,
        zmq_relay_port=zmq_relay_port,
    )
    reader.start()

    # === BEGIN TEMP DEBUG STUB — synthetic Quest motion (delete these lines) ===
    # Wrap get_latest so calibration AND streaming both see the fake motion.
    _real_get_latest = reader.get_latest
    reader.get_latest = lambda: _debug_rewrite_quest_data(_real_get_latest())
    print("[Manager] *** DEBUG STUB ACTIVE — streaming synthetic Quest motion ***")
    # === END TEMP DEBUG STUB ===================================================

    # --- 3-pt pose processor ---
    three_point = QuestThreePointPose()

    # --- Feedback reader (for measured-joint recalibration) ---
    feedback = FeedbackReader(zmq_host=zmq_feedback_host, zmq_port=zmq_feedback_port)

    # --- Wait for subscriber ---
    print("[Manager] Waiting 2 s for C++ subscriber to connect...")
    time.sleep(2.0)

    # --- Keyboard setup ---
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print("\nKeyboard controls:")
    print("  s - Start policy + enter VR_3PT mode")
    print("  r - Recalibrate VR 3-pt tracking (stand in rest pose first)")
    print("  f - Toggle finger retargeting on/off")
    print("  p - Pause/resume teleoperation (freeze robot; move freely; resume w/o jump)")
    print("  c - Toggle data collection")
    print("  x - Toggle data abort")
    print("  q - Stop policy and exit")
    print()

    current_mode = StreamMode.OFF
    finger_tracking = False

    # Pause/freeze state: while paused, the robot holds the last commanded pose
    # and live operator motion is ignored. On resume, the robot eases back to the
    # live operator pose over RESUME_RAMP_SEC (original calibration, absolute tracking).
    paused = False
    last_vr_3pt_pose = None
    last_left_hand = None
    last_right_hand = None
    ramping = False
    ramp_start = 0.0
    ramp_from_pose = None

    # Edge-triggered data collection flags
    toggle_dc = False
    toggle_da = False

    try:
        while True:
            # --- Keyboard ---
            toggle_dc = False
            toggle_da = False
            if _is_data():
                c = sys.stdin.read(1)
                if c == "s" and current_mode == StreamMode.OFF:
                    print("[Manager] 's' pressed: starting policy, entering VR_3PT mode...")
                    latest = reader.get_latest()
                    if not three_point.is_calibrated:
                        print("[Manager] Running initial calibration (stand in rest pose)...")
                        three_point.calibrate_now(latest)
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                    current_mode = StreamMode.PLANNER_VR_3PT
                    print(f"[Manager] Mode: {current_mode.name}")
                elif c == "r":
                    print("[Manager] 'r' pressed: recalibrating (stand in rest pose)...")
                    latest = reader.get_latest()
                    three_point.reset()
                    # Use measured robot joints for FK reference if feedback is available
                    if feedback.poll() and feedback.full_body_q is not None:
                        three_point.reset_with_measured_q(feedback.full_body_q)
                    three_point.calibrate_now(latest)
                elif c == "f":
                    finger_tracking = not finger_tracking
                    print(f"[Manager] Finger retargeting {'ENABLED' if finger_tracking else 'DISABLED'}")
                elif c == "p" and current_mode == StreamMode.PLANNER_VR_3PT:
                    paused = not paused
                    if paused:
                        print(
                            "[Manager] PAUSED — robot frozen at last pose. "
                            "Move freely; press 'p' to resume."
                        )
                    else:
                        # Resume: ease back to the live operator pose (absolute tracking,
                        # original calibration) over RESUME_RAMP_SEC to avoid a hard jump.
                        if last_vr_3pt_pose is not None:
                            ramping = True
                            ramp_start = time.time()
                            ramp_from_pose = last_vr_3pt_pose.copy()
                            print(
                                f"[Manager] RESUMED — easing to live pose over {RESUME_RAMP_SEC:.1f}s..."
                            )
                        else:
                            print("[Manager] RESUMED teleoperation.")
                elif c == "c":
                    toggle_dc = True
                    print("[Manager] 'c' pressed: toggling data collection")
                elif c == "x":
                    toggle_da = True
                    print("[Manager] 'x' pressed: toggling data abort")
                elif c == "q":
                    print("[Manager] 'q' pressed: stopping policy and exiting...")
                    break

            # --- Get Quest data ---
            latest = reader.get_latest()

            if current_mode == StreamMode.PLANNER_VR_3PT:
                if paused:
                    # Hold the last commanded pose/hands; ignore live operator motion.
                    if last_vr_3pt_pose is None:
                        last_vr_3pt_pose = three_point.process_quest_pose(latest)
                    vr_3pt_pose = last_vr_3pt_pose
                    left_hand = last_left_hand
                    right_hand = last_right_hand
                else:
                    # Live operator pose (absolute, original calibration).
                    live_pose = three_point.process_quest_pose(latest)

                    # Finger retargeting
                    left_hand = None
                    right_hand = None
                    if finger_tracking:
                        try:
                            left_hand = retarget_hand(latest["left_landmarks"], "left")
                        except Exception:
                            left_hand = None
                        try:
                            right_hand = retarget_hand(latest["right_landmarks"], "right")
                        except Exception:
                            right_hand = None

                    # On resume, ease from the frozen pose toward the live pose.
                    if ramping:
                        alpha = (time.time() - ramp_start) / RESUME_RAMP_SEC
                        if alpha >= 1.0:
                            ramping = False
                            vr_3pt_pose = live_pose
                        else:
                            vr_3pt_pose = _blend_vr3pt_pose(ramp_from_pose, live_pose, alpha)
                    else:
                        vr_3pt_pose = live_pose

                    # Remember latest so 'p' can freeze on these values.
                    last_vr_3pt_pose = vr_3pt_pose
                    last_left_hand = left_hand
                    last_right_hand = right_hand

                vr_3pt_pos = vr_3pt_pose[:, :3].flatten().tolist()
                vr_3pt_quat = vr_3pt_pose[:, 3:].flatten().tolist()

                # Send manager_state
                socket.send(
                    pack_pose_message(
                        {
                            "stream_mode": np.array([current_mode.value], dtype=np.int32),
                            "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                            "toggle_data_abort": np.array([toggle_da], dtype=bool),
                        },
                        topic="manager_state",
                    )
                )

                # Send planner
                socket.send(
                    build_planner_message(
                        mode=LocomotionMode.IDLE.value,
                        movement=[0.0, 0.0, 0.0],
                        facing=[1.0, 0.0, 0.0],
                        speed=-1.0,
                        height=-1.0,
                        upper_body_position=None,
                        left_hand_position=left_hand,
                        right_hand_position=right_hand,
                        vr_3pt_position=vr_3pt_pos,
                        vr_3pt_orientation=vr_3pt_quat,
                        vr_3pt_compliance=None,
                    )
                )

            time.sleep(0.02)  # 50 Hz

    except KeyboardInterrupt:
        print("\n[Manager] KeyboardInterrupt — stopping...")
    finally:
        print("[Manager] Sending STOP command...")
        try:
            socket.send(build_command_message(start=False, stop=True, planner=True))
            time.sleep(0.1)
        except Exception:
            pass
        reader.stop()
        socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("[Manager] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meta Quest + BrainCo hand teleoperation server."
    )
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (default: 5556)")
    parser.add_argument(
        "--head-topic",
        type=str,
        default="/quest/pose/headset",
        help="ROS2 topic for headset pose (default: /quest/pose/headset)",
    )
    parser.add_argument(
        "--left-hand-topic",
        type=str,
        default="/quest/hand_pose/left",
        help="ROS2 topic for left MANO landmarks (default: /quest/hand_pose/left)",
    )
    parser.add_argument(
        "--right-hand-topic",
        type=str,
        default="/quest/hand_pose/right",
        help="ROS2 topic for right MANO landmarks (default: /quest/hand_pose/right)",
    )
    parser.add_argument(
        "--zmq-feedback-host",
        type=str,
        default="localhost",
        help="ZMQ feedback host for robot state (default: localhost)",
    )
    parser.add_argument(
        "--zmq-feedback-port",
        type=int,
        default=5557,
        help="ZMQ feedback port for robot state (default: 5557)",
    )
    parser.add_argument(
        "--zmq-relay-host",
        type=str,
        default=None,
        help=(
            "Host of the quest-relay Docker container (e.g. localhost). "
            "When set, reads Quest data over ZMQ instead of subscribing to ROS2 topics directly. "
            "No rclpy installation needed. Start the relay with: "
            "docker run -p 10000:10000 -p 5559:5559 quest-relay"
        ),
    )
    parser.add_argument(
        "--zmq-relay-port",
        type=int,
        default=5559,
        help="ZMQ port of the quest-relay container (default: 5559)",
    )
    args = parser.parse_args()

    run_quest_manager(
        port=args.port,
        head_topic=args.head_topic,
        left_hand_topic=args.left_hand_topic,
        right_hand_topic=args.right_hand_topic,
        zmq_feedback_host=args.zmq_feedback_host,
        zmq_feedback_port=args.zmq_feedback_port,
        zmq_relay_host=args.zmq_relay_host,
        zmq_relay_port=args.zmq_relay_port,
    )
