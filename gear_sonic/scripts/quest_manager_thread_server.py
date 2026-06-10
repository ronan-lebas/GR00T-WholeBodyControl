"""
Meta Quest + BrainCo hand teleoperation server.

Reads head and wrist poses from Quest ROS2 topics, retargets MANO-21 finger landmarks
to BrainCo 6-DOF motor commands, and streams 3-point VR teleoperation data to the
C++ WBC policy via ZMQ — the same wire format as pico_manager_thread_server.py.

Controller buttons are replaced by local keyboard input.

Usage:
    python quest_manager_thread_server.py

Keyboard controls:
    s  - Start policy + enter VR_3PT mode (live: calibrates after --calib-delay-sec
         so the operator can assume the rest pose; replay: calibrates on frame 0,
         which is the prepended rest pose)
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

# Brainco retargeters — pure-numpy and optimization-based (dex_retargeting)
_BRAINCO_PKG  = Path(__file__).resolve().parents[2] / "third_party" / "brainco-retargeting" / "brainco_retargeting"
_BRAINCO_ROOT = _BRAINCO_PKG.parent  # exposes the brainco_retargeting package itself
sys.path.insert(0, str(_BRAINCO_PKG))   # for bare `import np_retargeting`
sys.path.insert(0, str(_BRAINCO_ROOT))  # for `from brainco_retargeting import ...`
import np_retargeting  # noqa: E402

try:
    from brainco_retargeting import BrainCoRetargeter
    from brainco_retargeting._utils import mp21_to_xr25
    _HAS_DEX_RETARGETER = True
except ImportError:
    print("Warning: BrainCoRetargeter (dex) not available — np_retargeting will be used.")
    _HAS_DEX_RETARGETER = False

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
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import (
        G1_KEY_FRAME_OFFSETS,
        get_g1_key_frame_poses,
    )
    _HAS_ROBOT_MODEL = True
except ImportError:
    print("Warning: Robot model not available. Calibration will use fixed fallback offsets.")
    instantiate_g1_robot_model = None
    get_g1_key_frame_poses = None
    G1_KEY_FRAME_OFFSETS = None
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

# Kinematic chain constants for the head row (same as pico ThreePointPose):
# root -> torso_link (+Z) -> head (along head's local Z).
_TORSO_LINK_OFFSET_Z = 0.05
_NECK_LINK_LENGTH = 0.35

# VR 3-point key-frame offsets, in each wrist LINK's local frame (must match
# G1_KEY_FRAME_OFFSETS and the deploy VR_3POINT_OFFSETS). The policy's wrist target
# is link_pos + R_link @ offset, so the offset must be re-applied with the *live*
# commanded orientation — otherwise a wrist rotation (palm flip) makes the offset
# arm swing and the link translates instead of rotating in place.
if G1_KEY_FRAME_OFFSETS is not None:
    _WRIST_OFFSET = {
        "left": np.asarray(G1_KEY_FRAME_OFFSETS["left_wrist"], dtype=float),
        "right": np.asarray(G1_KEY_FRAME_OFFSETS["right_wrist"], dtype=float),
    }
else:
    _WRIST_OFFSET = {
        "left": np.array([0.18, -0.025, 0.0]),
        "right": np.array([0.18, 0.025, 0.0]),
    }

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


def _read_key() -> str | None:
    """Read one keypress (including arrow-key escape sequences).

    Returns a single-character string for normal keys, or one of the strings
    "left" / "right" / "up" / "down" for arrow keys.  Returns None when there
    is no input available.

    Arrow keys send a 3-byte escape sequence (\x1b [ D/C/A/B).  After reading
    the leading \x1b we wait up to 50 ms for the rest to arrive — the zero-
    timeout select used by _is_data() is too tight and misses the trailing bytes.
    """
    if not _is_data():
        return None
    c = sys.stdin.read(1)
    if c == "\x1b":
        if select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
            c2 = sys.stdin.read(1)
            if c2 == "[" and select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
                c3 = sys.stdin.read(1)
                return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(c3, "\x1b")
    return c


# ---------------------------------------------------------------------------
# Hand retargeting
# ---------------------------------------------------------------------------


def correct_landmark_frame(landmarks: np.ndarray) -> np.ndarray:
    """Transform Quest MANO landmarks into the wrist TF frame.

    Quest hand-tracking publishes landmarks in a frame that is rotated +90 deg
    around world +Z relative to the wrist TF frame used by the controller.
    Applying Rz(-90 deg) to every landmark position realigns the hand skeleton
    with the wrist pose.

    Args:
        landmarks: (..., 3) landmark positions in Quest world frame.

    Returns:
        (..., 3) corrected positions in wrist-TF-compatible world frame.
    """
    R = sRot.from_euler("z", -np.pi / 2)
    return R.apply(landmarks.reshape(-1, 3)).reshape(landmarks.shape)


def retarget_hand(landmarks: np.ndarray, side: str, retargeter=None) -> list[float]:
    """MANO-21 landmarks (21, 3) → 7-element BrainCo wire list.

    Two backends (selected by the caller via `retargeter`):
      retargeter=None  — pure-numpy (np_retargeting), no extra dependencies.
      retargeter=BrainCoRetargeter instance — optimization-based (dex_retargeting):
          MANO-21 → XR-25 → canonicalize → dex solver → normalized [0, 1].

    Both paths return 6 normalized [0, 1] values + 1 padding zero.
    """
    if retargeter is not None:
        xr25 = retargeter.canonicalize(mp21_to_xr25(landmarks), side)
        fn = retargeter.retarget_left if side == "left" else retargeter.retarget_right
        normalized = list(fn(xr25))
        return normalized + [0.0]

    # pure-numpy fallback
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


# Robot head kinematic point in root frame (root -> torso_link -> head along +Z),
# matching QuestThreePointPose's head row. Used to express the FK rest wrist
# positions as head-relative vectors for the synthetic operator rest pose.
_ROBOT_HEAD_REF = np.array([0.0, 0.0, _TORSO_LINK_OFFSET_Z + _NECK_LINK_LENGTH])

# Fallback synthetic rest geometry (head-relative, head-yaw frame) if the robot
# model / FK is unavailable. Approximates the G1 rest wrist LINK: arms forward,
# palms in (key-frame minus the +0.18 m offset).
_FALLBACK_REST_VEC = {
    "left": np.array([0.38, 0.16, -0.30]) - np.array([0.18, -0.025, 0.0]),
    "right": np.array([0.38, -0.16, -0.30]) - np.array([0.18, 0.025, 0.0]),
}

_FK_REST_CACHE: dict | None = None


def _g1_fk_rest() -> dict | None:
    """Robot FK rest wrist LINK poses (cached, offset removed). None if unavailable.

    Uses apply_offset=False so the synthetic rest operator wrist is placed at the
    robot wrist LINK (matching the link-anchored calibration); the key-frame offset
    is re-applied at runtime with the live orientation.
    """
    global _FK_REST_CACHE
    if _FK_REST_CACHE is not None:
        return _FK_REST_CACHE
    if not _HAS_ROBOT_MODEL or instantiate_g1_robot_model is None:
        return None
    try:
        _FK_REST_CACHE = get_g1_key_frame_poses(instantiate_g1_robot_model(), apply_offset=False)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[prepend_rest_pose] Could not load FK rest poses: {e}")
        _FK_REST_CACHE = None
    return _FK_REST_CACHE


def prepend_rest_pose(
    data: dict,
    timestamps: np.ndarray,
    hold_sec: float,
    interp_sec: float,
) -> tuple[dict, np.ndarray]:
    """Prepend a synthetic rest pose (held, then interpolated) to a recording.

    Recorded Quest trajectories rarely start in a clean rest pose, so calibrating
    on frame 0 of the raw data gives a garbage reference. This builds a neutral
    rest frame from the recording's first head pose (upright, yaw preserved, arms
    relaxed), holds it for `hold_sec`, then interpolates to the original first
    frame over `interp_sec`. After this, frame 0 is a valid calibration pose and
    the robot eases from rest into the recorded motion.

    Args:
        data: dict of (n, ...) arrays — head/wrist pos+quat and landmarks.
        timestamps: (n,) recording timestamps.
        hold_sec: seconds to hold the rest pose (>= one frame so frame 0 is rest).
        interp_sec: seconds to interpolate rest -> original first frame.

    Returns:
        (extended_data, extended_timestamps) with the prefix prepended.
    """
    n = timestamps.shape[0]
    if n == 0 or (hold_sec <= 0.0 and interp_sec <= 0.0):
        return data, timestamps

    dt = float(np.median(np.diff(timestamps))) if n > 1 else 1.0 / 90.0
    n_hold = max(1, int(round(hold_sec / dt)))
    n_interp = max(0, int(round(interp_sec / dt)))

    # --- build the synthetic rest frame ---
    # The synthetic operator rest pose is the G1 FK rest pose lifted into the Quest
    # world frame via the operator's head yaw. Matching the robot rest exactly makes
    # the orientation map M = O_cal^-1 . R_rest = identity, so the robot mirrors the
    # operator's wrist orientation with no offset (and the position vector maps 1:1).
    head_pos0 = data["head_pos"][0]
    yaw0 = _quat_yaw(data["head_quat"][0])
    r0 = _yaw_rot(yaw0)
    head_quat_rest = r0.as_quat(scalar_first=True)  # upright, operator's heading

    fk = _g1_fk_rest()
    rest = {
        "head_pos": head_pos0.copy(),
        "head_quat": head_quat_rest,
        # Hold the first-frame fingers so finger retargeting does not jump.
        "left_landmarks": data["left_landmarks"][0].copy(),
        "right_landmarks": data["right_landmarks"][0].copy(),
    }
    for side, pos_key, quat_key in (
        ("left", "left_wrist_pos", "left_wrist_quat"),
        ("right", "right_wrist_pos", "right_wrist_quat"),
    ):
        if fk is not None:
            rest_vec = fk[f"{side}_wrist"]["position"] - _ROBOT_HEAD_REF
            r_rest = sRot.from_quat(fk[f"{side}_wrist"]["orientation_wxyz"], scalar_first=True)
        else:
            rest_vec = _FALLBACK_REST_VEC[side]
            r_rest = sRot.identity()
        rest[pos_key] = head_pos0 + r0.apply(rest_vec)
        # World wrist orientation W_cal = R0 . R_rest, so O_cal = R0^-1 . W_cal = R_rest.
        rest[quat_key] = (r0 * r_rest).as_quat(scalar_first=True)
    first = {k: data[k][0] for k in data}

    pos_keys = ("head_pos", "left_wrist_pos", "right_wrist_pos")
    quat_keys = ("head_quat", "left_wrist_quat", "right_wrist_quat")
    lm_keys = ("left_landmarks", "right_landmarks")

    pre = {k: [] for k in data}
    # Hold the rest pose (frame 0 is guaranteed to be the rest pose).
    for _ in range(n_hold):
        for k in data:
            pre[k].append(rest[k].copy())
    # Interpolate rest -> original first frame (exclusive of both endpoints).
    for i in range(1, n_interp + 1):
        alpha = i / (n_interp + 1)
        for k in pos_keys:
            pre[k].append((1.0 - alpha) * rest[k] + alpha * first[k])
        for k in quat_keys:
            pre[k].append(_nlerp_quat(rest[k], first[k], alpha))
        for k in lm_keys:
            pre[k].append((1.0 - alpha) * rest[k] + alpha * first[k])

    n_pre = n_hold + n_interp
    ext = {k: np.concatenate([np.asarray(pre[k]), data[k]], axis=0) for k in data}
    # Continuous, evenly-spaced timestamps for the prefix, ending one dt before t0.
    pre_ts = timestamps[0] - dt * np.arange(n_pre, 0, -1)
    ext_ts = np.concatenate([pre_ts, timestamps])
    print(
        f"[NpzReplayReader] Prepended rest pose: {n_hold} hold + {n_interp} interp "
        f"frames ({n_pre * dt:.1f}s @ {1.0 / dt:.0f} Hz)."
    )
    return ext, ext_ts


class NpzReplayReader:
    """Drop-in replacement for QuestReader that replays a recorded NPZ trajectory.

    Exposes the same duck-typed interface (start / stop / get_latest / has_data)
    so it can be passed directly to run_quest_manager without any other changes.

    Playback is time-anchored to the original recording timestamps so the replay
    runs at the same speed as the original capture.  When the end is reached the
    recording loops back to the beginning automatically.

    Extra methods for interactive control (called from the keyboard handler):
        toggle_pause() — pause / resume playback
        step(delta)    — move delta frames while paused (can be negative)
    """

    def __init__(
        self,
        npz_path: str,
        rest_hold_sec: float = 1.0,
        rest_interp_sec: float = 1.5,
    ) -> None:
        raw = np.load(npz_path, allow_pickle=True)
        self._data = {
            "head_pos":         raw["head_pos"].astype(np.float64),
            "head_quat":        raw["head_quat"].astype(np.float64),
            "left_wrist_pos":   raw["left_wrist_pos"].astype(np.float64),
            "left_wrist_quat":  raw["left_wrist_quat"].astype(np.float64),
            "right_wrist_pos":  raw["right_wrist_pos"].astype(np.float64),
            "right_wrist_quat": raw["right_wrist_quat"].astype(np.float64),
            "left_landmarks":   raw["left_landmarks"].astype(np.float64),
            "right_landmarks":  raw["right_landmarks"].astype(np.float64),
        }
        self._timestamps = raw["timestamp"].astype(np.float64)
        # Prepend a synthetic rest pose so calibration on frame 0 is valid.
        self._data, self._timestamps = prepend_rest_pose(
            self._data, self._timestamps, rest_hold_sec, rest_interp_sec
        )
        self._n = int(self._timestamps.shape[0])
        duration = float(self._timestamps[-1] - self._timestamps[0])
        self._frame_idx: int = 0
        self._paused: bool = False
        self._started: bool = False
        self._t0_wall: float = 0.0   # wall-clock time at last start/resume
        self._t0_data: float = 0.0   # recording timestamp at last start/resume
        print(f"[NpzReplayReader] Loaded {self._n} frames, {duration:.1f} s  ←  {npz_path}")

    # ---- QuestReader interface ----

    def start(self) -> None:
        self._t0_wall = time.time()
        self._t0_data = self._timestamps[0]
        self._paused = True  # start paused; press 's' to begin playback
        self._started = True
        print("[NpzReplayReader] Loaded — paused at frame 0  |  s=start  k=pause/resume  ←/→=step")

    def stop(self) -> None:
        pass  # nothing to tear down

    def get_latest(self) -> dict:
        if self._started and not self._paused:
            target = self._t0_data + (time.time() - self._t0_wall)
            if target >= self._timestamps[-1]:
                # loop: restart from the beginning
                self._t0_wall = time.time()
                self._t0_data = self._timestamps[0]
                self._frame_idx = 0
            else:
                idx = int(np.searchsorted(self._timestamps, target, side="right")) - 1
                self._frame_idx = max(0, min(idx, self._n - 1))
        return self._frame_at(self._frame_idx)

    @property
    def has_data(self) -> bool:
        return self._started

    # ---- replay control ----

    def toggle_pause(self) -> None:
        if self._paused:
            # re-anchor wall clock so playback continues from the current frame
            self._t0_wall = time.time()
            self._t0_data = self._timestamps[self._frame_idx]
            self._paused = False
            print(f"[NpzReplayReader] Resumed  (frame {self._frame_idx}/{self._n - 1})")
        else:
            self._paused = True
            print(f"[NpzReplayReader] Paused   (frame {self._frame_idx}/{self._n - 1})")

    def step(self, delta: int) -> None:
        """Move delta frames. Works during both playback and pause.

        Automatically pauses the clock on the first arrow-key press so the
        clock does not fight the step direction.  Press 'k' to resume
        clock-based playback from the current frame.
        """
        if not self._paused:
            self._paused = True
            print("[NpzReplayReader] Arrow key — paused for manual frame stepping  (k=resume)")
        self._frame_idx = max(0, min(self._frame_idx + delta, self._n - 1))
        print(f"[NpzReplayReader] Step {delta:+d}  →  frame {self._frame_idx}/{self._n - 1}")

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ---- internal ----

    def _frame_at(self, idx: int) -> dict:
        return {
            "head_pos":         self._data["head_pos"][idx].copy(),
            "head_quat":        self._data["head_quat"][idx].copy(),
            "left_wrist_pos":   self._data["left_wrist_pos"][idx].copy(),
            "left_wrist_quat":  self._data["left_wrist_quat"][idx].copy(),
            "right_wrist_pos":  self._data["right_wrist_pos"][idx].copy(),
            "right_wrist_quat": self._data["right_wrist_quat"][idx].copy(),
            "left_landmarks":   self._data["left_landmarks"][idx].copy(),
            "right_landmarks":  self._data["right_landmarks"][idx].copy(),
            "timestamp":        float(self._timestamps[idx]),
        }


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

def _quat_yaw(quat_wxyz: np.ndarray) -> float:
    """Yaw (rotation about world +Z, radians) of a scalar-first quaternion.

    Uses the forward-axis (local +X) projection onto the XY plane, which stays
    well-defined under moderate head pitch/roll — unlike an Euler decomposition,
    which is ambiguous near gimbal lock. The Quest head frame's local +X points
    forward (ROS FLU), so this returns the operator's heading.
    """
    fwd = sRot.from_quat(quat_wxyz, scalar_first=True).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(fwd[1], fwd[0]))


def _yaw_rot(yaw: float) -> sRot:
    """Rotation about world +Z by `yaw` radians."""
    return sRot.from_euler("z", yaw)


class QuestThreePointPose:
    """
    Converts Quest wrist + head poses into the calibrated 3-point VR pose format
    expected by the C++ WBC planner.

    Output shape: (3, 7) — rows are [L-wrist, R-wrist, Head].
    Each row: [x, y, z, qw, qx, qy, qz] in robot frame, scalar-first quaternion.

    Calibration aligns Quest vr_origin coordinates to the robot root frame. The
    operator should be in a neutral rest pose when calibrating. We capture, in a
    yaw-only head frame (R0 = Rz(head_yaw_cal)):

      Position — the head->wrist *vector* (not the absolute wrist position):
        v_cal = R0^-1 . (p_wrist_cal - p_head_cal)
      At runtime the wrist LINK tracks the *change* of that vector since
      calibration, scaled by `pos_scale` and added to the robot rest LINK position:
        link(t) = link_rest + pos_scale * [ R0^-1 . (p_wrist(t) - p_head(t)) - v_cal ]
      The value SENT is the policy's key-frame point, link + R(t) @ offset, with the
      +0.18 m offset re-applied using the live orientation R(t). This matters: the
      policy reconstructs link = sent_pos - R(t) @ offset, so freezing the offset at
      the rest orientation would make a palm flip swing the offset arm and translate
      the link (hands drift together) instead of rotating in place.
      Using the head->wrist vector makes tracking invariant to the operator
      walking / leaning / turning (head and wrist translate together), which a
      previous absolute-position mapping got wrong.

      Orientation — a WORLD-frame delta. With O(t) = R0^-1 . W_wrist(t) the
      operator wrist in the calibration-yaw frame and M = O_cal^-1 . R_rest:
        R_robot(t) = O(t) . M
      so the robot wrist mirrors the operator's world-frame rotation since
      calibration, anchored at the robot rest orientation. (A body-frame delta
      M . O(t) rotates about the wrong axis.)

      Head — yaw only. The robot has no neck joint, so the head row drives the
      torso heading. We pass the operator's relative head yaw and zero pitch/roll.
    """

    def __init__(self, pos_scale: float = 1.0):
        # Scale applied to the operator head->wrist displacement to account for the
        # robot vs. operator arm-length / form-factor difference. 1.0 = 1:1 mapping;
        # <1.0 shrinks the commanded workspace so a smaller robot does not saturate.
        # Orientation is dimensionless and therefore unaffected by this factor.
        self._pos_scale = float(pos_scale)

        # Calibration-yaw frame R0 = Rz(head_yaw_cal), stored as its inverse rotation.
        self._calib_yaw_inv: sRot | None = None
        # Head->wrist vectors at calibration (in the R0 frame).
        self._calib_lwrist_vec: np.ndarray | None = None
        self._calib_rwrist_vec: np.ndarray | None = None
        # Robot rest wrist positions the calibration vectors map to.
        self._calib_lwrist_rest: np.ndarray | None = None
        self._calib_rwrist_rest: np.ndarray | None = None
        # Orientation maps M = O_cal^-1 . R_rest, applied at runtime as R(t) = O(t) . M.
        self._calib_lwrist_rot_map: sRot | None = None
        self._calib_rwrist_rot_map: sRot | None = None

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
        return self._calib_yaw_inv is not None

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
        self._calib_yaw_inv = None
        self._calib_lwrist_vec = None
        self._calib_rwrist_vec = None
        self._calib_lwrist_rest = None
        self._calib_rwrist_rest = None
        self._calib_lwrist_rot_map = None
        self._calib_rwrist_rot_map = None
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
        head_pos = latest["head_pos"].copy()
        head_quat = latest["head_quat"].copy()  # [w,x,y,z] scalar-first
        lw_pos = latest["left_wrist_pos"].copy()
        rw_pos = latest["right_wrist_pos"].copy()
        lw_quat = latest["left_wrist_quat"].copy()
        rw_quat = latest["right_wrist_quat"].copy()

        # Calibration-yaw frame: R0 = Rz(head_yaw). We only use the head *yaw* so a
        # tilted/nodding head at calibration does not skew the wrist mapping.
        r0_inv = _yaw_rot(_quat_yaw(head_quat)).inv()
        self._calib_yaw_inv = r0_inv

        # Head->wrist vectors at calibration, expressed in the R0 frame.
        lw_vec = r0_inv.apply(lw_pos - head_pos)
        rw_vec = r0_inv.apply(rw_pos - head_pos)
        # Operator wrist orientation in the R0 frame (O_cal).
        lw_o_cal = r0_inv * sRot.from_quat(lw_quat, scalar_first=True)
        rw_o_cal = r0_inv * sRot.from_quat(rw_quat, scalar_first=True)

        # G1 FK reference LINK poses (apply_offset=False) — use override joints if set.
        # We anchor to the wrist LINK, not the key-frame point: the +0.18 m key-frame
        # offset is re-applied at runtime with the live orientation (see _apply_calibration).
        if self._robot_model is not None and get_g1_key_frame_poses is not None:
            fk_q = None
            if self._override_robot_q is not None:
                fk_q = self._robot_model.get_configuration_from_actuated_joints(
                    body_actuated_joint_values=self._override_robot_q[:29]
                )
                self._override_robot_q = None  # consumed
            g1_poses = get_g1_key_frame_poses(self._robot_model, q=fk_q, apply_offset=False)
            g1_lw_pos = g1_poses["left_wrist"]["position"]
            g1_rw_pos = g1_poses["right_wrist"]["position"]
            g1_lw_rot = sRot.from_quat(g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True)
            g1_rw_rot = sRot.from_quat(g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True)
        else:
            # Fallback: approximate G1 rest wrist LINK positions (key frame minus offset).
            g1_lw_pos = np.array([0.38, 0.16, 0.10]) - _WRIST_OFFSET["left"]
            g1_rw_pos = np.array([0.38, -0.16, 0.10]) - _WRIST_OFFSET["right"]
            g1_lw_rot = sRot.from_quat([1, 0, 0, 0], scalar_first=True)
            g1_rw_rot = sRot.from_quat([1, 0, 0, 0], scalar_first=True)

        # Position: store the calibration head->wrist vector and the robot rest pose
        # it maps to. Runtime tracks the change in the vector scaled by pos_scale.
        self._calib_lwrist_vec = lw_vec
        self._calib_rwrist_vec = rw_vec
        self._calib_lwrist_rest = g1_lw_pos
        self._calib_rwrist_rest = g1_rw_pos
        # Orientation: M = O_cal^-1 . R_rest, so R(t) = O(t) . M maps O_cal -> R_rest.
        self._calib_lwrist_rot_map = lw_o_cal.inv() * g1_lw_rot
        self._calib_rwrist_rot_map = rw_o_cal.inv() * g1_rw_rot

        print(
            f"[QuestThreePointPose] Calibration captured (pos_scale={self._pos_scale}):\n"
            f"  L head->wrist {np.round(lw_vec, 3)} -> rest {np.round(g1_lw_pos, 3)}\n"
            f"  R head->wrist {np.round(rw_vec, 3)} -> rest {np.round(g1_rw_pos, 3)}"
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
        head_pos = latest["head_pos"]
        head_quat = latest["head_quat"]
        lw_pos = latest["left_wrist_pos"]
        rw_pos = latest["right_wrist_pos"]
        lw_quat = latest["left_wrist_quat"]
        rw_quat = latest["right_wrist_quat"]

        r0_inv = self._calib_yaw_inv
        s = self._pos_scale

        # Orientation: world-frame delta — R(t) = O(t) . M, with O(t) the operator
        # wrist in the calibration-yaw frame and M = O_cal^-1 . R_rest.
        lw_rot_calib = (r0_inv * sRot.from_quat(lw_quat, scalar_first=True)) * self._calib_lwrist_rot_map
        rw_rot_calib = (r0_inv * sRot.from_quat(rw_quat, scalar_first=True)) * self._calib_rwrist_rot_map

        # Position of the wrist LINK: rest + scaled change of the head->wrist vector
        # since calibration (vector in the calibration-yaw frame). Subtracting the head
        # position makes this invariant to the operator translating.
        lw_vec = r0_inv.apply(lw_pos - head_pos)
        rw_vec = r0_inv.apply(rw_pos - head_pos)
        lw_link = self._calib_lwrist_rest + s * (lw_vec - self._calib_lwrist_vec)
        rw_link = self._calib_rwrist_rest + s * (rw_vec - self._calib_rwrist_vec)

        # Key-frame target = link + R(t) @ offset. Re-applying the offset with the live
        # orientation means a wrist rotation rotates the key frame in place (the link
        # stays put) instead of swinging the offset arm and translating the link — which
        # is exactly how the policy reconstructs the link from (position, orientation).
        lw_pos_calib = lw_link + lw_rot_calib.apply(_WRIST_OFFSET["left"])
        rw_pos_calib = rw_link + rw_rot_calib.apply(_WRIST_OFFSET["right"])

        # Head: yaw only. The G1 has no neck joint, so the head row drives the torso
        # heading — we pass the operator's relative head yaw and discard pitch/roll.
        head_rel = r0_inv * sRot.from_quat(head_quat, scalar_first=True)
        head_rot_calib = _yaw_rot(_quat_yaw(head_rel.as_quat(scalar_first=True)))

        # Head position via kinematic chain (same as pico's ThreePointPose). With a
        # yaw-only head this stays directly above the torso.
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
# def _debug_make_landmarks(curl: float) -> np.ndarray:
#     """Synthetic MANO-21 landmarks; curl in [0, 1] flexes all fingers (0=open, 1=fist)."""
#     seg = 0.035  # segment length (meters); only relative geometry matters
#     lm = np.zeros((21, 3), dtype=np.float64)
#     lm[0] = [0.0, 0.0, 0.0]  # wrist
#     beta = curl * 1.4  # per-joint bend angle (rad)
#     # index(5), middle(9), ring(13), pinky(17): MCP, PIP, DIP, TIP chains curling in -z
#     for mcp, x in {5: 0.03, 9: 0.01, 13: -0.01, 17: -0.03}.items():
#         p_mcp = np.array([x, 0.06, 0.0])
#         p_pip = p_mcp + seg * np.array([0.0, 1.0, 0.0])
#         p_dip = p_pip + seg * np.array([0.0, np.cos(beta), -np.sin(beta)])
#         p_tip = p_dip + seg * np.array([0.0, np.cos(2 * beta), -np.sin(2 * beta)])
#         lm[mcp], lm[mcp + 1], lm[mcp + 2], lm[mcp + 3] = p_mcp, p_pip, p_dip, p_tip
#     # thumb: CMC(1), MCP(2), IP(3), TIP(4) splayed out then curling toward palm
#     d = np.array([0.6, 0.6, 0.0])
#     d = d / np.linalg.norm(d)
#     tb = curl * 0.9

#     def _rotz(v, a):
#         c, s = np.cos(a), np.sin(a)
#         return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

#     t_cmc = np.array([0.03, 0.02, 0.0])
#     t_mcp = t_cmc + seg * d
#     t_ip = t_mcp + seg * _rotz(d, tb)
#     t_tip = t_ip + seg * _rotz(d, 2 * tb)
#     lm[1], lm[2], lm[3], lm[4] = t_cmc, t_mcp, t_ip, t_tip
#     return lm


# def _debug_rewrite_quest_data(latest: dict) -> dict:
#     """Overwrite head/wrist/landmark fields with smooth synthetic motion."""
#     t = time.time()
#     # Head: gentle yaw oscillation, fixed position.
#     latest["head_pos"] = np.array([0.0, 0.0, 0.0])
#     latest["head_quat"] = sRot.from_euler("z", 0.35 * np.sin(0.4 * t)).as_quat(scalar_first=True)
#     # Wrists: oscillate position (relative motion drives the arms after calibration)
#     # plus a slow pitch so orientation tracking is visible.
#     latest["left_wrist_pos"] = np.array(
#         [0.30 + 0.08 * np.sin(0.5 * t), 0.20, 0.20 + 0.08 * np.sin(0.7 * t)]
#     )
#     latest["right_wrist_pos"] = np.array(
#         [0.30 + 0.08 * np.sin(0.5 * t + np.pi), -0.20, 0.20 + 0.08 * np.cos(0.7 * t)]
#     )
#     latest["left_wrist_quat"] = sRot.from_euler("y", 0.3 * np.sin(0.8 * t)).as_quat(scalar_first=True)
#     latest["right_wrist_quat"] = sRot.from_euler("y", -0.3 * np.sin(0.8 * t)).as_quat(scalar_first=True)
#     # Fingers: open/close all fingers together.
#     curl = 0.5 * (1.0 + np.sin(1.5 * t))
#     latest["left_landmarks"] = _debug_make_landmarks(curl)
#     latest["right_landmarks"] = _debug_make_landmarks(curl)
#     return latest
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
    replay_npz: str | None = None,
    np_retarget: bool = False,
    pos_scale: float = 1.0,
    calib_delay_sec: float = 3.0,
    rest_hold_sec: float = 1.0,
    rest_interp_sec: float = 1.5,
) -> None:
    """Start the Quest teleoperation manager.

    When replay_npz is set, a recorded NPZ trajectory is used as the Quest data
    source instead of subscribing to ROS2 topics or a ZMQ relay.  All downstream
    processing (calibration, 3-pt pose, finger retargeting, ZMQ streaming) runs
    unchanged — the recording replaces the human operator. A synthetic rest pose
    is prepended to the recording (rest_hold_sec + rest_interp_sec) so calibration
    on frame 0 is valid.

    When zmq_relay_host is set (and replay_npz is None), reads Quest data from the
    relay Docker container over ZMQ instead of subscribing to ROS2 topics directly.

    For live operation, pressing 's' starts a calib_delay_sec countdown before
    capturing calibration, giving the operator time to assume the rest pose.
    """
    # --- ZMQ setup ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f"[Manager] ZMQ PUB socket bound to port {port}")

    # --- Quest reader (live or replay) ---
    if replay_npz is not None:
        reader: QuestReader | NpzReplayReader = NpzReplayReader(
            replay_npz, rest_hold_sec=rest_hold_sec, rest_interp_sec=rest_interp_sec
        )
    else:
        reader = QuestReader(
            head_topic=head_topic,
            left_hand_topic=left_hand_topic,
            right_hand_topic=right_hand_topic,
            zmq_relay_host=zmq_relay_host,
            zmq_relay_port=zmq_relay_port,
        )
    reader.start()

    # === BEGIN TEMP DEBUG STUB — synthetic Quest motion (delete these lines) ===
    # # Wrap get_latest so calibration AND streaming both see the fake motion.
    # # Skipped in replay mode because the NPZ data is already the motion source.
    # if replay_npz is None:
    #     _real_get_latest = reader.get_latest
    #     reader.get_latest = lambda: _debug_rewrite_quest_data(_real_get_latest())
    #     print("[Manager] *** DEBUG STUB ACTIVE — streaming synthetic Quest motion ***")
    # === END TEMP DEBUG STUB ===================================================

    # --- 3-pt pose processor ---
    three_point = QuestThreePointPose(pos_scale=pos_scale)

    # --- Feedback reader (for measured-joint recalibration) ---
    feedback = FeedbackReader(zmq_host=zmq_feedback_host, zmq_port=zmq_feedback_port)

    # --- Finger retargeter ---
    if np_retarget:
        hand_retargeter = None
        print("[Manager] Finger retargeting: np_retargeting (pure numpy)")
    elif _HAS_DEX_RETARGETER:
        hand_retargeter = BrainCoRetargeter()
        print("[Manager] Finger retargeting: BrainCoRetargeter (dex, optimization-based)")
    else:
        hand_retargeter = None
        print("[Manager] Finger retargeting: BrainCoRetargeter unavailable, falling back to np_retargeting")

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
    if replay_npz is not None:
        print("  k - Pause/resume replay")
        print("  ← / →  - Step back / forward 1 frame")
    print()

    current_mode = StreamMode.OFF
    finger_tracking = True
    is_replay = replay_npz is not None

    # Deferred calibration: on live Quest, 's'/'r' arm a countdown so the operator
    # can assume the rest pose before the frame is captured. Replay calibrates
    # immediately (frame 0 is the prepended rest pose).
    calib_pending = False
    calib_deadline = 0.0
    calib_next_tick = 0.0
    calib_start_policy = False  # also send the start command when the timer fires

    def _capture_calibration_now(use_measured: bool) -> None:
        latest_c = reader.get_latest()
        three_point.reset()
        if use_measured and feedback.poll() and feedback.full_body_q is not None:
            three_point.reset_with_measured_q(feedback.full_body_q)
        three_point.calibrate_now(latest_c)

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
            key = _read_key()
            if key is not None:
                if key == "s" and current_mode == StreamMode.OFF and not calib_pending:
                    if is_replay or calib_delay_sec <= 0.0:
                        # Replay frame 0 is the prepended rest pose — calibrate now.
                        print("[Manager] 's' pressed: starting policy, entering VR_3PT mode...")
                        if not three_point.is_calibrated:
                            three_point.calibrate_now(reader.get_latest())
                        socket.send(build_command_message(start=True, stop=False, planner=True))
                        current_mode = StreamMode.PLANNER_VR_3PT
                        print(f"[Manager] Mode: {current_mode.name}")
                        if isinstance(reader, NpzReplayReader) and reader.is_paused:
                            reader.toggle_pause()
                    else:
                        calib_pending = True
                        calib_start_policy = True
                        calib_deadline = time.time() + calib_delay_sec
                        calib_next_tick = int(np.ceil(calib_delay_sec))
                        print(
                            f"[Manager] 's' pressed: calibrating in {calib_delay_sec:.0f}s — "
                            "assume the rest pose and hold still..."
                        )
                elif key == "r" and not calib_pending:
                    if is_replay or calib_delay_sec <= 0.0:
                        print("[Manager] 'r' pressed: recalibrating (rest pose)...")
                        _capture_calibration_now(use_measured=True)
                    else:
                        calib_pending = True
                        calib_start_policy = False
                        calib_deadline = time.time() + calib_delay_sec
                        calib_next_tick = int(np.ceil(calib_delay_sec))
                        print(
                            f"[Manager] 'r' pressed: recalibrating in {calib_delay_sec:.0f}s — "
                            "assume the rest pose and hold still..."
                        )
                elif key == "f":
                    finger_tracking = not finger_tracking
                    print(f"[Manager] Finger retargeting {'ENABLED' if finger_tracking else 'DISABLED'}")
                elif key == "p" and current_mode == StreamMode.PLANNER_VR_3PT:
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
                elif key == "c":
                    toggle_dc = True
                    print("[Manager] 'c' pressed: toggling data collection")
                elif key == "x":
                    toggle_da = True
                    print("[Manager] 'x' pressed: toggling data abort")
                elif key == "q":
                    print("[Manager] 'q' pressed: stopping policy and exiting...")
                    break
                elif key == "k" and replay_npz is not None:
                    reader.toggle_pause()
                elif key == "left" and replay_npz is not None:
                    reader.step(-1)
                elif key == "right" and replay_npz is not None:
                    reader.step(1)

            # --- Deferred calibration countdown (live Quest) ---
            if calib_pending:
                remaining = calib_deadline - time.time()
                if remaining <= 0.0:
                    print("[Manager] Capturing calibration now (hold still)...")
                    _capture_calibration_now(use_measured=calib_start_policy is False)
                    if calib_start_policy and current_mode == StreamMode.OFF:
                        socket.send(build_command_message(start=True, stop=False, planner=True))
                        current_mode = StreamMode.PLANNER_VR_3PT
                        print(f"[Manager] Mode: {current_mode.name}")
                    calib_pending = False
                else:
                    sec = int(np.ceil(remaining))
                    if sec < calib_next_tick:
                        print(f"[Manager]   calibrating in {sec}s...")
                        calib_next_tick = sec

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
                            left_hand = retarget_hand(correct_landmark_frame(latest["left_landmarks"]), "left", hand_retargeter)
                        except Exception:
                            left_hand = None
                        try:
                            right_hand = retarget_hand(correct_landmark_frame(latest["right_landmarks"]), "right", hand_retargeter)
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
    parser.add_argument(
        "--np-retarget",
        action="store_true",
        help=(
            "Use the pure-numpy retargeter (np_retargeting) for finger tracking instead of "
            "the default optimization-based BrainCoRetargeter (dex_retargeting). "
            "Useful when dex_retargeting / sapien is not installed."
        ),
    )
    parser.add_argument(
        "--replay-from-data",
        type=str,
        default=None,
        metavar="NPZ_PATH",
        help=(
            "Replay a recorded NPZ trajectory instead of connecting to ROS2 / ZMQ. "
            "The recording replaces the live Quest operator; all other processing "
            "(calibration, 3-pt pose, finger retargeting, ZMQ streaming) runs unchanged. "
            "Replay controls: s=start  k=pause/resume  ←/→=step-back/forward"
        ),
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor on the operator head->wrist displacement to account for the "
            "robot vs operator arm length (form factor). 1.0 = 1:1; e.g. 0.8 shrinks the "
            "reach so a smaller robot does not saturate. Orientation is unaffected."
        ),
    )
    parser.add_argument(
        "--calib-delay-sec",
        type=float,
        default=3.0,
        help=(
            "Live Quest only: seconds to wait after pressing 's'/'r' before capturing "
            "calibration, so the operator can assume the rest pose. 0 disables the delay. "
            "Ignored in --replay-from-data mode (frame 0 is the prepended rest pose)."
        ),
    )
    parser.add_argument(
        "--rest-hold-sec",
        type=float,
        default=1.0,
        help=(
            "Replay only: seconds to hold the synthetic rest pose prepended to the "
            "recording (frame 0 is the rest pose used for calibration)."
        ),
    )
    parser.add_argument(
        "--rest-interp-sec",
        type=float,
        default=1.5,
        help=(
            "Replay only: seconds to interpolate from the synthetic rest pose to the "
            "recording's first frame, so the robot eases into the motion."
        ),
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
        replay_npz=args.replay_from_data,
        np_retarget=args.np_retarget,
        pos_scale=args.pos_scale,
        calib_delay_sec=args.calib_delay_sec,
        rest_hold_sec=args.rest_hold_sec,
        rest_interp_sec=args.rest_interp_sec,
    )
