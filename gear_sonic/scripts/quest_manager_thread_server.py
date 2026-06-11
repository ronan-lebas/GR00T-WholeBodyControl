# Quest manager thread server
#
# Streams Meta Quest 3-point VR poses (head + both wrists) and BrainCo finger
# commands to the WBC deploy policy over ZMQ, replacing the Pico manager for
# Quest-based teleoperation.

"""
Data flow:

    Quest (Unity) --TCP--> ros_tcp_endpoint --ROS1--> quest_relay (Docker)
        --ZMQ SUB "quest_data" (msgpack, port 5559)--> THIS SCRIPT
        --ZMQ PUB "command"/"planner"/"manager_state" (port 5556)--> deploy policy

Input (quest_relay snapshot, ROS FLU frame: X-forward, Y-left, Z-up;
quaternions scalar-first [w, x, y, z]):
    head_pos[3], head_quat[4], left/right_wrist_pos[3], left/right_wrist_quat[4],
    left/right_landmarks[21][3] (MANO-21), left/right_tracked, timestamp

Output ("planner" topic, robot ROOT frame, X-forward, Y-left, Z-up):
    vr_position[9]     rows = [L-wrist, R-wrist, head], KEY-FRAME points, i.e.
                       the local offsets ([0.18, -/+0.025, 0] for the wrists,
                       [0, 0, 0.35] above the torso for the head) are ALREADY
                       applied, rotated by the live commanded orientation —
                       the deploy uses the sent values directly
                       (see g1_deploy_onnx_ref.cpp GatherVR3PointPosition).
    vr_orientation[12] scalar-first quaternions for the same three rows.
    left/right_hand_joints[7]  BrainCo: 6 normalized motors [0=open, 1=closed]
                       + one 0.0 padding slot.
    mode + movement[3] locomotion: the operator's head planar velocity (in the
                       fixed calibration frame, same frame as facing) drives a
                       SLOW_WALK/WALK command so the robot walks when the
                       operator walks. Disable with --disable-walk (robot then
                       only turns in place via facing).

Usage:
    # Live Quest (relay container must be running, see run_quest_relay.py)
    python quest_manager_thread_server.py

    # Turn in place only, no base translation
    python quest_manager_thread_server.py --disable-walk

    # Replay a recorded trajectory (record_quest_data.py NPZ format)
    python quest_manager_thread_server.py --replay data/quest/traj_xxx.npz

Keyboard:
    s          start policy + enter VR_3PT mode (countdown on live Quest)
    r          recalibrate (countdown; uses measured robot joints as FK ref)
    f          toggle finger retargeting
    p          pause / resume teleop (freeze robot, smooth resume)
    c          toggle data collection        x   toggle data abort
    q          stop policy and exit
    k          (replay) pause / resume playback
    <- / ->    (replay) step one frame back / forward
"""

from __future__ import annotations

import argparse
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import msgpack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import (
    G1_KEY_FRAME_OFFSETS,
    get_g1_key_frame_poses,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)
from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller

try:
    from brainco_retargeting.retargeter import BrainCoRetargeter
except ImportError:
    BrainCoRetargeter = None

try:
    from brainco_retargeting import np_retargeting
except ImportError:
    np_retargeting = None


# StreamMode values understood by the deploy state machine (see
# pico_manager_thread_server.StreamMode / mock_quest_streamer.py).
STREAM_MODE_OFF = 0
STREAM_MODE_PLANNER_VR_3PT = 5

# LocomotionMode values (see localmotion_kplanner.hpp). IDLE keeps the robot
# stationary (only turning via the facing command); SLOW_WALK / WALK translate
# the base along the planner movement_direction.
LOCOMOTION_IDLE = 0
LOCOMOTION_SLOW_WALK = 1
LOCOMOTION_WALK = 2
_WALK_MODES = {"slow": LOCOMOTION_SLOW_WALK, "walk": LOCOMOTION_WALK}

# Wire format always carries 7 hand values; BrainCo uses the first 6
# (normalized [0=open, 1=closed]) and the 7th slot is padding.
WIRE_HAND_DOF = 7

_SIDES = ("left", "right")


# ---------------------------------------------------------------------------
# Small rotation helpers (quaternions scalar-first [w, x, y, z] throughout)
# ---------------------------------------------------------------------------


def _rot(quat_wxyz: np.ndarray) -> sRot:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return sRot.identity()
    return sRot.from_quat(q / norm, scalar_first=True)


def _quat(rot: sRot) -> np.ndarray:
    return rot.as_quat(scalar_first=True)


def _yaw_of(quat_wxyz: np.ndarray) -> float:
    """Heading angle: projection of the rotated forward axis (+X) on the XY plane."""
    fwd = _rot(quat_wxyz).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(fwd[1], fwd[0]))


def _rz(yaw: float) -> sRot:
    return sRot.from_euler("z", yaw)


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path normalized lerp between two wxyz quaternions (fine for short ramps)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / max(np.linalg.norm(q), 1e-9)


# ---------------------------------------------------------------------------
# Robot FK reference (rest pose of the wrist links in the root frame)
# ---------------------------------------------------------------------------


class RobotRestReference:
    """FK of the G1 wrist links + torso, used as the calibration anchor."""

    def __init__(self):
        self._robot_model = instantiate_g1_robot_model(hand_type="brainco")
        self._hand_convention = {
            side: self._compute_hand_convention(side) for side in _SIDES
        }
        print("[QuestManager] G1 robot model loaded for FK calibration")

    def _compute_hand_convention(self, side: str) -> sRot:
        """Constant local rotation N from the wrist LINK frame to the PHYSICAL
        hand frame (X = fingers, Z = back of hand), from the brainco finger FK.

        The Quest wrist TF uses the physical convention (verified against the
        MANO landmarks in recorded data), but the G1 wrist link frame is rolled
        ~90 deg about the forearm axis relative to it (back of hand = link -/+Y,
        not link Z). Comparing the two raw frames without N commands every
        wrist orientation 90 deg off — palms-down operator showed up as the
        robot's palms-inward rest pose.
        """
        rm = self._robot_model
        rm.cache_forward_kinematics(rm.default_body_pose, auto_clip=False)
        # Left and right hand URDFs capitalize the proximal link names differently.
        suffix = "proximal_Link" if side == "left" else "proximal_link"
        mid = np.asarray(rm.frame_placement(f"{side}_middle_{suffix}").translation)
        idx = np.asarray(rm.frame_placement(f"{side}_index_{suffix}").translation)
        pky = np.asarray(rm.frame_placement(f"{side}_pinky_{suffix}").translation)
        wrist = rm.frame_placement(f"{side}_wrist_yaw_link")
        wrist_pos = np.asarray(wrist.translation)
        wrist_rot = sRot.from_matrix(np.asarray(wrist.rotation))

        fingers = mid - wrist_pos
        fingers /= np.linalg.norm(fingers)
        across = idx - pky  # index->pinky spans the knuckles, thumb side first
        across /= np.linalg.norm(across)
        back = np.cross(fingers, across) if side == "right" else np.cross(across, fingers)
        y = np.cross(back, fingers)
        y /= np.linalg.norm(y)
        back = np.cross(fingers, y)
        phys = sRot.from_matrix(np.stack([fingers, y, back], axis=1))
        return wrist_rot.inv() * phys

    def hand_convention(self, side: str) -> sRot:
        """Wrist link frame -> physical hand frame (constant in link coords)."""
        return self._hand_convention[side]

    def compute(self, body_q_29: np.ndarray | None = None) -> dict:
        """FK key-frame poses in the root frame.

        Args:
            body_q_29: measured 29-DOF body joints, or None for the model's
                       default (rest) pose.

        Returns dict:
            {"left": (link_pos, link_rot), "right": (...), "torso_pos": (3,)}
            Positions are the wrist LINK origins (no key-frame offset applied);
            the offset is re-applied at runtime with the live orientation.
        """
        q = None
        if body_q_29 is not None:
            q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=np.asarray(body_q_29, dtype=np.float64)[:29]
            )
        poses = get_g1_key_frame_poses(self._robot_model, q=q, apply_offset=False)
        return {
            "left": (
                np.asarray(poses["left_wrist"]["position"], dtype=np.float64),
                _rot(poses["left_wrist"]["orientation_wxyz"]),
            ),
            "right": (
                np.asarray(poses["right_wrist"]["position"], dtype=np.float64),
                _rot(poses["right_wrist"]["orientation_wxyz"]),
            ),
            "torso_pos": np.asarray(poses["torso"]["position"], dtype=np.float64),
        }


class RobotFeedback:
    """Reads measured robot joints from the deploy g1_debug ZMQ stream."""

    def __init__(self, host: str, port: int):
        self._poller = ZMQPoller(host=host, port=port, topic="g1_debug")

    def measured_body_q(self) -> np.ndarray | None:
        data = self._poller.get_data()
        if data is None:
            return None
        try:
            unpacked = msgpack.unpackb(data, raw=False)
        except Exception as e:
            print(f"[QuestManager] Failed to unpack g1_debug feedback: {e}")
            return None
        if "body_q_measured" not in unpacked:
            print("[QuestManager] body_q_measured missing from g1_debug feedback")
            return None
        return np.array(unpacked["body_q_measured"], dtype=np.float64)


# ---------------------------------------------------------------------------
# 3-point pose tracking + calibration
# ---------------------------------------------------------------------------


class QuestThreePointTracker:
    """Maps Quest head/wrist poses to robot-root-frame VR 3-point targets.

    Calibration (operator in rest pose, mirroring the robot's reference pose):
      - R0 = Rz(head yaw): the operator's heading frame at calibration. All
        runtime quantities are expressed in this fixed frame, which is taken
        to coincide with the robot root frame.
      - v_cal = R0^-1 (p_wrist - p_head): the head->wrist offset vector. Using
        the head-relative delta (rather than absolute wrist position) makes
        tracking invariant to the operator walking or leaning.
      - W_cal: operator wrist orientation. Captures the (unknown, fixed)
        Quest-hand-frame vs robot-wrist-frame convention offset.
      - (link_ref, R_ref): robot wrist LINK pose from FK — the robot pose that
        the operator's rest pose maps onto.

    Runtime, per wrist, with Rh(t) = Rz(head_yaw(t)) the CURRENT heading frame:
        v(t)    = Rh(t)^-1 (p_wrist(t) - p_head(t))
        link(t) = link_ref + pos_scale * (v(t) - v_cal)
        R(t)    = Rh(t)^-1 W(t) W_cal^-1 R0 * R_ref       (world-frame delta)
        sent(t) = link(t) + R(t) @ key_frame_offset

    Using the CURRENT heading frame (not the fixed R0) makes the targets
    body-relative: when the operator turns their whole body about Z, v(t) and
    R(t) are unchanged, so the arms do not chase the rotation. The rotation
    itself is returned as yaw_rel and must be sent as the planner ``facing``
    command, which makes the robot's whole body turn to follow the operator.

    The world-frame delta makes R(t) independent of the Quest hand-frame
    convention: any fixed offset C in W = W_physical * C cancels in
    W(t) W_cal^-1.

    Head row: identity orientation — the torso target stays aligned with the
    root, which itself rotates via the ``facing`` command; position via the
    fixed kinematic chain torso + [0, 0, 0.35].

    Locomotion: the operator's head also drives base translation. compute()
    returns the head's planar velocity expressed in the fixed R0 frame (the
    same frame as ``facing``), low-pass filtered and computed from sensor
    timestamps so it is independent of this loop's polling rate. The manager
    turns that velocity into a planner movement_direction + speed so the robot
    walks when the operator walks. (The head ROW of vr_position stays fixed;
    walking is a separate planner command, not a moving torso target.)
    """

    # Head-velocity low-pass time constant (s) and the staleness window after
    # which a frozen / dropped Quest stream decays the velocity back to zero.
    _VEL_TAU = 0.25
    _VEL_STALE_SEC = 0.3

    _WRIST_OFFSET = {
        "left": np.asarray(G1_KEY_FRAME_OFFSETS["left_wrist"], dtype=np.float64),
        "right": np.asarray(G1_KEY_FRAME_OFFSETS["right_wrist"], dtype=np.float64),
    }
    _HEAD_OFFSET = np.asarray(G1_KEY_FRAME_OFFSETS["torso"], dtype=np.float64)

    def __init__(self, robot_ref: RobotRestReference, pos_scale: float = 0.8):
        self._robot_ref = robot_ref
        self.pos_scale = float(pos_scale)
        self._calibrated = False
        self._yaw_cal = 0.0
        self._r0 = sRot.identity()
        self._r0_inv = sRot.identity()
        self._v_cal: dict[str, np.ndarray] = {}
        self._w_cal_inv: dict[str, sRot] = {}
        self._link_ref: dict[str, np.ndarray] = {}
        self._rot_ref: dict[str, sRot] = {}
        self._torso_pos = np.zeros(3)
        # Head planar velocity tracking (R0 frame), reset on each calibration.
        self._prev_head_pos: np.ndarray | None = None
        self._prev_ts: float | None = None
        self._head_vel = np.zeros(3)
        self._last_vel_wall = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, frame: dict, body_q_29: np.ndarray | None = None) -> None:
        """Capture calibration from one Quest frame against the robot FK reference."""
        fk = self._robot_ref.compute(body_q_29)
        self._yaw_cal = _yaw_of(frame["head_quat"])
        self._r0 = _rz(self._yaw_cal)
        self._r0_inv = self._r0.inv()

        head_pos = np.asarray(frame["head_pos"], dtype=np.float64)
        for side in _SIDES:
            wrist_pos = np.asarray(frame[f"{side}_wrist_pos"], dtype=np.float64)
            self._v_cal[side] = self._r0_inv.apply(wrist_pos - head_pos)
            self._w_cal_inv[side] = _rot(frame[f"{side}_wrist_quat"]).inv()
            self._link_ref[side], self._rot_ref[side] = fk[side]
        self._torso_pos = fk["torso_pos"]
        self._prev_head_pos = None
        self._prev_ts = None
        self._head_vel = np.zeros(3)
        self._last_vel_wall = time.monotonic()
        self._calibrated = True

        src = "default rest pose" if body_q_29 is None else "measured robot joints"
        print(
            f"[QuestManager] Calibration captured (FK ref: {src}, head yaw "
            f"{np.degrees(self._yaw_cal):+.1f} deg)"
        )
        for side in _SIDES:
            v = self._v_cal[side]
            print(f"  {side}: head->wrist cal vector [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")

    def _update_head_velocity(self, head_pos: np.ndarray, ts: float) -> None:
        """Low-pass head planar velocity in the fixed R0 frame.

        Velocity is differenced over sensor timestamps (not loop time), so a
        repeated frame (sample-and-hold relay, paused replay) contributes no
        spurious motion; if no fresh frame arrives within ``_VEL_STALE_SEC``
        the estimate decays to zero so the robot does not keep walking.
        """
        now = time.monotonic()
        if self._prev_head_pos is None or self._prev_ts is None:
            self._prev_head_pos = head_pos
            self._prev_ts = ts
            self._last_vel_wall = now
            return
        dt = ts - self._prev_ts
        if dt < -1e-4:
            # Sensor time jumped backwards (replay looped or stepped back):
            # re-anchor on this frame instead of freezing on a stale prev
            # timestamp or differencing across the discontinuity.
            self._prev_head_pos = head_pos
            self._prev_ts = ts
            self._last_vel_wall = now
            return
        if dt > 1e-4:
            raw = self._r0_inv.apply(head_pos - self._prev_head_pos) / dt
            raw[2] = 0.0
            alpha = dt / (self._VEL_TAU + dt)
            self._head_vel = (1.0 - alpha) * self._head_vel + alpha * raw
            self._prev_head_pos = head_pos
            self._prev_ts = ts
            self._last_vel_wall = now
        elif now - self._last_vel_wall > self._VEL_STALE_SEC:
            self._head_vel = np.zeros(3)

    def compute(self, frame: dict) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """One Quest frame -> (vr_position (9,), vr_orientation (12,), yaw_rel,
        head_vel (3,)).

        Positions/orientations are body-relative targets in the robot root
        frame; yaw_rel is the operator heading change since calibration and
        must be sent as the planner ``facing`` direction to turn the robot;
        head_vel is the operator's planar head velocity (m/s, z=0) in the fixed
        R0 frame, used to drive base locomotion.
        """
        if not self._calibrated:
            raise RuntimeError("compute() called before calibrate()")

        yaw_rel = _wrap(_yaw_of(frame["head_quat"]) - self._yaw_cal)
        rh_inv = _rz(self._yaw_cal + yaw_rel).inv()  # current heading frame

        pos = np.zeros((3, 3), dtype=np.float64)
        quat = np.zeros((3, 4), dtype=np.float64)
        head_pos = np.asarray(frame["head_pos"], dtype=np.float64)
        self._update_head_velocity(head_pos, float(frame["timestamp"]))

        for i, side in enumerate(_SIDES):
            wrist_pos = np.asarray(frame[f"{side}_wrist_pos"], dtype=np.float64)
            v = rh_inv.apply(wrist_pos - head_pos)
            link = self._link_ref[side] + self.pos_scale * (v - self._v_cal[side])

            w = _rot(frame[f"{side}_wrist_quat"])
            r_cmd = rh_inv * w * self._w_cal_inv[side] * self._r0 * self._rot_ref[side]

            pos[i] = link + r_cmd.apply(self._WRIST_OFFSET[side])
            quat[i] = _quat(r_cmd)

        # Head row: torso aligned with the root (rotation goes through facing)
        pos[2] = self._torso_pos + self._HEAD_OFFSET
        quat[2] = np.array([1.0, 0.0, 0.0, 0.0])

        return pos.flatten(), quat.flatten(), yaw_rel, self._head_vel.copy()


# ---------------------------------------------------------------------------
# Finger retargeting (MANO-21 -> 6 BrainCo motors)
# ---------------------------------------------------------------------------

# XR-25 layout (OpenXR without palm): 0 wrist; then per finger
# [metacarpal, proximal, intermediate, distal, tip] — thumb has 4 joints
# (1-4), the other fingers 5 each (index 5-9, middle 10-14, ring 15-19,
# pinky 20-24). MANO-21 (MediaPipe) has no metacarpal points, so those are
# synthesized as the wrist->MCP midpoint.
_XR25_FROM_MANO = [
    (1, 1), (2, 2), (3, 3), (4, 4),  # thumb CMC/MCP/IP/TIP
    (6, 5), (7, 6), (8, 7), (9, 8),  # index MCP/PIP/DIP/TIP
    (11, 9), (12, 10), (13, 11), (14, 12),  # middle
    (16, 13), (17, 14), (18, 15), (19, 16),  # ring
    (21, 17), (22, 18), (23, 19), (24, 20),  # pinky
]
_XR25_METACARPALS = [(5, 5), (10, 9), (15, 13), (20, 17)]  # (xr_idx, mano_mcp_idx)


def mano21_to_xr25(landmarks21: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks21, dtype=np.float64)
    xr = np.zeros((25, 3), dtype=np.float64)
    xr[0] = lm[0]
    for xr_i, mano_i in _XR25_FROM_MANO:
        xr[xr_i] = lm[mano_i]
    for xr_i, mano_mcp in _XR25_METACARPALS:
        xr[xr_i] = 0.5 * (lm[0] + lm[mano_mcp])
    return xr


class FingerRetargeting:
    """MANO-21 landmarks -> 7-element wire vector (6 normalized motors + pad).

    Uses the optimization-based BrainCoRetargeter when available; otherwise
    falls back to the pure-numpy angle-based retargeter.
    """

    # Wire order matches the BrainCo firmware / mock streamer convention.
    _NP_JOINT_KEYS = [
        "thumb_metacarpal",
        "thumb_proximal",
        "index_proximal",
        "middle_proximal",
        "ring_proximal",
        "pinky_proximal",
    ]

    def __init__(self, force_np: bool = False):
        self._opt = None
        if not force_np and BrainCoRetargeter is not None:
            try:
                self._opt = BrainCoRetargeter()
                print("[QuestManager] Finger retargeting: optimization-based BrainCoRetargeter")
            except Exception as e:
                print(f"[QuestManager] BrainCoRetargeter init failed ({e}), using numpy fallback")
        if self._opt is None:
            if np_retargeting is None:
                raise ImportError(
                    "Neither BrainCoRetargeter nor np_retargeting is available. "
                    "Install third_party/brainco-retargeting."
                )
            print("[QuestManager] Finger retargeting: pure-numpy fallback")

    def __call__(self, landmarks21: np.ndarray, side: str) -> list[float]:
        if self._opt is not None:
            xr = mano21_to_xr25(landmarks21)
            canon = self._opt.canonicalize(xr, side)
            if side == "left":
                motors = self._opt.retarget_left(canon)
            else:
                motors = self._opt.retarget_right(canon)
        else:
            angles = np_retargeting.retarget(np.asarray(landmarks21, dtype=np.float64), side)
            motors = [
                angles[f"{side}_{k}_joint"] / np_retargeting._JOINT_LIMITS[k][1]
                for k in self._NP_JOINT_KEYS
            ]
        return [float(np.clip(m, 0.0, 1.0)) for m in motors] + [0.0]


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def _frame_from_snapshot(snap: dict) -> dict:
    """Normalize a relay snapshot into a dict of numpy arrays."""
    return {
        "head_pos": np.asarray(snap["head_pos"], dtype=np.float64),
        "head_quat": np.asarray(snap["head_quat"], dtype=np.float64),
        "left_wrist_pos": np.asarray(snap["left_wrist_pos"], dtype=np.float64),
        "left_wrist_quat": np.asarray(snap["left_wrist_quat"], dtype=np.float64),
        "right_wrist_pos": np.asarray(snap["right_wrist_pos"], dtype=np.float64),
        "right_wrist_quat": np.asarray(snap["right_wrist_quat"], dtype=np.float64),
        "left_landmarks": np.asarray(snap["left_landmarks"], dtype=np.float64),
        "right_landmarks": np.asarray(snap["right_landmarks"], dtype=np.float64),
        "left_tracked": bool(snap["left_tracked"]),
        "right_tracked": bool(snap["right_tracked"]),
        "timestamp": float(snap["timestamp"]),
    }


class LiveQuestSource:
    """Background ZMQ SUB thread holding the latest quest_data snapshot."""

    is_replay = False

    def __init__(self, host: str, port: int):
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{host}:{port}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"quest_data")
        self._sub.setsockopt(zmq.RCVTIMEO, 100)
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[QuestManager] Subscribed to quest relay at tcp://{host}:{port}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                parts = self._sub.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                frame = _frame_from_snapshot(msgpack.unpackb(parts[1], raw=False))
            except Exception as e:
                print(f"[QuestManager] Bad quest_data message: {e}")
                continue
            # timestamp 0.0 means the relay has not received any Quest topic
            # yet — not usable for tracking or calibration.
            if frame["timestamp"] <= 0.0:
                continue
            with self._lock:
                self._latest = frame

    def get_frame(self) -> dict | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sub.close()
        self._ctx.term()


class ReplaySource:
    """Replays a recorded NPZ trajectory with a synthetic rest-pose prefix.

    The prefix makes calibration on frame 0 always valid: the first frames put
    the (virtual) operator exactly in the pose the calibration expects — head
    at the recording's initial head pose, wrists where the robot's rest wrist
    links sit relative to the head — held for ``rest_hold_sec`` and then
    interpolated to the recording's real first frame over ``rest_interp_sec``.

    Playback is time-anchored (wall clock against recorded timestamps), loops
    at the end, and supports pause ('k') and single-frame stepping (arrows).
    """

    is_replay = True

    def __init__(
        self,
        path: str,
        robot_ref: RobotRestReference,
        rest_hold_sec: float = 1.0,
        rest_interp_sec: float = 1.0,
    ):
        raw = np.load(path, allow_pickle=True)
        n = raw["head_pos"].shape[0]
        if n < 2:
            raise ValueError(f"Recording {path} has fewer than 2 frames")
        ts = np.asarray(raw["timestamp"], dtype=np.float64)
        dt = float(np.median(np.diff(ts)))
        dt = dt if np.isfinite(dt) and dt > 1e-4 else 1.0 / 50.0

        frames = {
            k: np.asarray(raw[k])
            for k in (
                "head_pos",
                "head_quat",
                "left_wrist_pos",
                "left_wrist_quat",
                "right_wrist_pos",
                "right_wrist_quat",
                "left_landmarks",
                "right_landmarks",
                "left_tracked",
                "right_tracked",
            )
        }
        prefix = self._build_rest_prefix(frames, robot_ref, dt, rest_hold_sec, rest_interp_sec)
        self._data = {k: np.concatenate([prefix[k], frames[k]], axis=0) for k in frames}
        n_prefix = prefix["head_pos"].shape[0]
        self._ts = np.concatenate([np.arange(n_prefix) * dt, n_prefix * dt + (ts - ts[0])])
        self._n = self._ts.shape[0]
        self._duration = float(self._ts[-1]) + dt

        self._paused = False
        self._media_time = 0.0
        self._anchor = time.monotonic()
        print(
            f"[QuestManager] Replay {Path(path).name}: {n} frames + {n_prefix} rest-prefix "
            f"frames ({self._duration:.1f}s total, {1.0 / dt:.0f} fps, looping)"
        )

    @staticmethod
    def _build_rest_prefix(
        frames: dict,
        robot_ref: RobotRestReference,
        dt: float,
        hold_sec: float,
        interp_sec: float,
    ) -> dict:
        """Synthesize rest frames matching the robot FK rest pose, then blend to frame 0."""
        fk = robot_ref.compute(None)
        head_pos0 = np.asarray(frames["head_pos"][0], dtype=np.float64)
        head_quat0 = np.asarray(frames["head_quat"][0], dtype=np.float64)
        r0 = _rz(_yaw_of(head_quat0))
        head_ref = fk["torso_pos"] + QuestThreePointTracker._HEAD_OFFSET

        # Rest frame: virtual operator whose head matches the recording's first
        # head pose and whose wrists sit where the robot's rest wrist LINKS sit
        # relative to its own head reference, lifted into the Quest world by the
        # head heading. Calibrating on this frame maps it exactly onto FK rest.
        rest = {
            "head_pos": head_pos0,
            "head_quat": head_quat0,
            "left_landmarks": np.asarray(frames["left_landmarks"][0], dtype=np.float64),
            "right_landmarks": np.asarray(frames["right_landmarks"][0], dtype=np.float64),
        }
        for side in _SIDES:
            link_pos, link_rot = fk[side]
            rest[f"{side}_wrist_pos"] = head_pos0 + r0.apply(link_pos - head_ref)
            # The synthetic wrist quat must be what the QUEST would publish for
            # an operator physically holding the robot's rest pose: the link
            # orientation expressed in the physical hand convention (N), since
            # the Quest wrist TF is the physical frame, not the link frame.
            rest[f"{side}_wrist_quat"] = _quat(r0 * link_rot * robot_ref.hand_convention(side))

        n_hold = max(1, int(round(hold_sec / dt)))
        n_interp = max(0, int(round(interp_sec / dt)))
        n_total = n_hold + n_interp

        prefix: dict[str, np.ndarray] = {}
        for key in ("head_pos", "left_wrist_pos", "right_wrist_pos"):
            first = np.asarray(frames[key][0], dtype=np.float64)
            arr = np.tile(rest[key], (n_total, 1))
            for j in range(n_interp):
                a = (j + 1) / (n_interp + 1)
                arr[n_hold + j] = (1.0 - a) * rest[key] + a * first
            prefix[key] = arr
        for key in ("head_quat", "left_wrist_quat", "right_wrist_quat"):
            first = np.asarray(frames[key][0], dtype=np.float64)
            arr = np.tile(rest[key], (n_total, 1))
            for j in range(n_interp):
                a = (j + 1) / (n_interp + 1)
                arr[n_hold + j] = _slerp(rest[key], first, a)
            prefix[key] = arr
        for key in ("left_landmarks", "right_landmarks"):
            prefix[key] = np.tile(rest[key][None], (n_total, 1, 1))
        # No finger retargeting during the synthetic prefix.
        prefix["left_tracked"] = np.zeros(n_total, dtype=bool)
        prefix["right_tracked"] = np.zeros(n_total, dtype=bool)
        return prefix

    # -- playback controls --------------------------------------------------

    def _current_media_time(self) -> float:
        if self._paused:
            return self._media_time
        return (time.monotonic() - self._anchor) % self._duration

    def toggle_pause(self) -> None:
        if self._paused:
            self._anchor = time.monotonic() - self._media_time
            self._paused = False
            print("[QuestManager] Replay resumed")
        else:
            self._media_time = self._current_media_time()
            self._paused = True
            print(f"[QuestManager] Replay paused at t={self._media_time:.2f}s")

    def step(self, delta: int) -> None:
        """Step playback by delta frames; pauses the clock."""
        if not self._paused:
            self._media_time = self._current_media_time()
            self._paused = True
        idx = (self._current_index() + delta) % self._n
        self._media_time = float(self._ts[idx])
        print(f"[QuestManager] Replay step -> frame {idx}/{self._n - 1}")

    def restart(self) -> None:
        self._media_time = 0.0
        self._anchor = time.monotonic()

    def _current_index(self) -> int:
        idx = int(np.searchsorted(self._ts, self._current_media_time(), side="right")) - 1
        return int(np.clip(idx, 0, self._n - 1))

    def get_frame(self) -> dict:
        i = self._current_index()
        return {
            "head_pos": self._data["head_pos"][i],
            "head_quat": self._data["head_quat"][i],
            "left_wrist_pos": self._data["left_wrist_pos"][i],
            "left_wrist_quat": self._data["left_wrist_quat"][i],
            "right_wrist_pos": self._data["right_wrist_pos"][i],
            "right_wrist_quat": self._data["right_wrist_quat"][i],
            "left_landmarks": self._data["left_landmarks"][i],
            "right_landmarks": self._data["right_landmarks"][i],
            "left_tracked": bool(self._data["left_tracked"][i]),
            "right_tracked": bool(self._data["right_tracked"][i]),
            "timestamp": float(self._ts[i]),
        }

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


class KeyboardListener:
    """cbreak single-key reader; arrow keys are decoded to 'LEFT' / 'RIGHT'."""

    def __init__(self) -> None:
        self.keys: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self._old_term = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "KeyboardListener":
        if self._fd is not None:
            self._old_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            print("[QuestManager] stdin is not a TTY — keyboard controls disabled")
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            ch = sys.stdin.read(1)
            if ch != "\x1b":
                if ch:
                    self.keys.put(ch)
                continue
            # Escape sequence (arrow keys: ESC [ C / ESC [ D)
            seq = ""
            while select.select([sys.stdin], [], [], 0.01)[0] and len(seq) < 2:
                seq += sys.stdin.read(1)
            if seq == "[C":
                self.keys.put("RIGHT")
            elif seq == "[D":
                self.keys.put("LEFT")

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)


# ---------------------------------------------------------------------------
# Manager loop
# ---------------------------------------------------------------------------


class QuestManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot_ref = RobotRestReference()
        self.tracker = QuestThreePointTracker(self.robot_ref, pos_scale=args.pos_scale)
        self.retargeter = FingerRetargeting(force_np=args.np_retarget)
        self.feedback = RobotFeedback(args.feedback_host, args.feedback_port)

        if args.replay:
            self.source = ReplaySource(
                args.replay,
                self.robot_ref,
                rest_hold_sec=args.rest_hold_sec,
                rest_interp_sec=args.rest_interp_sec,
            )
        else:
            self.source = LiveQuestSource(args.relay_host, args.relay_port)

        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{args.port}")
        time.sleep(0.3)  # let subscribers connect before the first messages
        print(f"[QuestManager] ZMQ PUB bound to port {args.port}")

        self.stream_mode = STREAM_MODE_OFF
        self.finger_tracking = True
        self.teleop_paused = False
        # Locomotion (head-position walking). `_walking` adds hysteresis around
        # the speed deadband so the robot does not flap between IDLE and walk.
        self.walk_mode = _WALK_MODES[args.walk_mode]
        self._walking = False
        self.resume_ramp_start: float | None = None
        # Countdown state: (deadline, kind) with kind in {"start", "recalib"}
        self.pending_calib: tuple[float, str] | None = None
        self.last_countdown_print = 0

        # Last commanded targets; reused while paused / blended on resume.
        self.frozen_pos: np.ndarray | None = None
        self.frozen_quat: np.ndarray | None = None
        self.frozen_yaw: float = 0.0
        self.frozen_hands: dict[str, list[float] | None] = {"left": None, "right": None}
        self.last_hands: dict[str, list[float] | None] = {"left": None, "right": None}
        # facing_yaw = yaw_rel - yaw_offset: the offset rebases the operator's
        # heading after pauses/recalibrations so the commanded facing never jumps.
        self.yaw_offset: float = 0.0
        self.last_facing_yaw: float = 0.0
        self.resume_rebase_pending = False

    # -- calibration triggers -------------------------------------------------

    def _arm_calibration(self, kind: str) -> None:
        if self.source.is_replay:
            # The synthetic rest prefix stands in for the operator's rest pose,
            # so no countdown is needed: restart and calibrate on frame 0.
            if kind == "start":
                self.source.restart()
            self._do_calibration(kind, self.source.get_frame())
            return
        delay = self.args.calib_delay_sec
        self.pending_calib = (time.monotonic() + delay, kind)
        self.last_countdown_print = int(np.ceil(delay)) + 1
        print(
            f"[QuestManager] Assume the rest pose — calibrating in {delay:.0f}s "
            f"({'start' if kind == 'start' else 'recalibration'})"
        )

    def _do_calibration(self, kind: str, frame: dict | None) -> None:
        if frame is None:
            print("[QuestManager] WARNING: no Quest data — calibration aborted")
            return
        body_q = None
        if kind == "recalib":
            body_q = self.feedback.measured_body_q()
            if body_q is None:
                print(
                    "[QuestManager] WARNING: no g1_debug feedback — "
                    "recalibrating against the default rest pose instead"
                )
        self.tracker.calibrate(frame, body_q_29=body_q)
        if kind == "start":
            self.yaw_offset = 0.0
            self.last_facing_yaw = 0.0
            self.pub.send(build_command_message(start=True, stop=False, planner=True))
            self.stream_mode = STREAM_MODE_PLANNER_VR_3PT
            print("[QuestManager] Policy START sent — entering VR_3PT mode")
        else:
            # Recalibration zeroes yaw_rel; keep the commanded facing continuous
            # so the robot does not turn back to its start heading.
            self.yaw_offset = _wrap(-self.last_facing_yaw)

    # -- keyboard handling ------------------------------------------------------

    def _handle_key(self, key: str) -> tuple[bool, bool, bool]:
        """Returns (quit, toggle_data_collection, toggle_data_abort)."""
        toggle_dc = toggle_da = False
        if key == "q":
            return True, False, False
        elif key == "s":
            if self.stream_mode == STREAM_MODE_OFF:
                self._arm_calibration("start")
            else:
                print("[QuestManager] Already started ('r' to recalibrate, 'q' to stop)")
        elif key == "r":
            self._arm_calibration("recalib")
        elif key == "f":
            self.finger_tracking = not self.finger_tracking
            print(f"[QuestManager] Finger retargeting {'ON' if self.finger_tracking else 'OFF'}")
        elif key == "p":
            self.teleop_paused = not self.teleop_paused
            if self.teleop_paused:
                self.frozen_pos = None  # captured from the next computed targets
                self.resume_ramp_start = None
                print("[QuestManager] Teleop PAUSED — robot frozen at last pose")
            else:
                self.resume_ramp_start = time.monotonic()
                self.resume_rebase_pending = True
                print(
                    f"[QuestManager] Teleop RESUMED — easing to live pose over "
                    f"{self.args.resume_ramp_sec:.1f}s"
                )
        elif key == "c":
            toggle_dc = True
            print("[QuestManager] Data collection toggle sent")
        elif key == "x":
            toggle_da = True
            print("[QuestManager] Data abort toggle sent")
        elif key == "k" and self.source.is_replay:
            self.source.toggle_pause()
        elif key == "LEFT" and self.source.is_replay:
            self.source.step(-1)
        elif key == "RIGHT" and self.source.is_replay:
            self.source.step(+1)
        return False, toggle_dc, toggle_da

    # -- per-loop helpers ---------------------------------------------------------

    def _update_countdown(self, frame: dict | None) -> None:
        if self.pending_calib is None:
            return
        deadline, kind = self.pending_calib
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            self.pending_calib = None
            self._do_calibration(kind, frame)
        elif int(np.ceil(remaining)) < self.last_countdown_print:
            self.last_countdown_print = int(np.ceil(remaining))
            print(f"[QuestManager] ... {self.last_countdown_print}")

    def _compute_hands(self, frame: dict) -> dict[str, list[float] | None]:
        if not self.finger_tracking:
            return {"left": None, "right": None}
        hands: dict[str, list[float] | None] = {}
        for side in _SIDES:
            if frame[f"{side}_tracked"]:
                try:
                    self.last_hands[side] = self.retargeter(frame[f"{side}_landmarks"], side)
                except Exception as e:
                    print(f"[QuestManager] {side} retargeting error: {e}")
            hands[side] = self.last_hands[side]  # hold last command while untracked
        return hands

    def _apply_pause_resume(
        self, pos: np.ndarray, quat: np.ndarray, yaw_rel: float, hands: dict
    ) -> tuple[np.ndarray, np.ndarray, float, dict]:
        facing_yaw = _wrap(yaw_rel - self.yaw_offset)

        if self.teleop_paused:
            if self.frozen_pos is None:
                self.frozen_pos = pos.copy()
                self.frozen_quat = quat.copy()
                self.frozen_yaw = facing_yaw
                self.frozen_hands = dict(hands)
            return self.frozen_pos, self.frozen_quat, self.frozen_yaw, self.frozen_hands

        if self.resume_ramp_start is not None and self.frozen_pos is not None:
            if self.resume_rebase_pending:
                # Rebase heading: the operator may have turned while paused;
                # continue from the frozen facing instead of jumping.
                self.yaw_offset = _wrap(yaw_rel - self.frozen_yaw)
                self.resume_rebase_pending = False
            facing_yaw = _wrap(yaw_rel - self.yaw_offset)
            alpha = (time.monotonic() - self.resume_ramp_start) / max(
                self.args.resume_ramp_sec, 1e-3
            )
            if alpha >= 1.0:
                self.resume_ramp_start = None
                self.frozen_pos = None
            else:
                pos = (1.0 - alpha) * self.frozen_pos + alpha * pos
                quat = np.concatenate(
                    [
                        _slerp(self.frozen_quat[4 * i : 4 * i + 4], quat[4 * i : 4 * i + 4], alpha)
                        for i in range(3)
                    ]
                )
        return pos, quat, facing_yaw, hands

    def _walk_command(self, head_vel: np.ndarray) -> tuple[int, list[float], float]:
        """Map operator head planar velocity (R0 frame) to a planner movement.

        Returns (locomotion_mode, movement_direction[3], speed). While walking
        is disabled or teleop is paused — and below the speed deadband — this
        is IDLE with a zero movement vector, i.e. exactly the previous
        stationary behavior (the robot still turns via the facing command).
        """
        if self.args.disable_walk or self.teleop_paused:
            self._walking = False
            return LOCOMOTION_IDLE, [0.0, 0.0, 0.0], -1.0

        speed = float(np.hypot(head_vel[0], head_vel[1]))
        # Hysteresis: start above the deadband, keep going until well below it.
        if self._walking:
            self._walking = speed >= 0.5 * self.args.walk_deadband
        else:
            self._walking = speed >= self.args.walk_deadband
        if not self._walking:
            return LOCOMOTION_IDLE, [0.0, 0.0, 0.0], -1.0

        direction = np.asarray(head_vel[:2], dtype=np.float64) / max(speed, 1e-6)
        cmd_speed = float(
            np.clip(
                speed * self.args.walk_speed_scale,
                self.args.walk_min_speed,
                self.args.walk_max_speed,
            )
        )
        return self.walk_mode, [float(direction[0]), float(direction[1]), 0.0], cmd_speed

    # -- main loop ------------------------------------------------------------------

    def run(self) -> None:
        period = 1.0 / max(1, self.args.target_fps)
        print(
            "[QuestManager] Controls: s=start  r=recalibrate  f=fingers  p=pause  "
            "c=collect  x=abort  q=quit"
            + ("  k=replay-pause  arrows=step" if self.source.is_replay else "")
        )
        last_report = time.time()
        sent = 0
        with KeyboardListener() as kb:
            while True:
                t_start = time.monotonic()
                quit_requested = toggle_dc = toggle_da = False
                while not kb.keys.empty():
                    q, dc, da = self._handle_key(kb.keys.get())
                    quit_requested |= q
                    toggle_dc |= dc
                    toggle_da |= da
                if quit_requested:
                    break

                frame = self.source.get_frame()
                self._update_countdown(frame)

                if self.stream_mode == STREAM_MODE_PLANNER_VR_3PT and frame is not None:
                    pos, quat, yaw_rel, head_vel = self.tracker.compute(frame)
                    hands = self._compute_hands(frame)
                    pos, quat, facing_yaw, hands = self._apply_pause_resume(
                        pos, quat, yaw_rel, hands
                    )
                    self.last_facing_yaw = facing_yaw
                    mode, movement, speed = self._walk_command(head_vel)
                    self.pub.send(
                        build_planner_message(
                            mode=mode,
                            movement=movement,
                            facing=[float(np.cos(facing_yaw)), float(np.sin(facing_yaw)), 0.0],
                            speed=speed,
                            height=-1.0,
                            left_hand_position=hands["left"],
                            right_hand_position=hands["right"],
                            vr_3pt_position=pos.tolist(),
                            vr_3pt_orientation=quat.tolist(),
                        )
                    )
                    sent += 1

                self.pub.send(
                    pack_pose_message(
                        {
                            "stream_mode": np.array([self.stream_mode], dtype=np.int32),
                            "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                            "toggle_data_abort": np.array([toggle_da], dtype=bool),
                        },
                        topic="manager_state",
                    )
                )

                now = time.time()
                if now - last_report >= 5.0:
                    state = (
                        "OFF"
                        if self.stream_mode == STREAM_MODE_OFF
                        else ("PAUSED" if self.teleop_paused else "VR_3PT")
                    )
                    data = "ok" if frame is not None else "waiting"
                    walk = (
                        "off"
                        if self.args.disable_walk
                        else ("walk" if self._walking else "idle")
                    )
                    print(
                        f"[QuestManager] {state} | quest={data} | "
                        f"{sent / (now - last_report):.1f} planner msg/s | "
                        f"fingers={'on' if self.finger_tracking else 'off'} | "
                        f"walk={walk}"
                    )
                    last_report = now
                    sent = 0

                elapsed = time.monotonic() - t_start
                if elapsed < period:
                    time.sleep(period - elapsed)

        print("[QuestManager] Sending policy STOP...")
        self.pub.send(build_command_message(start=False, stop=True, planner=True))
        time.sleep(0.1)

    def close(self) -> None:
        self.source.close()
        self.pub.close()
        self.ctx.term()
        print("[QuestManager] Shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quest teleop manager: relay/NPZ -> WBC policy ZMQ stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (deploy subscribes)")
    parser.add_argument("--relay-host", default="localhost", help="quest_relay host")
    parser.add_argument("--relay-port", type=int, default=5559, help="quest_relay ZMQ port")
    parser.add_argument("--feedback-host", default="localhost", help="g1_debug feedback host")
    parser.add_argument("--feedback-port", type=int, default=5557, help="g1_debug feedback port")
    parser.add_argument("--target-fps", type=int, default=50, help="planner stream rate")
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=0.8,
        help="scale on operator arm reach (orientation unaffected)",
    )
    parser.add_argument(
        "--calib-delay-sec",
        type=float,
        default=3.0,
        help="countdown after 's'/'r' before the calibration frame is captured (live only)",
    )
    parser.add_argument(
        "--resume-ramp-sec",
        type=float,
        default=1.0,
        help="ease-in duration from frozen to live pose when resuming after pause",
    )
    parser.add_argument(
        "--disable-walk",
        action="store_true",
        help="disable head-position walking; robot only turns in place (facing) "
        "and never translates, reproducing the pre-walk behavior",
    )
    parser.add_argument(
        "--walk-mode",
        choices=tuple(_WALK_MODES),
        default="slow",
        help="locomotion mode used while the operator walks",
    )
    parser.add_argument(
        "--walk-deadband",
        type=float,
        default=0.08,
        help="operator head speed (m/s) above which a walk command is sent",
    )
    parser.add_argument(
        "--walk-speed-scale",
        type=float,
        default=1.0,
        help="scale from operator head speed to commanded robot walk speed",
    )
    parser.add_argument(
        "--walk-min-speed",
        type=float,
        default=0.2,
        help="lower clamp on commanded walk speed (m/s) once walking",
    )
    parser.add_argument(
        "--walk-max-speed",
        type=float,
        default=0.8,
        help="upper clamp on commanded walk speed (m/s)",
    )
    parser.add_argument(
        "--np-retarget",
        action="store_true",
        help="force the pure-numpy finger retargeter",
    )
    parser.add_argument(
        "--replay", default=None, help="NPZ trajectory to replay instead of live Quest"
    )
    parser.add_argument(
        "--rest-hold-sec",
        type=float,
        default=1.0,
        help="(replay) duration the synthetic rest pose is held before interpolation",
    )
    parser.add_argument(
        "--rest-interp-sec",
        type=float,
        default=1.0,
        help="(replay) duration of the blend from rest pose to the recording's first frame",
    )
    args = parser.parse_args()

    manager = QuestManager(args)
    try:
        manager.run()
    except KeyboardInterrupt:
        print("\n[QuestManager] Interrupted — sending policy STOP...")
        manager.pub.send(build_command_message(start=False, stop=True, planner=True))
        time.sleep(0.1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
