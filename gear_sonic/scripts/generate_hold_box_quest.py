#!/usr/bin/env python3
"""Generate a synthetic "hold a box and sway it" Quest trajectory (joint-space).

Produces a single ``.npz`` in the exact ``record_quest_data`` schema (so it
replays through ``quest_manager_thread_server.py --replay``), driving the G1 to
hold the joint-space ``arm_pose`` from ``hold_box_params.yaml`` with a small
sinusoid added to one shoulder joint (``motion``). Combined with
``run_sim_loop.py --held-box`` (which anchors the box to the two-hand midpoint),
this moves a box in the robot's hands so the FoundationPose pipeline can be
exercised on the real in-hand use case: no headset, no grasping.

How a joint pose becomes a replayable trajectory
------------------------------------------------
We have forward kinematics but not a convenient IK, and the replay drives the WBC
policy with wrist *targets*, not joint angles. So we author the motion in joint
space, run FK on each frame to get the desired wrist LINK pose, and convert that
to the Quest wrist pose the policy will track.

The replay (``ReplaySource``) prepends a rest prefix and calibrates on its frame 0
against the robot FK rest pose, giving ``v_cal = link_ref - head_ref`` and a
runtime command ``link = link_ref + pos_scale * (v - v_cal)``. With the head held
static at identity yaw (R0 = I) we therefore emit, per frame and per side,

    wrist_pos  = head_pos0 + (link_ref - head_ref) + (link_fk - link_ref) / pos_scale
    wrist_quat = R_fk * N                      # N = wrist link -> physical hand frame

so the commanded wrist link lands exactly at the FK pose ``link_fk`` of the
authored joint configuration. ``pos_scale`` MUST match the manager's
``--pos-scale`` (default 0.8).

Run it (in .venv_sim):
    python gear_sonic/scripts/generate_hold_box_quest.py

Then replay with the base frozen and fingers tracked:
    python gear_sonic/scripts/quest_manager_thread_server.py \
        --replay data/quest/hold_box_<stamp>.npz --static-base --pos-scale 0.8
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from gear_sonic.scripts.quest_manager_thread_server import RobotRestReference
from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import G1_KEY_FRAME_OFFSETS

# Reuse the synthetic-landmark machinery from the mock Quest generator (pure
# numpy; lives outside the package, so add its directory to sys.path).
_QUEST_RELAY_DIR = (
    Path(__file__).resolve().parents[2] / "gear_sonic_deploy" / "docker" / "quest_relay"
)
sys.path.insert(0, str(_QUEST_RELAY_DIR))
import generate_mock_quest_data as gmq  # noqa: E402

_SIDES = ("left", "right")
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_ARRAY_FIELDS = gmq._ARRAY_FIELDS

# Left -> right arm mirror mask over the 7 arm joints
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw].
# Joints about Y (pitch/elbow/wrist_pitch) keep sign; joints about X/Z (roll/yaw) flip
# across the sagittal plane — matches the model's mirrored shoulder_roll joint ranges.
ARM_MIRROR_MASK = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def arm_pose_deg(params: dict) -> dict[str, np.ndarray]:
    """Left arm from the YAML; right arm as its mirror image. Returns degrees per side."""
    left = np.asarray(params["arm_pose"]["left_deg"], dtype=np.float64)
    return {"left": left, "right": left * ARM_MIRROR_MASK}

# Index of each arm's 7 joints within the 29-element body-actuated vector
# (see instantiate_g1_robot_model body_actuated_joints ordering).
_ARM_SLICE = {"left": slice(15, 22), "right": slice(22, 29)}

# Head held static at identity orientation so R0 = I (see module docstring).
_HEAD_POS0 = np.array([0.0, 0.0, 1.40])
_HEAD_QUAT0 = np.array([1.0, 0.0, 0.0, 0.0])


def _params_path(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    return Path(__file__).resolve().parent / "hold_box_params.yaml"


def generate(params: dict) -> dict[str, np.ndarray]:
    """Build the stacked-array trajectory dict for the hold + sine motion."""
    m = params["motion"]
    curl = float(params["grip"]["curl"])

    pos_scale = float(m["pos_scale"])
    hz = float(m["hz"])
    n = max(2, int(round(float(m["duration"]) * hz)))
    t = np.arange(n) / hz

    ji = int(m["joint_index"])
    amp = np.deg2rad(float(m["amplitude_deg"]))
    sine = amp * np.sin(2.0 * np.pi * t / max(1e-3, float(m["period"])))
    signs = {"left": 1.0, "right": -1.0 if m.get("mirror") else 1.0}

    pose_deg = arm_pose_deg(params)

    ref = RobotRestReference()
    rm = ref._robot_model
    default_29 = np.asarray(rm.get_body_actuated_joints(rm.default_body_pose), dtype=np.float64)
    base_29 = default_29.copy()
    for side in _SIDES:
        base_29[_ARM_SLICE[side]] = np.deg2rad(pose_deg[side])

    # Rest reference (root frame): wrist LINK position + head ref + hand convention N.
    fk0 = ref.compute(None)
    head_ref = np.asarray(fk0["torso_pos"], dtype=np.float64) + np.asarray(
        G1_KEY_FRAME_OFFSETS["torso"], dtype=np.float64
    )
    link_ref = {s: np.asarray(fk0[s][0], dtype=np.float64) for s in _SIDES}
    n_conv = {s: ref.hand_convention(s) for s in _SIDES}

    out: dict[str, np.ndarray] = {
        "timestamp": time.time() + t.astype(np.float64),
        "head_pos": np.tile(_HEAD_POS0, (n, 1)),
        "head_quat": np.tile(_HEAD_QUAT0, (n, 1)),
    }
    for side in _SIDES:
        out[f"{side}_wrist_pos"] = np.zeros((n, 3))
        out[f"{side}_wrist_quat"] = np.zeros((n, 4))
        out[f"{side}_tracked"] = np.ones(n, dtype=bool)

    # FK the authored joint trajectory frame by frame -> wrist targets.
    for i in range(n):
        q29 = base_29.copy()
        for side in _SIDES:
            q29[_ARM_SLICE[side]][ji] += signs[side] * sine[i]
        fk = ref.compute(q29)
        for side in _SIDES:
            link_fk = np.asarray(fk[side][0], dtype=np.float64)
            r_fk = fk[side][1]
            v = (link_ref[side] - head_ref) + (link_fk - link_ref[side]) / pos_scale
            out[f"{side}_wrist_pos"][i] = _HEAD_POS0 + v  # R0 = I
            out[f"{side}_wrist_quat"][i] = (r_fk * n_conv[side]).as_quat(scalar_first=True)

    # Synthetic closed-grip landmarks (constant curl), placed at the per-frame wrist
    # pose in the landmarks-topic frame, exactly as generate_mock_quest_data does. ### The hand curl doesn't work for some reason
    rest_canon = gmq._canonical_hand()
    curls = {f: curl for f in _FINGERS}
    hand_local = {
        "right": gmq._curl_hand(rest_canon, curls),
        "left": (gmq._REFLECT_Y @ gmq._curl_hand(rest_canon, curls).T).T,
    }
    for side in _SIDES:
        rpalm = gmq._R_PALM[side]
        wpos = out[f"{side}_wrist_pos"]
        wquat = out[f"{side}_wrist_quat"]
        lm = np.zeros((n, 21, 3))
        local = hand_local[side]
        for i in range(n):
            w = (gmq._quat_to_matrix(wquat[i]) @ rpalm @ local.T).T + wpos[i]
            lm[i] = (gmq._R_TOPIC @ w.T).T
        out[f"{side}_landmarks"] = lm

    return {k: out[k] for k in _ARRAY_FIELDS}


def save(traj: dict[str, np.ndarray], out_dir: Path, params: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"hold_box_{stamp}.npz"
    meta = json.dumps(
        {
            "num_frames": int(traj["timestamp"].shape[0]),
            "hz": float(params["motion"]["hz"]),
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "synthetic": True,
            "style": "hold_box",
            "hold_box_params": params,
        }
    )
    np.savez_compressed(path, metadata=np.array(meta), **traj)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a hold-a-box Quest trajectory .npz from a joint-space pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params", default=None, help="Path to hold_box_params.yaml (default: alongside this script)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/quest"), help="Where to write the .npz."
    )
    args = parser.parse_args()

    with open(_params_path(args.params)) as f:
        params = yaml.safe_load(f)

    traj = generate(params)
    path = save(traj, args.output_dir, params)

    m = params["motion"]
    print(
        f"[gen_hold_box] Saved {path} "
        f"({traj['timestamp'].shape[0]} frames, {m['duration']}s, {m['hz']} Hz).\n"
        f"  joint {m['joint_index']} sine +/-{m['amplitude_deg']} deg @ {m['period']}s "
        f"(mirror={bool(m.get('mirror'))}); grip curl={params['grip']['curl']}.\n"
        f"  Replay with: python gear_sonic/scripts/quest_manager_thread_server.py "
        f"--replay {path} --static-base --pos-scale {m['pos_scale']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
