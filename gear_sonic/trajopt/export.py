"""Write a generated reference trajectory in the formats the rest of the stack reads.

Three artifacts per run, in ``<out>/``:

``trajectory.npz``   everything, self-describing: time base, MuJoCo ``qpos`` for the whole
                     scene (robot + chair, directly replayable), the 51-DOF Pinocchio
                     ``whole_q`` the recorder/visualizer use as ``observation.state``, the
                     chair pose in *both* the canonical proxy frame and the staged asset's
                     frame (the latter is what the sim's ``object`` free joint takes), and
                     the contact schedule.
``motion_lib.pkl``   SONIC training format (``root_trans_offset`` / ``root_rot`` /
                     ``dof`` / ``pose_aa`` / ``fps``, see
                     ``data_process/convert_soma_csv_to_motion_lib.py``), with the object
                     trajectory and contact flags carried along as extra keys so an
                     object-aware tracking reward can use them.
``report.json``      what was searched, what was rejected and why, and every feasibility
                     margin of the accepted trajectory.
``scene.xml``        the exact MuJoCo scene the trajectory was generated against.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from gear_sonic.trajopt.scene import BODY_JOINT_NAMES, HAND_JOINT_NAMES, TrajOptScene
from gear_sonic.trajopt.se3 import Pose, rot_exp
from gear_sonic.trajopt.trajectory import Trajectory

# pelvis + 29 actuated links, and the rotation axis of each of those joints; kept in sync
# with data_process/convert_soma_csv_to_motion_lib.py (imported from there when possible).
NUM_BODIES = 30


def _dof_axis() -> np.ndarray:
    """Per-joint rotation axes in MuJoCo order, from the converter if it imports."""
    try:
        from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS

        return np.asarray(DOF_AXIS, dtype=np.float32)
    except Exception:
        # Same table, derived from the model instead (axis of each hinge in its own frame).
        return None


def _whole_q(scene: TrajOptScene, traj: Trajectory) -> Optional[np.ndarray]:
    """(T, 51) Pinocchio configuration, the layout the recorder stores as observation.state."""
    try:
        from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

        rm = instantiate_g1_robot_model(hand_type="brainco")
    except Exception:
        return None
    out = np.zeros((traj.n, rm.num_dofs), dtype=np.float32)
    for j, name in enumerate(BODY_JOINT_NAMES):
        out[:, rm.dof_index(name)] = traj.qj[:, j]
    for k, side in enumerate(("left", "right")):
        for j, name in enumerate(HAND_JOINT_NAMES[side]):
            out[:, rm.dof_index(name)] = traj.hand_q[:, 6 * k + j]
    return out


def scene_qpos(scene: TrajOptScene, traj: Trajectory) -> np.ndarray:
    """(T, nq) full-scene MuJoCo qpos — robot floating base + joints + chair free joint."""
    out = np.zeros((traj.n, scene.model.nq))
    for i in range(traj.n):
        scene.set_state(
            traj.base_p[i], traj.base_rv[i], traj.qj[i],
            {"left": traj.hand_q[i, :6], "right": traj.hand_q[i, 6:]},
            traj.object_pose(i),
        )
        out[i] = scene.qpos()
    return out


def _motion_lib_entry(traj: Trajectory, qpos: np.ndarray, obj_quat: np.ndarray) -> Dict:
    from scipy.spatial.transform import Rotation as R

    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    dof = traj.qj.astype(np.float32)
    pose_aa = np.zeros((traj.n, NUM_BODIES, 3), dtype=np.float32)
    axis = _dof_axis()
    if axis is not None:
        pose_aa[:, 1:NUM_BODIES, :] = axis[None, :, :] * dof[:, :, None]
    pose_aa[:, 0, :] = R.from_quat(root_quat_xyzw).as_rotvec()
    return {
        "root_trans_offset": qpos[:, 0:3].astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),  # xyzw (scipy convention)
        "dof": dof,  # MuJoCo/MJCF joint order, 29 body joints
        "pose_aa": pose_aa,
        "smpl_joints": np.zeros((traj.n, 24, 3), dtype=np.float32),
        "fps": int(round(traj.fps)),
        # Extras beyond the SONIC schema: the manipulated object and the contact schedule.
        "object_trans": traj.obj_p.astype(np.float32),
        "object_rot": obj_quat[:, [1, 2, 3, 0]].astype(np.float32),  # xyzw
        "hand_dof": traj.hand_q.astype(np.float32),
        "contact": traj.contact,
        "foot_contact": traj.foot_contact,
        "stage": traj.stage,
    }


def save(
    out_dir: Path,
    scene: TrajOptScene,
    traj: Trajectory,
    report: Dict,
    name: str = "chair_reference",
) -> Dict[str, Path]:
    """Write every artifact; returns the paths that were written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    qpos = scene_qpos(scene, traj)
    obj_quat = np.stack([Pose(traj.obj_p[i], rot_exp(traj.obj_rv[i])).quat_wxyz()
                         for i in range(traj.n)])
    # The sim's `object` body is the staged asset, whose own frame is a yaw offset from the
    # canonical proxy frame the optimizer works in (ChairSpec.asset_yaw).
    asset_pose = [scene.chair.to_asset_frame(Pose(traj.obj_p[i], rot_exp(traj.obj_rv[i])))
                  for i in range(traj.n)]
    obj_quat_asset = np.stack([p.quat_wxyz() for p in asset_pose])
    obj_pos_asset = np.stack([p.p for p in asset_pose])
    whole_q = _whole_q(scene, traj)

    npz = out_dir / "trajectory.npz"
    payload = dict(
        fps=np.float64(traj.fps),
        t=traj.t,
        qpos=qpos,
        root_pos=qpos[:, 0:3],
        root_quat_wxyz=qpos[:, 3:7],
        dof29=traj.qj,
        hand_dof=traj.hand_q,
        object_pos=traj.obj_p,
        object_quat_wxyz=obj_quat,
        object_pos_asset_frame=obj_pos_asset,
        object_quat_wxyz_asset_frame=obj_quat_asset,
        contact=traj.contact,
        foot_contact=traj.foot_contact,
        foot_placement=traj.foot_place,
        stage=traj.stage,
        stage_names=np.array(traj.stage_names),
        durations=traj.durations,
        body_joint_names=np.array(BODY_JOINT_NAMES),
        hand_joint_names=np.array(
            list(HAND_JOINT_NAMES["left"]) + list(HAND_JOINT_NAMES["right"])
        ),
    )
    if whole_q is not None:
        payload["whole_q"] = whole_q
    np.savez_compressed(npz, **payload)
    written["trajectory"] = npz

    entry = _motion_lib_entry(traj, qpos, obj_quat)
    pkl = out_dir / "motion_lib.pkl"
    try:
        import joblib

        joblib.dump({name: entry}, pkl)
        written["motion_lib"] = pkl
    except Exception as exc:  # pragma: no cover - joblib is in every env here
        report.setdefault("warnings", []).append(f"motion_lib not written: {exc}")

    rep = out_dir / "report.json"
    rep.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    written["report"] = rep

    xml = out_dir / "scene.xml"
    xml.write_text(scene._build_xml(scene.chair, traj.object_pose(0), absolute_include=True))
    written["scene"] = xml
    return written


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
