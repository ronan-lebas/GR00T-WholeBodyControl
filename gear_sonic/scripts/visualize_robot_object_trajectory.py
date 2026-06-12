"""Replay a recorded episode in MuJoCo: G1 robot + FoundationPose-tracked object.

Purely kinematic replay — nothing is actuated and gravity is disabled. Each frame we
set the robot joints from the recording and place the tracked object from its estimated
6D pose, then ``mj_forward`` + ``viewer.sync``.

Data sources for a recording folder ``outputs/<ts>/``:
  - robot joints : ``data/**/episode_*.parquet`` column ``observation.state`` (51-DOF
    Pinocchio whole_q for the brainco G1).
  - object pose  : ``foundation_pose_data/episode_<NNNNNN>/ob_in_cam/NNNNNN.txt`` (4x4
    object-in-camera, OpenCV optical frame) — produced by FoundationPose
    (``foundation_pose/pose_estimation.py``).
  - object mesh  : ``foundation_pose_data/box.obj``.

The robot base keeps the model's default standing *position* (the recording stores no base
world position) but uses the recorded base *orientation* (`observation.root_orientation`,
yaw-zeroed). Getting the orientation right matters: the head camera is pitched down, so a few
degrees of base pitch levers the camera-anchored object by several cm in height — without it the
object visibly sinks into the floor. The object is placed relative to the **live head_camera FK
pose**, so the hand/object geometry stays consistent.

The robot trajectory (50 Hz) and the object trajectory (lower rate) generally differ in
length; with no stored frame map they are aligned by uniform nearest-neighbour
(robot is the master timeline).

Usage (needs gear_sonic[sim] — mujoco, pin, scipy, pyarrow):
    python gear_sonic/scripts/visualize_robot_object_trajectory.py
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --trajectory outputs/2026-06-12-19-32-55
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --episode 0 --check   # headless sanity

Controls: space = pause/resume, left/right arrows = step one frame (while paused).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
SCENE_XML = (
    REPO_ROOT
    / "gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml"
)
# OpenCV optical frame (X-right, Y-down, Z-forward) -> MuJoCo/OpenGL camera frame
# (X-right, Y-up, Z-back). Matches third_party/FoundationPose/Utils.py:glcam_in_cvcam.
GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# GLFW key codes used by the passive viewer key callback.
KEY_SPACE, KEY_RIGHT, KEY_LEFT = 32, 262, 263


# --------------------------------------------------------------------------- #
# Path resolution + data loading
# --------------------------------------------------------------------------- #


def latest_output_dir() -> Path:
    if not OUTPUTS_DIR.is_dir():
        sys.exit(f"[error] outputs directory not found: {OUTPUTS_DIR}")
    subdirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        sys.exit(f"[error] no recording folders found in {OUTPUTS_DIR}")
    return sorted(subdirs)[-1]  # timestamp folders sort lexicographically


def resolve_paths(args) -> tuple[Path, Path, Path, Path]:
    traj = Path(args.trajectory).resolve() if args.trajectory else latest_output_dir()
    if not traj.is_dir():
        sys.exit(f"[error] trajectory folder does not exist: {traj}")

    parquets = sorted((traj / "data").rglob("*.parquet"))
    if not parquets:
        sys.exit(f"[error] no parquet under {traj / 'data'}")
    # Prefer the requested episode's parquet if it can be matched by name.
    parquet = next(
        (p for p in parquets if p.name == f"episode_{args.episode:06d}.parquet"),
        parquets[0],
    )

    fp_data = traj / "foundation_pose_data"
    mesh = fp_data / "box.obj"
    ob_dir = fp_data / f"episode_{args.episode:06d}" / "ob_in_cam"
    if not mesh.is_file():
        sys.exit(f"[error] object mesh not found: {mesh}")
    if not ob_dir.is_dir() or not any(ob_dir.glob("*.txt")):
        sys.exit(
            f"[error] no FoundationPose poses in {ob_dir}\n"
            "        Run foundation_pose/pose_estimation.py on this recording first."
        )
    return traj, parquet, mesh, ob_dir


def load_robot_states(parquet: Path) -> np.ndarray:
    """Return (N, 51) whole_q array from the parquet's observation.state column."""
    table = pq.read_table(parquet, columns=["observation.state"])
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    if states.ndim != 2:
        sys.exit(f"[error] unexpected observation.state shape {states.shape}")
    return states


def load_base_quats(parquet: Path) -> np.ndarray | None:
    """Return (N, 4) base world orientation (wxyz, yaw-zeroed), or None if unavailable.

    Only pitch/roll are kept (yaw is arbitrary/drifting and irrelevant to a fixed-origin
    replay). Pitch/roll are what align the head camera correctly so the object rests on
    the floor instead of sinking into it.
    """
    names = pq.read_table(parquet).column_names
    if "observation.root_orientation" not in names:
        return None
    quats = np.asarray(
        pq.read_table(parquet, columns=["observation.root_orientation"])
        .column(0)
        .to_pylist(),
        dtype=np.float64,
    )
    euler = R.from_quat(quats, scalar_first=True).as_euler("ZYX")
    euler[:, 0] = 0.0  # zero yaw
    return R.from_euler("ZYX", euler).as_quat(scalar_first=True)


def load_object_poses(ob_dir: Path) -> np.ndarray:
    """Return (M, 4, 4) object-in-camera poses, sorted by frame index."""
    files = sorted(ob_dir.glob("*.txt"))
    return np.stack([np.loadtxt(f).reshape(4, 4) for f in files], axis=0)


def read_fps(traj: Path, default: float = 50.0) -> float:
    info = traj / "meta" / "info.json"
    if info.is_file():
        try:
            return float(json.loads(info.read_text()).get("fps", default))
        except Exception:
            pass
    return default


# --------------------------------------------------------------------------- #
# MuJoCo model (robot scene + tracked object mesh)
# --------------------------------------------------------------------------- #


def build_model(mesh_path: Path) -> mujoco.MjModel:
    """Inject the tracked object (mesh + freejoint) into the brainco scene."""
    tree = ET.parse(SCENE_XML)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    mesh_el = ET.SubElement(asset, "mesh")
    mesh_el.set("name", "tracked_object")
    mesh_el.set("file", str(mesh_path.resolve()))

    body = ET.SubElement(root.find("worldbody"), "body")
    body.set("name", "tracked_object")
    ET.SubElement(body, "freejoint")
    geom = ET.SubElement(body, "geom")
    geom.set("type", "mesh")
    geom.set("mesh", "tracked_object")
    geom.set("contype", "0")  # visual only — no collision (we never step physics)
    geom.set("conaffinity", "0")
    geom.set("rgba", "1 0.5 0 0.6")

    # Write next to the scene so the relative <include> stays valid.
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".xml", dir=SCENE_XML.parent
    ) as f:
        tree.write(f.name)
        tmp = f.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)

    model.opt.gravity[:] = 0.0
    return model


def build_joint_map(model: mujoco.MjModel, robot_model) -> list[tuple[int, int]]:
    """(qpos_addr, whole_q_index) for every non-free joint present in the Pinocchio model."""
    mapping = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue  # robot floating base + the object freejoint
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        try:
            didx = robot_model.dof_index(name)
        except Exception:
            continue  # joint not in the Pinocchio model
        mapping.append((int(model.jnt_qposadr[j]), int(didx)))
    return mapping


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


class TrajectoryReplay:
    def __init__(
        self,
        states: np.ndarray,
        obj_poses: np.ndarray,
        mesh_path: Path,
        base_quats: np.ndarray | None = None,
    ):
        self.states = states
        self.obj_poses = obj_poses
        self.base_quats = base_quats
        self.n = states.shape[0]
        self.m = obj_poses.shape[0]

        self.model = build_model(mesh_path)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0  # fixed standing base

        robot_model = instantiate_g1_robot_model(hand_type="brainco")
        self.joint_map = build_joint_map(self.model, robot_model)

        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tracked_object")
        obj_jnt = int(self.model.body_jntadr[obj_body])
        self.obj_qadr = int(self.model.jnt_qposadr[obj_jnt])

    def object_index(self, i: int) -> int:
        if self.n <= 1 or self.m <= 1:
            return 0
        return int(round(i * (self.m - 1) / (self.n - 1)))

    def set_frame(self, i: int) -> None:
        i = int(np.clip(i, 0, self.n - 1))
        whole_q = self.states[i]
        for adr, didx in self.joint_map:
            self.data.qpos[adr] = whole_q[didx]
        # Recorded base orientation (pitch/roll) — keeps the camera correctly tilted so
        # the object rests on the floor. Base position stays at the default qpos0.
        if self.base_quats is not None:
            self.data.qpos[3:7] = self.base_quats[i]
        # Camera FK depends on the robot pose set above.
        mujoco.mj_forward(self.model, self.data)

        t_wc = np.eye(4)
        t_wc[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        t_wc[:3, 3] = self.data.cam_xpos[self.cam_id]
        t_wo = t_wc @ GL_FROM_CV @ self.obj_poses[self.object_index(i)]

        self.data.qpos[self.obj_qadr : self.obj_qadr + 3] = t_wo[:3, 3]
        self.data.qpos[self.obj_qadr + 3 : self.obj_qadr + 7] = R.from_matrix(
            t_wo[:3, :3]
        ).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model, self.data)

    def run(self, fps: float) -> None:
        state = {"frame": 0.0, "paused": False, "step": 0}

        def key_callback(key: int) -> None:
            if key == KEY_SPACE:
                state["paused"] = not state["paused"]
                print(f"[replay] {'paused' if state['paused'] else 'playing'}")
            elif key == KEY_RIGHT:
                state["step"] += 1
            elif key == KEY_LEFT:
                state["step"] -= 1

        print(
            f"[replay] {self.n} robot frames, {self.m} object frames @ {fps:.0f} fps | "
            "space=pause  arrows=step"
        )
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_callback,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            self.set_frame(0)
            last = time.time()
            while viewer.is_running():
                now = time.time()
                dt = now - last
                last = now
                if state["paused"]:
                    if state["step"]:
                        idx = (int(round(state["frame"])) + state["step"]) % self.n
                        state["frame"] = float(idx)
                        state["step"] = 0
                        self.set_frame(idx)
                        viewer.sync()
                else:
                    state["frame"] = (state["frame"] + dt * fps) % self.n
                    self.set_frame(int(state["frame"]))
                    viewer.sync()
                time.sleep(max(0.0, 1.0 / fps - (time.time() - now)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trajectory", default=None,
        help="recording folder under outputs/ (default: most recent)",
    )
    parser.add_argument("--episode", type=int, default=0, help="episode index (default: 0)")
    parser.add_argument(
        "--check", action="store_true",
        help="headless sanity check (no viewer): print stats for the first/last frame",
    )
    args = parser.parse_args()

    traj, parquet, mesh, ob_dir = resolve_paths(args)
    states = load_robot_states(parquet)
    base_quats = load_base_quats(parquet)
    obj_poses = load_object_poses(ob_dir)
    fps = read_fps(traj)

    print(f"[info] trajectory : {traj}")
    print(f"[info] parquet    : {parquet.relative_to(traj)} ({states.shape[0]} frames)")
    print(f"[info] object     : {ob_dir.relative_to(traj)} ({obj_poses.shape[0]} frames)")

    replay = TrajectoryReplay(states, obj_poses, mesh, base_quats)

    if args.check:
        for i in (0, replay.n - 1):
            replay.set_frame(i)
            obj_pos = replay.data.qpos[replay.obj_qadr : replay.obj_qadr + 3].copy()
            cam_pos = replay.data.cam_xpos[replay.cam_id].copy()
            print(
                f"[check] frame {i:4d} -> obj idx {replay.object_index(i):3d} | "
                f"cam {np.round(cam_pos, 3)} | obj_world {np.round(obj_pos, 3)} | "
                f"obj-cam dist {np.linalg.norm(obj_pos - cam_pos):.3f} m"
            )
        applied = sum(
            replay.data.qpos[adr] != replay.model.qpos0[adr] for adr, _ in replay.joint_map
        )
        print(f"[check] joints mapped: {len(replay.joint_map)} (nonzero-vs-default: {applied})")
        print("[check] OK")
        return

    replay.run(fps)


if __name__ == "__main__":
    main()
