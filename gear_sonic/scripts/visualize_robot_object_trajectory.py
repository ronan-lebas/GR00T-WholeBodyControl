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

The recording stores no base world *position*, so the base translation is solved each frame to
keep the feet planted on the floor (during recording the feet — not the pelvis — are fixed). The
base *orientation* comes from the recording (`observation.root_orientation`) with only the
initial yaw removed: the yaw *variation* is real camera azimuth motion and must be kept, or the
static object appears to swing in azimuth as the robot turns. Orientation accuracy matters: the
head camera is pitched down, so a few degrees of base pitch levers the camera-anchored object by
several cm. The object is placed relative to the **live head_camera FK pose**, so the
hand/object geometry stays consistent.

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


def resolve_paths(args) -> tuple[Path, Path, Path | None, Path | None]:
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

    # Ground truth (sim-only): poses come from a single object_gt parquet, not the
    # FoundationPose per-frame txt folders. The mesh stays the shared box.obj.
    if getattr(args, "ground_truth", False):
        gt_parquet = traj / "object_gt" / f"episode_{args.episode:06d}.parquet"
        if not mesh.is_file() or not gt_parquet.is_file():
            print(
                "[info] no ground-truth object data "
                f"(mesh={mesh.is_file()}, gt={gt_parquet.is_file()}); replaying robot only"
            )
            return traj, parquet, None, None
        return traj, parquet, mesh, gt_parquet

    episode_dir = fp_data / f"episode_{args.episode:06d}"
    sub = "ob_in_world_filtered" if getattr(args, "filtered", False) else "ob_in_cam"
    ob_dir = episode_dir / sub
    if getattr(args, "filtered", False) and not ob_dir.is_dir():
        sys.exit(
            f"[error] no filtered poses in {ob_dir}\n"
            "        Run foundation_pose/filter_object_pose.py on this recording first."
        )
    if not mesh.is_file() or not ob_dir.is_dir() or not any(ob_dir.glob("*.txt")):
        print(
            "[info] no object data found "
            f"(mesh={mesh.is_file()}, poses in {ob_dir}={ob_dir.is_dir() and any(ob_dir.glob('*.txt'))}); "
            "replaying robot only"
        )
        return traj, parquet, None, None
    return traj, parquet, mesh, ob_dir


def load_robot_states(parquet: Path) -> np.ndarray:
    """Return (N, 51) whole_q array from the parquet's observation.state column."""
    table = pq.read_table(parquet, columns=["observation.state"])
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    if states.ndim != 2:
        sys.exit(f"[error] unexpected observation.state shape {states.shape}")
    return states


def load_base_quats(parquet: Path) -> np.ndarray | None:
    """Return (N, 4) base world orientation (wxyz), with only the INITIAL yaw removed.

    The *absolute* yaw is arbitrary (the recording has no world frame), but its *variation*
    over the trajectory is real camera azimuth motion: when the robot turns, the head camera
    sweeps and a stationary object sweeps across the image with it. That rotation must be
    reproduced here so it cancels when the object is projected back to world — otherwise the
    (truly static) object appears to swing in azimuth. So we subtract only frame 0's yaw (a
    constant offset that merely orients the replay) and keep the per-frame yaw delta. Pitch/roll
    are kept as-is (they also tilt the head camera so the object rests on the floor).
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
    euler[:, 0] -= euler[0, 0]  # remove only the initial yaw; keep the yaw *variation*
    return R.from_euler("ZYX", euler).as_quat(scalar_first=True)


def load_object_poses(ob_dir: Path) -> np.ndarray:
    """Return (M, 4, 4) object-in-camera poses, sorted by frame index."""
    files = sorted(ob_dir.glob("*.txt"))
    return np.stack([np.loadtxt(f).reshape(4, 4) for f in files], axis=0)


def load_object_gt(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ((M, 4, 4) object-in-head-camera poses, (M,) proprio-row indices).

    Ground truth recorded by ObjectGtWriter: exact MuJoCo box poses in the head_camera
    frame, one row per recorded frame. The proprio_frame_index column *is* the object->
    robot frame map (rows are dense and monotonic), so it feeds object_index directly.
    """
    table = pq.read_table(parquet, columns=["proprio_frame_index", "ob_in_cam"])
    idx = np.asarray(table.column("proprio_frame_index").to_pylist(), dtype=int)
    poses = np.asarray(table.column("ob_in_cam").to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
    return poses, idx


def load_frame_map(ob_dir: Path, n_obj: int) -> np.ndarray | None:
    """Return (M,) proprio/parquet row index per object frame, or None if absent.

    FP frames are recorded sparser than the proprio stream, so this exact map
    (written by FoundationPoseWriter) replaces uniform nearest-neighbour pairing.
    Old recordings without ``frame_map.txt`` fall back to uniform resampling.
    """
    path = ob_dir.parent / "frame_map.txt"
    if not path.is_file():
        return None
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        return None
    proprio = rows[:, 1].astype(int)
    if proprio.shape[0] != n_obj:
        print(
            f"[warn] frame_map has {proprio.shape[0]} rows but {n_obj} object poses; "
            "ignoring map and falling back to uniform resampling"
        )
        return None
    return proprio


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


def build_model(mesh_path: Path | None) -> mujoco.MjModel:
    """Inject the tracked object (mesh + freejoint) into the brainco scene, if a mesh is given."""
    tree = ET.parse(SCENE_XML)
    root = tree.getroot()

    if mesh_path is not None:
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
        obj_poses: np.ndarray | None,
        mesh_path: Path | None,
        base_quats: np.ndarray | None = None,
        obj_to_robot: np.ndarray | None = None,
        obj_in_world: bool = False,
        obj_gt_cam: bool = False,
    ):
        self.states = states
        self.obj_poses = obj_poses
        self.base_quats = base_quats
        self.obj_to_robot = obj_to_robot
        # If True, obj_poses are already 4x4 world poses (filtered) and are placed
        # directly; otherwise they are object-in-camera and go through camera FK.
        self.obj_in_world = obj_in_world
        # If True, obj_poses are ground-truth object-in-camera in MuJoCo camera frame
        # (no OpenCV-optical conversion) — placed via camera FK without GL_FROM_CV.
        self.obj_gt_cam = obj_gt_cam
        self.n = states.shape[0]
        self.has_object = obj_poses is not None
        self.m = obj_poses.shape[0] if self.has_object else 0

        self.model = build_model(mesh_path)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0  # fixed standing base

        robot_model = instantiate_g1_robot_model(hand_type="brainco")
        self.joint_map = build_joint_map(self.model, robot_model)

        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        self.obj_qadr: int | None = None
        if self.has_object:
            obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tracked_object")
            obj_jnt = int(self.model.body_jntadr[obj_body])
            self.obj_qadr = int(self.model.jnt_qposadr[obj_jnt])

        # During recording the feet are planted on the floor while the pelvis translates;
        # the recording stores no base world position, so we keep the feet planted by solving
        # the (only) free base DOF — translation — to hold the feet midpoint at its frame-0
        # location (see set_frame). Anchoring to the feet rather than pinning the pelvis is what
        # stops the feet sliding and the object drifting with the unmodelled pelvis sway. The
        # target is taken from frame 0 so it is identical for the visualizer and the filter
        # (which both go through set_frame), keeping their world frames aligned.
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        self.foot_target: np.ndarray | None = None
        if all(fid >= 0 for fid in self.foot_ids):
            self._apply_robot_pose(0)
            self.foot_target = 0.5 * (
                self.data.xpos[self.foot_ids[0]] + self.data.xpos[self.foot_ids[1]]
            ).copy()
        else:
            self.foot_ids = None

    def _apply_robot_pose(self, i: int) -> None:
        """Set robot joints + base orientation for frame i and run FK (no foot anchor/object)."""
        i = int(np.clip(i, 0, self.n - 1))
        whole_q = self.states[i]
        for adr, didx in self.joint_map:
            self.data.qpos[adr] = whole_q[didx]
        # Recorded base orientation (pitch/roll + relative yaw) reproduces how the head camera
        # actually swept during recording, so the static object cancels out on reprojection.
        if self.base_quats is not None:
            self.data.qpos[3:7] = self.base_quats[i]
        mujoco.mj_forward(self.model, self.data)

    def object_index(self, i: int) -> int:
        if self.obj_to_robot is not None:
            # Hold-last: the latest object frame whose proprio row is <= i. This
            # uses the exact FP<->proprio map instead of uniform resampling.
            j = int(np.searchsorted(self.obj_to_robot, i, side="right")) - 1
            return int(np.clip(j, 0, self.m - 1))
        if self.n <= 1 or self.m <= 1:
            return 0
        return int(round(i * (self.m - 1) / (self.n - 1)))

    def set_frame(self, i: int) -> None:
        i = int(np.clip(i, 0, self.n - 1))
        self._apply_robot_pose(i)

        # Keep the feet planted: translate the (only) free base DOF so the feet midpoint
        # returns to its frame-0 world location. Pinning the pelvis instead would let the feet
        # slide and inject the pelvis's unmodelled translation into the camera (hence object) pose.
        if self.foot_target is not None:
            mid = 0.5 * (self.data.xpos[self.foot_ids[0]] + self.data.xpos[self.foot_ids[1]])
            self.data.qpos[0:3] += self.foot_target - mid
            mujoco.mj_forward(self.model, self.data)

        if not self.has_object:
            mujoco.mj_forward(self.model, self.data)
            return

        obj = self.obj_poses[self.object_index(i)]
        if self.obj_in_world:
            # Filtered poses are already world 4x4 — place directly, NOT through the
            # per-frame camera FK (which would re-inject the camera wobble).
            t_wo = obj
        else:
            t_wc = np.eye(4)
            t_wc[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            t_wc[:3, 3] = self.data.cam_xpos[self.cam_id]
            # Ground truth is already in the MuJoCo camera frame; FoundationPose poses are
            # in the OpenCV optical frame and need GL_FROM_CV first.
            t_wo = t_wc @ obj if self.obj_gt_cam else t_wc @ GL_FROM_CV @ obj

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
        "--filtered", action="store_true",
        help="use ob_in_world_filtered/ (from filter_object_pose.py) instead of raw ob_in_cam/",
    )
    parser.add_argument(
        "--ground-truth", dest="ground_truth", action="store_true",
        help="use the sim ground-truth object pose (object_gt/episode_*.parquet) instead of "
        "the FoundationPose estimate — sim-only, for comparison / exact replay",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="headless sanity check (no viewer): print stats for the first/last frame",
    )
    args = parser.parse_args()

    if args.ground_truth and args.filtered:
        sys.exit("[error] --ground-truth and --filtered are mutually exclusive")

    traj, parquet, mesh, ob_src = resolve_paths(args)
    states = load_robot_states(parquet)
    base_quats = load_base_quats(parquet)
    if ob_src is None:
        obj_poses, obj_to_robot = None, None
    elif args.ground_truth:
        obj_poses, obj_to_robot = load_object_gt(ob_src)
    else:
        obj_poses = load_object_poses(ob_src)
        obj_to_robot = load_frame_map(ob_src, obj_poses.shape[0])
    fps = read_fps(traj)

    print(f"[info] trajectory : {traj}")
    print(f"[info] parquet    : {parquet.relative_to(traj)} ({states.shape[0]} frames)")
    if obj_poses is not None:
        print(f"[info] object     : {ob_src.relative_to(traj)} ({obj_poses.shape[0]} frames)")
        if args.ground_truth:
            print("[info] frame map  : exact (object_gt proprio_frame_index)")
            print("[info] object pose: ground truth, camera frame, via FK")
        else:
            fmap = "exact (frame_map.txt)" if obj_to_robot is not None else "uniform resampling (no map)"
            pose = "world frame, placed directly (filtered)" if args.filtered else "camera frame, via FK"
            print(f"[info] frame map  : {fmap}")
            print(f"[info] object pose: {pose}")
    else:
        print("[info] object     : none — replaying robot body only")

    replay = TrajectoryReplay(
        states, obj_poses, mesh, base_quats, obj_to_robot,
        obj_in_world=args.filtered, obj_gt_cam=args.ground_truth,
    )

    if args.check:
        for i in (0, replay.n - 1):
            replay.set_frame(i)
            cam_pos = replay.data.cam_xpos[replay.cam_id].copy()
            if replay.has_object:
                obj_pos = replay.data.qpos[replay.obj_qadr : replay.obj_qadr + 3].copy()
                print(
                    f"[check] frame {i:4d} -> obj idx {replay.object_index(i):3d} | "
                    f"cam {np.round(cam_pos, 3)} | obj_world {np.round(obj_pos, 3)} | "
                    f"obj-cam dist {np.linalg.norm(obj_pos - cam_pos):.3f} m"
                )
            else:
                print(f"[check] frame {i:4d} | cam {np.round(cam_pos, 3)} | no object data")
        applied = sum(
            replay.data.qpos[adr] != replay.model.qpos0[adr] for adr, _ in replay.joint_map
        )
        print(f"[check] joints mapped: {len(replay.joint_map)} (nonzero-vs-default: {applied})")
        print("[check] OK")
        return

    replay.run(fps)


if __name__ == "__main__":
    main()
