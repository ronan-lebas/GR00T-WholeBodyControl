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

Two ways to place the robot base:

  - **legacy (default)** — the LeRobot recording stores no base world *position*, so the base
    translation is solved each frame to keep the feet planted on the floor (during recording the
    feet — not the pelvis — are fixed). The base *orientation* comes from the recording
    (`observation.root_orientation`) with only the initial yaw removed: the yaw *variation* is real
    camera azimuth motion and must be kept, or the static object appears to swing in azimuth as the
    robot turns. Orientation accuracy matters: the head camera is pitched down, so a few degrees of
    base pitch levers the camera-anchored object by several cm. With `--ground-truth` the box is
    mapped into this feet-planted world by a single constant frame-0 anchor.

  - **`--exact-base`** (needs `--ground-truth` + a recording with the `pelvis_in_world` column,
    see new_data_collection_report.md) — the robot base is placed **directly** at its recorded true
    world pose (full orientation + translation) and the box **directly** at `ob_in_world`. Both are
    in the true sim world, so this is an exact reproduction of the sim: no feet-planting, no
    yaw-zeroing, no anchor. Prefer this for new recordings.

The robot trajectory (50 Hz) and the object trajectory (lower rate) generally differ in
length; with no stored frame map they are aligned by uniform nearest-neighbour
(robot is the master timeline).

Usage (needs gear_sonic[sim] — mujoco, pin, scipy, pyarrow):
    python gear_sonic/scripts/visualize_robot_object_trajectory.py
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --trajectory outputs/2026-06-12-19-32-55
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --episode 0 --check   # headless sanity
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --ground-truth --contacts  # + red contact dots
    python gear_sonic/scripts/visualize_robot_object_trajectory.py --ground-truth --exact-base  # exact sim replay

With ``--contacts`` the finger<->object contact points from the ``contacts/`` sidecar (written by
``gear_sonic/scripts/process_contacts.py``) are overlaid as red spheres. They are stored in the
object-local frame, so they ride on the box surface; pair with ``--ground-truth`` (the pose they
were computed against) for the exact geometry.

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
    # object.obj is the staged mesh asset (--object-asset recordings); box.obj is the synthesized
    # cube of the tabletop task. Older recordings only have box.obj.
    mesh = fp_data / "object.obj"
    if not mesh.is_file():
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


def load_object_gt(
    parquet: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Return ((M,4,4) box-in-world, (M,4,4) ref-body-in-world, (M,4,4) pelvis-in-world|None,
    (M,) proprio rows).

    Ground truth recorded by ObjectGtWriter: exact MuJoCo box poses in the *world* frame
    (so a static box stays static — no robot-link motion is folded in), plus the reference
    body's world pose at the same instant, used once to anchor the sim world to the replay's
    feet-planted world. The proprio_frame_index column *is* the object->robot frame map (rows
    are dense and monotonic), so it feeds object_index directly.

    ``pelvis_in_world`` is present only in recordings made after the reset-state schema change
    (see new_data_collection_report.md); it is the robot's true floating-base world pose and lets
    ``--exact-base`` place the base directly instead of feet-planting. Returns None when absent.
    """
    cols = pq.read_table(parquet).column_names
    want = ["proprio_frame_index", "ob_in_world", "ref_in_world"]
    if "pelvis_in_world" in cols:
        want.append("pelvis_in_world")
    table = pq.read_table(parquet, columns=want)
    idx = np.asarray(table.column("proprio_frame_index").to_pylist(), dtype=int)
    box = np.asarray(table.column("ob_in_world").to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
    ref = np.asarray(table.column("ref_in_world").to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
    pelvis = (
        np.asarray(table.column("pelvis_in_world").to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
        if "pelvis_in_world" in cols
        else None
    )
    return box, ref, pelvis, idx


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


def load_contacts(traj: Path, episode: int, n_frames: int) -> np.ndarray | None:
    """Return (N, S, 3) per-frame object-local contact points (NaN where no contact), or None.

    Reads the ``contacts/episode_*.parquet`` sidecar written by ``process_contacts.py``. Points
    are stored in the OBJECT-LOCAL frame; the replay converts them to world each frame with the
    placed box pose (see ``set_frame``), so the drawn dots stay glued to the box surface whatever
    pose mode is used. The parquet is one row per robot frame keyed by ``proprio_frame_index``, so
    we scatter into a robot-frame-aligned array. Contacts pair naturally with ``--ground-truth``
    (the pose they were computed against), but also render fine over the filtered/FP box.
    """
    path = traj / "contacts" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=["proprio_frame_index", "contact_points"])
    idx = np.asarray(table.column("proprio_frame_index").to_pylist(), dtype=int)
    pts = np.asarray(table.column("contact_points").to_pylist(), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] % 3 != 0:
        print(f"[warn] unexpected contact_points shape {pts.shape}; ignoring contacts")
        return None
    s = pts.shape[1] // 3
    pts = pts.reshape(-1, s, 3)
    out = np.full((n_frames, s, 3), np.nan)
    valid = (idx >= 0) & (idx < n_frames)
    out[idx[valid]] = pts[valid]
    return out


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


def build_model(
    mesh_path: Path | None,
    collidable_object: bool = False,
    object_margin: float = 0.0,
) -> mujoco.MjModel:
    """Inject the tracked object (mesh + freejoint) into the brainco scene, if a mesh is given.

    By default the object is *visual only* (contype/conaffinity 0): the visualizer and the pose
    filter never step physics, so it must not collide. Pass ``collidable_object=True`` to instead
    make it collidable with all-ones contype/conaffinity (so it overlaps whatever collision bitmask
    the hand geoms carry) and ``margin=object_margin`` — this lets ``mj_forward``'s collision phase
    report near-contacts within ``object_margin`` (used by ``process_contacts.py`` to recover
    finger<->object contacts offline, without stepping physics).
    """
    tree = ET.parse(SCENE_XML)
    root = tree.getroot()

    if mesh_path is not None:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        body = ET.SubElement(root.find("worldbody"), "body")
        body.set("name", "tracked_object")
        ET.SubElement(body, "freejoint")

        # A non-convex object (chair) is recorded together with its convex decomposition,
        # since MuJoCo collides meshes as convex hulls — colliding against the single visual
        # mesh would report contacts against the chair's overall hull, i.e. thin air.
        hulls = sorted(mesh_path.parent.glob("object_collision_*.stl"))
        collision_meshes = [(f"tracked_object_col_{i}", p) for i, p in enumerate(hulls)]
        if collidable_object and collision_meshes:
            visual_names = [("tracked_object", mesh_path)]
        else:
            collision_meshes = [("tracked_object", mesh_path)]
            visual_names = []

        for name, path in collision_meshes + visual_names:
            mesh_el = ET.SubElement(asset, "mesh")
            mesh_el.set("name", name)
            mesh_el.set("file", str(path.resolve()))

        for name, _ in collision_meshes:
            geom = ET.SubElement(body, "geom")
            geom.set("type", "mesh")
            geom.set("mesh", name)
            if collidable_object:
                # All-ones masks overlap both hands' bitmasks (left contype=2/conaffinity=5,
                # right contype=4/conaffinity=3); margin makes mj_forward emit contacts up to
                # `object_margin` away. Callers filter results to object<->finger pairs by body.
                all_ones = str((1 << 31) - 1)
                geom.set("contype", all_ones)
                geom.set("conaffinity", all_ones)
                geom.set("margin", str(object_margin))
                geom.set("rgba", "1 0.5 0 0.4")
            else:
                geom.set("contype", "0")  # visual only — no collision (we never step physics)
                geom.set("conaffinity", "0")
                geom.set("rgba", "1 0.5 0 0.6")

        for name, _ in visual_names:
            geom = ET.SubElement(body, "geom")
            geom.set("type", "mesh")
            geom.set("mesh", name)
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("rgba", "1 0.5 0 0.4")

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
        obj_gt_ref: bool = False,
        ref_poses: np.ndarray | None = None,
        collidable_object: bool = False,
        object_margin: float = 0.0,
        contacts_local: np.ndarray | None = None,
        pelvis_poses: np.ndarray | None = None,
        exact_base: bool = False,
    ):
        self.states = states
        self.obj_poses = obj_poses
        self.base_quats = base_quats
        self.obj_to_robot = obj_to_robot
        # (N, S, 3) per-frame contact points in the OBJECT-LOCAL frame (NaN where no contact),
        # from process_contacts.py. Converted to world in set_frame and drawn as red spheres.
        self.contacts_local = contacts_local
        self.current_contact_points = np.zeros((0, 3))
        # If True, obj_poses are already 4x4 world poses (filtered) and are placed
        # directly; otherwise they are object-in-camera and go through camera FK.
        self.obj_in_world = obj_in_world
        # If True, obj_poses are ground-truth box-in-*world* poses and ref_poses are the
        # reference body's world pose at the same instant. The box is placed via a single
        # constant anchor (self.gt_anchor, built at frame 0) that maps the sim world to the
        # replay's feet-planted world — so a static box stays exactly static and the box's
        # true motion is reproduced with no per-frame robot motion leaking into the cube.
        self.obj_gt_ref = obj_gt_ref
        self.ref_poses = ref_poses
        self.gt_anchor = np.eye(4)
        # Exact-base mode: place the robot's floating base directly from the recorded
        # pelvis_in_world (full orientation + translation, true W_sim) instead of the
        # feet-planting + yaw-zeroing reconstruction. The box then goes straight to ob_in_world
        # (gt_anchor stays identity) since robot and box are both in the true sim world — an
        # exact reproduction of the sim, no anchor, no feet-planting approximation. Requires the
        # ground-truth path and the new pelvis_in_world column.
        self.pelvis_poses = pelvis_poses
        self.exact_base = bool(exact_base) and pelvis_poses is not None
        self.n = states.shape[0]
        self.has_object = obj_poses is not None
        self.m = obj_poses.shape[0] if self.has_object else 0

        self.model = build_model(mesh_path, collidable_object, object_margin)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0  # fixed standing base

        robot_model = instantiate_g1_robot_model(hand_type="brainco")
        self.joint_map = build_joint_map(self.model, robot_model)
        # BrainCo fingers are underactuated: each *_distal joint is passive, driven by its
        # *_proximal joint through a mimic coupling (multiplier ~1.155 for fingers, 1.0 for the
        # thumb). The recorded whole_q stores those passive joints as 0 (only the 6 actuated
        # joints/hand are filled), and mj_forward does NOT project qpos onto equality constraints,
        # so without this the distal segments stay straight and the fingers never close on the
        # object. We read the coupling straight from the model's joint-equality constraints so it
        # stays in sync with the MJCF, and re-apply it every frame in _apply_robot_pose.
        self.joint_couplings = self._build_joint_couplings()

        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        # Ground-truth box poses are stored relative to this body (must match base_sim's
        # BOX_GT_REFERENCE_BODY); placed via its FK pose in set_frame.
        self.gt_ref_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
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
        if self.exact_base:
            # Base comes straight from pelvis_in_world; no feet-planting solve, no anchor.
            self.foot_ids = None
        elif all(fid >= 0 for fid in self.foot_ids):
            self._apply_robot_pose(0)
            self.foot_target = 0.5 * (
                self.data.xpos[self.foot_ids[0]] + self.data.xpos[self.foot_ids[1]]
            ).copy()
        else:
            self.foot_ids = None

        # Ground-truth anchor: one constant transform mapping the sim world into the replay's
        # feet-planted world. Built at frame 0 as (replay reference pose) @ inv(sim reference
        # pose), so placing the box with gt_anchor @ box_in_world reproduces the box's true
        # world trajectory relative to the robot — with no per-frame robot motion folded in.
        # Skipped in exact-base mode: gt_anchor stays identity, so the box is placed at its
        # recorded ob_in_world directly (robot base is already in the true sim world).
        if self.obj_gt_ref and self.ref_poses is not None and not self.exact_base:
            ref0 = self._replay_ref_pose(0)  # replay reference body pose at frame 0
            sim_ref0 = self.ref_poses[self.object_index(0)]
            self.gt_anchor = ref0 @ np.linalg.inv(sim_ref0)

    def _replay_ref_pose(self, i: int) -> np.ndarray:
        """World pose (4x4) of the GT reference body at frame ``i`` in the replay, feet planted.

        Mirrors ``set_frame``'s robot placement (joints + base orientation + foot-planting
        translation) but only up to and including reading the reference body — used to build
        the constant ground-truth anchor without needing the anchor itself (chicken-and-egg).
        """
        self._apply_robot_pose(i)
        if self.foot_target is not None:
            mid = 0.5 * (self.data.xpos[self.foot_ids[0]] + self.data.xpos[self.foot_ids[1]])
            self.data.qpos[0:3] += self.foot_target - mid
            mujoco.mj_forward(self.model, self.data)
        t = np.eye(4)
        t[:3, :3] = self.data.xmat[self.gt_ref_body_id].reshape(3, 3)
        t[:3, 3] = self.data.xpos[self.gt_ref_body_id]
        return t

    def _build_joint_couplings(self) -> list[tuple[int, int, np.ndarray, float, float]]:
        """Passive-joint couplings from the model's joint-equality constraints.

        Each returned tuple is ``(dep_qadr, ref_qadr, polycoef, dep_q0, ref_q0)`` for a MuJoCo
        ``mjEQ_JOINT`` constraint, encoding ``qpos[dep] = dep_q0 + poly(qpos[ref] - ref_q0)`` with
        ``poly`` the (lowest-order-first) 5-coefficient polynomial in ``eq_data``. This is exactly
        the BrainCo distal<-proximal mimic, but read generically so any coupling in the MJCF works.
        """
        couplings = []
        for e in range(self.model.neq):
            if self.model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j_dep = int(self.model.eq_obj1id[e])  # joint1: the passive/driven joint
            j_ref = int(self.model.eq_obj2id[e])  # joint2: the actuated/reference joint
            dep_qadr = int(self.model.jnt_qposadr[j_dep])
            ref_qadr = int(self.model.jnt_qposadr[j_ref])
            poly = np.asarray(self.model.eq_data[e][:5], dtype=np.float64)
            couplings.append(
                (dep_qadr, ref_qadr, poly, float(self.model.qpos0[dep_qadr]),
                 float(self.model.qpos0[ref_qadr]))
            )
        return couplings

    def _apply_joint_couplings(self) -> None:
        """Drive each passive joint from its reference joint (BrainCo distal <- proximal mimic)."""
        for dep_qadr, ref_qadr, poly, dep_q0, ref_q0 in self.joint_couplings:
            d = self.data.qpos[ref_qadr] - ref_q0
            self.data.qpos[dep_qadr] = dep_q0 + np.polyval(poly[::-1], d)

    def _apply_robot_pose(self, i: int) -> None:
        """Set robot joints + base orientation for frame i and run FK (no foot anchor/object)."""
        i = int(np.clip(i, 0, self.n - 1))
        whole_q = self.states[i]
        for adr, didx in self.joint_map:
            self.data.qpos[adr] = whole_q[didx]
        # Fill the passive finger joints from their actuated counterparts (recorded whole_q has
        # them at 0); without this the fingers don't curl around the object.
        self._apply_joint_couplings()
        if self.exact_base:
            # Place the floating base directly at its recorded true world pose (position + full
            # orientation) — no feet-planting, no yaw-zeroing. Exact reproduction of the sim.
            pel = self.pelvis_poses[self.object_index(i)]
            self.data.qpos[0:3] = pel[:3, 3]
            self.data.qpos[3:7] = R.from_matrix(pel[:3, :3]).as_quat(scalar_first=True)
        elif self.base_quats is not None:
            # Recorded base orientation (pitch/roll + relative yaw) reproduces how the head camera
            # actually swept during recording, so the static object cancels out on reprojection.
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
            self.current_contact_points = np.zeros((0, 3))  # contacts attach to the object
            mujoco.mj_forward(self.model, self.data)
            return

        obj = self.obj_poses[self.object_index(i)]
        if self.obj_in_world:
            # Filtered poses are already world 4x4 — place directly, NOT through the
            # per-frame camera FK (which would re-inject the camera wobble).
            t_wo = obj
        elif self.obj_gt_ref:
            # Ground truth is the box's absolute world pose; map it into the replay world with
            # the single constant anchor built at frame 0. No dependence on the live robot pose,
            # so a static box stays exactly static and its true motion is reproduced faithfully.
            t_wo = self.gt_anchor @ obj
        else:
            # FoundationPose poses are object-in-camera in the OpenCV optical frame; convert
            # to the MuJoCo camera frame (GL_FROM_CV) then to world via the head-camera FK.
            t_wc = np.eye(4)
            t_wc[:3, :3] = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            t_wc[:3, 3] = self.data.cam_xpos[self.cam_id]
            t_wo = t_wc @ GL_FROM_CV @ obj

        self.data.qpos[self.obj_qadr : self.obj_qadr + 3] = t_wo[:3, 3]
        self.data.qpos[self.obj_qadr + 3 : self.obj_qadr + 7] = R.from_matrix(
            t_wo[:3, :3]
        ).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model, self.data)

        # Object-local contact points -> world, using the box pose we just placed, so the red
        # dots ride on the box surface (finite rows are in contact; NaN rows are skipped).
        if self.contacts_local is not None:
            local = self.contacts_local[int(np.clip(i, 0, self.contacts_local.shape[0] - 1))]
            finite = np.isfinite(local).all(axis=1)
            if finite.any():
                self.current_contact_points = local[finite] @ t_wo[:3, :3].T + t_wo[:3, 3]
            else:
                self.current_contact_points = np.zeros((0, 3))

    def _render_contacts(self, viewer, radius: float) -> None:
        """Draw the current frame's contact points as red spheres in the viewer overlay scene."""
        scn = viewer.user_scn
        scn.ngeom = 0
        rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        size = np.array([radius, radius, radius], dtype=np.float64)
        eye = np.eye(3, dtype=np.float64).reshape(9)
        for p in self.current_contact_points:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size,
                pos=np.asarray(p, dtype=np.float64),
                mat=eye,
                rgba=rgba,
            )
            scn.ngeom += 1

    def run(self, fps: float, contact_radius: float = 0.008) -> None:
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
            self._render_contacts(viewer, contact_radius)
            viewer.sync()
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
                        self._render_contacts(viewer, contact_radius)
                        viewer.sync()
                else:
                    state["frame"] = (state["frame"] + dt * fps) % self.n
                    self.set_frame(int(state["frame"]))
                    self._render_contacts(viewer, contact_radius)
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
        "--exact-base", dest="exact_base", action="store_true",
        help="place the robot base directly from the recorded pelvis_in_world (true sim world, no "
        "feet-planting / yaw-zeroing) and the box at ob_in_world directly — an exact replay of the "
        "sim. Requires --ground-truth and a recording with the pelvis_in_world column "
        "(new_data_collection_report.md). Default is the legacy feet-planted foot-anchor replay.",
    )
    parser.add_argument(
        "--contacts", action="store_true",
        help="overlay red spheres at the finger<->object contact points from the contacts/ "
        "sidecar (run gear_sonic/scripts/process_contacts.py first); best with --ground-truth",
    )
    parser.add_argument(
        "--contact-radius", type=float, default=0.008,
        help="radius (m) of the contact spheres (default: 0.008)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="headless sanity check (no viewer): print stats for the first/last frame",
    )
    args = parser.parse_args()

    if args.ground_truth and args.filtered:
        sys.exit("[error] --ground-truth and --filtered are mutually exclusive")
    if args.exact_base and not args.ground_truth:
        sys.exit("[error] --exact-base requires --ground-truth (pelvis_in_world is in object_gt/)")

    traj, parquet, mesh, ob_src = resolve_paths(args)
    states = load_robot_states(parquet)
    base_quats = load_base_quats(parquet)
    ref_poses = None
    pelvis_poses = None
    if ob_src is None:
        obj_poses, obj_to_robot = None, None
    elif args.ground_truth:
        obj_poses, ref_poses, pelvis_poses, obj_to_robot = load_object_gt(ob_src)
    else:
        obj_poses = load_object_poses(ob_src)
        obj_to_robot = load_frame_map(ob_src, obj_poses.shape[0])

    exact_base = args.exact_base
    if exact_base and pelvis_poses is None:
        print(
            "[warn] --exact-base set but this recording has no pelvis_in_world column "
            "(pre-reset-state schema); falling back to the feet-planted foot-anchor replay"
        )
        exact_base = False
    fps = read_fps(traj)

    print(f"[info] trajectory : {traj}")
    print(f"[info] parquet    : {parquet.relative_to(traj)} ({states.shape[0]} frames)")
    if obj_poses is not None:
        print(f"[info] object     : {ob_src.relative_to(traj)} ({obj_poses.shape[0]} frames)")
        if args.ground_truth:
            print("[info] frame map  : exact (object_gt proprio_frame_index)")
            if exact_base:
                print("[info] base pose  : exact (recorded pelvis_in_world, true sim world)")
                print("[info] object pose: ground truth, world frame, placed directly (no anchor)")
            else:
                print("[info] base pose  : feet-planted foot anchor (legacy)")
                print("[info] object pose: ground truth, world frame, via constant frame-0 anchor")
        else:
            fmap = "exact (frame_map.txt)" if obj_to_robot is not None else "uniform resampling (no map)"
            pose = "world frame, placed directly (filtered)" if args.filtered else "camera frame, via FK"
            print(f"[info] frame map  : {fmap}")
            print(f"[info] object pose: {pose}")
    else:
        print("[info] object     : none — replaying robot body only")

    contacts_local = None
    if args.contacts:
        contacts_local = load_contacts(traj, args.episode, states.shape[0])
        if contacts_local is None:
            print(
                "[warn] --contacts set but no contacts/episode_"
                f"{args.episode:06d}.parquet found; run process_contacts.py first"
            )
        else:
            n_hit = int(np.isfinite(contacts_local).all(axis=2).any(axis=1).sum())
            print(
                f"[info] contacts   : {contacts_local.shape[1]} segments, "
                f"{n_hit}/{states.shape[0]} frames with contact (red spheres)"
            )

    replay = TrajectoryReplay(
        states, obj_poses, mesh, base_quats, obj_to_robot,
        obj_in_world=args.filtered, obj_gt_ref=args.ground_truth, ref_poses=ref_poses,
        contacts_local=contacts_local, pelvis_poses=pelvis_poses, exact_base=exact_base,
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

    replay.run(fps, contact_radius=args.contact_radius)


if __name__ == "__main__":
    main()
