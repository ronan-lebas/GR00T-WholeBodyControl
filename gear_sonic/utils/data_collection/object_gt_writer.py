"""Writes the cube's ground-truth pose (sim-only) alongside the LeRobot dataset.

In simulation the object's exact 6D pose is known, so we can record it directly instead
of (or in parallel to) estimating it offline with FoundationPose. This is impossible on
real hardware, so it lives behind a flag and in a *separate* file — never mixed into the
robot-state parquet.

For each recorded episode this produces, under ``<dataset_root>/object_gt/``::

    episode_000000.parquet          # one row per recorded frame:
                                     #   proprio_frame_index : int   (parquet/robot row)
                                     #   timestamp           : float
                                     #   ob_in_world         : 16 floats (4x4 row-major)
                                     #   ref_in_world        : 16 floats (4x4 row-major)
                                     #   pelvis_in_world     : 16 floats (4x4 row-major)
                                     #   base_vel            : 6 floats  (freejoint qvel)
                                     #   object_vel          : 6 floats  (freejoint qvel)
                                     #   joint_vel           : N floats  (observation.state order)

``ob_in_world`` is the box's absolute pose in the MuJoCo world frame; ``ref_in_world`` is
the pose of the ground-truth reference body (``right_ankle_roll_link``) in that same world,
sampled at the same instant. Storing the box in the *world* (not relative to a robot link)
keeps a physically static cube's recorded pose constant — expressing it relative to a
moving link would inject that link's motion into the cube. The replay visualizer anchors
the sim world to its own feet-planted world using ``ref_in_world`` once (frame 0), a single
constant transform, so the cube lands correctly relative to the robot without re-injecting
per-frame robot motion. ``proprio_frame_index`` links each pose to the robot row it was
captured alongside (same convention as ``FoundationPoseWriter``'s ``frame_map.txt``),
letting the replay align the two timelines and compare ground truth against the estimate.

``pelvis_in_world`` is the robot floating-base (pelvis) world pose — the base translation the
LeRobot parquet never records — so downstream can place the robot in the world exactly, with no
FK reconstruction. ``base_vel`` / ``object_vel`` are the base and box freejoint velocities and
``joint_vel`` the joint velocities in ``observation.state`` (whole_q) ordering; together they let
a training env reset to any recorded frame (reference-state initialization) without a
zero-velocity jump. All are sampled from the same physics step as the poses. See
``new_data_collection_report.md`` for the reset recipe.

The writer also ensures the shared object mesh exists (under ``foundation_pose_data/``, where the
visualizer looks for it) so ground-truth-only runs still have a mesh to render: a synthesized
colored ``box.obj`` for the primitive cube, or a copy of the staged asset (``object.obj`` plus its
convex hulls as ``object_collision_*.stl``) when the sim ran with a mesh object (``--object-asset``).
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear_sonic.utils.box_appearance import write_colored_box_obj


class ObjectGtWriter:
    """Incrementally buffers ground-truth box poses, one parquet per recorded episode."""

    def __init__(self, dataset_root):
        self.base = Path(dataset_root) / "object_gt"
        # box.obj is shared with the FoundationPose layout so the visualizer finds it
        # in one place regardless of which writer produced it.
        self._mesh_path = Path(dataset_root) / "foundation_pose_data" / "box.obj"
        self._episode_index: int | None = None
        self._rows: list[dict] = []
        self._identity_checked = False

    def is_active(self) -> bool:
        """Whether an episode is currently open for writing."""
        return self._episode_index is not None

    def start_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._rows = []

    def write_frame(
        self,
        ob_in_world,
        ref_in_world,
        proprio_frame_index: int,
        timestamp: float,
        box_half_extents=None,
        object_mesh_dir=None,
        object_name=None,
        pelvis_in_world=None,
        base_vel=None,
        object_vel=None,
        joint_vel=None,
    ) -> None:
        """Buffer one ground-truth frame.

        Poses are 4x4 (or flat-16) world transforms; velocities are 1-D arrays. The reset-state
        fields (``pelvis_in_world``, ``base_vel``, ``object_vel``, ``joint_vel``) are optional so
        that legacy publishers that only send poses still record; missing ones are written as
        zeros of the expected width (a missing ``object_vel`` — e.g. a held/kinematic box — is a
        genuine scripted zero). ``joint_vel`` sets this recording's joint count.
        """
        if self._episode_index is None:
            return
        box = np.asarray(ob_in_world, dtype=np.float64).reshape(16)
        ref = np.asarray(ref_in_world, dtype=np.float64).reshape(16)
        jvel = None if joint_vel is None else np.asarray(joint_vel, dtype=np.float64).reshape(-1)
        self._rows.append(
            {
                "proprio_frame_index": int(proprio_frame_index),
                "timestamp": float(timestamp),
                "ob_in_world": box,
                "ref_in_world": ref,
                "pelvis_in_world": (
                    None
                    if pelvis_in_world is None
                    else np.asarray(pelvis_in_world, dtype=np.float64).reshape(16)
                ),
                "base_vel": (
                    None if base_vel is None else np.asarray(base_vel, dtype=np.float64).reshape(-1)
                ),
                "object_vel": (
                    None
                    if object_vel is None
                    else np.asarray(object_vel, dtype=np.float64).reshape(-1)
                ),
                "joint_vel": jvel,
            }
        )
        # Write the shared object mesh once (needed for ground-truth-only replay).
        self._check_identity(object_name, box_half_extents)
        if object_mesh_dir:
            self._ensure_asset_mesh(object_mesh_dir)
        elif box_half_extents is not None and len(box_half_extents) == 3:
            self._ensure_box_mesh(box_half_extents)

    def _check_identity(self, object_name, box_half_extents) -> None:
        """Record which object this dataset holds, and warn if it ever changes.

        The mesh writers below skip when their output already exists, so recording a *different*
        object into an existing dataset root would silently keep the first object's mesh and make
        every replay of the new episodes wrong. Datasets are named per run by default, so this
        only bites on an explicit --dataset-name reuse — but it fails silently, hence the shout.
        """
        if self._identity_checked or not object_name:
            return
        self._identity_checked = True
        path = self._mesh_path.parent / "object_meta.json"
        meta = {
            "name": str(object_name),
            "box_half_extents": [float(v) for v in (box_half_extents or [])],
        }
        if path.exists():
            previous = json.loads(path.read_text()).get("name")
            if previous != meta["name"]:
                print(
                    f"[ObjectGt] WARNING: this dataset was recorded with object '{previous}' but "
                    f"the sim is now running '{meta['name']}'. The stored mesh is NOT updated, so "
                    "replays of these episodes will show the wrong object — record into a fresh "
                    "dataset root instead."
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2) + "\n")

    def _ensure_box_mesh(self, box_half_extents) -> None:
        if self._mesh_path.exists():
            return
        self._mesh_path.parent.mkdir(parents=True, exist_ok=True)
        write_colored_box_obj(self._mesh_path, box_half_extents)

    def _ensure_asset_mesh(self, object_mesh_dir) -> None:
        """Copy a staged mesh asset (``--object-asset``) next to the dataset for replay.

        Landed as ``object.obj`` + ``object_collision_*.stl``, which the replay/contact tooling
        prefers over the synthesized ``box.obj``. The hulls travel too so ``process_contacts.py``
        collides against the decomposition rather than the mesh's overall convex hull.
        """
        out = self._mesh_path.parent / "object.obj"
        if out.exists():
            return
        src = Path(object_mesh_dir)
        meta_path = src / "object.json"
        if not meta_path.exists():
            print(f"[ObjectGt] staged asset {src} has no object.json; skipping mesh copy")
            return
        meta = json.loads(meta_path.read_text())
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / meta["visual_mesh"], out)
        for i, fname in enumerate(meta.get("collision_meshes", [])):
            shutil.copyfile(src / fname, out.parent / f"object_collision_{i:03d}.stl")

    def discard_episode(self) -> None:
        """Drop the current (partial) episode without writing it."""
        self._episode_index = None
        self._rows = []

    def close_episode(self) -> None:
        """Write the buffered poses to ``object_gt/episode_%06d.parquet`` and reset."""
        if self._episode_index is None:
            return
        if self._rows:
            self.base.mkdir(parents=True, exist_ok=True)

            # Widths for the optional velocity/pose fields: infer joint_vel width from the first
            # row that has it (51 for brainco), fall back to fixed sizes for the twists.
            n_joints = next(
                (r["joint_vel"].shape[0] for r in self._rows if r["joint_vel"] is not None), 0
            )

            def _flat(key: str, width: int) -> list[list[float]]:
                zeros = [0.0] * width
                return [
                    (r[key].tolist() if r[key] is not None else zeros) for r in self._rows
                ]

            table = pa.table(
                {
                    "proprio_frame_index": pa.array(
                        [r["proprio_frame_index"] for r in self._rows], pa.int64()
                    ),
                    "timestamp": pa.array([r["timestamp"] for r in self._rows], pa.float64()),
                    "ob_in_world": pa.array(_flat("ob_in_world", 16), pa.list_(pa.float64(), 16)),
                    "ref_in_world": pa.array(_flat("ref_in_world", 16), pa.list_(pa.float64(), 16)),
                    "pelvis_in_world": pa.array(
                        _flat("pelvis_in_world", 16), pa.list_(pa.float64(), 16)
                    ),
                    "base_vel": pa.array(_flat("base_vel", 6), pa.list_(pa.float64(), 6)),
                    "object_vel": pa.array(_flat("object_vel", 6), pa.list_(pa.float64(), 6)),
                    "joint_vel": pa.array(
                        _flat("joint_vel", n_joints), pa.list_(pa.float64(), n_joints)
                    ),
                }
            )
            pq.write_table(table, self.base / f"episode_{self._episode_index:06d}.parquet")
        self._episode_index = None
        self._rows = []
