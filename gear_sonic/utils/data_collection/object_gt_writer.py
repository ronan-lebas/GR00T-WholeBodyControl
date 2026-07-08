"""Writes the cube's ground-truth pose (sim-only) alongside the LeRobot dataset.

In simulation the object's exact 6D pose is known, so we can record it directly instead
of (or in parallel to) estimating it offline with FoundationPose. This is impossible on
real hardware, so it lives behind a flag and in a *separate* file — never mixed into the
robot-state parquet.

For each recorded episode this produces, under ``<dataset_root>/object_gt/``::

    episode_000000.parquet          # one row per recorded frame:
                                     #   proprio_frame_index : int   (parquet/robot row)
                                     #   timestamp           : float
                                     #   ob_in_ref           : 16 floats (4x4 row-major)

``ob_in_ref`` is the box pose in the right-foot frame (``right_ankle_roll_link``), a
planted, near-static reference — so the replayed cube doesn't jitter (the head camera, the
fast-moving tip of the chain, did). The replay visualizer places the box with that same
body's pose. ``proprio_frame_index`` links each pose to the robot row it was captured
alongside (same convention as ``FoundationPoseWriter``'s ``frame_map.txt``), letting the
replay align the two timelines and compare ground truth against the estimate.

The writer also ensures the shared colored ``box.obj`` mesh exists (under
``foundation_pose_data/``, where the visualizer looks for it) so ground-truth-only runs
still have a mesh to render.
"""

from pathlib import Path

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
        self._rows: list[tuple[int, float, np.ndarray]] = []

    def is_active(self) -> bool:
        """Whether an episode is currently open for writing."""
        return self._episode_index is not None

    def start_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._rows = []

    def write_frame(
        self,
        ob_in_ref,
        proprio_frame_index: int,
        timestamp: float,
        box_half_extents=None,
    ) -> None:
        """Buffer one ground-truth pose. ``ob_in_ref`` is a 4x4 (or flat-16) array."""
        if self._episode_index is None:
            return
        pose = np.asarray(ob_in_ref, dtype=np.float64).reshape(16)
        self._rows.append((int(proprio_frame_index), float(timestamp), pose))
        # Write the shared colored mesh once (needed for ground-truth-only replay).
        if box_half_extents is not None and len(box_half_extents) == 3:
            self._ensure_box_mesh(box_half_extents)

    def _ensure_box_mesh(self, box_half_extents) -> None:
        if self._mesh_path.exists():
            return
        self._mesh_path.parent.mkdir(parents=True, exist_ok=True)
        write_colored_box_obj(self._mesh_path, box_half_extents)

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
            table = pa.table(
                {
                    "proprio_frame_index": pa.array([r[0] for r in self._rows], pa.int64()),
                    "timestamp": pa.array([r[1] for r in self._rows], pa.float64()),
                    "ob_in_ref": pa.array(
                        [r[2].tolist() for r in self._rows], pa.list_(pa.float64(), 16)
                    ),
                }
            )
            pq.write_table(table, self.base / f"episode_{self._episode_index:06d}.parquet")
        self._episode_index = None
        self._rows = []
