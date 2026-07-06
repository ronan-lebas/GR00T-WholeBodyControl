"""Writes per-episode FoundationPose scene folders alongside the LeRobot dataset.

For each recorded episode this produces, under ``<dataset_root>/foundation_pose_data/``::

    episode_000000/
        cam_K.txt                   # 3x3 intrinsics, row-major
        cam_extrinsics.txt          # 4x4 depth->color transform, row-major (if available)
        rgb/000000.png ...          # 8-bit RGB, every frame
        depth/000000.png ...        # 16-bit depth in millimeters, every frame
        frame_map.txt               # FP-frame -> proprio-row correspondence

This is the (partial) input layout expected by FoundationPose's ``run_demo.py`` (one
scene folder per sequence); mask + mesh are supplied separately. The writer is
inactive until ``start_episode`` is called, so it is a no-op when depth is not being
streamed.

``frame_map.txt`` is the exact correspondence between each FP frame and the
proprio/parquet row it was captured alongside. Depth is rendered/streamed at a
reduced rate on some setups, so FP frames may be sparser than the 50 Hz proprio
stream; this map lets the replay / pose-jitter diagnostic pair each object pose
with the precise robot state (camera FK).
"""

from pathlib import Path
import shutil

import cv2
import numpy as np


class FoundationPoseWriter:
    """Incrementally writes FoundationPose scene folders, one per recorded episode."""

    def __init__(self, dataset_root):
        self.base = Path(dataset_root) / "foundation_pose_data"
        self._episode_dir: Path | None = None
        self._frame = 0
        # Rows of (fp_frame, proprio_frame_index, timestamp) for this episode.
        self._frame_map: list[tuple[int, int, float]] = []

    def is_active(self) -> bool:
        """Whether an episode folder is currently open for writing."""
        return self._episode_dir is not None

    def start_episode(self, episode_index: int) -> None:
        """Create the folder structure for a new episode and reset the frame counter."""
        self._episode_dir = self.base / f"episode_{episode_index:06d}"
        for sub in ("rgb", "depth"):
            (self._episode_dir / sub).mkdir(parents=True, exist_ok=True)
        self._frame = 0
        self._frame_map = []

    def write_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        cam_K,
        proprio_frame_index: int,
        cam_extrinsics_R=None,
        cam_extrinsics_t=None,
        timestamp: float | None = None,
    ) -> None:
        """Write one RGB+depth frame; on frame 0 also write cam_K (and extrinsics if given).

        ``proprio_frame_index`` is the parquet row (within the episode) captured in
        the same loop iteration as this FP frame; together with ``timestamp`` it is
        appended to ``frame_map.txt`` so downstream tools can recover the exact
        FP-frame -> robot-state correspondence.

        ``cam_extrinsics_R``/``cam_extrinsics_t`` are the depth->color transform
        (RealSense depth and color are physically separate sensors); pass ``None``
        when depth and RGB already share the same optical frame (e.g. sim).
        """
        if self._episode_dir is None:
            return

        # Depth may be rendered/streamed at a different resolution; upscale (nearest) to match RGB.
        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        if self._frame == 0:
            if cam_K is None:
                print(
                    "[FoundationPoseWriter] Warning: no cam_K in the camera stream "
                    "(fp_meta missing) — recording rgb/depth without calibration."
                )
            else:
                K = np.asarray(cam_K, dtype=np.float64).reshape(3, 3)
                np.savetxt(self._episode_dir / "cam_K.txt", K)
            if cam_extrinsics_R is not None and cam_extrinsics_t is not None:
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = np.asarray(cam_extrinsics_R, dtype=np.float64).reshape(3, 3)
                T[:3, 3] = np.asarray(cam_extrinsics_t, dtype=np.float64).reshape(3)
                np.savetxt(self._episode_dir / "cam_extrinsics.txt", T)

        name = f"{self._frame:06d}.png"
        # cv2 expects BGR; the camera client delivers RGB.
        cv2.imwrite(str(self._episode_dir / "rgb" / name), rgb[..., ::-1])
        cv2.imwrite(str(self._episode_dir / "depth" / name), depth.astype(np.uint16))

        self._frame_map.append(
            (
                self._frame,
                int(proprio_frame_index),
                float(timestamp) if timestamp is not None else float("nan"),
            )
        )
        self._write_frame_map()
        self._frame += 1

    def _write_frame_map(self) -> None:
        """(Re)write frame_map.txt for the current episode.

        Rewritten in full each FP frame (FP rate is low, so this is cheap) so the
        file stays consistent up to the last written frame even if recording is
        interrupted.
        """
        if self._episode_dir is None:
            return
        lines = ["# fp_frame proprio_frame_index timestamp"]
        lines += [f"{f} {p} {t:.6f}" for f, p, t in self._frame_map]
        (self._episode_dir / "frame_map.txt").write_text("\n".join(lines) + "\n")

    def discard_episode(self) -> None:
        """Remove the current (partial) episode folder, e.g. after an abort."""
        if self._episode_dir is not None and self._episode_dir.is_dir():
            shutil.rmtree(self._episode_dir, ignore_errors=True)
        self._episode_dir = None
        self._frame_map = []

    def close_episode(self) -> None:
        """Mark the current episode finished, keeping its data (unlike discard_episode)."""
        self._episode_dir = None
        self._frame_map = []
