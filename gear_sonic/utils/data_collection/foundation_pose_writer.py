"""Writes per-episode FoundationPose scene folders alongside the LeRobot dataset.

For each recorded episode this produces, under ``<dataset_root>/foundation_pose_data/``::

    box.obj                         # shared object mesh (meters), written once
    episode_000000/
        cam_K.txt                   # 3x3 intrinsics, row-major
        rgb/000000.png ...          # 8-bit RGB, every frame
        depth/000000.png ...        # 16-bit depth in millimeters, every frame
        masks/000000.png            # binary box mask, frame 0 only

This is the input layout expected by FoundationPose's ``run_demo.py`` (one scene
folder per sequence, plus a separate ``--mesh_file``). The writer is inactive until
``start_episode`` is called, so it is a no-op when depth/seg are not being streamed.
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

    def start_episode(self, episode_index: int) -> None:
        """Create the folder structure for a new episode and reset the frame counter."""
        self._episode_dir = self.base / f"episode_{episode_index:06d}"
        for sub in ("rgb", "depth", "masks"):
            (self._episode_dir / sub).mkdir(parents=True, exist_ok=True)
        self._frame = 0

    def write_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        cam_K,
        box_half_extents,
    ) -> None:
        """Write one RGB+depth frame; on frame 0 also write cam_K, the mask and box.obj."""
        if self._episode_dir is None:
            return

        if self._frame == 0:
            K = np.asarray(cam_K, dtype=np.float64).reshape(3, 3)
            np.savetxt(self._episode_dir / "cam_K.txt", K)
            cv2.imwrite(str(self._episode_dir / "masks" / "000000.png"), mask)
            self._ensure_box_mesh(box_half_extents)

        name = f"{self._frame:06d}.png"
        # cv2 expects BGR; the camera client delivers RGB.
        cv2.imwrite(str(self._episode_dir / "rgb" / name), rgb[..., ::-1])
        cv2.imwrite(str(self._episode_dir / "depth" / name), depth.astype(np.uint16))
        self._frame += 1

    def discard_episode(self) -> None:
        """Remove the current (partial) episode folder, e.g. after an abort."""
        if self._episode_dir is not None and self._episode_dir.is_dir():
            shutil.rmtree(self._episode_dir, ignore_errors=True)
        self._episode_dir = None

    def _ensure_box_mesh(self, box_half_extents) -> None:
        """Write a centered axis-aligned box mesh (meters) once, shared across episodes."""
        out = self.base / "box.obj"
        if out.exists() or box_half_extents is None:
            return
        hx, hy, hz = (float(s) for s in box_half_extents)
        verts = [
            (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
            (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
        ]
        # 1-indexed triangles, outward winding.
        faces = [
            (1, 3, 2), (1, 4, 3),  # bottom (-z)
            (5, 6, 7), (5, 7, 8),  # top (+z)
            (1, 2, 6), (1, 6, 5),  # -y
            (3, 4, 8), (3, 8, 7),  # +y
            (1, 5, 8), (1, 8, 4),  # -x
            (2, 3, 7), (2, 7, 6),  # +x
        ]
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        lines += [f"f {a} {b} {c}" for a, b, c in faces]
        out.write_text("\n".join(lines) + "\n")
