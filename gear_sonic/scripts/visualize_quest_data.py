"""
Visualize recorded Quest head + hand pose data from NPZ trajectories.

Shows head and wrist positions with orientation axes (X=red, Y=green, Z=blue)
and MANO-21 hand landmark skeletons animated over time.

Data convention: ROS FLU (X-forward, Y-left, Z-up). Quaternions are scalar-first [w,x,y,z].

Usage:
    python visualize_quest_data.py                             # first file in data/quest/, GUI
    python visualize_quest_data.py data/quest/traj_*.npz       # specific file
    python visualize_quest_data.py --video                     # save MP4
    python visualize_quest_data.py --video --output out.mp4
    python visualize_quest_data.py --downsample 3 --video      # faster render, every 3rd frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# MANO-21 skeleton: (parent, child) index pairs
_MANO_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),   # wrist → finger roots
    (1, 2), (2, 3), (3, 4),                       # thumb
    (5, 6), (6, 7), (7, 8),                       # index
    (9, 10), (10, 11), (11, 12),                  # middle
    (13, 14), (14, 15), (15, 16),                 # ring
    (17, 18), (18, 19), (19, 20),                 # pinky
]

_AXIS_LEN = 0.06  # meters — length of orientation axis arrows


def _draw_coord_frame(ax, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    """Draw RGB = XYZ coordinate axes at pos using the given orientation."""
    from scipy.spatial.transform import Rotation as sRot

    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-6:
        return
    rot = sRot.from_quat(quat_wxyz / norm, scalar_first=True)
    axes = rot.apply(np.eye(3)) * _AXIS_LEN
    colors = ("red", "green", "blue")
    for i, c in enumerate(colors):
        ax.quiver(
            pos[0], pos[1], pos[2],
            axes[i, 0], axes[i, 1], axes[i, 2],
            color=c, linewidth=1.5, arrow_length_ratio=0.3,
        )


def _draw_hand(ax, landmarks: np.ndarray, color: str) -> None:
    """Draw MANO-21 hand skeleton from (21, 3) landmark array."""
    for i, j in _MANO_CONNECTIONS:
        ax.plot(
            [landmarks[i, 0], landmarks[j, 0]],
            [landmarks[i, 1], landmarks[j, 1]],
            [landmarks[i, 2], landmarks[j, 2]],
            color=color, linewidth=1.0, alpha=0.8,
        )
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
               color=color, s=6, zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D animation of Quest head/hand pose recordings."
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="NPZ file to visualize (default: first file found in data/quest/)",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Render to video file instead of opening the GUI",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video path (default: <input_stem>.mp4)",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Playback / render FPS (default: 30)",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="Use every N-th frame — speeds up rendering (default: 1 = all frames)",
    )
    parser.add_argument(
        "--data-dir", default="data/quest",
        help="Directory to search for NPZ files when no file is specified (default: data/quest)",
    )
    args = parser.parse_args()

    # ---- resolve input file ----
    if args.file is None:
        candidates = sorted(Path(args.data_dir).glob("*.npz"))
        if not candidates:
            print(f"No NPZ files found in '{args.data_dir}'. Pass a file path as argument.")
            sys.exit(1)
        args.file = str(candidates[0])
        print(f"Using: {args.file}")

    if args.output is None:
        args.output = Path(args.file).stem + ".mp4"

    # ---- load data ----
    raw = np.load(args.file, allow_pickle=True)
    idx = slice(None, None, args.downsample)

    head_pos   = raw["head_pos"][idx]          # (T, 3)
    head_quat  = raw["head_quat"][idx]          # (T, 4) [w,x,y,z]
    lw_pos     = raw["left_wrist_pos"][idx]     # (T, 3)
    lw_quat    = raw["left_wrist_quat"][idx]    # (T, 4)
    rw_pos     = raw["right_wrist_pos"][idx]    # (T, 3)
    rw_quat    = raw["right_wrist_quat"][idx]   # (T, 4)
    l_lm       = raw["left_landmarks"][idx]     # (T, 21, 3)
    r_lm       = raw["right_landmarks"][idx]    # (T, 21, 3)
    l_tracked  = raw["left_tracked"][idx]       # (T,)
    r_tracked  = raw["right_tracked"][idx]      # (T,)
    timestamps = raw["timestamp"][idx]          # (T,)

    n_frames = head_pos.shape[0]
    duration  = timestamps[-1] - timestamps[0]
    print(f"Loaded {n_frames} frames  |  duration {duration:.1f}s  |  ~{n_frames/duration:.1f} fps")

    # ---- matplotlib backend (must be set before pyplot import) ----
    if args.video:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.animation as anim_mod

    # ---- compute fixed axis limits ----
    all_pts = np.concatenate(
        [head_pos, lw_pos, rw_pos, l_lm.reshape(-1, 3), r_lm.reshape(-1, 3)], axis=0
    )
    margin = 0.05
    xlim = (float(all_pts[:, 0].min()) - margin, float(all_pts[:, 0].max()) + margin)
    ylim = (float(all_pts[:, 1].min()) - margin, float(all_pts[:, 1].max()) + margin)
    zlim = (float(all_pts[:, 2].min()) - margin, float(all_pts[:, 2].max()) + margin)

    # ---- figure setup ----
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    def _reset_ax() -> None:
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X  (forward)")
        ax.set_ylabel("Y  (left)")
        ax.set_zlabel("Z  (up)")

    def update(fi: int):
        _reset_ax()
        t_rel = timestamps[fi] - timestamps[0]
        ax.set_title(
            f"{Path(args.file).name}  |  frame {fi}/{n_frames - 1}  |  t = {t_rel:.2f} s",
            fontsize=10,
        )

        # Head — grey
        ax.scatter(*head_pos[fi], color="dimgrey", s=80, zorder=10, label="Head")
        _draw_coord_frame(ax, head_pos[fi], head_quat[fi])

        # Left wrist — blue family
        ax.scatter(*lw_pos[fi], color="royalblue", s=60, zorder=10, label="L-wrist")
        _draw_coord_frame(ax, lw_pos[fi], lw_quat[fi])

        # Right wrist — red family
        ax.scatter(*rw_pos[fi], color="crimson", s=60, zorder=10, label="R-wrist")
        _draw_coord_frame(ax, rw_pos[fi], rw_quat[fi])

        # Hand landmarks
        if l_tracked[fi]:
            _draw_hand(ax, l_lm[fi], "cornflowerblue")
        if r_tracked[fi]:
            _draw_hand(ax, r_lm[fi], "salmon")

        ax.legend(loc="upper left", fontsize=8, framealpha=0.5)

        # Axis legend in corner
        ax.text2D(0.01, 0.01,
                  "Axes: X=red  Y=green  Z=blue",
                  transform=ax.transAxes, fontsize=7, color="grey")
        return []

    interval_ms = 1000.0 / args.fps
    animation = anim_mod.FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms, blit=False
    )

    if args.video:
        print(f"Rendering {n_frames} frames → {args.output} ...")
        try:
            writer = anim_mod.FFMpegWriter(fps=args.fps, bitrate=1800)
            animation.save(args.output, writer=writer, dpi=150)
            print(f"Saved: {args.output}")
        except Exception as exc:
            gif_path = str(Path(args.output).with_suffix(".gif"))
            print(f"FFMpeg failed ({exc}), falling back to GIF → {gif_path}")
            writer = anim_mod.PillowWriter(fps=args.fps)
            animation.save(gif_path, writer=writer)
            print(f"Saved: {gif_path}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
