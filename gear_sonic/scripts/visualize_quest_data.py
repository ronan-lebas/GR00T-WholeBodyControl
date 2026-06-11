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


# ---------------------------------------------------------------------------
# Landmark frame correction  (extract this block when integrating into the pipeline)
# ---------------------------------------------------------------------------

def correct_landmark_frame(landmarks: np.ndarray) -> np.ndarray:
    """Transform Quest MANO landmarks into the wrist TF frame.

    Quest hand-tracking publishes landmarks in a frame that is rotated +90 deg
    around world +Z relative to the wrist TF frame used by the controller.
    Applying Rz(-90 deg) to every landmark position realigns the hand skeleton
    with the wrist pose.

    Args:
        landmarks: (..., 3) landmark positions in Quest world frame.

    Returns:
        (..., 3) corrected positions in wrist-TF-compatible world frame.
    """
    from scipy.spatial.transform import Rotation as sRot

    R = sRot.from_euler("z", -np.pi / 2)
    return R.apply(landmarks.reshape(-1, 3)).reshape(landmarks.shape)


# ---------------------------------------------------------------------------


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


def _process_file(npz_path: str, args, plt, anim_mod) -> None:
    """Load one NPZ file and either show the GUI or save a video/GIF."""
    raw = np.load(npz_path, allow_pickle=True)
    idx = slice(None, None, args.downsample)

    head_pos   = raw["head_pos"][idx]
    head_quat  = raw["head_quat"][idx]
    lw_pos     = raw["left_wrist_pos"][idx]
    lw_quat    = raw["left_wrist_quat"][idx]
    rw_pos     = raw["right_wrist_pos"][idx]
    rw_quat    = raw["right_wrist_quat"][idx]
    l_lm       = correct_landmark_frame(raw["left_landmarks"][idx])
    r_lm       = correct_landmark_frame(raw["right_landmarks"][idx])
    l_tracked  = raw["left_tracked"][idx]
    r_tracked  = raw["right_tracked"][idx]
    timestamps = raw["timestamp"][idx]

    n_frames = head_pos.shape[0]
    duration  = timestamps[-1] - timestamps[0]
    print(f"{Path(npz_path).name}  |  {n_frames} frames  |  {duration:.1f}s  |  ~{n_frames/duration:.1f} fps")

    all_pts = np.concatenate(
        [head_pos, lw_pos, rw_pos, l_lm.reshape(-1, 3), r_lm.reshape(-1, 3)], axis=0
    )
    margin = 0.05
    xlim = (float(all_pts[:, 0].min()) - margin, float(all_pts[:, 0].max()) + margin)
    ylim = (float(all_pts[:, 1].min()) - margin, float(all_pts[:, 1].max()) + margin)
    zlim = (float(all_pts[:, 2].min()) - margin, float(all_pts[:, 2].max()) + margin)

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
            f"{Path(npz_path).name}  |  frame {fi}/{n_frames - 1}  |  t = {t_rel:.2f} s",
            fontsize=10,
        )
        ax.scatter(*head_pos[fi], color="dimgrey",   s=80, zorder=10, label="Head")
        _draw_coord_frame(ax, head_pos[fi], head_quat[fi])
        ax.scatter(*lw_pos[fi],   color="royalblue", s=60, zorder=10, label="L-wrist")
        _draw_coord_frame(ax, lw_pos[fi], lw_quat[fi])
        ax.scatter(*rw_pos[fi],   color="crimson",   s=60, zorder=10, label="R-wrist")
        _draw_coord_frame(ax, rw_pos[fi], rw_quat[fi])
        if l_tracked[fi]:
            _draw_hand(ax, l_lm[fi], "cornflowerblue")
        if r_tracked[fi]:
            _draw_hand(ax, r_lm[fi], "salmon")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.5)
        ax.text2D(0.01, 0.01, "Axes: X=red  Y=green  Z=blue",
                  transform=ax.transAxes, fontsize=7, color="grey")
        return []

    animation = anim_mod.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / args.fps, blit=False
    )

    if args.video:
        output = str(Path(npz_path).stem + ".mp4") if args.output is None else args.output
        print(f"Rendering → {output} ...")
        try:
            writer = anim_mod.FFMpegWriter(fps=args.fps, bitrate=1800)
            animation.save(output, writer=writer, dpi=150)
            print(f"Saved: {output}")
        except Exception as exc:
            gif_path = str(Path(output).with_suffix(".gif"))
            print(f"FFMpeg failed ({exc}), falling back to GIF → {gif_path}")
            writer = anim_mod.PillowWriter(fps=args.fps)
            animation.save(gif_path, writer=writer)
            print(f"Saved: {gif_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D animation of Quest head/hand pose recordings."
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="NPZ file to visualize (default: first file found in data/quest/)",
    )
    parser.add_argument(
        "--all", dest="process_all", action="store_true",
        help="Process every NPZ file in --data-dir (ignores positional file argument)",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Render to video file instead of opening the GUI",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video path — single-file mode only (default: <input_stem>.mp4)",
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
        help="Directory to search for NPZ files (default: data/quest)",
    )
    args = parser.parse_args()

    # ---- resolve file list ----
    if args.process_all:
        files = sorted(Path(args.data_dir).glob("*.npz"))
        if not files:
            print(f"No NPZ files found in '{args.data_dir}'.")
            sys.exit(1)
        if args.output is not None:
            print("Warning: --output is ignored with --all (each file gets its own output name).")
            args.output = None
        files = [str(f) for f in files]
    else:
        if args.file is None:
            candidates = sorted(Path(args.data_dir).glob("*.npz"))
            if not candidates:
                print(f"No NPZ files found in '{args.data_dir}'. Pass a file path as argument.")
                sys.exit(1)
            args.file = str(candidates[0])
            print(f"Using: {args.file}")
        files = [args.file]

    # ---- matplotlib backend (must be set before pyplot import) ----
    if args.video:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.animation as anim_mod

    for i, path in enumerate(files):
        if args.process_all:
            print(f"\n[{i + 1}/{len(files)}] {path}")
        _process_file(path, args, plt, anim_mod)


if __name__ == "__main__":
    main()
