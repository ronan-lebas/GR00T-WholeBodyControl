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


def _draw_coord_frame(ax, pos: np.ndarray, quat_wxyz: np.ndarray,
                      axis_len: float = _AXIS_LEN, lw: float = 1.5,
                      alpha: float = 1.0, linestyle: str = "-") -> None:
    """Draw RGB = XYZ coordinate axes at pos using the given orientation."""
    from scipy.spatial.transform import Rotation as sRot

    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-6:
        return
    rot = sRot.from_quat(quat_wxyz / norm, scalar_first=True)
    axes = rot.apply(np.eye(3)) * axis_len
    colors = ("red", "green", "blue")
    for i, c in enumerate(colors):
        ax.quiver(
            pos[0], pos[1], pos[2],
            axes[i, 0], axes[i, 1], axes[i, 2],
            color=c, linewidth=lw, alpha=alpha, linestyle=linestyle,
            arrow_length_ratio=0.3,
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


def _process_calibrated(npz_path: str, args, plt, anim_mod) -> None:
    """Overlay the operator (transformed) and the calibrated robot targets.

    Runs the recording through the exact teleop pipeline (rest-pose prefix ->
    QuestThreePointPose calibrated on frame 0 -> (3,7) robot-root targets), then
    plots, in the robot root frame:

      * ROBOT command (filled balls, solid axes): what is sent to the policy.
      * OPERATOR (open balls, dashed axes): the raw Quest head/wrists transformed
        into the same frame — head->wrist vector in the calibration-yaw frame,
        anchored at the robot head point, with the operator's yaw-compensated wrist
        orientation. Operator hand landmarks (MANO skeleton) are drawn too.
      * Robot rest key frames (grey squares) for reference.

    When the calibration is correct the operator (open/dashed) and robot (filled/
    solid) coincide — any visible gap is the residual offset to diagnose. This is
    the offline check for wrist tracking (no Quest or robot needed).

    Convention: robot root frame, X-forward / Y-left / Z-up; axes X=red Y=green Z=blue.
    """
    import importlib.util
    from scipy.spatial.transform import Rotation as sRot

    _spec = importlib.util.spec_from_file_location(
        "quest_manager_thread_server", str(Path(__file__).with_name("quest_manager_thread_server.py"))
    )
    qm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(qm)

    raw = np.load(npz_path, allow_pickle=True)
    keys = ["head_pos", "head_quat", "left_wrist_pos", "left_wrist_quat",
            "right_wrist_pos", "right_wrist_quat", "left_landmarks", "right_landmarks"]
    data = {k: raw[k].astype(np.float64) for k in keys}
    ts = raw["timestamp"].astype(np.float64)
    data, ts = qm.prepend_rest_pose(data, ts, args.rest_hold_sec, args.rest_interp_sec)
    idx = np.arange(0, ts.shape[0], args.downsample)

    tp = qm.QuestThreePointPose(pos_scale=args.pos_scale)
    tp.calibrate_now({k: data[k][0].copy() for k in data})  # calibrate on rest pose
    r0_inv = tp._calib_yaw_inv  # calibration-yaw inverse rotation
    head_ref = np.array([0.0, 0.0, qm._TORSO_LINK_OFFSET_Z + qm._NECK_LINK_LENGTH])

    def _op_in_robot_frame(d, i):
        """Transform operator head/wrists/landmarks into the robot root frame."""
        hp = d["head_pos"][i]
        out = {}
        for side, pk, qk, lk in (
            ("left", "left_wrist_pos", "left_wrist_quat", "left_landmarks"),
            ("right", "right_wrist_pos", "right_wrist_quat", "right_landmarks"),
        ):
            out[f"{side}_pos"] = head_ref + r0_inv.apply(d[pk][i] - hp)
            out[f"{side}_quat"] = (r0_inv * sRot.from_quat(d[qk][i], scalar_first=True)).as_quat(
                scalar_first=True
            )
            lm = correct_landmark_frame(d[lk][i])
            out[f"{side}_lm"] = head_ref + r0_inv.apply(lm - hp)
        return out

    targets = np.stack([tp.process_quest_pose({k: data[k][i].copy() for k in data}) for i in idx])
    op = [_op_in_robot_frame(data, i) for i in idx]
    timestamps = ts[idx]
    n_frames = targets.shape[0]

    rest = None
    if qm.get_g1_key_frame_poses is not None and tp._robot_model is not None:
        poses = qm.get_g1_key_frame_poses(tp._robot_model)
        rest = {"left": poses["left_wrist"], "right": poses["right_wrist"], "torso": poses["torso"]}

    labels = ["L-wrist", "R-wrist", "Head"]
    colors = ["royalblue", "crimson", "orange"]
    lm_colors = {"left": "cornflowerblue", "right": "salmon"}

    all_pts = [targets[:, :, :3].reshape(-1, 3)]
    for o in op:
        all_pts.append(np.stack([o["left_pos"], o["right_pos"]]))
        all_pts.append(o["left_lm"]); all_pts.append(o["right_lm"])
    all_pts = np.concatenate(all_pts, axis=0)
    margin = 0.1
    lims = [(float(all_pts[:, a].min()) - margin, float(all_pts[:, a].max()) + margin) for a in range(3)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(fi: int):
        ax.cla()
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel("X (forward)"); ax.set_ylabel("Y (left)"); ax.set_zlabel("Z (up)")
        t_rel = timestamps[fi] - timestamps[0]
        ax.set_title(
            f"{Path(npz_path).name} — robot cmd vs operator | frame {fi}/{n_frames - 1} | t={t_rel:.2f}s",
            fontsize=10,
        )
        if rest is not None:
            for key in ("left", "right", "torso"):
                ax.scatter(*rest[key]["position"], color="grey", s=40, alpha=0.4, marker="s")
        # Robot wrist LINK (filled ball + solid axes) reconstructed from the sent
        # key-frame: link = key_frame - R @ offset (how the policy recovers the link).
        # A small dotted arm link->key-frame shows the offset; it should rotate in
        # place around a still link when the operator flips a palm.
        for j, side in enumerate(("left", "right")):
            kf = targets[fi, j, :3]
            rot = sRot.from_quat(targets[fi, j, 3:], scalar_first=True)
            link = kf - rot.apply(qm._WRIST_OFFSET[side])
            ax.scatter(*link, color=colors[j], s=70, zorder=10, label=f"{labels[j]} link cmd")
            _draw_coord_frame(ax, link, targets[fi, j, 3:], lw=2.0)
            ax.plot([link[0], kf[0]], [link[1], kf[1]], [link[2], kf[2]],
                    color=colors[j], linewidth=1.0, linestyle=":", alpha=0.7)
            ax.scatter(*kf, color=colors[j], s=15, alpha=0.5)  # key-frame point
        # Head command.
        ax.scatter(*targets[fi, 2, :3], color=colors[2], s=70, zorder=10, label="Head cmd")
        _draw_coord_frame(ax, targets[fi, 2, :3], targets[fi, 2, 3:], lw=2.0)
        # Operator: open balls + dashed axes + hand landmarks (wrist == link).
        o = op[fi]
        for j, side in enumerate(("left", "right")):
            opos = o[f"{side}_pos"]
            ax.scatter(*opos, facecolors="none", edgecolors=colors[j], s=80, zorder=9,
                       label=f"{labels[j]} operator")
            _draw_coord_frame(ax, opos, o[f"{side}_quat"], axis_len=_AXIS_LEN * 0.8,
                              lw=1.2, alpha=0.6, linestyle="--")
            _draw_hand(ax, o[f"{side}_lm"], lm_colors[side])
        ax.scatter(0, 0, 0, color="white", edgecolors="black", s=60, label="root")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.5)
        ax.text2D(0.01, 0.01,
                  "filled+solid = robot wrist link   dotted arm -> key-frame point   "
                  "open+dashed = operator   grey = rest   (link should overlap operator)",
                  transform=ax.transAxes, fontsize=7, color="grey")
        return []

    animation = anim_mod.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / args.fps, blit=False
    )
    if args.video:
        output = str(Path(npz_path).stem + "_calibrated.mp4") if args.output is None else args.output
        print(f"Rendering → {output} ...")
        try:
            animation.save(output, writer=anim_mod.FFMpegWriter(fps=args.fps, bitrate=1800), dpi=150)
            print(f"Saved: {output}")
        except Exception as exc:
            gif_path = str(Path(output).with_suffix(".gif"))
            print(f"FFMpeg failed ({exc}), falling back to GIF → {gif_path}")
            animation.save(gif_path, writer=anim_mod.PillowWriter(fps=args.fps))
            print(f"Saved: {gif_path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


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
    parser.add_argument(
        "--show-calibrated", action="store_true",
        help=(
            "Instead of raw Quest data, run the recording through the teleop "
            "calibration pipeline (rest-pose prefix + QuestThreePointPose) and "
            "animate the resulting robot-frame 3-pt targets vs. the robot rest pose."
        ),
    )
    parser.add_argument("--pos-scale", type=float, default=1.0,
                        help="--show-calibrated: head->wrist displacement scale (form factor)")
    parser.add_argument("--rest-hold-sec", type=float, default=1.0,
                        help="--show-calibrated: seconds to hold the prepended rest pose")
    parser.add_argument("--rest-interp-sec", type=float, default=1.5,
                        help="--show-calibrated: seconds to interpolate rest -> first frame")
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

    per_file = _process_calibrated if args.show_calibrated else _process_file
    for i, path in enumerate(files):
        if args.process_all:
            print(f"\n[{i + 1}/{len(files)}] {path}")
        per_file(path, args, plt, anim_mod)


if __name__ == "__main__":
    main()
