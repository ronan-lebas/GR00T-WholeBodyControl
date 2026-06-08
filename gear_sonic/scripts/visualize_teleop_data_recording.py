"""
Visualize a recorded teleop episode: commanded VR 3-point targets vs. the pose
the robot actually achieved.

Unlike visualize_quest_data.py (which only shows the raw Quest stream), this script
reads a LeRobot recording produced by run_data_exporter.py and overlays, in the
**robot root frame**, for L-wrist / R-wrist / Head:

  * COMMANDED  — the teleop target sent to the WBC
                 (teleop.vr_3pt_position + teleop.vr_3pt_orientation)
  * ACHIEVED   — where the robot's corresponding key frame ended up, obtained by
                 running FK on the recorded joints (observation.state) through the
                 *same* key-frame + offset convention used to build the command
                 (get_g1_key_frame_poses → left_wrist_yaw_link / right_wrist_yaw_link /
                  torso_link + the force.yaml offsets).

This apples-to-apples framing is the whole point: the commanded targets are defined
on the wrist-yaw key frames (with a +0.18 m forward offset), while
observation.eef_state is the *hand* frame — so naively plotting the command against
eef_state shows a spurious ~0.18 m gap that looks like a tracking failure but is just
a frame-convention difference. Pass --show-eef to additionally see the hand frame.

A per-key-point summary (mean position error, mean orientation error in degrees) is
printed to the console — a large, frame-varying orientation error points at a missing
coordinate-frame transform in the wrist-orientation mapping rather than a WBC
tracking problem.

Coordinate convention: robot root frame, X-forward / Y-left / Z-up. Orientation axes
are X=red, Y=green, Z=blue. Quaternions are scalar-first [w, x, y, z].

Usage:
    python gear_sonic/scripts/visualize_teleop_data_recording.py                       # latest dir in outputs/, GUI
    python gear_sonic/scripts/visualize_teleop_data_recording.py outputs/2026-06-04-14-24-05
    python gear_sonic/scripts/visualize_teleop_data_recording.py <dir> --video
    python gear_sonic/scripts/visualize_teleop_data_recording.py <dir> --show-eef --downsample 3
    python gear_sonic/scripts/visualize_teleop_data_recording.py --all --video         # every dir in outputs/
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.utils.data_collection.transforms import rot6d_to_quat
from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses

# Key-point order is shared everywhere: [L-wrist, R-wrist, Head].
_KEY_LABELS = ["L-wrist", "R-wrist", "Head"]
_KEY_COLORS = ["royalblue", "crimson", "dimgrey"]
# FK key-frame names returned by get_g1_key_frame_poses, aligned to _KEY_LABELS.
_FK_KEYS = ["left_wrist", "right_wrist", "torso"]

_AXIS_LEN_CMD = 0.08   # commanded orientation axis length (m)
_AXIS_LEN_ACH = 0.055  # achieved orientation axis length (m)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _find_parquet(dataset_dir: str, episode: int) -> str:
    """Return the parquet path for the requested episode index inside a recording dir."""
    files = sorted(glob.glob(str(Path(dataset_dir) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under '{dataset_dir}/data/'.")
    if episode >= len(files):
        raise IndexError(f"Episode {episode} requested but only {len(files)} episode(s) found.")
    return files[episode]


def _stack(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack an object column of per-frame arrays into a single (N, D) float array."""
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[col]], axis=0)


def _scalar(df: pd.DataFrame, col: str) -> np.ndarray:
    """Read a scalar-per-frame column to a (N,) array."""
    return np.array([np.asarray(v).flat[0] for v in df[col]])


def _rows_to_quat(rot6d_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, 6) rot6d → (N, 3, 4) wxyz quats plus a (N, 3) validity mask.

    Rows that are exactly zero (unset orientation) are flagged invalid and left as
    identity so downstream drawing/error code can skip them.
    """
    n, k, _ = rot6d_rows.shape
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, k, 1)).astype(np.float64)
    valid = np.zeros((n, k), dtype=bool)
    for i in range(n):
        for j in range(k):
            if np.abs(rot6d_rows[i, j]).sum() > 1e-6:
                quats[i, j] = rot6d_to_quat(rot6d_rows[i, j])
                valid[i, j] = True
    return quats, valid


def _load_recording(parquet_path: str, hand_type: str | None):
    """Load one episode and precompute commanded + achieved key-point poses.

    Returns a dict of stacked arrays ready for animation / summary.
    """
    df = pd.read_parquet(parquet_path)
    n = len(df)

    cmd_pos = _stack(df, "teleop.vr_3pt_position").reshape(n, 3, 3)
    cmd_quat, cmd_valid = _rows_to_quat(_stack(df, "teleop.vr_3pt_orientation").reshape(n, 3, 6))

    eef = _stack(df, "observation.eef_state")  # (n, 14): Lpos,Lquat,Rpos,Rquat
    eef_pos = np.stack([eef[:, 0:3], eef[:, 7:10]], axis=1)    # (n, 2, 3)
    eef_quat = np.stack([eef[:, 3:7], eef[:, 10:14]], axis=1)  # (n, 2, 4) wxyz

    state = _stack(df, "observation.state")  # (n, num_joints)
    stream_mode = _scalar(df, "teleop.stream_mode").astype(int)
    timestamps = _scalar(df, "timestamp").astype(np.float64)

    # Pick the model variant from the joint-config width (43 = dex3, 51 = brainco).
    if hand_type is None:
        hand_type = "brainco" if state.shape[1] == 51 else "dex3"
    print(f"[teleop-vis] hand_type={hand_type}  (observation.state dim {state.shape[1]})")
    robot_model = get_g1_robot_model(hand_type=hand_type)

    # Achieved key-point poses via FK in the SAME convention as the command.
    ach_pos = np.zeros((n, 3, 3))
    ach_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 3, 1)).astype(np.float64)
    for i in range(n):
        poses = get_g1_key_frame_poses(robot_model, q=state[i])
        for j, fk_key in enumerate(_FK_KEYS):
            ach_pos[i, j] = poses[fk_key]["position"]
            ach_quat[i, j] = poses[fk_key]["orientation_wxyz"]

    return {
        "name": Path(parquet_path).parent.parent.parent.name,
        "n": n,
        "cmd_pos": cmd_pos,
        "cmd_quat": cmd_quat,
        "cmd_valid": cmd_valid,
        "ach_pos": ach_pos,
        "ach_quat": ach_quat,
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "stream_mode": stream_mode,
        "timestamps": timestamps,
    }


# ---------------------------------------------------------------------------
# Quantitative summary
# ---------------------------------------------------------------------------


def _print_error_summary(data: dict) -> None:
    """Print mean position / orientation error (commanded vs achieved) per key point.

    Restricted to VR_3PT frames (stream_mode == 5) with a valid commanded orientation.
    """
    sm = data["stream_mode"]
    mask = sm == 5
    if not mask.any():
        print("[teleop-vis] No VR_3PT (stream_mode==5) frames — skipping error summary.")
        return

    print(f"\n[teleop-vis] Command-vs-achieved error over {mask.sum()} VR_3PT frames:")
    print(f"  {'key':<9}{'pos err (m)':>14}{'  ':>2}{'ori err (deg)':>16}")
    for j, label in enumerate(_KEY_LABELS):
        pos_err = np.linalg.norm(data["cmd_pos"][mask, j] - data["ach_pos"][mask, j], axis=1)
        ovalid = mask & data["cmd_valid"][:, j]
        if ovalid.any():
            cmd_r = sRot.from_quat(data["cmd_quat"][ovalid, j], scalar_first=True)
            ach_r = sRot.from_quat(data["ach_quat"][ovalid, j], scalar_first=True)
            ang = np.degrees((cmd_r.inv() * ach_r).magnitude())
            ori_str = f"{ang.mean():6.1f} (max {ang.max():.0f})"
        else:
            ori_str = "   n/a"
        print(f"  {label:<9}{pos_err.mean():>9.3f} (max {pos_err.max():.2f})  {ori_str:>16}")
    print(
        "  NOTE: large, frame-varying orientation error => likely a missing wrist-frame\n"
        "        transform in the manager; small/constant => tracking is fine.\n"
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_frame_axes(ax, pos, quat_wxyz, axis_len, alpha, lw) -> None:
    """Draw RGB=XYZ orientation axes at pos."""
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-6:
        return
    rot = sRot.from_quat(quat_wxyz / norm, scalar_first=True)
    axes = rot.apply(np.eye(3)) * axis_len
    for i, c in enumerate(("red", "green", "blue")):
        ax.quiver(
            pos[0], pos[1], pos[2],
            axes[i, 0], axes[i, 1], axes[i, 2],
            color=c, linewidth=lw, alpha=alpha, arrow_length_ratio=0.3,
        )


def _process_file(data: dict, args, plt, anim_mod) -> None:
    name = data["name"]
    n = data["n"]
    idx = np.arange(0, n, args.downsample)
    duration = float(data["timestamps"][-1] - data["timestamps"][0]) if n > 1 else 0.0
    print(f"{name}  |  {n} frames  |  {duration:.1f}s  |  ~{n / max(duration, 1e-6):.1f} fps")

    pt_sets = [data["cmd_pos"][idx].reshape(-1, 3), data["ach_pos"][idx].reshape(-1, 3)]
    if args.show_eef:
        pt_sets.append(data["eef_pos"][idx].reshape(-1, 3))
    all_pts = np.concatenate(pt_sets, axis=0)
    margin = 0.05
    lims = [
        (float(all_pts[:, d].min()) - margin, float(all_pts[:, d].max()) + margin) for d in range(3)
    ]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def _reset_ax() -> None:
        ax.cla()
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel("X  (forward)"); ax.set_ylabel("Y  (left)"); ax.set_zlabel("Z  (up)")

    def update(fi: int):
        _reset_ax()
        t_rel = data["timestamps"][fi] - data["timestamps"][0]
        ax.set_title(
            f"{name}  |  frame {fi}/{n - 1}  |  t={t_rel:.2f}s  |  stream_mode={data['stream_mode'][fi]}",
            fontsize=10,
        )
        for j, (label, color) in enumerate(zip(_KEY_LABELS, _KEY_COLORS)):
            cpos = data["cmd_pos"][fi, j]
            apos = data["ach_pos"][fi, j]
            # Commanded: filled ball + bold axes. (cla() clears each frame, so label every frame.)
            ax.scatter(*cpos, color=color, s=70, zorder=10, label=f"{label} cmd")
            _draw_frame_axes(ax, cpos, data["cmd_quat"][fi, j], _AXIS_LEN_CMD, alpha=1.0, lw=2.0)
            # Achieved: open ball + faint axes.
            ax.scatter(*apos, facecolors="none", edgecolors=color, s=70, zorder=9,
                       label=f"{label} achieved")
            _draw_frame_axes(ax, apos, data["ach_quat"][fi, j], _AXIS_LEN_ACH, alpha=0.5, lw=1.2)
            # Error line connecting commanded ↔ achieved.
            ax.plot([cpos[0], apos[0]], [cpos[1], apos[1]], [cpos[2], apos[2]],
                    color="grey", linestyle="--", linewidth=1.0, alpha=0.7)

        if args.show_eef:
            for j, label in enumerate(["L-wrist", "R-wrist"]):
                epos = data["eef_pos"][fi, j]
                ax.scatter(*epos, marker="x", color=_KEY_COLORS[j], s=50, zorder=8,
                           label=f"{label} eef(hand)")
                _draw_frame_axes(ax, epos, data["eef_quat"][fi, j], _AXIS_LEN_ACH, alpha=0.4, lw=1.0)

        ax.legend(loc="upper left", fontsize=7, framealpha=0.5)
        ax.text2D(0.01, 0.01,
                  "filled=cmd  open=achieved(FK)  x=eef(hand)   axes X=red Y=green Z=blue",
                  transform=ax.transAxes, fontsize=7, color="grey")
        return []

    animation = anim_mod.FuncAnimation(
        fig, update, frames=idx, interval=1000.0 / args.fps, blit=False
    )

    if args.video:
        output = str(Path(name).stem + "_teleop.mp4") if args.output is None else args.output
        print(f"Rendering → {output} ...")
        try:
            writer = anim_mod.FFMpegWriter(fps=args.fps, bitrate=1800)
            animation.save(output, writer=writer, dpi=150)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare commanded VR 3-pt teleop targets vs. achieved robot poses."
    )
    parser.add_argument(
        "dataset", nargs="?", default=None,
        help="Recording directory (default: most recent under --data-dir)",
    )
    parser.add_argument(
        "--all", dest="process_all", action="store_true",
        help="Process every recording directory under --data-dir",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index within the recording")
    parser.add_argument("--video", action="store_true", help="Render to a file instead of GUI")
    parser.add_argument("--output", default=None, help="Output video path (single-recording mode)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback / render FPS")
    parser.add_argument("--downsample", type=int, default=1, metavar="N", help="Use every N-th frame")
    parser.add_argument("--data-dir", default="outputs", help="Root dir of recordings (default: outputs)")
    parser.add_argument(
        "--hand-type", choices=["dex3", "brainco"], default=None,
        help="Override hand type (default: inferred from observation.state width)",
    )
    parser.add_argument(
        "--show-eef", action="store_true",
        help="Also overlay observation.eef_state (hand frame) — offset from the cmd key frame",
    )
    args = parser.parse_args()

    # ---- resolve recording directories ----
    if args.process_all:
        dirs = sorted(
            d for d in glob.glob(str(Path(args.data_dir) / "*"))
            if glob.glob(str(Path(d) / "data" / "**" / "*.parquet"), recursive=True)
        )
        if not dirs:
            print(f"No recordings with parquet data found in '{args.data_dir}'.")
            sys.exit(1)
        if args.output is not None:
            print("Warning: --output is ignored with --all.")
            args.output = None
    elif args.dataset is not None:
        dirs = [args.dataset]
    else:
        candidates = sorted(
            (d for d in glob.glob(str(Path(args.data_dir) / "*"))
             if glob.glob(str(Path(d) / "data" / "**" / "*.parquet"), recursive=True)),
            key=lambda d: Path(d).stat().st_mtime,
        )
        if not candidates:
            print(f"No recordings found in '{args.data_dir}'. Pass a directory path.")
            sys.exit(1)
        dirs = [candidates[-1]]
        print(f"Using: {dirs[0]}")

    # ---- matplotlib backend (must be set before pyplot import) ----
    if args.video:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.animation as anim_mod
    import matplotlib.pyplot as plt

    for i, d in enumerate(dirs):
        if args.process_all:
            print(f"\n[{i + 1}/{len(dirs)}] {d}")
        try:
            data = _load_recording(_find_parquet(d, args.episode), args.hand_type)
        except (FileNotFoundError, IndexError) as e:
            print(f"  Skipping '{d}': {e}")
            continue
        _print_error_summary(data)
        _process_file(data, args, plt, anim_mod)


if __name__ == "__main__":
    main()
