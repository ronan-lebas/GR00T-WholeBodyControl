#!/usr/bin/env python3
"""Diagnose where the tracked-object jitter comes from: FoundationPose error vs.
camera-pose (T_world_cam) mismatch.

The world pose of the box in ``visualize_robot_object_trajectory.py`` is

    T_world_obj(t) = T_world_cam(t) @ GL_FROM_CV @ ob_in_cam(t)

The camera vibration lives in *both* terms. If T_world_cam(t) were exact, the
two would cancel and the world box would be stable without any filtering, so any
residual world-frame jitter is

    (FoundationPose estimation error)  +  (T_world_cam reconstruction error)

This script quantifies the high-frequency ("jitter") content of three signals,
sampled at the object-tracking rate, and reports how strongly the *world-box*
jitter tracks the *camera ego-motion*:

  - camera-frame box   : ob_in_cam translation/rotation (FoundationPose output)
  - world-frame box     : the reconstructed T_world_obj (what the viz shows)
  - camera ego-motion   : T_world_cam from robot FK (yaw-zeroed base, like viz)

Interpretation heuristic (printed at the end):
  * world-box jitter strongly correlated with camera ego-motion  -> the
    cancellation is failing, i.e. T_world_cam is the problem. Biggest win:
    record exact per-frame camera extrinsics at collection time instead of
    reconstructing approximate FK here.
  * world-box jitter ~ camera-frame box jitter and *uncorrelated* with camera
    motion -> intrinsic FoundationPose noise dominates. Fix by filtering the
    world-frame pose (EMA / constant-velocity Kalman with outlier gating).

Note: the viz FK uses the model's default base *position* (qpos0) and a
yaw-zeroed base *orientation* — any linear torso vibration is simply absent from
T_world_cam and therefore cannot cancel. That class of error shows up as
world-box jitter that does NOT correlate with the camera motion we can see here.

Usage (needs gear_sonic[sim] — mujoco, pin, scipy, pyarrow; matplotlib optional):
    python foundation_pose/diagnose_pose_jitter.py
    python foundation_pose/diagnose_pose_jitter.py --trajectory outputs/2026-06-15-11-30-13
    python foundation_pose/diagnose_pose_jitter.py --episode 0 --window 9 --no-plot
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent.parent
VIS_SCRIPT = REPO_ROOT / "gear_sonic" / "scripts" / "visualize_robot_object_trajectory.py"


def _load_viz_module():
    """Import the visualizer script as a module so we reuse its exact FK / paths."""
    if not VIS_SCRIPT.is_file():
        sys.exit(f"[error] visualizer script not found: {VIS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("viz_traj", VIS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Jitter metrics
# --------------------------------------------------------------------------- #


def high_freq_residual(x: np.ndarray, window: int) -> np.ndarray:
    """High-frequency part of a (T, D) signal = x minus its moving average.

    The moving average is the low-frequency 'true' motion; the residual is the
    jitter we want to measure. ``mode='nearest'`` avoids edge ringing.
    """
    smooth = uniform_filter1d(x, size=window, axis=0, mode="nearest")
    return x - smooth


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def per_frame_norm(resid: np.ndarray) -> np.ndarray:
    """(T, D) residual -> (T,) per-frame euclidean norm (a jitter time series)."""
    return np.linalg.norm(resid, axis=1)


def rot_residual_deg(rots: R, window: int) -> tuple[np.ndarray, float]:
    """High-frequency angular residual of a rotation sequence.

    Express each rotation as a rotvec relative to the sequence mean (locally
    linear for small jitter), smooth, and take the residual. Returns the
    per-frame residual angle (deg) and its RMS (deg).
    """
    mean = rots.mean()
    rotvec = (mean.inv() * rots).as_rotvec()  # (T, 3), radians
    resid = high_freq_residual(rotvec, window)
    ang = np.degrees(np.linalg.norm(resid, axis=1))
    return ang, float(np.sqrt(np.mean(np.square(ang))))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else 0.0


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
        "--window", type=int, default=9,
        help="moving-average window (frames) separating signal from jitter (odd; default 9)",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    parser.add_argument(
        "--out", default=None,
        help="PNG path for the figure (default: <trajectory>/pose_jitter_ep<NN>.png)",
    )
    args = parser.parse_args()

    viz = _load_viz_module()

    traj, parquet, mesh, ob_dir = viz.resolve_paths(args)
    states = viz.load_robot_states(parquet)
    base_quats = viz.load_base_quats(parquet)
    obj_poses = viz.load_object_poses(ob_dir)  # (M, 4, 4) object-in-camera
    obj_to_robot = viz.load_frame_map(ob_dir, obj_poses.shape[0])
    fps = viz.read_fps(traj)

    replay = viz.TrajectoryReplay(states, obj_poses, mesh, base_quats, obj_to_robot)
    n, m = replay.n, replay.m

    # Robot frame paired with each object frame: exact map if present, else the
    # uniform nearest-neighbour pairing (the source of timing-induced jitter).
    if obj_to_robot is not None:
        robot_of_obj = np.clip(obj_to_robot, 0, n - 1).astype(int)
        pairing = "exact (frame_map.txt)"
    else:
        robot_of_obj = np.zeros(m, dtype=int) if (n <= 1 or m <= 1) else np.round(
            np.arange(m) * (n - 1) / (m - 1)
        ).astype(int)
        pairing = "uniform resampling (no map)"

    # Effective time of each object frame and inter-frame spacing (FP is sparse).
    obj_t = robot_of_obj / fps
    obj_dt = float(np.median(np.diff(obj_t))) if m > 1 else 1.0 / fps

    print(f"[info] trajectory : {traj}")
    print(f"[info] robot frames: {n} | object frames: {m} | fps: {fps:.0f}")
    print(f"[info] FP<->proprio pairing: {pairing}")
    print(f"[info] jitter window: {args.window} obj frames (~{args.window * obj_dt * 1000:.0f} ms)")

    cam_pos = np.zeros((m, 3))
    cam_rot = np.zeros((m, 3, 3))
    box_cam_pos = obj_poses[:, :3, 3].copy()  # FoundationPose, camera frame
    box_world_pos = np.zeros((m, 3))
    box_cam_rot = np.zeros((m, 3, 3))
    box_world_rot = np.zeros((m, 3, 3))

    for j in range(m):
        i = int(robot_of_obj[j])
        replay.set_frame(i)  # sets robot joints + base, runs mj_forward
        t_wc = np.eye(4)
        t_wc[:3, :3] = replay.data.cam_xmat[replay.cam_id].reshape(3, 3)
        t_wc[:3, 3] = replay.data.cam_xpos[replay.cam_id]
        cam_pos[j] = t_wc[:3, 3]
        cam_rot[j] = t_wc[:3, :3]

        t_wo = t_wc @ viz.GL_FROM_CV @ obj_poses[j]
        box_world_pos[j] = t_wo[:3, 3]
        box_world_rot[j] = t_wo[:3, :3]
        box_cam_rot[j] = obj_poses[j][:3, :3]

    w = max(3, args.window | 1)  # force odd, >=3

    # Translation jitter (RMS, mm) and per-frame jitter time series.
    cam_box_resid = high_freq_residual(box_cam_pos, w)
    world_box_resid = high_freq_residual(box_world_pos, w)
    cam_ego_resid = high_freq_residual(cam_pos, w)

    cam_box_jit = per_frame_norm(cam_box_resid)
    world_box_jit = per_frame_norm(world_box_resid)
    cam_ego_jit = per_frame_norm(cam_ego_resid)

    # Rotation jitter (RMS, deg) and per-frame angular jitter time series.
    cam_box_ang, cam_box_ang_rms = rot_residual_deg(R.from_matrix(box_cam_rot), w)
    world_box_ang, world_box_ang_rms = rot_residual_deg(R.from_matrix(box_world_rot), w)
    cam_ego_ang, cam_ego_ang_rms = rot_residual_deg(R.from_matrix(cam_rot), w)

    # Correlation: does the world-box jitter track the camera ego-motion?
    corr_pos = pearson(world_box_jit, cam_ego_jit)
    corr_ang = pearson(world_box_ang, cam_ego_ang)

    print("\n=== translation jitter (high-freq RMS) ===")
    print(f"  camera-frame box (FoundationPose) : {rms(cam_box_resid) * 1000:7.2f} mm")
    print(f"  world-frame box (visualized)       : {rms(world_box_resid) * 1000:7.2f} mm")
    print(f"  camera ego-motion (FK)             : {rms(cam_ego_resid) * 1000:7.2f} mm")
    print("\n=== rotation jitter (high-freq RMS) ===")
    print(f"  camera-frame box (FoundationPose) : {cam_box_ang_rms:7.3f} deg")
    print(f"  world-frame box (visualized)       : {world_box_ang_rms:7.3f} deg")
    print(f"  camera ego-motion (FK)             : {cam_ego_ang_rms:7.3f} deg")
    print("\n=== correlation: world-box jitter vs camera ego-motion ===")
    print(f"  translation : r = {corr_pos:+.2f}")
    print(f"  rotation    : r = {corr_ang:+.2f}")

    # Heuristic verdict.
    print("\n=== verdict ===")
    if corr_pos > 0.5 or corr_ang > 0.5:
        print(
            "  World-box jitter tracks the camera ego-motion (r > 0.5): the\n"
            "  T_world_cam @ ob_in_cam cancellation is failing. Biggest win is to\n"
            "  record exact per-frame camera extrinsics at collection time rather\n"
            "  than reconstructing approximate FK (yaw-zeroed, default base position)."
        )
    elif rms(world_box_resid) > 1.5 * rms(cam_box_resid):
        print(
            "  World-box jitter exceeds camera-frame box jitter but is uncorrelated\n"
            "  with the camera motion we can see — consistent with the missing base\n"
            "  *translation* (not in the FK). Recording true extrinsics would still help."
        )
    else:
        print(
            "  World-box jitter ~ camera-frame box jitter and uncorrelated with camera\n"
            "  motion: intrinsic FoundationPose noise dominates. Filter the world-frame\n"
            "  pose (EMA / constant-velocity Kalman with outlier gating)."
        )

    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not available — skipping figure (--no-plot to silence)")
        return

    t = obj_t
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    labels = ["x", "y", "z"]

    def plot_xyz(a, raw, title):
        sm = uniform_filter1d(raw, size=w, axis=0, mode="nearest")
        for d in range(3):
            line, = a.plot(t, raw[:, d] * 100, alpha=0.45, lw=0.8)
            a.plot(t, sm[:, d] * 100, color=line.get_color(), lw=1.8, label=labels[d])
        a.set_title(title)
        a.set_ylabel("cm")
        a.legend(loc="upper right", ncol=3, fontsize=8)
        a.grid(alpha=0.3)

    plot_xyz(ax[0], box_cam_pos, "Camera-frame box (FoundationPose ob_in_cam)")
    plot_xyz(ax[1], box_world_pos, "World-frame box (visualized T_world_obj)")
    plot_xyz(ax[2], cam_pos, "Camera ego-motion (T_world_cam, FK)")

    ax[3].plot(t, world_box_jit * 1000, label=f"world-box jitter (r={corr_pos:+.2f})", lw=1.2)
    ax[3].plot(t, cam_ego_jit * 1000, label="camera ego-motion jitter", lw=1.2, alpha=0.8)
    ax[3].set_title("Per-frame translation jitter (high-freq residual norm)")
    ax[3].set_ylabel("mm")
    ax[3].set_xlabel("time (s)")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out) if args.out else traj / f"pose_jitter_ep{args.episode:02d}.png"
    fig.savefig(out, dpi=120)
    print(f"\n[plot] saved figure to {out}")


if __name__ == "__main__":
    main()
