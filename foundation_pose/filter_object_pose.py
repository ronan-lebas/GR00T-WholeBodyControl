#!/usr/bin/env python3
"""Temporally filter the FoundationPose object trajectory in the body-anchored frame.

FoundationPose estimates the object only in the *camera* frame (``ob_in_cam``), and
the camera is bolted to the head — so when the arms move, ``ob_in_cam`` swings
mostly because the *camera* shakes, not the box. We first cancel that with forward
kinematics (the same transform the visualizer uses):

    T_anchor_obj(t) = T_anchor_cam(t) @ GL_FROM_CV @ ob_in_cam(t)

``T_anchor_cam`` comes from the robot joints (encoders) + the gravity-aligned IMU
base orientation, i.e. only quantities available on the real robot — never the
world position. In this body-anchored frame the *true* object motion is slow, so
the leftover high-frequency content is genuine FoundationPose noise, which we
attenuate with:

  * translation : a constant-velocity Kalman filter (predicts through the lag,
    gives less lag than a plain EMA for the same smoothing), and
  * orientation : a SLERP exponential moving average,

both with **outlier gating** — measurements whose innovation is implausibly large
(FoundationPose symmetry flips / spikes) are rejected and the filter coasts on its
prediction instead of smearing the spike.

The filtered poses are written in the body-anchored WORLD frame to
``ob_in_world_filtered/`` (4x4 per object frame). They are NOT re-projected to the
camera frame: the visualizer places them directly, so it never re-multiplies by
the per-frame (vibrating) camera FK — doing so would re-inject the camera wobble
(amplified by the ~1.8 m lever to the object) that the filter just removed, which
is why a camera-frame round-trip leaves the replay looking unfiltered. View with
``visualize_robot_object_trajectory.py --filtered``.

Usage (needs gear_sonic[sim] — mujoco, pin, scipy, pyarrow):
    python foundation_pose/filter_object_pose.py
    python foundation_pose/filter_object_pose.py --trajectory outputs/2026-06-15-12-54-46
    python foundation_pose/filter_object_pose.py --sigma-a 1.5 --ori-tau 0.08
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
    if not VIS_SCRIPT.is_file():
        sys.exit(f"[error] visualizer script not found: {VIS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("viz_traj", VIS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def _cv_process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """6x6 process noise for a constant-velocity model driven by white accel."""
    I = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = (dt**4 / 4) * I
    Q[:3, 3:] = (dt**3 / 2) * I
    Q[3:, :3] = (dt**3 / 2) * I
    Q[3:, 3:] = (dt**2) * I
    return sigma_a**2 * Q


def kalman_cv(pos: np.ndarray, dt: np.ndarray, sigma_a: float, sigma_m: float,
              gate_chi2: float) -> tuple[np.ndarray, int]:
    """Constant-velocity Kalman smoothing of a (M,3) position track.

    State [p(3), v(3)]. ``sigma_a`` = process accel std (m/s^2, larger -> tracks
    motion more tightly), ``sigma_m`` = measurement std (m, larger -> smoother).
    ``gate_chi2`` is the 3-DOF Mahalanobis threshold above which a measurement is
    treated as an outlier (rejected; the filter coasts on its prediction).
    """
    m = len(pos)
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    Rm = (sigma_m**2) * np.eye(3)

    x = np.zeros(6)
    x[:3] = pos[0]
    P = np.eye(6)
    P[:3, :3] *= sigma_m**2
    P[3:, 3:] *= 1.0  # 1 (m/s)^2 initial velocity uncertainty

    out = np.zeros((m, 3))
    out[0] = pos[0]
    n_rej = 0
    for k in range(1, m):
        d = float(dt[k])
        F = np.eye(6)
        F[:3, 3:] = d * np.eye(3)
        x = F @ x
        P = F @ P @ F.T + _cv_process_noise(d, sigma_a)

        y = pos[k] - H @ x
        S = H @ P @ H.T + Rm
        if float(y @ np.linalg.solve(S, y)) <= gate_chi2:
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P
        else:
            n_rej += 1
        out[k] = x[:3]
    return out, n_rej


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-path normalized lerp between wxyz quaternions (fine for short steps)."""
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - a) * q0 + a * q1
    return q / max(np.linalg.norm(q), 1e-9)


def slerp_ema(quats: np.ndarray, dt: np.ndarray, tau: float,
              gate_deg: float) -> tuple[np.ndarray, int]:
    """SLERP exponential moving average over a (M,4) wxyz quaternion track.

    alpha = dt / (tau + dt) (frame-rate independent). Measurements more than
    ``gate_deg`` from the current estimate are rejected (FP flips); the estimate
    is held. ``tau <= 0`` disables smoothing.
    """
    m = len(quats)
    out = np.zeros_like(quats)
    qf = quats[0].copy()
    out[0] = qf
    n_rej = 0
    for k in range(1, m):
        qm = quats[k]
        if np.dot(qf, qm) < 0.0:
            qm = -qm
        ang = 2.0 * np.degrees(np.arccos(np.clip(abs(np.dot(qf, qm)), -1.0, 1.0)))
        if ang > gate_deg:
            n_rej += 1
            out[k] = qf
            continue
        a = 1.0 if tau <= 0.0 else float(dt[k]) / (tau + float(dt[k]))
        qf = _slerp(qf, qm, a)
        out[k] = qf
    return out, n_rej


# --------------------------------------------------------------------------- #
# Jitter metric (high-frequency residual)
# --------------------------------------------------------------------------- #


def trans_jitter_mm(pos: np.ndarray, w: int) -> float:
    resid = pos - uniform_filter1d(pos, size=w, axis=0, mode="nearest")
    return float(np.sqrt(np.mean(np.square(resid)))) * 1000.0


def rot_jitter_deg(quats: np.ndarray, w: int) -> float:
    rots = R.from_quat(quats, scalar_first=True)
    rotvec = (rots.mean().inv() * rots).as_rotvec()
    resid = rotvec - uniform_filter1d(rotvec, size=w, axis=0, mode="nearest")
    return float(np.degrees(np.sqrt(np.mean(np.sum(np.square(resid), axis=1)))))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory", default=None, help="recording folder (default: most recent)")
    parser.add_argument("--episode", type=int, default=0, help="episode index (default: 0)")
    parser.add_argument("--sigma-a", type=float, default=0.3,
                        help="KF process accel std (m/s^2); larger tracks motion tighter "
                             "(less lag, less smoothing)")
    parser.add_argument("--sigma-m", type=float, default=0.02,
                        help="KF measurement std (m); larger = smoother (more lag)")
    parser.add_argument("--pos-gate", type=float, default=16.27,
                        help="position outlier gate (3-DOF chi^2; 16.27 ~ p=0.999)")
    parser.add_argument("--ori-tau", type=float, default=0.1,
                        help="orientation SLERP-EMA time constant (s); 0 disables")
    parser.add_argument("--ori-gate-deg", type=float, default=30.0,
                        help="orientation outlier gate (deg) against the current estimate")
    parser.add_argument("--window", type=int, default=9,
                        help="moving-average window (obj frames) for the before/after metric")
    parser.add_argument("--out-name", default="ob_in_world_filtered",
                        help="output subfolder under the episode dir")
    args = parser.parse_args()

    viz = _load_viz_module()

    traj, parquet, mesh, ob_dir = viz.resolve_paths(args)
    states = viz.load_robot_states(parquet)
    base_quats = viz.load_base_quats(parquet)
    obj_poses = viz.load_object_poses(ob_dir)  # (M,4,4) object-in-camera
    obj_to_robot = viz.load_frame_map(ob_dir, obj_poses.shape[0])
    fps = viz.read_fps(traj)

    replay = viz.TrajectoryReplay(states, obj_poses, mesh, base_quats, obj_to_robot)
    n, m = replay.n, replay.m

    if obj_to_robot is not None:
        robot_of_obj = np.clip(obj_to_robot, 0, n - 1).astype(int)
        pairing = "exact (frame_map.txt)"
    else:
        robot_of_obj = np.zeros(m, dtype=int) if (n <= 1 or m <= 1) else np.round(
            np.arange(m) * (n - 1) / (m - 1)
        ).astype(int)
        pairing = "uniform resampling (no map)"

    obj_t = robot_of_obj / fps
    dt = np.diff(obj_t, prepend=obj_t[0] - 1.0 / fps)
    dt[dt <= 0] = 1.0 / fps  # guard repeated/duplicate rows

    print(f"[info] trajectory : {traj}")
    print(f"[info] object frames: {m} | fps: {fps:.0f} | pairing: {pairing}")

    # Body-anchored object pose + camera pose per object frame (FK cancellation).
    box_pos = np.zeros((m, 3))
    box_quat = np.zeros((m, 4))
    t_wc = np.zeros((m, 4, 4))
    for j in range(m):
        replay.set_frame(int(robot_of_obj[j]))
        T = np.eye(4)
        T[:3, :3] = replay.data.cam_xmat[replay.cam_id].reshape(3, 3)
        T[:3, 3] = replay.data.cam_xpos[replay.cam_id]
        t_wc[j] = T
        t_wo = T @ viz.GL_FROM_CV @ obj_poses[j]
        box_pos[j] = t_wo[:3, 3]
        box_quat[j] = R.from_matrix(t_wo[:3, :3]).as_quat(scalar_first=True)

    # Filter in the body-anchored frame.
    f_pos, n_rej_p = kalman_cv(box_pos, dt, args.sigma_a, args.sigma_m, args.pos_gate)
    f_quat, n_rej_o = slerp_ema(box_quat, dt, args.ori_tau, args.ori_gate_deg)

    # Write the filtered poses in the body-anchored WORLD frame (4x4 per obj
    # frame). Crucially NOT re-projected to camera frame: the visualizer places
    # these directly, so it never re-multiplies by the per-frame (vibrating)
    # camera FK — which would re-inject the camera wobble (x the ~1.8 m lever to
    # the object) the filter just removed.
    out_dir = ob_dir.parent / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [f.stem for f in sorted(ob_dir.glob("*.txt"))]
    for j in range(m):
        t_wo = np.eye(4)
        t_wo[:3, :3] = R.from_quat(f_quat[j], scalar_first=True).as_matrix()
        t_wo[:3, 3] = f_pos[j]
        np.savetxt(out_dir / f"{names[j]}.txt", t_wo.reshape(4, 4))

    w = max(3, args.window | 1)
    print("\n=== body-anchored jitter (high-freq RMS): before -> after ===")
    print(f"  translation : {trans_jitter_mm(box_pos, w):6.2f} -> "
          f"{trans_jitter_mm(f_pos, w):6.2f} mm")
    print(f"  rotation    : {rot_jitter_deg(box_quat, w):6.3f} -> "
          f"{rot_jitter_deg(f_quat, w):6.3f} deg")
    print(f"  outliers gated: {n_rej_p} pos / {n_rej_o} ori (of {m} frames)")
    print(f"\n[done] filtered poses -> {out_dir}")
    print("       view with: python gear_sonic/scripts/visualize_robot_object_trajectory.py "
          f"--trajectory outputs/{traj.name} --episode {args.episode} --filtered")


if __name__ == "__main__":
    main()
