"""Convert legacy ``object_gt`` recordings to the world-frame schema.

The ground-truth object writer used to store the box relative to a robot link
(``ob_in_cam`` — head-camera frame — or ``ob_in_ref`` — right-foot frame). Expressing the
box relative to a *moving* link folds that link's motion into the stored pose, so a
physically static cube drifts/jitters on replay, and the placement depends on how faithfully
the replay reconstructs that link from proprio. The current schema instead stores the box's
absolute world pose (``ob_in_world``) plus the reference body's world pose (``ref_in_world``)
sampled at the same instant, and the replay anchors with a single constant transform.

Legacy recordings never stored the *sim's* link pose, so an exact conversion is impossible.
This script reconstructs the reference-body world pose the same way the replay does — robot
FK from the recorded proprio (joints + base orientation), feet planted at the frame-0
location — and writes::

    ref_in_world = replay_reference_pose(frame)
    ob_in_world  = replay_reference_pose(frame) @ ob_in_ref     # (or GL-converted ob_in_cam)

Because the reference body (right foot) is near-static, ``replay_reference_pose`` closely
matches the sim's link pose, so the reconstructed world poses reproduce what the *old*
visualizer showed — but now stored absolutely, so the fixed visualizer treats them as the
ground truth and no longer re-injects robot motion. Freshly recorded data (sim publishing
``ob_in_world``/``ref_in_world`` directly) is exact and needs no conversion.

Usage (needs gear_sonic[sim]):
    python gear_sonic/scripts/convert_object_gt_to_world.py --trajectory outputs/<ts>
    python gear_sonic/scripts/convert_object_gt_to_world.py --trajectory outputs/<ts> --episode 4
    python gear_sonic/scripts/convert_object_gt_to_world.py --trajectory outputs/<ts> --dry-run

Each converted parquet is backed up to ``<name>.parquet.legacy`` before being overwritten.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import gear_sonic.scripts.visualize_robot_object_trajectory as V


def _resolve_proprio_parquet(traj: Path, episode: int) -> Path:
    parquets = sorted((traj / "data").rglob("*.parquet"))
    if not parquets:
        sys.exit(f"[error] no proprio parquet under {traj / 'data'}")
    return next(
        (p for p in parquets if p.name == f"episode_{episode:06d}.parquet"),
        parquets[0],
    )


def convert_episode(traj: Path, gt_parquet: Path, dry_run: bool) -> bool:
    """Convert one object_gt parquet in place. Returns True if it was (re)written."""
    episode = int(gt_parquet.stem.split("_")[-1])
    table = pq.read_table(gt_parquet)
    cols = set(table.column_names)

    if {"ob_in_world", "ref_in_world"}.issubset(cols):
        print(f"[skip] {gt_parquet.name}: already world-frame")
        return False

    if "ob_in_ref" in cols:
        legacy_col, is_camera = "ob_in_ref", False
    elif "ob_in_cam" in cols:
        legacy_col, is_camera = "ob_in_cam", True
    else:
        print(f"[skip] {gt_parquet.name}: no legacy pose column (ob_in_ref/ob_in_cam)")
        return False

    idx = np.asarray(table.column("proprio_frame_index").to_pylist(), dtype=int)
    ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    legacy = np.asarray(table.column(legacy_col).to_pylist(), dtype=np.float64).reshape(-1, 4, 4)

    proprio = _resolve_proprio_parquet(traj, episode)
    states = V.load_robot_states(proprio)
    base_quats = V.load_base_quats(proprio)

    # Robot-only replay: reuse the exact FK + feet-planting the visualizer uses so the
    # reconstructed reference pose matches how the box was placed before.
    rep = V.TrajectoryReplay(states, None, None, base_quats)

    if is_camera:
        # Legacy camera-frame ground truth was stored already in the MuJoCo camera frame
        # (no GL_FROM_CV); world = head_camera_FK @ pose. We still anchor via the foot, so we
        # reconstruct world here and let the foot be the reference for the constant anchor.
        print(f"[info] {gt_parquet.name}: converting {legacy_col} (camera frame)")
    else:
        print(f"[info] {gt_parquet.name}: converting {legacy_col} (foot frame)")

    box_world = np.empty((len(legacy), 4, 4), dtype=np.float64)
    ref_world = np.empty((len(legacy), 4, 4), dtype=np.float64)
    for k, p in enumerate(idx):
        ref = rep._replay_ref_pose(int(p))
        ref_world[k] = ref
        if is_camera:
            rep.set_frame(int(p))
            t_wc = np.eye(4)
            t_wc[:3, :3] = rep.data.cam_xmat[rep.cam_id].reshape(3, 3)
            t_wc[:3, 3] = rep.data.cam_xpos[rep.cam_id]
            box_world[k] = t_wc @ legacy[k]
            # re-read the (foot-planted) reference after set_frame so anchor stays consistent
            ref_world[k] = rep._replay_ref_pose(int(p))
        else:
            box_world[k] = ref @ legacy[k]

    if dry_run:
        drift = box_world[:, :3, 3].max(0) - box_world[:, :3, 3].min(0)
        print(f"[dry-run] {gt_parquet.name}: {len(legacy)} rows, "
              f"box world-pos range {np.round(drift, 3)} m (not written)")
        return False

    new_table = pa.table(
        {
            "proprio_frame_index": pa.array(idx.tolist(), pa.int64()),
            "timestamp": pa.array(ts.tolist(), pa.float64()),
            "ob_in_world": pa.array(
                [m.reshape(16).tolist() for m in box_world], pa.list_(pa.float64(), 16)
            ),
            "ref_in_world": pa.array(
                [m.reshape(16).tolist() for m in ref_world], pa.list_(pa.float64(), 16)
            ),
        }
    )
    backup = gt_parquet.with_suffix(".parquet.legacy")
    if not backup.exists():
        gt_parquet.replace(backup)
    else:
        print(f"[warn] {backup.name} already exists; keeping it, overwriting parquet only")
    pq.write_table(new_table, gt_parquet)
    print(f"[ok]   {gt_parquet.name}: wrote {len(legacy)} rows (backup: {backup.name})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory", required=True, help="recording folder under outputs/")
    parser.add_argument(
        "--episode", type=int, default=None,
        help="convert only this episode (default: all episodes in object_gt/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change without writing"
    )
    args = parser.parse_args()

    traj = Path(args.trajectory).resolve()
    gt_dir = traj / "object_gt"
    if not gt_dir.is_dir():
        sys.exit(f"[error] no object_gt/ under {traj}")

    if args.episode is not None:
        parquets = [gt_dir / f"episode_{args.episode:06d}.parquet"]
        if not parquets[0].is_file():
            sys.exit(f"[error] {parquets[0]} not found")
    else:
        parquets = sorted(gt_dir.glob("episode_*.parquet"))
        if not parquets:
            sys.exit(f"[error] no episode_*.parquet under {gt_dir}")

    n = sum(convert_episode(traj, p, args.dry_run) for p in parquets)
    print(f"[done] {n} parquet(s) converted")


if __name__ == "__main__":
    main()
