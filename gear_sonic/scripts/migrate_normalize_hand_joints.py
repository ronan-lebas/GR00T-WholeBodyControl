"""Migrate a recording whose BrainCo hand joints were stored NORMALIZED [0,1] to radians.

Recordings made before the recorder fix stored the BrainCo hand STATE/COMMAND columns as the raw
normalized [0,1] value the sim bridge publishes (``(q - lower) / span``), while the body joints are
in radians. Any FK on ``observation.state`` therefore under-curls the fingers (~1.47x over the
1.466 rad finger range) — the fingers visibly miss a grasped object on replay. This script maps the
hand joints in ``observation.state`` and ``action.wbc`` back to radians in place, so the whole 51-DOF
vector is uniformly radians and the replay / contacts / training FK are correct.

What is and isn't touched, per data parquet (``data/chunk-*/episode_*.parquet``):
  - ``observation.state`` / ``action.wbc`` : the ACTUATED hand joints (6/hand) are de-normalized
    ``q_rad = lower + norm * (upper - lower)``; body joints and the passive distal-finger joints
    (stored as 0) are left untouched.
  - ``teleop.{left,right}_hand_joints``    : left as-is — that column IS the normalized command.
  - ``object_gt/`` ``joint_vel``           : left as-is — it comes from the sim in true rad/s.

Idempotent: sets ``meta/info.json['hand_joints_denormalized'] = true`` and refuses to run again.
Make a backup of the folder before running (this rewrites parquet files in place).

Usage (needs gear_sonic[sim] — pin, pyarrow):
    python gear_sonic/scripts/migrate_normalize_hand_joints.py --trajectory outputs/<ts>
    python gear_sonic/scripts/migrate_normalize_hand_joints.py --trajectory outputs/<ts> --hand-type brainco
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

HAND_COLUMNS = ("observation.state", "action.wbc")
MARKER = "hand_joints_denormalized"


def denorm_column(values: np.ndarray, act_lower_upper) -> np.ndarray:
    """De-normalize the actuated hand joints of an (N, DOF) whole_q array, in place-safe copy."""
    out = np.array(values, dtype=np.float64, copy=True)
    for idx, lower, upper in act_lower_upper:
        out[:, idx] = lower + out[:, idx] * (upper - lower)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--trajectory", required=True, help="recording folder under outputs/")
    ap.add_argument("--hand-type", default="brainco", choices=["brainco", "dex3"])
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    traj = Path(args.trajectory).resolve()
    if not traj.is_dir():
        sys.exit(f"[error] trajectory folder does not exist: {traj}")
    if args.hand_type != "brainco":
        sys.exit("[error] only BrainCo recordings were affected (Dex3 already stores radians)")

    info_path = traj / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if info.get(MARKER):
        sys.exit(f"[skip] {traj.name} already migrated (meta/info.json['{MARKER}'] is true)")

    rm = instantiate_g1_robot_model(hand_type=args.hand_type)
    lower = np.asarray(rm.lower_joint_limits, dtype=np.float64)
    upper = np.asarray(rm.upper_joint_limits, dtype=np.float64)
    act_lu = []  # (whole_q index, lower, upper) for every actuated hand joint, both hands
    for side in ("left", "right"):
        for i in rm.get_hand_actuated_joint_indices(side):
            act_lu.append((int(i), float(lower[i]), float(upper[i])))

    parquets = sorted((traj / "data").rglob("*.parquet"))
    if not parquets:
        sys.exit(f"[error] no data parquet under {traj / 'data'}")
    print(f"[info] trajectory : {traj}")
    print(f"[info] hand type  : {args.hand_type} | actuated hand joints: {len(act_lu)}")
    print(f"[info] parquets   : {len(parquets)}")

    for p in parquets:
        table = pq.read_table(p)
        for col in HAND_COLUMNS:
            if col not in table.column_names:
                continue
            field = table.schema.field(col)
            arr = np.asarray(table.column(col).to_pylist(), dtype=np.float64)  # (N, DOF)
            before = arr[:, [i for i, _, _ in act_lu]].max(axis=0)
            new = denorm_column(arr, act_lu)
            after = new[:, [i for i, _, _ in act_lu]].max(axis=0)
            if col == HAND_COLUMNS[0]:
                print(
                    f"  {p.name} [{col}] hand max {np.round(before,3)} -> {np.round(after,3)} rad"
                )
            new_arr = pa.array(new.tolist(), type=field.type)
            table = table.set_column(table.schema.get_field_index(col), field, new_arr)
        if not args.dry_run:
            pq.write_table(table, p)

    if args.dry_run:
        print("[dry-run] no files written")
        return

    info[MARKER] = True
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[done] migrated {len(parquets)} parquet(s); marked meta/info.json['{MARKER}']=true")


if __name__ == "__main__":
    main()
