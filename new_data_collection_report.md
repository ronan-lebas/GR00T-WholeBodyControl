# New Data Collection — Reset-State Fields (for the training repo)

**Audience.** The agent adapting the training-repo data loading + env reset. This is a *delta*
on top of `frame_report.md` (the full frame reference). Read this first; fall back to
`frame_report.md` for anything about frames, quaternions, or the FoundationPose path.

**What changed and why.** The recorder now stores everything needed to reset a sim env to any
recorded frame (reference-state initialization) **without a zero-velocity discontinuity**, and to
place the robot in the world **without any FK reconstruction**. Previously the base world
*translation* was never recorded (a real-hardware limitation that had bled into the sim path) and
no velocities were stored at all. In sim we have all of it for free from `mj_data`, so we now
record it.

All new fields are **sim-only ground truth**, written to the existing `object_gt/` sidecar. Nothing
in `data/chunk-000/*.parquet` (the LeRobot robot parquet) changed. Old recordings are unaffected;
new fields simply won't exist in them (and are written as zeros if a pre-change sim publisher is
paired with a new exporter).

---

## 1. New columns in `object_gt/episode_NNNNNN.parquet`

Dense, one row per robot frame, keyed by `proprio_frame_index` (→ robot parquet row). Same join
convention as before. Existing columns (`ob_in_world`, `ref_in_world`) are unchanged.

| Column | Shape | Frame / convention | Meaning |
|---|---|---|---|
| `ob_in_world` | 16 (4×4 row-major) | `W_sim`, world | Box pose *(unchanged)* |
| `ref_in_world` | 16 (4×4 row-major) | `W_sim`, world | `right_ankle_roll_link` pose *(unchanged; still needed by the replay anchor)* |
| **`pelvis_in_world`** | 16 (4×4 row-major) | `W_sim`, world | **Robot floating-base (pelvis) pose.** The base translation the LeRobot parquet never stored. |
| **`base_vel`** | 6 | freejoint qvel | Base velocity: `[lin_xyz, ang_xyz]` |
| **`object_vel`** | 6 | freejoint qvel | Box velocity, same layout |
| **`joint_vel`** | 51 (brainco) | joint space | Joint velocities in **`observation.state` (whole_q) ordering** — `joint_vel[i]` pairs with `observation.state[i]` |

All are sampled from the **same `mj_data` (one physics step)** as `ob_in_world`, so poses and
velocities are mutually time-consistent.

### Conventions you must respect

- **4×4 matrices are row-major.** `np.asarray(col).reshape(4, 4)`. All in the MuJoCo world frame
  `W_sim` (Z up). No quaternion ambiguity since they're matrices.
- **Freejoint velocity convention (MuJoCo):** `qvel[0:3]` is **linear velocity in the world
  frame**; `qvel[3:6]` is **angular velocity in the body-local frame**. This holds for both
  `base_vel` and `object_vel`. If you reset into another **MuJoCo** model, copy them straight into
  the corresponding freejoint `qvel` (identical convention). If you reset into **Isaac Lab**,
  convert as needed (Isaac typically expects both linear and angular root velocity in the world
  frame → rotate the angular part by the base orientation).
- **`joint_vel` is in whole_q ordering** (the same 51-DOF Pinocchio ordering as
  `observation.state`; see `frame_report.md` §3.1 for the index→joint table). It includes the
  passive distal-finger joints at their true sim velocities (unlike `observation.state`, which
  stores passive-joint *positions* as 0).
- **`object_vel` may be zeros** for a held/kinematic box (collision-disabled, scripted onto the
  hands) — that's a genuine scripted zero, not missing data. For the free-box manipulation setup
  it's the real box twist.

---

## 2. Reset recipe (reference-state initialization)

To reset a MuJoCo env to recorded frame `i`:

```python
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

d  = pq.read_table("data/chunk-000/episode_000001.parquet")
q  = np.asarray(d.column("observation.state").to_pylist())            # (N,51) whole_q positions

gt = pq.read_table("object_gt/episode_000001.parquet")
gidx = np.asarray(gt.column("proprio_frame_index").to_pylist(), int)
pel  = np.asarray(gt.column("pelvis_in_world").to_pylist()).reshape(-1, 4, 4)
ob   = np.asarray(gt.column("ob_in_world").to_pylist()).reshape(-1, 4, 4)
bvel = np.asarray(gt.column("base_vel").to_pylist())                  # (M,6)
ovel = np.asarray(gt.column("object_vel").to_pylist())               # (M,6)
jvel = np.asarray(gt.column("joint_vel").to_pylist())                 # (M,51)

i = 400
g = int(np.searchsorted(gidx, i, side="right")) - 1                   # object row (hold-last)

# --- robot base freejoint (qpos[:7], qvel[:6]) ---
data.qpos[0:3] = pel[g][:3, 3]
data.qpos[3:7] = R.from_matrix(pel[g][:3, :3]).as_quat(scalar_first=True)   # MuJoCo wxyz
data.qvel[0:6] = bvel[g]

# --- robot joints (positions from whole_q, velocities from joint_vel) ---
for mj_qadr, whole_q_idx in joint_pos_map:      # your MuJoCo qposadr <-> whole_q index map
    data.qpos[mj_qadr] = q[i][whole_q_idx]
for mj_dofadr, whole_q_idx in joint_vel_map:    # analogous dof-address map
    data.qvel[mj_dofadr] = jvel[g][whole_q_idx]

# --- CRITICAL: re-apply the BrainCo distal<-proximal mimic on positions (whole_q stores
#     passive distals as 0). See frame_report.md §3.1 and _apply_joint_couplings in
#     visualize_robot_object_trajectory.py. Skipping it = fingers never close (1.4-2.6 cm error).
apply_distal_mimic(data)

# --- object freejoint (world pose + twist) ---
data.qpos[obj_qadr:obj_qadr+3]   = ob[g][:3, 3]
data.qpos[obj_qadr+3:obj_qadr+7] = R.from_matrix(ob[g][:3, :3]).as_quat(scalar_first=True)
data.qvel[obj_dofadr:obj_dofadr+6] = ovel[g]

mujoco.mj_forward(model, data)   # once, after everything is set
```

Build `joint_pos_map` / `joint_vel_map` exactly like
`visualize_robot_object_trajectory.build_joint_map` (MuJoCo joint name → `robot_model.dof_index`),
one for `jnt_qposadr` and one for `jnt_dofadr`.

**Why this is now exact.** Base pose + joint positions + object pose fully specify configuration,
and base twist + joint velocities + object twist fully specify velocity — all from one physics
step, so an in-hand cube resets in the hand with the correct relative velocity and the balancer
sees no jump.

### For an egocentric (base-relative) observation

Store/reset in world (above), but compute the policy's egocentric object pose on the fly — this is
exact, no anchor, no feet-planting:

```python
obj_in_pelvis = np.linalg.inv(pel[g]) @ ob[g]
```

This supersedes `frame_report.md`'s "Recipe A" (which reconstructed the base from `ref_in_world`
+ FK because the base pose wasn't stored). `pelvis_in_world` now gives the base directly; keep
Recipe A only for *old* recordings that lack the new columns.

---

## 3. On the SONIC-balancing "desync"

SONIC auto-balances, so the actual hand (and grasped object) ends up offset from the commanded
target. **This is learnable data, not a bug.** The only requirement is consistency: the robot
state and object state are both *measured* and from the *same* `mj_data`, which they are. Train
the policy to reproduce the *measured* robot+object relationship; never pair a *commanded* teleop
target with the *measured* object pose. No special handling needed beyond that.

---

## 3b. Hand joints are in RADIANS (post-fix) — old recordings were normalized

The BrainCo hand joints in `observation.state` and `action.wbc` are in **radians**, uniform with
the body joints — but only after a July 2026 fix. Before it, they were stored **normalized [0,1]**
(the sim bridge publishes `(q-lower)/span`), which silently under-curls the fingers by ~1.47× in
any FK (fingers miss a grasped object; the thumb looks near-right by coincidence). Consequences:

- **Detect old recordings** by `meta/info.json`: migrated ones have `"hand_joints_denormalized": true`.
  A recording produced by the fixed recorder is radians from the start (no marker needed — the
  marker only tags *migrated* older data). If you have a recording with neither the marker nor the
  fixed recorder, its hand joints are normalized.
- **To fix an old recording:** `python gear_sonic/scripts/migrate_normalize_hand_joints.py
  --trajectory <folder>` (idempotent; back it up first). It de-normalizes only `observation.state`
  and `action.wbc`; `teleop.*_hand_joints` stays the raw [0,1] command by design.
- `object_gt`'s `joint_vel` was always true rad/s and is unaffected.

If your reset/FK sees fingers that never close on the object, this is the first thing to check.

## 4. Checklist / gotchas

1. **Re-apply the distal-finger mimic** (×1.155 fingers, ×1.0 thumb) on positions after setting
   `whole_q`, every reset. `mj_forward` does **not** project qpos onto equality constraints.
2. **`joint_vel` pairs 1:1 with `observation.state`** (same 51-DOF whole_q ordering). Positions
   come from the robot parquet, velocities from `object_gt`.
3. **Freejoint qvel convention**: world-frame linear, body-local angular. Direct copy into MuJoCo;
   convert for Isaac.
4. **`object_vel == 0`** means a held/kinematic box (or an old recording), not missing data.
5. **Backward compat**: recordings made before this change have none of the four new columns; new
   recordings paired with an old sim publisher get zeros. Detect by column presence and fall back
   to `frame_report.md` Recipe A for the base pose; you can't recover velocities for old data.
6. **Join key is `proprio_frame_index`**, dense and monotonic (`searchsorted`, hold-last). Never
   align by timestamp.

---

## 5. Source of the new fields (this repo)

| Concern | File / symbol |
|---|---|
| Gathers GT state from `mj_data` | `gear_sonic/utils/mujoco_sim/base_sim.py::BaseSimulator.get_gt_state` |
| whole_q velocity ordering map | `base_sim.py::_ensure_gt_jointvel_map` |
| Publishes over ZMQ (`box_gt` topic) | `base_sim.py::_publish_box_gt` |
| Writes the parquet | `gear_sonic/utils/data_collection/object_gt_writer.py::ObjectGtWriter` |
| Threads fields through the exporter | `gear_sonic/scripts/run_data_exporter.py::_write_object_gt_frame` |

Recording is gated behind `--record-object-gt` (exporter) + `--record-box-gt` (sim), unchanged.
