"""Recompute BrainCo-finger <-> object contacts for a recorded episode, offline.

This post-processes an existing sim recording (like ``foundation_pose/filter_object_pose.py``
post-processes object poses) and writes a per-frame *contact* sidecar next to it, in the same
*form* the ConTrack repo uses for its reference-motion contacts:

  * a **binary** ``is_contact`` flag per hand finger segment, per frame, and
  * the **contact point in the object-local frame** (meters), NaN where not in contact,

both derived with a **5 mm mesh-proximity threshold** (ConTrack's ``contact_threshold = 0.005``).
ConTrack ships no contact-generation code (its contacts are baked into ``.h5`` files by an
upstream dataset pipeline), so we replicate the *convention* rather than import logic. Our robot
(G1 + BrainCo, ~16 finger segments/hand) differs from ConTrack's (xArm7 + XHand), so a separate
converter — out of scope here — later remaps our ``segment_names`` onto ConTrack's links.

How contacts are recovered (no physics stepping):
  We reuse the replay in ``visualize_robot_object_trajectory.py``. Each frame it poses the robot
  from the recorded 51-DOF ``whole_q`` and places the box at its object pose, so the *relative*
  hand/box geometry is physically correct (up to the accuracy of that object pose — see below). We
  build that model with the injected object made **collidable with margin = threshold**, so
  ``mj_forward``'s collision phase (which runs even without ``mj_step``) reports every
  object<->finger contact whose surface gap is <= the threshold. We read those from
  ``mj_data.contact``, keep the closest per finger segment, and express its midpoint in the
  object-local frame. This works for both free-box and held-box recordings (held-box disables
  *sim* collision, but here we recompute purely from geometry).

Object pose source: by default this uses the sim ground truth (``object_gt/``) — exact, when
available. Pass ``--from-vision`` to instead use the FoundationPose estimate (``ob_in_cam`` per
frame via the camera FK, or ``ob_in_world_filtered`` with ``--filtered``) — needed for recordings
where the ground truth is unavailable or unreliable (e.g. reconstructed from a legacy schema and
contaminated by unrecorded base motion — see ``convert_object_gt_to_world.py``). Output format,
path, and schema are identical either way; re-running overwrites any existing contacts file for
that episode, whichever source produced it.

Usage (needs gear_sonic[sim] — mujoco, pin, scipy, pyarrow):
    python gear_sonic/scripts/process_contacts.py
    python gear_sonic/scripts/process_contacts.py --trajectory outputs/2026-07-09-14-54-30 --episode 0
    python gear_sonic/scripts/process_contacts.py --episode 0 --check   # headless sanity, no writes
    python gear_sonic/scripts/process_contacts.py --episode 4 --from-vision            # raw ob_in_cam
    python gear_sonic/scripts/process_contacts.py --episode 4 --from-vision --filtered # ob_in_world_filtered
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.mujoco_sim.sim_utils import get_body_geom_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_SCRIPT = REPO_ROOT / "gear_sonic" / "scripts" / "visualize_robot_object_trajectory.py"

DEFAULT_THRESHOLD = 0.005  # 5 mm, matching ConTrack's contacts/contact_threshold
OBJECT_BODY_NAME = "tracked_object"  # the replay injects the object under this name
HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load_viz_module():
    """Import visualize_robot_object_trajectory.py as a module (same pattern as the pose filter)."""
    if not VIS_SCRIPT.is_file():
        sys.exit(f"[error] visualizer script not found: {VIS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("viz_traj", VIS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Segment enumeration + contact extraction
# --------------------------------------------------------------------------- #


def _finger_segment_names() -> list[str]:
    """Deterministic finger-segment body names, left hand then right hand.

    The thumb has an extra ``metacarpal`` link; every finger also has a fixed ``tip`` link, which
    carries a collision geom in the BrainCo XML, so tips are real contact segments here.
    """
    names = []
    for side in HANDS:
        for finger in FINGERS:
            segs = (
                ("metacarpal", "proximal", "distal", "tip")
                if finger == "thumb"
                else ("proximal", "distal", "tip")
            )
            for seg in segs:
                names.append(f"{side}_{finger}_{seg}_Link")
    return names


def build_segment_map(model) -> tuple[list[str], dict[int, int], set[int]]:
    """Map each finger-segment *collision* geom -> a segment index, and collect the object geoms.

    Returns ``(segment_names, geom_to_segment, object_geom_ids)`` where ``segment_names`` is the
    ordered list of finger bodies that actually have a collision geom in this model (left then
    right). Visual geoms (contype == conaffinity == 0) are ignored, so only true collision geoms
    map to a segment.
    """
    import mujoco

    segment_names: list[str] = []
    geom_to_segment: dict[int, int] = {}
    for name in _finger_segment_names():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        collision_geoms = [
            g
            for g in get_body_geom_ids(model, body_id)
            if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        ]
        if not collision_geoms:
            continue
        seg = len(segment_names)
        segment_names.append(name)
        for g in collision_geoms:
            geom_to_segment[g] = seg

    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    if obj_body < 0:
        sys.exit(f"[error] object body '{OBJECT_BODY_NAME}' not found in the replay model")
    object_geom_ids = set(get_body_geom_ids(model, obj_body))
    return segment_names, geom_to_segment, object_geom_ids


def extract_frame_contacts(
    data,
    geom_to_segment: dict[int, int],
    object_geom_ids: set[int],
    n_segments: int,
    box_qadr: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment (is_contact, object-local point, distance) for the current ``mj_forward`` state.

    Iterates the frame's contacts, keeps the closest object<->segment contact per segment, gates on
    ``dist <= threshold``, and expresses the contact midpoint in the object-local frame (matching
    ConTrack's convention). Non-contacting segments get NaN point/dist.
    """
    best_dist = np.full(n_segments, np.inf)
    best_pos_world = np.full((n_segments, 3), np.nan)

    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in object_geom_ids and g2 in geom_to_segment:
            seg = geom_to_segment[g2]
        elif g2 in object_geom_ids and g1 in geom_to_segment:
            seg = geom_to_segment[g1]
        else:
            continue  # not an object<->finger pair (drops object<->floor, cross-hand, etc.)
        if c.dist > threshold:
            continue
        if c.dist < best_dist[seg]:
            best_dist[seg] = c.dist
            best_pos_world[seg] = np.asarray(c.pos, dtype=np.float64)

    is_contact = (best_dist <= threshold).astype(np.uint8)
    points_local = np.full((n_segments, 3), np.nan)
    if is_contact.any():
        t = np.asarray(data.qpos[box_qadr : box_qadr + 3], dtype=np.float64)
        q_wxyz = np.asarray(data.qpos[box_qadr + 3 : box_qadr + 7], dtype=np.float64)
        rot_inv = R.from_quat(q_wxyz, scalar_first=True).inv()
        idx = np.flatnonzero(is_contact)
        points_local[idx] = rot_inv.apply(best_pos_world[idx] - t)
    dists = np.where(is_contact.astype(bool), best_dist, np.nan)
    return is_contact, points_local, dists


def compute_all(replay, geom_to_segment, object_geom_ids, n_segments, threshold):
    """Run the full trajectory, returning (is_contact (N,S), points (N,S,3), dist (N,S), max_ncon)."""
    n = replay.n
    all_contact = np.zeros((n, n_segments), dtype=np.uint8)
    all_points = np.full((n, n_segments, 3), np.nan)
    all_dist = np.full((n, n_segments), np.nan)
    max_ncon = 0
    for i in range(n):
        replay.set_frame(i)
        max_ncon = max(max_ncon, int(replay.data.ncon))
        c, p, d = extract_frame_contacts(
            replay.data, geom_to_segment, object_geom_ids, n_segments, replay.obj_qadr, threshold
        )
        all_contact[i] = c
        all_points[i] = p
        all_dist[i] = d
    return all_contact, all_points, all_dist, max_ncon


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def load_frame_index(parquet: Path, n: int) -> np.ndarray:
    """Robot ``frame_index`` column (the proprio row id), or 0..n-1 if the column is absent."""
    names = pq.read_table(parquet).column_names
    if "frame_index" not in names:
        return np.arange(n, dtype=np.int64)
    vals = np.asarray(
        pq.read_table(parquet, columns=["frame_index"]).column(0).to_pylist(), dtype=np.int64
    ).reshape(-1)
    return vals if vals.shape[0] == n else np.arange(n, dtype=np.int64)


def write_outputs(out_dir, episode, proprio_idx, is_contact, points, dist, meta):
    """Write ``contacts/episode_%06d.parquet`` (one row/frame) + ``contacts/meta.json`` (once)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n, s = is_contact.shape
    table = pa.table(
        {
            "proprio_frame_index": pa.array([int(x) for x in proprio_idx], pa.int64()),
            "is_contact": pa.array(
                [is_contact[i].tolist() for i in range(n)], pa.list_(pa.uint8(), s)
            ),
            "contact_points": pa.array(
                [points[i].reshape(-1).tolist() for i in range(n)], pa.list_(pa.float64(), s * 3)
            ),
            "contact_dist": pa.array(
                [dist[i].tolist() for i in range(n)], pa.list_(pa.float64(), s)
            ),
        }
    )
    pq.write_table(table, out_dir / f"episode_{episode:06d}.parquet")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# Headless check
# --------------------------------------------------------------------------- #


def run_check(segment_names, is_contact, points, dist, box_rbound, threshold, max_ncon):
    n, s = is_contact.shape
    n_per_hand = sum(1 for name in segment_names if name.startswith("left_"))
    print(f"[check] segments: {s} ({n_per_hand} left / {s - n_per_hand} right)")
    print(f"[check] max ncon over trajectory: {max_ncon}")

    counts = is_contact.sum(axis=1)
    frames_with_contact = int((counts > 0).sum())
    print(f"[check] frames with >=1 contact: {frames_with_contact} / {n}")

    # Show the frames where the contact set changes (compact but informative).
    prev = None
    shown = 0
    for i in range(n):
        active = tuple(np.flatnonzero(is_contact[i]).tolist())
        if active != prev:
            names = [segment_names[j] for j in active]
            print(f"[check] frame {i:5d}: {len(active)} contact(s) {names}")
            prev = active
            shown += 1
            if shown >= 40:
                print("[check] ... (further changes suppressed)")
                break

    # Per-segment frequency — a segment that never fires may signal a bitmask issue.
    freq = is_contact.sum(axis=0)
    left_hit = any(freq[j] > 0 for j, nm in enumerate(segment_names) if nm.startswith("left_"))
    right_hit = any(freq[j] > 0 for j, nm in enumerate(segment_names) if nm.startswith("right_"))
    print(f"[check] any contact on left hand: {left_hit} | right hand: {right_hit}")
    if not (left_hit or right_hit):
        print("[check][WARN] neither hand ever contacts — check collision bitmasks / threshold")
    elif not (left_hit and right_hit):
        print(
            "[check][note] only one hand contacts — expected for one-handed manipulation; "
            "the other hand's masks are fine as long as its side is used in some episode"
        )

    # Sanity: object-local contact points must lie within the object's bounding radius (+margin).
    finite = np.isfinite(points).all(axis=2)
    if finite.any():
        mags = np.linalg.norm(points[finite], axis=1)
        bound = box_rbound + threshold + 1e-6
        n_bad = int((mags > bound).sum())
        print(
            f"[check] object-local |point|: max {mags.max():.4f} m (bound {bound:.4f} m); "
            f"out-of-bound {n_bad}"
        )
        if n_bad:
            print("[check][WARN] contact points exceed the object bound — check pose/frame")
    else:
        print("[check][WARN] no contacts anywhere in the trajectory")
    print("[check] OK")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trajectory", default=None, help="recording folder under outputs/ (default: most recent)"
    )
    parser.add_argument("--episode", type=int, default=0, help="episode index (default: 0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="mesh-proximity contact threshold in meters (default: 0.005, matching ConTrack)",
    )
    parser.add_argument(
        "--out-name",
        default="contacts",
        help="output subfolder under the trajectory (default: contacts)",
    )
    parser.add_argument(
        "--from-vision",
        action="store_true",
        help="use the FoundationPose object-pose estimate instead of the sim ground truth "
        "(ob_in_cam by default, or ob_in_world_filtered with --filtered)",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="with --from-vision, use ob_in_world_filtered/ instead of raw ob_in_cam/",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="headless sanity summary (no viewer, no files written)",
    )
    args = parser.parse_args()

    if args.filtered and not args.from_vision:
        sys.exit("[error] --filtered only applies with --from-vision")

    viz = _load_viz_module()

    # resolve_paths reads .ground_truth / .filtered off the args namespace; force ground-truth
    # mode (default) so it returns the object_gt parquet, or vision mode with --from-vision so it
    # returns the FoundationPose ob_in_cam/ob_in_world_filtered folder instead.
    src_args = types.SimpleNamespace(
        trajectory=args.trajectory,
        episode=args.episode,
        ground_truth=not args.from_vision,
        filtered=args.filtered,
    )
    traj, parquet, mesh, ob_src = viz.resolve_paths(src_args)
    if mesh is None or ob_src is None:
        source_desc = (
            f"foundation_pose_data/episode_{args.episode:06d}/"
            f"{'ob_in_world_filtered' if args.filtered else 'ob_in_cam'}"
            if args.from_vision
            else f"object_gt/episode_{args.episode:06d}.parquet"
        )
        sys.exit(f"[error] contacts require the object pose ({source_desc}) and box.obj; not found under {traj}")

    states = viz.load_robot_states(parquet)
    base_quats = viz.load_base_quats(parquet)
    proprio_idx = load_frame_index(parquet, states.shape[0])
    fps = viz.read_fps(traj)

    pelvis_poses = None
    exact_base = False
    if args.from_vision:
        # Hardware path: no sim base pose exists, so place the base with the feet-planted anchor
        # and the object via camera FK (raw) or directly (filtered) — same as on real recordings.
        obj_poses = viz.load_object_poses(ob_src)
        obj_to_robot = viz.load_frame_map(ob_src, obj_poses.shape[0])
        obj_gt_ref, ref_poses = False, None
        obj_in_world = args.filtered
        source_tag = "vision_filtered" if args.filtered else "vision_raw"
    else:
        obj_poses, ref_poses, pelvis_poses, obj_to_robot = viz.load_object_gt(ob_src)
        obj_gt_ref, obj_in_world = True, False
        # New sim recordings store the robot's true base pose: place the base exactly (no
        # feet-planting) for the most accurate hand<->object geometry. Old GT recordings lacking
        # pelvis_in_world fall back to the feet-planted anchor automatically.
        exact_base = pelvis_poses is not None
        source_tag = "ground_truth"

    base_mode = "exact (pelvis_in_world)" if exact_base else "feet-planted anchor"
    replay = viz.TrajectoryReplay(
        states,
        obj_poses,
        mesh,
        base_quats,
        obj_to_robot=obj_to_robot,
        obj_in_world=obj_in_world,
        obj_gt_ref=obj_gt_ref,
        ref_poses=ref_poses,
        collidable_object=True,
        object_margin=args.threshold,
        pelvis_poses=pelvis_poses,
        exact_base=exact_base,
    )
    segment_names, geom_to_segment, object_geom_ids = build_segment_map(replay.model)
    n_segments = len(segment_names)

    print(f"[info] trajectory : {traj}")
    print(
        f"[info] parquet    : {parquet.relative_to(traj)} ({states.shape[0]} frames @ {fps:.0f} fps)"
    )
    print(f"[info] object pose: {ob_src.relative_to(traj)} ({obj_poses.shape[0]} frames, source={source_tag})")
    print(f"[info] base pose  : {base_mode}")
    print(
        f"[info] segments   : {n_segments} finger collision segments; threshold {args.threshold*1000:.1f} mm"
    )

    is_contact, points, dist, max_ncon = compute_all(
        replay, geom_to_segment, object_geom_ids, n_segments, args.threshold
    )

    # Bounding radius of the whole object (a decomposed mesh has one geom per convex hull).
    box_rbound = max(float(replay.model.geom_rbound[g]) for g in object_geom_ids)

    if args.check:
        run_check(segment_names, is_contact, points, dist, box_rbound, args.threshold, max_ncon)
        return

    n_per_hand = sum(1 for name in segment_names if name.startswith("left_"))
    meta = {
        "segment_names": segment_names,
        "num_segments": n_segments,
        "num_segments_per_hand": n_per_hand,
        "hand_order": list(HANDS),
        "num_hands": 2,
        "num_objects": 1,
        "object_body_name": OBJECT_BODY_NAME,
        "contact_threshold": args.threshold,
        "contact_frame": "object_local",
        "contact_point_units": "meters",
        "nan_convention": "NaN where segment not in contact",
        "source": source_tag,
        "base_placement": "exact_pelvis_in_world" if exact_base else "feet_planted_anchor",
        "fps": fps,
        "flat_layout": (
            "columns are flat length-num_segments (is_contact/dist) or num_segments*3 "
            "(contact_points, row-major xyz); segments ordered left-hand then right-hand. "
            "Reshape a (T, num_segments) column to (T, 2, num_segments_per_hand) to match "
            "ConTrack's (T, H, num_segments)."
        ),
    }
    out_dir = traj / args.out_name
    out_path = out_dir / f"episode_{args.episode:06d}.parquet"
    if out_path.exists():
        print(f"[info] {out_path.relative_to(traj)} already exists; overwriting (source={source_tag})")
    write_outputs(out_dir, args.episode, proprio_idx, is_contact, points, dist, meta)
    counts = is_contact.sum(axis=1)
    print(
        f"[done] wrote {out_path} "
        f"({is_contact.shape[0]} rows, {int((counts > 0).sum())} with contact) + meta.json"
    )


if __name__ == "__main__":
    main()
