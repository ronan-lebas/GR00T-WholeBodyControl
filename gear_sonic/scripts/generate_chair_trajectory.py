"""Generate a whole-body reference trajectory for moving the chair, by optimization.

Say where the chair should end up and get back a trajectory — robot joints + base pose +
chair pose, 50 Hz — that puts it there: the robot steps up to the chair, grasps it, moves it
(stepping again if one stance cannot cover the whole displacement) and lets go, with the
hands staying on the object and each planted foot staying exactly where it was put.
It is meant as the reference an RL tracking controller is trained on, so it is exact on
kinematics and contact *placement*, and only plausible (not certified) on the dynamics —
the policy re-derives the contact detail during training.

The method follows FARO (Ciebielski et al., arXiv:2607.18362) — see ``gear_sonic/trajopt/``
for the module-by-module mapping. In short: candidate contact-mode sequences are filtered
by increasingly expensive feasibility tests (single-configuration IK per mode, then per
transition, then a kinematic sequence optimization over all keyframes, then a dynamic
pass), and the first plan that survives is densified into a timed trajectory.

Usage (needs gear_sonic[sim] — mujoco, scipy; run in .venv_sim):

    # turn the chair 45 deg in place — the canonical request
    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 45

    # turn it and move it 15 cm to the robot's left, replay the result in the viewer
    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 45 \
        --translate 0.0 0.15 --replay

    # do it without moving the feet at all (the old static-stance behaviour)
    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 45 --walk-steps 0

    # play back a trajectory generated earlier (no optimization runs)
    python gear_sonic/scripts/generate_chair_trajectory.py \
        --replay-from data/generated_trajectories/chair_yawp45

    # see what a plan the actuation check refuses would have looked like
    python gear_sonic/scripts/generate_chair_trajectory.py --start 0.85 0 0 --no-torque

    # against a staged chair asset (uses its bounding box and draws its mesh)
    python gear_sonic/scripts/generate_chair_trajectory.py --rotate-deg 90 \
        --chair-asset data/objects/chair

Chair pose convention: (x, y, yaw) on the floor in the robot's world frame, with yaw 0
meaning the backrest faces the robot — the pose ``run_sim_loop.py --chair`` spawns. The
exported object trajectory carries both this canonical frame and the staged asset's own
frame (what the sim's ``object`` free joint takes), so nothing has to be re-derived
downstream.

Outputs (default ``data/generated_trajectories/<name>/``): ``trajectory.npz``,
``motion_lib.pkl``, ``report.json``, ``scene.xml`` — see ``gear_sonic/trajopt/export.py``.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from gear_sonic.trajopt import dynamics as dyn, kso as K, plans as P, trajectory as T
from gear_sonic.trajopt.chair import ChairSpec
from gear_sonic.trajopt.export import save, scene_qpos
from gear_sonic.trajopt.scene import TrajOptScene
from gear_sonic.trajopt.se3 import Pose

REPO_ROOT = Path(__file__).resolve().parents[2]
# Where run_sim_loop.py spawns the chair (CHAIR_POS_X), in the canonical frame.
DEFAULT_START = (0.55, 0.0, 0.0)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("task")
    g.add_argument("--rotate-deg", type=float, default=45.0,
                   help="chair yaw change about its own vertical axis (default: 45)")
    g.add_argument("--translate", type=float, nargs=2, metavar=("DX", "DY"),
                   default=(0.0, 0.0), help="chair translation on the floor, meters")
    g.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                   default=None, help=f"chair start pose (default: {DEFAULT_START} with "
                                      "yaw in degrees)")
    g.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                   default=None, help="absolute goal pose; overrides --rotate-deg/--translate")

    g = p.add_argument_group("chair")
    g.add_argument("--chair-asset", type=Path, default=REPO_ROOT / "data/objects/chair",
                   help="staged asset dir (object.json); falls back to the SANDSBERG proxy")
    g.add_argument("--chair-mass", type=float, default=None,
                   help="chair mass in kg (default: the sim's 0.2 kg)")
    g.add_argument("--seat-height", type=float, default=None,
                   help="override the proxy's seat height, meters")

    g = p.add_argument_group("planning")
    g.add_argument("--plan", default="auto",
                   choices=("auto", "push", "slide", "side_lift", "rail_lift"),
                   help="contact plan family to try (default: auto = all, best first)")
    g.add_argument("--lift", type=float, default=0.12,
                   help="how far the chair is lifted clear of the floor, meters")
    g.add_argument("--transport-knots", type=int, default=2,
                   help="keyframes across the transport stage (more = tighter object path)")
    g.add_argument("--walk-steps", type=int, default=2,
                   help="steps taken to approach the chair before reaching for it "
                        "(0 = stand where the robot spawns)")
    g.add_argument("--knots", type=int, default=9,
                   help="IK knots per contact stage when densifying the path")
    g.add_argument("--no-torque", action="store_true",
                   help="do not reject a plan for exceeding the joint torque limits. The "
                        "ratio is still computed and reported — this only stops it being a "
                        "veto, so a trajectory the actuation check refuses can be looked at")

    g = p.add_argument_group("output")
    g.add_argument("--fps", type=float, default=50.0, help="output sample rate")
    g.add_argument("--vel-scale", type=float, default=T.DEFAULT_VEL_SCALE,
                   help="fraction of the joint velocity limits the reference may use")
    g.add_argument("--out", type=Path, default=None,
                   help="output directory (default: data/generated_trajectories/<auto>)")
    g.add_argument("--name", default=None, help="motion name inside motion_lib.pkl")
    g.add_argument("--replay", action="store_true",
                   help="open the MuJoCo viewer and play the result on a loop")
    g.add_argument("--replay-from", type=Path, default=None, metavar="DIR",
                   help="play an already generated trajectory and exit — pass its output "
                        "directory (or the trajectory.npz inside it). Nothing is generated")
    g.add_argument("--no-save", action="store_true", help="do not write anything")
    return p.parse_args(argv)


def resolve_poses(args) -> tuple[Pose, Pose]:
    sx, sy, syaw = args.start if args.start else DEFAULT_START
    if args.start:
        syaw = np.deg2rad(syaw)
    start = Pose.from_xyz_yaw(sx, sy, 0.0, syaw)
    if args.goal:
        gx, gy, gyaw = args.goal
        goal = Pose.from_xyz_yaw(gx, gy, 0.0, np.deg2rad(gyaw))
    else:
        goal = Pose.from_xyz_yaw(
            sx + args.translate[0], sy + args.translate[1], 0.0,
            syaw + np.deg2rad(args.rotate_deg),
        )
    return start, goal


def auto_name(args, start: Pose, goal: Pose) -> str:
    dyaw = np.degrees(goal.yaw() - start.yaw())
    d = goal.p - start.p
    bits = [f"chair_yaw{dyaw:+.0f}"]
    if np.linalg.norm(d[:2]) > 1e-3:
        bits.append(f"dx{d[0]:+.02f}_dy{d[1]:+.02f}")
    return "_".join(bits).replace("+", "p").replace("-", "m").replace(".", "")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.replay_from is not None:
        return replay_saved(args.replay_from)
    t_start = time.perf_counter()

    start, goal = resolve_poses(args)
    chair = ChairSpec.resolve(args.chair_asset, mass=args.chair_mass)
    if args.seat_height is not None:
        chair.seat_height = args.seat_height
    print(f"[trajopt] {chair.describe()}")
    if args.no_torque:
        print("[trajopt] WARNING: --no-torque — the joint torque check is measured but not "
              "enforced. The result may ask for torques the robot does not have.")
    print(f"[trajopt] chair {np.round(start.p[:2], 3)} @ {np.degrees(start.yaw()):+.0f}deg "
          f"-> {np.round(goal.p[:2], 3)} @ {np.degrees(goal.yaw()):+.0f}deg")

    scene = TrajOptScene(chair, start)
    candidates = P.plan_by_name(
        args.plan, chair, start, goal, lift_height=args.lift,
        n_transport=args.transport_knots, walk_steps=args.walk_steps,
    )
    print(f"[trajopt] {len(candidates)} candidate contact plan(s): "
          f"{', '.join(c.name for c in candidates)}")

    cache = K.FeasibilityCache()
    attempts = []
    result = None
    for plan in candidates:
        t0 = time.perf_counter()
        ev = K.evaluate_plan(scene, plan, cache=cache)
        entry = {
            "plan": plan.name,
            "modes": [m.name for m in plan.modes],
            "passed_kinematics": ev.passed,
            "stopped_at": ev.stage,
            "detail": ev.detail,
            "filter_seconds": ev.timings,
        }
        print(f"[trajopt] {plan.name:26s} {'PASS' if ev.passed else 'reject'} "
              f"@{ev.stage}: {ev.detail} ({time.perf_counter() - t0:.1f}s)")
        if not ev.passed:
            attempts.append(entry)
            continue

        entry["kso_violations"] = ev.kso.report
        keyframes = K.unpack_keyframes(ev.problem, ev.kso.z)
        stages = T.solve_path(scene, plan, keyframes, knots_per_stage=args.knots)
        durations = T.stage_durations(
            stages, plan, scene.limits, scene, vel_scale=args.vel_scale, fps=args.fps
        )
        traj, report, durations = dyn.retime_until_feasible(
            scene, stages, plan, durations, fps=args.fps,
            enforce_torque=not args.no_torque,
        )
        entry["dynamics"] = report.summary()
        print(f"[trajopt] {plan.name:26s} dynamics: {report.summary()}")
        if not report.ok:
            # Same role as FARO's trajectory-optimization filter: a plan can be kinematically
            # fine and still be rejected here, and the search moves on.
            entry["passed_dynamics"] = False
            attempts.append(entry)
            continue
        entry["passed_dynamics"] = True
        attempts.append(entry)
        result = (plan, stages, traj, report, durations, ev)
        break

    if result is None:
        print("[trajopt] no candidate plan survived the feasibility cascade.", file=sys.stderr)
        print("[trajopt] try: a smaller displacement, --lift 0.08, a different --plan, "
              "or move the chair closer (--start).", file=sys.stderr)
        return 1

    plan, stages, traj, report, durations, ev = result
    drift = T.contact_drift(scene, traj, plan)
    print(f"[trajopt] sampled {traj.n} frames at {traj.fps:g} Hz over {traj.t[-1]:.2f}s")
    print("[trajopt] interpolation drift: "
          + ", ".join(
              f"{k}={v}" if isinstance(v, str)
              else (f"{k}={v:.1f}deg" if k.endswith("_deg")
                    else f"{k}={v:.3f}" if k == "contact_normal"
                    else f"{k}={v*1000:.1f}mm")
              for k, v in drift.items()))

    name = args.name or auto_name(args, start, goal)
    full_report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv[1:]),
        "chair": {
            "name": chair.name,
            "source": str(chair.asset_dir) if chair.asset_dir else "hardcoded SANDSBERG proxy",
            "seat_depth": chair.seat_depth, "seat_width": chair.seat_width,
            "seat_height": chair.seat_height, "total_height": chair.total_height,
            "mass": chair.mass, "asset_yaw": chair.asset_yaw,
        },
        "task": {
            "start": [*start.p[:2], np.degrees(start.yaw())],
            "goal": [*goal.p[:2], np.degrees(goal.yaw())],
        },
        "candidates": attempts,
        "accepted": {
            "plan": plan.name,
            "modes": [m.name for m in plan.modes],
            "stage_durations": durations,
            "total_time": float(traj.t[-1]),
            "frames": traj.n,
            "fps": traj.fps,
            "kso_violations": ev.kso.report,
            "path_knot_violations": {f"stage{i}": s.residual for i, s in enumerate(stages)},
            "interpolation_drift": drift,
            "dynamics": {
                "zmp_margin_m": report.zmp_margin,
                "friction_margin_N": report.friction_margin,
                "min_contact_normal_force_N": report.normal_force_min,
                "grasp_moment_ratio": report.grasp_moment_ratio,
                "torque_ratio": report.torque_ratio,
                "velocity_ratio": report.velocity_ratio,
                "max_hand_force_N": report.hand_force_max,
                "torque_enforced": report.torque_enforced,
                "ok": report.ok,
            },
        },
        "wall_seconds": time.perf_counter() - t_start,
    }

    if not args.no_save:
        out_dir = args.out or (REPO_ROOT / "data" / "generated_trajectories" / name)
        written = save(out_dir, scene, traj, full_report, name=name)
        print(f"[trajopt] wrote {out_dir}")
        for k, v in written.items():
            print(f"           {k:12s} {v.name}")
    else:
        print(json.dumps(full_report["accepted"], indent=2, default=str))

    print(f"[trajopt] done in {time.perf_counter() - t_start:.1f}s")

    if args.replay:
        replay(scene.model, scene.data, scene_qpos(scene, traj), traj.fps,
               labels=[f"{s} {traj.stage_names[s]}" for s in traj.stage])
    return 0


def replay(model, data, qpos: np.ndarray, fps: float, labels=None) -> None:
    """Kinematic playback of a full-scene ``qpos`` sequence in the MuJoCo viewer.

    Space = pause/resume, left/right = step a frame while paused.
    """
    import mujoco
    import mujoco.viewer

    n = len(qpos)
    state = {"i": 0, "paused": False, "said": None}

    def key_callback(key):
        if key == 32:  # space
            state["paused"] = not state["paused"]
        elif key == 262:  # right
            state["i"] = min(state["i"] + 1, n - 1)
        elif key == 263:  # left
            state["i"] = max(state["i"] - 1, 0)

    print(f"[trajopt] replay: {n} frames at {fps:g} Hz — "
          "space = pause/resume, left/right = step (while paused)")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False
    ) as viewer:
        dt = 1.0 / fps
        while viewer.is_running():
            i = state["i"]
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            if labels is not None and labels[i] != state["said"]:
                state["said"] = labels[i]
                print(f"           t={i / fps:6.2f}s  stage {labels[i]}")
            if not state["paused"]:
                state["i"] = (i + 1) % n
            time.sleep(dt)


def replay_saved(path: Path) -> int:
    """Play a trajectory written by a previous run, against its own exported scene."""
    import mujoco

    path = Path(path)
    npz_path = path / "trajectory.npz" if path.is_dir() else path
    xml_path = npz_path.parent / "scene.xml"
    for f in (npz_path, xml_path):
        if not f.is_file():
            print(f"[trajopt] {f} not found", file=sys.stderr)
            return 1

    data = np.load(npz_path, allow_pickle=True)
    names = [str(s) for s in data["stage_names"]]
    stage = data["stage"]
    print(f"[trajopt] {npz_path}: {len(data['t'])} frames, {float(data['t'][-1]):.2f}s, "
          f"{len(names)} stages")
    print("[trajopt] " + " -> ".join(names))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    replay(model, mujoco.MjData(model), data["qpos"], float(data["fps"]),
           labels=[f"{int(s)} {names[int(s)]}" for s in stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
