"""Generate a whole-body reference trajectory for moving the chair, by optimization.

Say where the chair should end up and get back a trajectory — robot joints + base pose +
chair pose, 50 Hz — that puts it there: the robot reaches, grasps the chair, moves it and
lets go, with the hands staying on the object and the feet staying planted throughout.
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
from gear_sonic.trajopt.export import save
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
    g.add_argument("--knots", type=int, default=9,
                   help="IK knots per contact stage when densifying the path")

    g = p.add_argument_group("output")
    g.add_argument("--fps", type=float, default=50.0, help="output sample rate")
    g.add_argument("--vel-scale", type=float, default=T.DEFAULT_VEL_SCALE,
                   help="fraction of the joint velocity limits the reference may use")
    g.add_argument("--out", type=Path, default=None,
                   help="output directory (default: data/generated_trajectories/<auto>)")
    g.add_argument("--name", default=None, help="motion name inside motion_lib.pkl")
    g.add_argument("--replay", action="store_true",
                   help="open the MuJoCo viewer and play the result on a loop")
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
    t_start = time.perf_counter()

    start, goal = resolve_poses(args)
    chair = ChairSpec.resolve(args.chair_asset, mass=args.chair_mass)
    if args.seat_height is not None:
        chair.seat_height = args.seat_height
    print(f"[trajopt] {chair.describe()}")
    print(f"[trajopt] chair {np.round(start.p[:2], 3)} @ {np.degrees(start.yaw()):+.0f}deg "
          f"-> {np.round(goal.p[:2], 3)} @ {np.degrees(goal.yaw()):+.0f}deg")

    scene = TrajOptScene(chair, start)
    candidates = P.plan_by_name(
        args.plan, chair, start, goal, lift_height=args.lift, n_transport=args.transport_knots
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
        durations = T.stage_durations(stages, plan, scene.limits, vel_scale=args.vel_scale)
        traj, report, durations = dyn.retime_until_feasible(
            scene, stages, plan, durations, fps=args.fps
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
              else (f"{k}={v:.3f}" if k == "contact_normal" else f"{k}={v*1000:.1f}mm")
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
        replay(scene, traj)
    return 0


def replay(scene: TrajOptScene, traj: T.Trajectory) -> None:
    """Kinematic playback in the MuJoCo viewer (space = pause, arrows = step)."""
    import mujoco
    import mujoco.viewer

    state = {"i": 0, "paused": False}

    def key_callback(key):
        if key == 32:  # space
            state["paused"] = not state["paused"]
        elif key == 262:  # right
            state["i"] = min(state["i"] + 1, traj.n - 1)
        elif key == 263:  # left
            state["i"] = max(state["i"] - 1, 0)

    print("[trajopt] replay: space = pause/resume, left/right = step (while paused)")
    with mujoco.viewer.launch_passive(
        scene.model, scene.data, key_callback=key_callback, show_left_ui=False
    ) as viewer:
        dt = 1.0 / traj.fps
        while viewer.is_running():
            i = state["i"]
            scene.set_state(
                traj.base_p[i], traj.base_rv[i], traj.qj[i],
                {"left": traj.hand_q[i, :6], "right": traj.hand_q[i, 6:]},
                traj.object_pose(i),
            )
            mujoco.mj_forward(scene.model, scene.data)
            viewer.sync()
            if not state["paused"]:
                state["i"] = (i + 1) % traj.n
            time.sleep(dt)


if __name__ == "__main__":
    raise SystemExit(main())
