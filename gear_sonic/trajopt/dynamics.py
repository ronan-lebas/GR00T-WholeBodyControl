"""Quasi-static / centroidal feasibility pass — this repo's stand-in for FARO eq. 17.

FARO closes its hierarchy with a full trajectory optimization: whole-body and object
dynamics, contact wrenches and time scaling solved together. That needs a direct-collocation
NLP stack (casadi + acados/Hippo) which is not in this repo's environment, and the reference
produced here does not need it: an RL tracking controller re-derives the contact forces
anyway. What it *does* need is a trajectory that is not physically absurd — one whose
required contact forces exist, whose ZMP stays under the feet, and whose joints are not
asked for impossible speeds or torques.

So this module checks exactly that, on the already-sampled trajectory:

  * **object Newton-Euler** — the wrench the hands must apply to move the chair as
    commanded, split over the active contacts, then tested against the friction cone,
    torsional friction and centre-of-pressure bounds of its patch (FARO eq. 7c/7d);
  * **robot centroidal balance** — the ZMP of the robot (carrying whatever the hands hold,
    and taking the reaction of whatever they push) must stay inside the foot polygon;
  * **actuation** — inverse dynamics with those contact wrenches, against the URDF's
    torque and velocity limits (FARO eq. 12/13).

Everything is a function of the timing alone (the geometry is fixed by the path), so the
single lever for fixing a violation is stage duration — which is exactly FARO's time-scaling
variable, and what ``retime_until_feasible`` searches over.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from gear_sonic.trajopt import kso as K
from gear_sonic.trajopt.plans import HANDS, ContactPlan
from gear_sonic.trajopt.scene import TrajOptScene
from gear_sonic.trajopt.trajectory import Trajectory, sample

GRAVITY = np.array([0.0, 0.0, -9.81])
MU_HAND = 0.7  # palm/chair sliding friction used for the cone test
MU_FLOOR = 0.5  # chair/floor friction, for the push case
# A "grasp" patch means fingers wrapped around it, so the contact can pull and can resist a
# moment through the grip itself, not only through the pressure distribution over a flat
# patch. Modelling that as an internal squeeze force is crude but keeps the check on the
# right side of reality: a two-hand grip on a backrest holds a chair, a flat palm does not.
GRIP_FORCE = 15.0  # N of internal grip a wrapped grasp can add to the normal force
GRASP_LEVER = 0.06  # m, effective moment arm of a wrapped grasp (palm/finger span)
MU_TORSION = 0.35  # torsional friction coefficient of a wrapped grasp, per meter of lever
TORQUE_FRACTION = 0.8  # of the URDF effort limit a reference may ask for
# How far outside the foot polygon the ZMP may stray before the trajectory is rejected.
# Not zero: this reference is tracked by a feedback controller with ankle authority and
# angular momentum of its own, and the check itself neglects the momentum rate. A
# centimetre for a fraction of a second is well inside what tracking absorbs; demanding a
# hard zero on a robot whose feet are 17 cm long, reaching a chair half a metre away,
# stretches every trajectory to a minute of slow motion instead.
ZMP_TOLERANCE = 0.01  # m
MAX_TOTAL_TIME = 25.0  # s: past this, slowing down is not fixing anything


@dataclass
class DynamicsReport:
    zmp_margin: float  # m, min distance of the ZMP inside the foot polygon (>0 is good)
    friction_margin: float  # min over contacts of (mu*fn - |ft|), N (>0 is good)
    normal_force_min: float  # N, most negative normal force asked of a contact
    grasp_moment_ratio: float  # required patch moment / what the patch can hold (<1 good)
    torque_ratio: float  # max |tau| / (fraction * limit) (<1 good)
    velocity_ratio: float  # max |qdot| / limit (<1 good)
    hand_force_max: float  # N
    duration: float  # s
    ok: bool
    detail: str = ""
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        return (
            f"T={self.duration:.2f}s zmp_margin={self.zmp_margin*100:.1f}cm "
            f"friction={self.friction_margin:.1f}N grasp_moment={self.grasp_moment_ratio:.2f} "
            f"torque={self.torque_ratio:.2f} vel={self.velocity_ratio:.2f} "
            f"|f|max={self.hand_force_max:.1f}N -> {'ok' if self.ok else self.detail}"
        )


def _derivatives(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference first and second derivatives of a sampled signal."""
    v = np.gradient(x, dt, axis=0, edge_order=2)
    a = np.gradient(v, dt, axis=0, edge_order=2)
    return v, a


def _angular_velocity(rot: List[np.ndarray], dt: float) -> np.ndarray:
    """Body-frame-free angular velocity from a sequence of rotation matrices."""
    n = len(rot)
    w = np.zeros((n, 3))
    for i in range(n):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        if j1 == j0:
            continue
        dR = rot[j1] @ rot[j0].T
        from gear_sonic.trajopt.se3 import rot_log

        w[i] = rot_log(dR) / ((j1 - j0) * dt)
    return w


def _grasp_split(
    points: np.ndarray, com: np.ndarray, wrench: np.ndarray, moment_weight: float = 40.0
) -> np.ndarray:
    """Minimum-effort split of a required wrench over rigid contacts.

    ``points`` are the contact positions, ``wrench`` = (force, torque about ``com``). Each
    contact may apply a force and a local moment (a wrapped grasp can); moments are
    penalized relative to forces so the solver prefers a force couple across the hands over
    twisting a single wrist, which is also what the hardware prefers.

    Returns ``(n_contacts, 6)`` of (force, moment).
    """
    n = len(points)
    G = np.zeros((6, 6 * n))
    for i, p in enumerate(points):
        G[0:3, 6 * i : 6 * i + 3] = np.eye(3)
        r = p - com
        G[3:6, 6 * i : 6 * i + 3] = np.array(
            [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]]
        )
        G[3:6, 6 * i + 3 : 6 * i + 6] = np.eye(3)
    scale = np.tile(np.array([1.0, 1.0, 1.0, moment_weight, moment_weight, moment_weight]), n)
    y, *_ = np.linalg.lstsq(G / scale, wrench, rcond=None)
    return (y / scale).reshape(n, 6)


def analyze(
    scene: TrajOptScene,
    traj: Trajectory,
    plan: ContactPlan,
    patch_half: float = 0.05,
) -> DynamicsReport:
    """Run the whole feasibility pass over a sampled trajectory."""
    dt = 1.0 / traj.fps
    n = traj.n
    patches = scene.chair.patches()
    chair = scene.chair

    # --- kinematics of every frame (robot CoM, palm poses, object motion) ------
    com_r = np.zeros((n, 3))
    palm = {side: np.zeros((n, 3)) for side in HANDS}
    obj_com = np.zeros((n, 3))
    obj_rot: List[np.ndarray] = []
    for i in range(n):
        pose = traj.object_pose(i)
        scene.set_state(
            traj.base_p[i], traj.base_rv[i], traj.qj[i],
            {"left": traj.hand_q[i, :6], "right": traj.hand_q[i, 6:]}, pose,
        )
        com_r[i] = scene.robot_com()
        for side in HANDS:
            palm[side][i] = scene.palm_pose(side).p
        obj_com[i] = pose.apply(chair.com)
        obj_rot.append(pose.Rm)

    _, acc_r = _derivatives(com_r, dt)
    vel_o, acc_o = _derivatives(obj_com, dt)
    omega = _angular_velocity(obj_rot, dt)
    alpha = np.gradient(omega, dt, axis=0, edge_order=2)
    qdot = np.gradient(traj.qj, dt, axis=0, edge_order=2)
    qddot = np.gradient(qdot, dt, axis=0, edge_order=2)
    base_v, base_a = _derivatives(traj.base_p, dt)
    base_w = np.gradient(traj.base_rv, dt, axis=0, edge_order=2)
    base_al = np.gradient(base_w, dt, axis=0, edge_order=2)
    hand_bodies = {s: scene.model.body(f"{s}_base_link").id for s in HANDS}

    support = K._convex_edges(scene.foot_polygon())

    zmp_margin = np.full(n, np.inf)
    fric_margin = np.full(n, np.inf)
    fn_min = np.inf
    moment_ratio = 0.0
    force_max = 0.0
    torque_ratio = 0.0

    for i in range(n):
        mode = plan.modes[int(traj.stage[i])]
        sides = [s for s in HANDS if mode.patch(s) is not None]
        pose = traj.object_pose(i)

        # ---- wrench the hands must apply to the object -----------------------
        hand_w = np.zeros((len(sides), 6))
        if sides:
            I_world = pose.Rm @ chair.inertia @ pose.Rm.T
            force = chair.mass * (acc_o[i] - GRAVITY)
            torque = I_world @ alpha[i] + np.cross(omega[i], I_world @ omega[i])
            if mode.grounded:
                # The floor carries the weight; the hands only have to overcome friction
                # and accelerate the chair (this is the push case).
                normal = max(0.0, chair.mass * 9.81)
                speed = np.linalg.norm(vel_o[i][:2])
                drag = np.zeros(3)
                if speed > 1e-4:
                    drag[:2] = -MU_FLOOR * normal * vel_o[i][:2] / speed
                force = chair.mass * acc_o[i] - drag
                force[2] = 0.0
            pts = np.stack([palm[s][i] for s in sides])
            hand_w = _grasp_split(pts, obj_com[i], np.concatenate([force, torque]))

        # ---- contact-level checks (friction cone, CoP, torsion) --------------
        for j, side in enumerate(sides):
            patch = patches[mode.patch(side)]
            grasped = patch.kind == "grasp"
            grip = GRIP_FORCE if grasped else 0.0
            lever = GRASP_LEVER if grasped else min(patch.half_extents)
            nrm = pose.rotate(patch.normal)
            f = hand_w[j, :3]
            mom = hand_w[j, 3:]
            # Force applied *by the hand on the chair* presses inward, i.e. along -normal.
            fn = -float(f @ nrm)
            ft = float(np.linalg.norm(f + fn * nrm))
            fn_min = min(fn_min, fn + grip)
            hold = max(fn, 0.0) + grip  # what actually presses the surfaces together
            fric_margin[i] = min(fric_margin[i], MU_HAND * hold - ft)
            force_max = max(force_max, float(np.linalg.norm(f)))
            # A contact resists a moment through its lever arm (centre of pressure inside
            # the patch / the finger wrap) and through torsional friction about its normal.
            mn = abs(float(mom @ nrm))
            mt = float(np.linalg.norm(mom - (mom @ nrm) * nrm))
            # A contact that is not pressing cannot hold any moment at all. That case is
            # already reported through fn_min ("would have to pull"); dividing by its
            # zero capacity here would only produce a meaningless astronomic ratio.
            if hold > 0.1:
                moment_ratio = max(
                    moment_ratio, mt / (hold * lever), mn / (hold * MU_TORSION * lever)
                )

        # ---- robot centroidal balance ---------------------------------------
        mass = scene.robot_mass
        com = com_r[i].copy()
        acc = acc_r[i].copy()
        if sides and not mode.grounded:
            # The chair rides with the robot: fold it into the centroidal state.
            mass_c = chair.mass
            com = (com * scene.robot_mass + obj_com[i] * mass_c) / (scene.robot_mass + mass_c)
            acc = (acc * scene.robot_mass + acc_o[i] * mass_c) / (scene.robot_mass + mass_c)
            mass += mass_c
            ext_f = np.zeros((0, 3))
            ext_p = np.zeros((0, 3))
            ext_b: List[int] = []
        else:
            # Reaction of whatever the hands push on, applied to the robot.
            ext_f = np.array([-hand_w[j, :3] for j in range(len(sides))]).reshape(-1, 3)
            ext_p = np.array([palm[s][i] for s in sides]).reshape(-1, 3)
            ext_b = [hand_bodies[s] for s in sides]

        grf = mass * (acc - GRAVITY) - ext_f.sum(0) if len(ext_f) else mass * (acc - GRAVITY)
        tau_ext = np.zeros(3)
        for p, f in zip(ext_p, ext_f):
            tau_ext += np.cross(p - com, f)
        if grf[2] > 1.0:
            zmp = np.array(
                [
                    com[0] + (tau_ext[1] - com[2] * grf[0]) / grf[2],
                    com[1] + (-tau_ext[0] - com[2] * grf[1]) / grf[2],
                ]
            )
            zmp_margin[i] = -_signed_distance(support, zmp)
        else:  # airborne robot: not a case this generator can produce
            zmp_margin[i] = -np.inf

        # ---- actuation -------------------------------------------------------
        torque_ratio = max(
            torque_ratio,
            _torque_ratio(scene, traj, i, qdot[i], qddot[i], base_v[i], base_a[i],
                          base_w[i], base_al[i], ext_b, ext_p, ext_f, grf),
        )

    vel_ratio = float(np.max(np.abs(qdot) / scene.limits.velocity))
    ok = True
    detail = []
    if float(zmp_margin.min()) < -ZMP_TOLERANCE:
        ok = False
        detail.append(f"ZMP leaves the support polygon by {-zmp_margin.min()*100:.1f} cm")
    if np.isfinite(fric_margin).any() and float(np.nanmin(fric_margin)) < 0.0:
        ok = False
        detail.append(f"friction cone violated by {-np.nanmin(fric_margin):.1f} N")
    if fn_min < 0.0:
        ok = False
        detail.append(f"a contact would have to pull ({fn_min:.1f} N)")
    if moment_ratio > 1.0:
        ok = False
        detail.append(f"patch moment {moment_ratio:.2f}x what the contact can hold")
    if torque_ratio > 1.0:
        ok = False
        detail.append(f"joint torque {torque_ratio:.2f}x the usable limit")
    if vel_ratio > 1.0:
        ok = False
        detail.append(f"joint velocity {vel_ratio:.2f}x the limit")

    return DynamicsReport(
        zmp_margin=float(zmp_margin.min()),
        friction_margin=float(np.nanmin(fric_margin)) if np.isfinite(fric_margin).any() else 0.0,
        normal_force_min=float(fn_min) if np.isfinite(fn_min) else 0.0,
        grasp_moment_ratio=float(moment_ratio),
        torque_ratio=float(torque_ratio),
        velocity_ratio=float(vel_ratio),
        hand_force_max=float(force_max),
        duration=float(traj.t[-1]),
        ok=ok,
        detail="; ".join(detail),
        per_frame={"zmp_margin": zmp_margin, "friction_margin": fric_margin},
    )


def _signed_distance(edges, p: np.ndarray) -> float:
    normals, offsets = edges
    return float(np.max(normals @ p - offsets))


def _torque_ratio(
    scene: TrajOptScene,
    traj: Trajectory,
    i: int,
    qdot: np.ndarray,
    qddot: np.ndarray,
    base_v: np.ndarray,
    base_a: np.ndarray,
    base_w: np.ndarray,
    base_al: np.ndarray,
    ext_b: List[int],
    ext_p: np.ndarray,
    ext_f: np.ndarray,
    grf: np.ndarray,
) -> float:
    """Rough inverse-dynamics torque check for one frame.

    ``M q'' + C`` comes from MuJoCo's RNE; the external wrenches (hand reactions and the
    ground reaction, split evenly between the feet) are mapped to joint space through the
    contact Jacobians. Angular-momentum rate is neglected — the reference is slow by
    construction — so this is a magnitude check, not a certificate.
    """
    m, d = scene.model, scene.data
    pose = traj.object_pose(i)
    scene.set_state(
        traj.base_p[i], traj.base_rv[i], traj.qj[i],
        {"left": traj.hand_q[i, :6], "right": traj.hand_q[i, 6:]}, pose,
    )
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    dofs = [int(m.jnt_dofadr[m.joint(name).id]) for name in _BODY_JOINTS(scene)]
    d.qvel[dofs] = qdot
    d.qacc[dofs] = qddot
    d.qvel[0:3], d.qvel[3:6] = base_v, base_w
    d.qacc[0:3], d.qacc[3:6] = base_a, base_al
    mujoco.mj_comPos(m, d)
    bias = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 1, bias)  # includes gravity and the requested acceleration

    tau = bias.copy()
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for bid, p, f in zip(ext_b, ext_p, ext_f):
        mujoco.mj_jac(m, d, jacp, jacr, p, bid)
        tau -= jacp.T @ f
    # Ground reaction, halved between the feet at their sole centres.
    for gid in scene.foot_sole_geoms:
        mujoco.mj_jac(m, d, jacp, jacr, d.geom_xpos[gid], int(m.geom_bodyid[gid]))
        tau -= jacp.T @ (grf / len(scene.foot_sole_geoms))
    used = np.abs(tau[dofs]) / np.maximum(scene.limits.effort * TORQUE_FRACTION, 1e-6)
    return float(used.max())


def _BODY_JOINTS(scene: TrajOptScene):
    from gear_sonic.trajopt.scene import BODY_JOINT_NAMES

    return BODY_JOINT_NAMES


def retime_until_feasible(
    scene: TrajOptScene,
    stages,
    plan: ContactPlan,
    durations: np.ndarray,
    fps: float = 50.0,
    max_rounds: int = 6,
    growth: float = 1.4,
    verbose: bool = True,
) -> Tuple[Trajectory, DynamicsReport, np.ndarray]:
    """Stretch the stage durations until the dynamic checks pass (FARO's time scaling).

    Slowing down shrinks every inertial term quadratically; what it cannot fix is a
    configuration whose *static* requirements are already impossible (a grasp that has to
    resist more moment than the patch can hold, a posture whose gravity torque exceeds the
    motors). Those stop the loop with the reason attached, so the caller can drop the plan.
    """
    dur = np.array(durations, dtype=float)
    traj = sample(stages, plan, dur, scene, fps=fps)
    report = analyze(scene, traj, plan)
    for _ in range(max_rounds):
        if report.ok:
            break
        if report.grasp_moment_ratio > 1.0 and report.zmp_margin >= 0.0:
            break  # static: no amount of slowing down helps
        if dur.sum() * growth > MAX_TOTAL_TIME:
            break
        dur = dur * growth
        if verbose:
            print(f"[trajopt] dynamics: {report.detail} -> stretching to "
                  f"{dur.sum():.1f}s total")
        traj = sample(stages, plan, dur, scene, fps=fps)
        report = analyze(scene, traj, plan)
    return traj, report, dur
