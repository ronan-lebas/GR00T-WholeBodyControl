"""From KSO keyframes to a dense, contact-consistent, time-parameterized trajectory.

The kinematic sequence optimization gives one configuration per keyframe. Interpolating
those directly in joint space would break every contact in between — the hands would leave
the chair as soon as it starts turning. So this module builds a *path* first, in a
stage-local path parameter u, by solving the same whole-body IK at intermediate knots with
the chair pose pinned to the interpolated object motion and the in-patch contact locations
frozen at the values KSO chose. Time is only introduced afterwards, which makes retiming
free: the geometry does not depend on it (FARO's per-stage time scaling T_s, eq. 17,
without having to re-solve anything).
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from gear_sonic.trajopt import kso as K
from gear_sonic.trajopt.plans import HANDS, ContactPlan
from gear_sonic.trajopt.scene import TrajOptScene
from gear_sonic.trajopt.se3 import Pose, interp_pose, rot_exp, rot_log

# Fractions of the URDF's motor limits a *reference* trajectory should use. The URDF
# numbers are motor maxima (20-32 rad/s); a demonstration to be tracked by RL must stay
# far below them.
DEFAULT_VEL_SCALE = 0.25
MAX_BASE_SPEED = 0.35  # m/s of pelvis translation
MAX_BASE_OMEGA = 1.0  # rad/s of pelvis rotation
MIN_STAGE_TIME = 0.4  # s
MAX_STAGE_TIME = 12.0  # s


@dataclass
class StagePath:
    """Configurations along one contact stage, sampled in the path parameter u ∈ [0, 1]."""

    u: np.ndarray  # (K,)
    base_p: np.ndarray  # (K, 3)
    base_rv: np.ndarray  # (K, 3)
    qj: np.ndarray  # (K, 29)
    closure: np.ndarray  # (K,)
    start: Pose  # object pose at u = 0
    end: Pose  # object pose at u = 1
    mode_index: int
    residual: Dict[str, float] = field(default_factory=dict)
    _spline: object = field(default=None, repr=False)

    def object_pose(self, u: float) -> Pose:
        return interp_pose(self.start, self.end, float(u))

    def config(self, u: float) -> np.ndarray:
        """Robot configuration at path parameter ``u``: [base_p(3), base_rv(3), qj(29)].

        Interpolated with a C2 spline, not linearly: the timing law is applied on top of
        this, and a piecewise-linear path would inject an acceleration spike at every knot,
        which the centroidal check would (correctly) read as a balance violation.
        """
        if self._spline is None:
            from scipy.interpolate import CubicSpline

            table = np.hstack([self.base_p, self.base_rv, self.qj])
            self._spline = CubicSpline(self.u, table, axis=0, bc_type="natural")
        return np.asarray(self._spline(float(np.clip(u, 0.0, 1.0))))


@dataclass
class Trajectory:
    """A time-sampled reference: robot configuration + object pose + contact schedule."""

    fps: float
    t: np.ndarray  # (T,)
    base_p: np.ndarray  # (T, 3)
    base_rv: np.ndarray  # (T, 3)
    qj: np.ndarray  # (T, 29)
    hand_q: np.ndarray  # (T, 12) left then right, BrainCo actuated joints
    obj_p: np.ndarray  # (T, 3) canonical chair frame
    obj_rv: np.ndarray  # (T, 3) canonical chair frame, rotation vector
    stage: np.ndarray  # (T,) index of the contact stage
    contact: np.ndarray  # (T, 2) bool, [left, right] hand in contact
    stage_names: List[str]
    durations: np.ndarray  # (n_stages,)

    @property
    def n(self) -> int:
        return len(self.t)

    def object_pose(self, i: int) -> Pose:
        return Pose(self.obj_p[i], rot_exp(self.obj_rv[i]))


# --------------------------------------------------------------------------- #
# Path construction (geometry only)
# --------------------------------------------------------------------------- #


def _stage_problem(
    scene: TrajOptScene,
    mode,
    chair_poses: Sequence[Pose],
    uv: Dict[str, np.ndarray],
    ends: Sequence[np.ndarray],
    weights: K.Weights,
) -> K.KinematicProblem:
    """Whole-stage IK: one configuration per knot, contacts frozen, endpoints pinned.

    Solving the knots *jointly* rather than one at a time is not an optimization detail.
    The robot is redundant, so an independent solve at each knot is free to land in a
    different part of the null space, and the resulting path — while every knot satisfies
    every constraint — zig-zags through configuration space. Sampled in time that reads as
    metre-per-second-squared CoM accelerations and throws the balance check off completely.
    The coupling term makes the sequence a path.
    """
    n = len(chair_poses)
    episodes = [
        K.Episode(side=side, patch=mode.patch(side), frames=list(range(n)), uv_index=0,
                  uv_fixed=np.asarray(uv[side], dtype=float))
        for side in HANDS
        if mode.patch(side) is not None
    ]
    free_hands = sum(mode.patch(s) is None for s in HANDS)
    specs = [
        K.FrameSpec(
            mode=mode, chair_ref=pose, posture_scale=1.0 + 2.0 * free_hands,
            fixed_config=(ends[0] if i == 0 else ends[1] if i == n - 1 else None),
        )
        for i, pose in enumerate(chair_poses)
    ]
    return K.KinematicProblem(scene, specs, episodes, weights)


def solve_path(
    scene: TrajOptScene,
    plan: ContactPlan,
    keyframes: Sequence[Dict],
    knots_per_stage: int = 6,
    weights: Optional[K.Weights] = None,
    tol: Optional[K.Tolerances] = None,
    verbose: bool = True,
) -> List[StagePath]:
    """Densify the KSO keyframes into per-stage paths of contact-consistent configurations.

    Interior knots are solved warm-started from their predecessor, so each costs a couple of
    Gauss-Newton steps rather than a fresh IK.
    """
    stages: List[StagePath] = []
    t0 = time.perf_counter()
    n_bad = 0
    # A dedicated weight set for the path solves: the same constraints as KSO, but leaning
    # much harder on inter-knot smoothness, which is the whole point of this pass.
    path_weights = K.Weights(**vars(weights)) if weights is not None else K.Weights()
    path_weights.smooth = 12.0
    for s, mode in enumerate(plan.modes):
        kf_a, kf_b = keyframes[s], keyframes[s + 1]
        # The in-patch contact locations are whatever KSO settled on for this stage's
        # episode (identical at both ends of the stage — that is what a no-slip contact is).
        uv = {side: kf_b["uv"][side] if side in kf_b["uv"] else kf_a["uv"][side]
              for side in HANDS if mode.patch(side) is not None}
        u = np.linspace(0.0, 1.0, max(3, knots_per_stage))
        ends = [
            np.concatenate([kf["base_p"], kf["base_rv"], kf["qj"]]) for kf in (kf_a, kf_b)
        ]
        poses = [interp_pose(kf_a["object"], kf_b["object"], ui) for ui in u]
        prob = _stage_problem(scene, mode, poses, uv, ends, path_weights)
        # Seed with the straight line between the two keyframes: already contact-plausible
        # (both ends are), so the solve is a correction rather than a search.
        guess = prob.initial_guess()
        for i, ui in enumerate(u):
            guess[prob.frame_slice(i)][: K.IDX_QJ.stop] = (1 - ui) * ends[0] + ui * ends[1]
        res = K.solve(prob, warm=guess, max_nfev=4000, tol=tol)
        if not res.success:
            n_bad += 1
        worst = dict(res.report)
        base_p = np.zeros((len(u), 3))
        base_rv = np.zeros((len(u), 3))
        qj = np.zeros((len(u), len(kf_a["qj"])))
        for i in range(len(u)):
            x = res.z[prob.frame_slice(i)]
            base_p[i], base_rv[i], qj[i] = x[K.IDX_BASE_P], x[K.IDX_BASE_R], x[K.IDX_QJ]
        closure = np.full(len(u), mode.closure)
        if s > 0:
            # Ramp the fingers between the previous stage's closure and this one so the
            # grasp closes over a stage instead of snapping shut in one frame.
            prev = plan.modes[s - 1].closure
            if abs(prev - mode.closure) > 1e-6:
                closure = prev + (mode.closure - prev) * (
                    6 * u**5 - 15 * u**4 + 10 * u**3
                )
        stages.append(
            StagePath(u=u, base_p=base_p, base_rv=base_rv, qj=qj, closure=closure,
                      start=kf_a["object"], end=kf_b["object"], mode_index=s, residual=worst)
        )
    if verbose:
        print(f"[trajopt] path: {len(stages)} stages, "
              f"{sum(len(s.u) for s in stages)} knots, {n_bad} stage(s) off-tolerance, "
              f"{time.perf_counter() - t0:.1f}s")
    return stages


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def _quintic(s: np.ndarray) -> np.ndarray:
    """Smoothstep with zero velocity and acceleration at both ends."""
    return 6 * s**5 - 15 * s**4 + 10 * s**3


def stage_durations(
    stages: Sequence[StagePath],
    plan: ContactPlan,
    limits,
    vel_scale: float = DEFAULT_VEL_SCALE,
) -> np.ndarray:
    """Per-stage duration: the plan's nominal, stretched until velocity limits hold.

    A quintic stage profile peaks at 1.875 * Δ/T, which is what the limits are applied to.
    """
    out = []
    for s, stage in enumerate(stages):
        # The bound has to come from the *fastest* piece of the path, not from its total
        # length: the knots are uniform in u but the configuration is not, so a stage that
        # is gentle on average can still contain a segment that moves several times faster.
        du = np.diff(stage.u)[:, None]
        rate_q = np.max(np.abs(np.diff(stage.qj, axis=0)) / du, axis=0)
        rate_p = float(np.max(np.linalg.norm(np.diff(stage.base_p, axis=0), axis=1) / du[:, 0]))
        rate_r = float(np.max(np.linalg.norm(np.diff(stage.base_rv, axis=0), axis=1) / du[:, 0]))
        vmax = limits.velocity * vel_scale
        need = [
            plan.durations[s],
            1.875 * float(np.max(rate_q / np.maximum(vmax, 1e-6))),
            1.875 * rate_p / MAX_BASE_SPEED,
            1.875 * rate_r / MAX_BASE_OMEGA,
        ]
        out.append(float(np.clip(max(need), MIN_STAGE_TIME, MAX_STAGE_TIME)))
    return np.asarray(out)


def sample(
    stages: Sequence[StagePath],
    plan: ContactPlan,
    durations: np.ndarray,
    scene: TrajOptScene,
    fps: float = 50.0,
) -> Trajectory:
    """Sample the path on a uniform time grid under the given stage durations."""
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    total = float(edges[-1])
    t = np.arange(0.0, total + 1e-9, 1.0 / fps)
    n = len(t)
    base_p = np.zeros((n, 3))
    base_rv = np.zeros((n, 3))
    qj = np.zeros((n, stages[0].qj.shape[1]))
    hand_q = np.zeros((n, 12))
    obj_p = np.zeros((n, 3))
    obj_rv = np.zeros((n, 3))
    stage_idx = np.zeros(n, dtype=int)
    contact = np.zeros((n, 2), dtype=bool)

    for i, ti in enumerate(t):
        s = int(np.clip(np.searchsorted(edges, ti, side="right") - 1, 0, len(stages) - 1))
        stage = stages[s]
        frac = 0.0 if durations[s] <= 0 else (ti - edges[s]) / durations[s]
        u = float(_quintic(np.clip(frac, 0.0, 1.0)))
        cfg = stage.config(u)
        base_p[i], base_rv[i], qj[i] = cfg[0:3], cfg[3:6], cfg[6:]
        closure = float(np.interp(u, stage.u, stage.closure))
        hp = scene.hand_posture(closure)
        hand_q[i] = np.concatenate([hp["left"], hp["right"]])
        pose = stage.object_pose(u)
        obj_p[i] = pose.p
        obj_rv[i] = rot_log(pose.Rm)
        stage_idx[i] = s
        mode = plan.modes[s]
        contact[i] = [mode.patch("left") is not None, mode.patch("right") is not None]

    return Trajectory(
        fps=fps, t=t, base_p=base_p, base_rv=base_rv, qj=qj, hand_q=hand_q,
        obj_p=obj_p, obj_rv=obj_rv, stage=stage_idx, contact=contact,
        stage_names=[m.name for m in plan.modes], durations=np.asarray(durations),
    )


# --------------------------------------------------------------------------- #
# Verification of the sampled trajectory
# --------------------------------------------------------------------------- #


def contact_drift(
    scene: TrajOptScene, traj: Trajectory, plan: ContactPlan
) -> Dict[str, float]:
    """Largest contact / foot error introduced by interpolating between path knots."""
    patches = scene.chair.patches()
    worst = {"contact_pos": 0.0, "contact_normal": 0.0, "feet": 0.0, "collision": 0.0}
    worst_pair = ""
    pairs = np.asarray(list(scene.chair_pairs) + list(scene.self_pairs), dtype=int)
    for i in range(traj.n):
        mode = plan.modes[int(traj.stage[i])]
        obj = traj.object_pose(i)
        scene.set_state(
            traj.base_p[i], traj.base_rv[i], traj.qj[i],
            {"left": traj.hand_q[i, :6], "right": traj.hand_q[i, 6:]}, obj,
        )
        for side in HANDS:
            name = mode.patch(side)
            if name is None:
                continue
            palm = scene.palm_pose(side)
            # The in-patch coordinate is not stored per frame; measure the perpendicular
            # error only (distance along the patch normal + normal misalignment), which is
            # what "the hand left the surface" means.
            n = obj.rotate(patches[name].normal)
            d = palm.p - obj.apply(patches[name].center)
            worst["contact_pos"] = max(worst["contact_pos"], abs(float(d @ n)))
            worst["contact_normal"] = max(
                worst["contact_normal"], float(np.linalg.norm(palm.Rm[:, 2] + n))
            )
        for name, target in zip(("left_ankle_roll_link", "right_ankle_roll_link"),
                                scene.foot_targets):
            pose = scene.body_pose(name)
            worst["feet"] = max(worst["feet"], float(np.abs(pose.p - target.p).max()))
        d = scene.distances(pairs, K.COLLISION_MARGIN + 0.05)
        j = int(np.argmax(K.COLLISION_MARGIN - d))
        if float(K.COLLISION_MARGIN - d[j]) > worst["collision"]:
            worst["collision"] = float(K.COLLISION_MARGIN - d[j])
            worst_pair = "{} / {}".format(
                *(scene.model.body(int(scene.model.geom_bodyid[g])).name for g in pairs[j])
            )
    if worst_pair:
        worst["closest_pair"] = worst_pair
    return worst
