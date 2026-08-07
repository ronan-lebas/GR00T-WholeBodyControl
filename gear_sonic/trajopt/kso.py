"""Kinematic feasibility optimization: mode/edge filters and the sequence solve.

Implements the two cheap layers of FARO's feasibility hierarchy:

  * **mode / edge feasibility** (eq. 14) — a single whole-body configuration satisfying one
    contact mode (or the union of two adjacent modes, i.e. a transition), subject to
    contact, collision and limit constraints;
  * **kinematic sequence optimization, KSO** (eq. 15) — one configuration per keyframe of a
    contact plan, coupled by the requirement that a persisting contact keeps the same point
    on the object patch (FARO's no-slip eq. 8, enforced here structurally: the in-patch
    coordinates are shared decision variables over a contact episode).

Both are the same nonlinear least-squares problem with a different number of frames, solved
with ``scipy.optimize.least_squares`` (trust-region reflective, analytic sparsity pattern,
finite-difference Jacobian; the model has no analytic derivatives available without casadi).
Hard constraints are entered either as bounds (joint/base limits, in-patch coordinates,
object slack) or as heavily weighted residuals; inequalities (collision, balance) as
one-sided hinges. Feasibility is then decided on the *unweighted* violations, so weights
only shape convergence, never the verdict.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from gear_sonic.trajopt.chair import ContactPatch
from gear_sonic.trajopt.plans import HANDS, ContactMode, ContactPlan, contact_episodes
from gear_sonic.trajopt.scene import BODY_JOINT_NAMES, TrajOptScene
from gear_sonic.trajopt.se3 import Pose, interp_pose, rot_error, rot_exp

NX = 41  # base pos 3 + base rotvec 3 + body joints 29 + object dp 3 + object drotvec 3
IDX_BASE_P = slice(0, 3)
IDX_BASE_R = slice(3, 6)
IDX_QJ = slice(6, 35)
IDX_OBJ_P = slice(35, 38)
IDX_OBJ_R = slice(38, 41)


@dataclass
class Weights:
    """Residual weights (sqrt-scale: least_squares minimizes 0.5*||r||^2)."""

    contact_pos: float = 120.0
    contact_normal: float = 40.0
    feet: float = 200.0
    chair_floor: float = 120.0
    balance: float = 200.0
    object_support: float = 25.0
    collision: float = 120.0
    hand_clearance: float = 120.0
    posture: float = 1.0
    base_reg: float = 2.0
    obj_reg: float = 8.0
    com_reg: float = 6.0
    smooth: float = 2.0


@dataclass
class Tolerances:
    """Feasibility thresholds on the *unweighted* constraint violations."""

    contact_pos: float = 0.012  # m
    contact_normal: float = 0.20  # ~11.5 deg of palm/patch normal misalignment
    # m and rad. The feet are planted, so this is drift, not motion — but it is one norm
    # over position and orientation of a foot the solver is also asked to balance on, and
    # tightening it much further starts rejecting configurations for millimetres.
    feet: float = 0.008
    chair_floor: float = 0.010  # m
    # The balance residual hinges on BALANCE_MARGIN, so a tolerance equal to that margin
    # means "the CoM is inside the true foot polygon"; the 2.5 cm margin itself is only the
    # shaping target the solver aims for. The G1's feet are small (17 x 30 cm) and the chair
    # stands half a metre away, so demanding much more than this rejects every plan — and
    # the real gate on standing up during the motion is the ZMP check in the dynamic pass.
    balance: float = 0.025
    object_support: float = 0.02  # m
    # Collision residuals hinge on COLLISION_MARGIN, so this requires 1 cm of real
    # clearance (0.035 - 0.025) at the solved knots, which leaves room for the few
    # millimetres the spline interpolation between them can eat.
    collision: float = 0.025
    # A hand on its way to a grasp is allowed to come right up to the chair — it is about
    # to touch it. All this forbids is passing *through* it (margin minus tolerance = 0).
    hand_clearance: float = 0.015


# Posture regularization weight per body joint: legs are free to do what balance needs,
# arms should stay near a natural pose unless a contact drags them.
_POSTURE_W = np.array(
    [0.4] * 12  # legs
    + [0.8, 1.5, 1.0]  # waist yaw / roll / pitch
    + [0.25] * 7  # left arm
    + [0.25] * 7  # right arm
)

# What decides kinematic feasibility. "object_support" — the chair's CoM lying between the
# palms — is deliberately advisory only: a wrapped two-hand grasp resists that moment
# instead, and whether it can is a question about forces, which only the dynamic pass can
# answer. Same layering as FARO, where dynamics enter at the trajectory-optimization filter
# and not before.
CONSTRAINT_TERMS = (
    "feet", "contact_pos", "contact_normal", "chair_floor", "collision", "hand_clearance",
    "balance",
)
ADVISORY_TERMS = CONSTRAINT_TERMS + ("object_support",)
# Terms whose residuals are 3-vectors, so violations are measured as vector norms.
_VECTOR_TERMS = frozenset({"feet", "contact_pos", "contact_normal"})

FILTER_MAX_NFEV = 600  # a mode/edge filter that has not converged by here is not going to
COLLISION_MARGIN = 0.035  # m: keep this much clear of the chair and of itself
HAND_CLEARANCE_MARGIN = 0.015  # m: a free hand near the chair it is reaching for
BALANCE_MARGIN = 0.025  # m: CoM must stay this far inside the foot polygon
CHAIR_CLEARANCE = 0.02  # m: minimum floor gap once the chair is airborne
GRASP_SPAN_MARGIN = 0.04  # m: chair CoM must stay this far inside the hand-to-hand span


@dataclass
class FrameSpec:
    """One keyframe of a kinematic problem."""

    mode: ContactMode
    chair_ref: Pose
    chair_pos_tol: float = 0.0  # object position slack around chair_ref (0 = hard)
    chair_rot_tol: float = 0.0
    posture_scale: float = 1.0
    # Pin the robot part of this frame (base pose + joints) to a given configuration —
    # used to clamp a stage path onto the keyframes KSO already solved.
    fixed_config: Optional[np.ndarray] = None


@dataclass
class Episode:
    """A contact episode: one (hand, patch) pairing held over a set of frames."""

    side: str
    patch: str
    frames: List[int]
    uv_index: int  # column offset of its two in-patch coordinates
    uv_fixed: Optional[np.ndarray] = None


@dataclass
class SolveResult:
    success: bool
    z: np.ndarray
    report: Dict[str, float]
    cost: float
    n_eval: int
    seconds: float
    message: str = ""
    tol: "Tolerances" = field(default_factory=lambda: Tolerances())

    def worst(self) -> Tuple[str, float]:
        """Worst constraint, ranked by how far past *its own* tolerance it is."""
        items = {k: v for k, v in self.report.items() if k in CONSTRAINT_TERMS}
        if not items:
            return ("", 0.0)
        k = max(items, key=lambda n: items[n] / getattr(self.tol, n))
        return (k, items[k])


class KinematicProblem:
    """Least-squares problem over F keyframes and the episodes' in-patch coordinates."""

    def __init__(
        self,
        scene: TrajOptScene,
        specs: Sequence[FrameSpec],
        episodes: Sequence[Episode],
        weights: Optional[Weights] = None,
        couple_frames: bool = True,
    ):
        self.scene = scene
        self.specs = list(specs)
        self.episodes = list(episodes)
        # Copied, never aliased: solve() escalates these in place during continuation, and
        # a shared instance would carry the inflated penalties into every later problem.
        self.w = Weights(**vars(weights)) if weights is not None else Weights()
        self.couple = couple_frames and len(specs) > 1
        self.patches = scene.chair.patches()
        self.n_frames = len(self.specs)
        self.n_uv = 2 * sum(1 for e in self.episodes if e.uv_fixed is None)
        self.n_z = NX * self.n_frames + self.n_uv

        # Episodes indexed by frame and side, for O(1) lookup in the inner loop.
        self._ep_at: Dict[Tuple[int, str], Episode] = {}
        for ep in self.episodes:
            for s in ep.frames:
                self._ep_at[(s, ep.side)] = ep

        # The feet are held at the nominal stance, so the support polygon is constant:
        # precompute its outward edge normals once (max over edges = signed distance).
        self._support_edges = _convex_edges(scene.foot_polygon())
        self._support_center = scene.foot_polygon().mean(0)
        self._pairs = [self._frame_pairs(s) for s in range(self.n_frames)]
        self._cache: Dict[int, Tuple[bytes, Dict[str, np.ndarray]]] = {}
        self._term_sizes = self._probe_sizes()
        self.n_eval = 0

    # ------------------------------------------------------------------ #
    # layout helpers
    # ------------------------------------------------------------------ #

    def _frame_pairs(self, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(structural, hand)`` geom pairs checked at frame ``s``.

        A grasping hand is not checked against the chair at all (the contact constraints
        place it); a free hand is, but only against penetration — it is on its way to touch
        the thing, so demanding the structural clearance there would forbid every approach.
        """
        mode = self.specs[s].mode
        structural = list(self.scene.chair_pairs) + list(self.scene.self_pairs)
        hand = []
        for side in HANDS:
            if mode.patch(side) is None:
                hand += self.scene.hand_chair_pairs[side]
        return (np.asarray(structural, dtype=int),
                np.asarray(hand, dtype=int).reshape(-1, 2))

    def frame_slice(self, s: int) -> slice:
        return slice(NX * s, NX * (s + 1))

    def uv_of(self, z: np.ndarray, ep: Episode) -> np.ndarray:
        if ep.uv_fixed is not None:
            return ep.uv_fixed
        base = NX * self.n_frames + ep.uv_index
        return z[base : base + 2]

    def object_pose(self, s: int, x: np.ndarray) -> Pose:
        ref = self.specs[s].chair_ref
        return Pose(ref.p + x[IDX_OBJ_P], ref.Rm @ rot_exp(x[IDX_OBJ_R]))

    def apply_frame(self, s: int, x: np.ndarray) -> Pose:
        """Push frame ``s``'s configuration into the scene; returns the object pose."""
        obj = self.object_pose(s, x)
        self.scene.set_state(
            x[IDX_BASE_P],
            x[IDX_BASE_R],
            x[IDX_QJ],
            self.scene.hand_posture(self.specs[s].mode.closure),
            obj,
        )
        return obj

    # ------------------------------------------------------------------ #
    # residual terms
    # ------------------------------------------------------------------ #

    def _frame_terms(self, s: int, x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
        """Unweighted residual terms of one keyframe (hinges already one-sided)."""
        spec = self.specs[s]
        mode = spec.mode
        if spec.fixed_config is not None:
            # Clamped knot (a stage endpoint, already solved by KSO): every term here is a
            # constant, so it can only add noise to the report — and would flag things like
            # "the hand is touching the chair" at the very knot where it lets go.
            return {k: np.zeros(0) for k in self.TERM_ORDER}
        obj = self.apply_frame(s, x)
        scene = self.scene
        out: Dict[str, np.ndarray] = {}

        # Feet planted at the nominal stance (the sim runs a static base: no stepping).
        feet = []
        for name, target in zip(("left_ankle_roll_link", "right_ankle_roll_link"),
                                scene.foot_targets):
            pose = scene.body_pose(name)
            feet.append(np.concatenate([pose.p - target.p, rot_error(pose.Rm, target.Rm)]))
        out["feet"] = np.concatenate(feet)

        # Contacts: palm pad on the patch point, palm normal opposing the patch normal.
        cpos, cnrm = [], []
        for side in HANDS:
            patch_name = mode.patch(side)
            if patch_name is None:
                continue
            patch: ContactPatch = self.patches[patch_name]
            uv = self.uv_of(z, self._ep_at[(s, side)])
            palm = scene.palm_pose(side)
            cpos.append(palm.p - obj.apply(patch.point(uv)))
            cnrm.append(palm.Rm[:, 2] + obj.rotate(patch.normal))
        out["contact_pos"] = np.concatenate(cpos) if cpos else np.zeros(0)
        out["contact_normal"] = np.concatenate(cnrm) if cnrm else np.zeros(0)

        # Chair floor state: legs on the ground, or a clear gap under all of them.
        tips_z = obj.apply(scene.chair.leg_tips())[:, 2]
        if spec.chair_pos_tol <= 0.0 and spec.chair_rot_tol <= 0.0:
            # The object pose is fixed here (a path knot), so this term is a constant —
            # keeping it would report the lift-off knot, which sits exactly on the floor,
            # as violating its own stage's airborne clearance.
            out["chair_floor"] = np.zeros(0)
        elif mode.grounded:
            out["chair_floor"] = tips_z
        else:
            out["chair_floor"] = np.array([max(0.0, CHAIR_CLEARANCE - float(tips_z.min()))])

        # Quasi-static balance: the CoM of the robot (plus the chair while it is carried)
        # must stay inside the foot polygon. This is the cheap stand-in for FARO's
        # centroidal dynamics; the dynamic pass later checks it under acceleration.
        com = scene.robot_com() * scene.robot_mass
        mass = scene.robot_mass
        if not mode.grounded:
            com = com + obj.apply(scene.chair.com) * scene.chair.mass
            mass += scene.chair.mass
        com = com / mass
        # Note this applies in every mode, including while a hand rests on the grounded
        # chair. A robot *can* lean on what it pushes, but only if that object can carry the
        # load — and the chair here weighs 0.2 kg. Extending the support polygon through a
        # hand contact would need the contact forces, i.e. the dynamic pass; keeping the
        # test unconditional is the conservative choice, at the price of rejecting pushes
        # that a heavy, well-braced object would allow.
        out["balance"] = np.array([_outside(self._support_edges, com[:2], BALANCE_MARGIN)])

        # Object support: while airborne, the chair's CoM must lie between the hands.
        pts = [
            scene.palm_pose(side).p for side in HANDS if mode.patch(side) is not None
        ]
        if mode.grounded or not pts:
            out["object_support"] = np.zeros(1)
        else:
            ccom = obj.apply(scene.chair.com)[:2]
            out["object_support"] = np.array(
                [_segment_excess(np.array([p[:2] for p in pts]), ccom, GRASP_SPAN_MARGIN)]
            )

        # Collision avoidance (FARO eq. 9, via MuJoCo's convex distance query).
        structural, hand_pairs = self._pairs[s]
        d = scene.distances(structural, COLLISION_MARGIN + 0.05)
        out["collision"] = np.maximum(0.0, COLLISION_MARGIN - d)
        if len(hand_pairs):
            dh = scene.distances(hand_pairs, HAND_CLEARANCE_MARGIN + 0.05)
            out["hand_clearance"] = np.maximum(0.0, HAND_CLEARANCE_MARGIN - dh)
        else:
            out["hand_clearance"] = np.zeros(0)

        # Regularization toward the ready stance and the plan's object pose.
        out["posture"] = _POSTURE_W * spec.posture_scale * (x[IDX_QJ] - scene.nominal["qj"])
        out["base_reg"] = np.concatenate(
            [x[IDX_BASE_P] - scene.nominal["base_p"], x[IDX_BASE_R] - scene.nominal["base_rv"]]
        )
        out["obj_reg"] = np.concatenate([x[IDX_OBJ_P], x[IDX_OBJ_R]])
        # Weak pull of the CoM toward the middle of the feet. The balance hinge above is
        # flat wherever it is satisfied, so on its own it gives the solver nothing to
        # follow until the configuration is already unstable; this term is what actually
        # makes the robot sit back as it reaches forward.
        out["com_reg"] = com[:2] - self._support_center
        return out

    TERM_ORDER = (
        "feet", "contact_pos", "contact_normal", "chair_floor", "balance",
        "object_support", "collision", "hand_clearance", "posture", "base_reg", "obj_reg",
        "com_reg",
    )
    CONSTRAINT_TERMS = CONSTRAINT_TERMS  # module-level (see above)
    ADVISORY_TERMS = ADVISORY_TERMS

    def _probe_sizes(self) -> List[Dict[str, int]]:
        z0 = self.initial_guess()
        sizes = []
        for s in range(self.n_frames):
            terms = self._frame_terms(s, z0[self.frame_slice(s)], z0)
            sizes.append({k: len(v) for k, v in terms.items()})
        self._cache.clear()
        return sizes

    def _weights_for(self, name: str) -> float:
        return getattr(self.w, name)

    def _frame_vector(self, s: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        key = x.tobytes() + b"|" + b"".join(
            np.asarray(self.uv_of(z, self._ep_at[(s, side)])).tobytes()
            for side in HANDS
            if (s, side) in self._ep_at
        )
        hit = self._cache.get(s)
        if hit is not None and hit[0] == key:
            return hit[1]
        terms = self._frame_terms(s, x, z)
        vec = np.concatenate(
            [self._weights_for(n) * terms[n] for n in self.TERM_ORDER]
        )
        self._cache[s] = (key, vec)
        return vec

    def residuals(self, z: np.ndarray) -> np.ndarray:
        self.n_eval += 1
        parts = [self._frame_vector(s, z[self.frame_slice(s)], z) for s in range(self.n_frames)]
        if self.couple:
            for s in range(self.n_frames - 1):
                a = z[self.frame_slice(s)]
                b = z[self.frame_slice(s + 1)]
                parts.append(self.w.smooth * (b[: IDX_QJ.stop] - a[: IDX_QJ.stop]))
        return np.concatenate(parts)

    def report(self, z: np.ndarray) -> Dict[str, float]:
        """Max unweighted violation per category over all frames (advisory ones included)."""
        out = {k: 0.0 for k in self.ADVISORY_TERMS}
        for s in range(self.n_frames):
            terms = self._frame_terms(s, z[self.frame_slice(s)], z)
            for k in self.ADVISORY_TERMS:
                if not len(terms[k]):
                    continue
                if k in _VECTOR_TERMS:
                    # These come in 3-vectors (one per contact / per foot); the meaningful
                    # error is the length of each, not its largest component — a 10 mm
                    # error in every axis is a 17 mm miss, not a 10 mm one.
                    v = float(np.linalg.norm(terms[k].reshape(-1, 3), axis=1).max())
                else:
                    v = float(np.abs(terms[k]).max())
                out[k] = max(out[k], v)
        return out

    def feasible(self, z: np.ndarray, tol: Tolerances = Tolerances()) -> Tuple[bool, Dict]:
        rep = self.report(z)
        ok = all(rep[k] <= getattr(tol, k) for k in self.CONSTRAINT_TERMS)
        return ok, rep

    # ------------------------------------------------------------------ #
    # bounds / guess / sparsity
    # ------------------------------------------------------------------ #

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        scene = self.scene
        lo = np.empty(self.n_z)
        hi = np.empty(self.n_z)
        margin = np.deg2rad(2.0)
        for s, spec in enumerate(self.specs):
            sl = self.frame_slice(s)
            low = np.empty(NX)
            h = np.empty(NX)
            low[IDX_BASE_P] = scene.nominal["base_p"] - np.array([0.35, 0.35, 0.30])
            h[IDX_BASE_P] = scene.nominal["base_p"] + np.array([0.35, 0.35, 0.12])
            low[IDX_BASE_R] = np.array([-0.35, -0.55, -0.9])
            h[IDX_BASE_R] = np.array([0.35, 0.55, 0.9])
            low[IDX_QJ] = scene.limits.lower + margin
            h[IDX_QJ] = scene.limits.upper - margin
            pt = max(spec.chair_pos_tol, 1e-9)
            rt = max(spec.chair_rot_tol, 1e-9)
            low[IDX_OBJ_P], h[IDX_OBJ_P] = -pt, pt
            low[IDX_OBJ_R], h[IDX_OBJ_R] = -rt, rt
            if spec.fixed_config is not None:
                low[: IDX_QJ.stop] = spec.fixed_config - 1e-9
                h[: IDX_QJ.stop] = spec.fixed_config + 1e-9
            lo[sl], hi[sl] = low, h
        off = NX * self.n_frames
        for ep in self.episodes:
            if ep.uv_fixed is not None:
                continue
            half = np.asarray(self.patches[ep.patch].half_extents, dtype=float) * 0.9
            i = off + ep.uv_index
            lo[i : i + 2] = -half
            hi[i : i + 2] = half
        return lo, hi

    def initial_guess(self, warm: Optional[np.ndarray] = None) -> np.ndarray:
        if warm is not None:
            return warm.copy()
        z = np.zeros(self.n_z)
        n = self.scene.nominal
        for s in range(self.n_frames):
            x = np.zeros(NX)
            x[IDX_BASE_P] = n["base_p"]
            x[IDX_BASE_R] = n["base_rv"]
            x[IDX_QJ] = n["qj"]
            z[self.frame_slice(s)] = x
        return z

    def sparsity(self) -> lil_matrix:
        rows_per_frame = [
            sum(self._term_sizes[s][k] for k in self.TERM_ORDER) for s in range(self.n_frames)
        ]
        n_rows = sum(rows_per_frame)
        edge_rows = IDX_QJ.stop
        if self.couple:
            n_rows += edge_rows * (self.n_frames - 1)
        S = lil_matrix((n_rows, self.n_z), dtype=np.int8)
        off = 0
        uv_off = NX * self.n_frames
        for s in range(self.n_frames):
            r = slice(off, off + rows_per_frame[s])
            S[r, self.frame_slice(s)] = 1
            for side in HANDS:
                ep = self._ep_at.get((s, side))
                if ep is not None and ep.uv_fixed is None:
                    S[r, uv_off + ep.uv_index : uv_off + ep.uv_index + 2] = 1
            off += rows_per_frame[s]
        if self.couple:
            for s in range(self.n_frames - 1):
                r = slice(off, off + edge_rows)
                S[r, NX * s : NX * s + edge_rows] = 1
                S[r, NX * (s + 1) : NX * (s + 1) + edge_rows] = 1
                off += edge_rows
        return S


def solve(
    problem: KinematicProblem,
    warm: Optional[np.ndarray] = None,
    max_nfev: int = 4000,
    tol: Optional[Tolerances] = None,
    xtol: float = 1e-7,
    rounds: int = 3,
) -> SolveResult:
    """Solve with penalty continuation.

    Everything here is a soft residual, so a single weighted solve settles wherever the
    weights balance — and hand-tuning weights until every constraint happens to land inside
    its tolerance is not a method. Instead: solve, look at what is still violated, multiply
    *those* weights, and re-solve warm-started. A few rounds drive the true constraints to
    satisfaction without the regularizers having to be made negligible (which would leave
    the solution shapeless).
    """
    tol = tol if tol is not None else Tolerances()
    t_start = time.perf_counter()
    original = problem.w
    problem.w = Weights(**vars(original))  # escalate a copy, never the caller's weights
    result = _solve_once(problem, warm, max_nfev, tol, xtol)
    for _ in range(max(0, rounds - 1)):
        if result.success:
            break
        bumped = False
        for name in CONSTRAINT_TERMS:
            if result.report.get(name, 0.0) > getattr(tol, name):
                w = min(getattr(problem.w, name) * 4.0, 1e4)
                setattr(problem.w, name, w)
                bumped = True
        if not bumped:
            break
        problem._cache.clear()  # cached residuals are weighted
        result = _solve_once(problem, result.z, max_nfev, tol, xtol)
    problem.w = original
    problem._cache.clear()
    result.seconds = time.perf_counter() - t_start
    return result


def _solve_once(
    problem: KinematicProblem,
    warm: Optional[np.ndarray],
    max_nfev: int,
    tol: Tolerances,
    xtol: float,
) -> SolveResult:
    lo, hi = problem.bounds()
    z0 = np.clip(problem.initial_guess(warm), lo, hi)
    t0 = time.perf_counter()
    res = least_squares(
        problem.residuals,
        z0,
        jac_sparsity=problem.sparsity(),
        bounds=(lo, hi),
        method="trf",
        x_scale="jac",
        # MuJoCo's convex distance queries are only accurate to ~1e-6, so the default
        # sqrt(eps) finite-difference step would differentiate numerical noise.
        diff_step=1e-5,
        ftol=1e-7,
        xtol=xtol,
        gtol=1e-9,
        max_nfev=max_nfev,
    )
    ok, rep = problem.feasible(res.x, tol)
    return SolveResult(
        success=ok,
        z=res.x,
        report=rep,
        cost=float(res.cost),
        n_eval=int(res.nfev),
        seconds=time.perf_counter() - t0,
        message=str(res.message),
        tol=tol,
    )


# --------------------------------------------------------------------------- #
# FARO-style feasibility filters
# --------------------------------------------------------------------------- #


def _episodes_for_mode(scene: TrajOptScene, mode: ContactMode) -> List[Episode]:
    eps, idx = [], 0
    for side in HANDS:
        patch = mode.patch(side)
        if patch is None:
            continue
        eps.append(Episode(side=side, patch=patch, frames=[0], uv_index=idx))
        idx += 2
    return eps


def mode_problem(
    scene: TrajOptScene, mode: ContactMode, chair_pose: Pose,
    weights: Optional[Weights] = None,
) -> KinematicProblem:
    """Single-configuration problem for one contact mode (FARO eq. 14)."""
    spec = FrameSpec(mode=mode, chair_ref=chair_pose, posture_scale=1.0)
    return KinematicProblem(scene, [spec], _episodes_for_mode(scene, mode), weights)


def _filter_solve(
    problem: "KinematicProblem", warm_frame: Optional[np.ndarray], tol: Tolerances
) -> SolveResult:
    """Solve a mode/edge filter, retrying cold if the warm start led nowhere.

    Warm starting from the previous mode is what keeps the cascade fast, but it also drags
    the solver into whatever branch that configuration was in. A false "infeasible" is
    expensive here: the verdict is cached, so it rejects every later plan that reuses the
    mode. One cold retry costs a second and removes most of them — FARO reports the same
    failure mode (its few false negatives come from poor initialization).
    """
    res = solve(problem, warm=_warm(problem, warm_frame), max_nfev=FILTER_MAX_NFEV, tol=tol)
    if not res.success and warm_frame is not None:
        cold = solve(problem, warm=None, max_nfev=FILTER_MAX_NFEV, tol=tol)
        if cold.success:
            return cold
    return res


def _cached_warm(cache: "FeasibilityCache", mode: ContactMode, pose: Pose,
                 fallback: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Warm start for the next filter, taken from a cached solution when there is one."""
    cfg = cache.config(mode, pose)
    return fallback if cfg is None else cfg[:NX].copy()


def _uv_slot(mode: ContactMode, side: str) -> int:
    """Index of a hand's in-patch coordinates inside a single-mode problem's variables."""
    return sum(1 for s in HANDS if HANDS.index(s) < HANDS.index(side) and mode.patch(s))


def _warm(problem: "KinematicProblem", frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Seed a single-frame problem with a previously solved configuration."""
    if frame is None:
        return None
    z = problem.initial_guess()
    z[:NX] = frame
    z[IDX_OBJ_P] = 0.0  # object slack is relative to *this* problem's reference pose
    z[IDX_OBJ_R] = 0.0
    return z


class FeasibilityCache:
    """Feasible / infeasible cache, keyed by mode identity and chair pose (FARO §III).

    Stores the solved configuration alongside the verdict, not just the verdict: a feasible
    single-frame answer is the natural starting point for the sequence solve that follows.
    """

    def __init__(self):
        self.entries: Dict[Tuple, bool] = {}
        self.configs: Dict[Tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(mode: ContactMode, pose: Pose) -> Tuple:
        p = tuple(np.round(pose.p, 3))
        r = tuple(np.round(pose.quat_wxyz(), 3))
        return (mode.key, p, r)

    def get(self, mode: ContactMode, pose: Pose) -> Optional[bool]:
        v = self.entries.get(self.key(mode, pose))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, mode: ContactMode, pose: Pose, ok: bool,
            config: Optional[np.ndarray] = None) -> None:
        k = self.key(mode, pose)
        self.entries[k] = ok
        if ok and config is not None:
            self.configs[k] = np.asarray(config).copy()

    def config(self, mode: ContactMode, pose: Pose) -> Optional[np.ndarray]:
        return self.configs.get(self.key(mode, pose))


@dataclass
class PlanEvaluation:
    plan: ContactPlan
    passed: bool
    stage: str  # which filter it reached / failed at
    detail: str
    kso: Optional[SolveResult] = None
    episodes: List[Episode] = field(default_factory=list)
    problem: Optional["KinematicProblem"] = None
    timings: Dict[str, float] = field(default_factory=dict)


def evaluate_plan(
    scene: TrajOptScene,
    plan: ContactPlan,
    weights: Optional[Weights] = None,
    tol: Optional[Tolerances] = None,
    cache: Optional[FeasibilityCache] = None,
    chair_pos_tol: float = 0.03,
    chair_rot_tol: float = 0.10,
    verbose: bool = True,
) -> PlanEvaluation:
    """Run the filter cascade M -> E -> KSO on one contact plan.

    Filters are ordered cheapest-first and evaluation stops at the first failure, which is
    the whole point of FARO: a plan that no single configuration can satisfy never reaches
    the sequence solve.
    """
    cache = cache if cache is not None else FeasibilityCache()
    tol = tol if tol is not None else Tolerances()
    timings: Dict[str, float] = {}
    # Single-frame filters are warm-started from the previous solved configuration: the
    # chair only moves a little between stages, so this cuts the filter cost several-fold.
    warm_frame: Optional[np.ndarray] = None

    # --- M: every mode must admit at least one configuration ------------------
    t0 = time.perf_counter()
    for i, mode in enumerate(plan.modes):
        # Halfway through the stage the mode governs: evaluating "lift" at the keyframe
        # where the chair is still on the floor would contradict the mode itself.
        pose = interp_pose(plan.frames[i].chair_pose, plan.frames[i + 1].chair_pose, 0.5)
        cached = cache.get(mode, pose)
        if cached is False:
            timings["mode"] = time.perf_counter() - t0
            return PlanEvaluation(plan, False, "mode", f"mode '{mode.name}' infeasible (cached)",
                                  timings=timings)
        if cached is True:
            warm_frame = _cached_warm(cache, mode, pose, warm_frame)
            continue
        prob = mode_problem(scene, mode, pose, weights)
        res = _filter_solve(prob, warm_frame, tol)
        cache.put(mode, pose, res.success, res.z)
        if res.success:
            warm_frame = res.z[:NX].copy()
        if not res.success:
            timings["mode"] = time.perf_counter() - t0
            k, v = res.worst()
            return PlanEvaluation(plan, False, "mode",
                                  f"mode '{mode.name}' infeasible ({k}={v:.3f})", timings=timings)
    timings["mode"] = time.perf_counter() - t0

    # --- E: every transition (union of adjacent modes) must admit one too -----
    t0 = time.perf_counter()
    for s, frame in enumerate(plan.frames):
        cached = cache.get(frame.mode, frame.chair_pose)
        if cached is False:
            timings["edge"] = time.perf_counter() - t0
            return PlanEvaluation(plan, False, "edge", f"transition {s} infeasible (cached)",
                                  timings=timings)
        if cached is True:
            warm_frame = _cached_warm(cache, frame.mode, frame.chair_pose, warm_frame)
            continue
        prob = mode_problem(scene, frame.mode, frame.chair_pose, weights)
        res = _filter_solve(prob, warm_frame, tol)
        cache.put(frame.mode, frame.chair_pose, res.success, res.z)
        if res.success:
            warm_frame = res.z[:NX].copy()
        if not res.success:
            timings["edge"] = time.perf_counter() - t0
            k, v = res.worst()
            return PlanEvaluation(plan, False, "edge",
                                  f"transition {s} ('{frame.mode.name}') infeasible "
                                  f"({k}={v:.3f})", timings=timings)
    timings["edge"] = time.perf_counter() - t0

    # --- KSO: the whole sequence, with shared in-patch contact locations ------
    episodes = [
        Episode(side=e["side"], patch=e["patch"], frames=e["frames"], uv_index=2 * i)
        for i, e in enumerate(contact_episodes(plan))
    ]
    specs = []
    for frame in plan.frames:
        free_hands = sum(frame.mode.patch(s) is None for s in HANDS)
        specs.append(
            FrameSpec(
                mode=frame.mode,
                chair_ref=frame.chair_pose,
                chair_pos_tol=0.0 if frame.chair_fixed else chair_pos_tol,
                chair_rot_tol=0.0 if frame.chair_fixed else chair_rot_tol,
                # With both hands free the arms should actually return to a natural pose.
                posture_scale=1.0 + 2.0 * free_hands,
            )
        )
    problem = KinematicProblem(scene, specs, episodes, weights)
    # Seed every keyframe with the configuration its own edge filter already found. Without
    # this the sequence solve starts from the nominal stance at every frame and can settle
    # into a different branch of the arm's null space than the one that actually works —
    # the smoothness coupling then keeps it there, and a keyframe the edge filter proved
    # feasible comes back infeasible.
    guess = problem.initial_guess()
    for s, frame in enumerate(plan.frames):
        cfg = cache.config(frame.mode, frame.chair_pose)
        if cfg is not None:
            guess[problem.frame_slice(s)][: IDX_QJ.stop] = cfg[: IDX_QJ.stop]
            for side in HANDS:
                ep = problem._ep_at.get((s, side))
                if ep is not None and ep.uv_fixed is None and s == ep.frames[0]:
                    off = NX * problem.n_frames + ep.uv_index
                    src = NX + 2 * _uv_slot(frame.mode, side)
                    guess[off : off + 2] = cfg[src : src + 2]
    t0 = time.perf_counter()
    res = solve(problem, warm=guess, max_nfev=6000, tol=tol)
    timings["kso"] = time.perf_counter() - t0
    if not res.success:
        k, v = res.worst()
        return PlanEvaluation(plan, False, "kso", f"KSO infeasible ({k}={v:.3f})",
                              kso=res, episodes=episodes, problem=problem, timings=timings)
    return PlanEvaluation(plan, True, "kso", "feasible", kso=res, episodes=episodes,
                          problem=problem, timings=timings)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def _convex_edges(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Outward edge normals + offsets of the convex hull of 2D ``points``."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    verts = points[hull.vertices]
    nxt = np.roll(verts, -1, axis=0)
    edge = nxt - verts
    normals = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.einsum("ij,ij->i", normals, verts)
    # Orient outward (the centroid must be strictly inside).
    c = verts.mean(0)
    flip = (normals @ c - offsets) > 0
    normals[flip] *= -1
    offsets[flip] *= -1
    return normals, offsets


def _outside(edges: Tuple[np.ndarray, np.ndarray], p: np.ndarray, margin: float) -> float:
    """How far ``p`` is outside the polygon shrunk by ``margin`` (0 if safely inside)."""
    normals, offsets = edges
    return float(max(0.0, np.max(normals @ p - offsets + margin)))


def _segment_excess(pts: np.ndarray, p: np.ndarray, margin: float) -> float:
    """Distance from ``p`` to the segment/point spanned by ``pts``, shrunk by ``margin``."""
    if len(pts) == 1:
        return float(max(0.0, np.linalg.norm(p - pts[0]) - margin))
    a, b = pts[0], pts[-1]
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(max(0.0, np.linalg.norm(p - (a + t * ab)) - margin))


def unpack_keyframes(problem: KinematicProblem, z: np.ndarray) -> List[Dict]:
    """Per-keyframe dict of base pose, joints, object pose and in-patch coordinates."""
    out = []
    for s in range(problem.n_frames):
        x = z[problem.frame_slice(s)]
        uv = {
            side: np.asarray(problem.uv_of(z, problem._ep_at[(s, side)]))
            for side in HANDS
            if (s, side) in problem._ep_at
        }
        out.append(
            {
                "base_p": x[IDX_BASE_P].copy(),
                "base_rv": x[IDX_BASE_R].copy(),
                "qj": x[IDX_QJ].copy(),
                "object": problem.object_pose(s, x),
                "uv": uv,
                "mode": problem.specs[s].mode,
            }
        )
    return out


assert len(_POSTURE_W) == len(BODY_JOINT_NAMES)
