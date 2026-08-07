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

from dataclasses import dataclass, field, replace
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from gear_sonic.trajopt.chair import ContactPatch
from gear_sonic.trajopt.plans import (
    FEET,
    HANDS,
    ContactMode,
    ContactPlan,
    contact_episodes,
    frame_support,
    stance_episodes,
)
from gear_sonic.trajopt.scene import BODY_JOINT_NAMES, FOOT_BODY, TrajOptScene
from gear_sonic.trajopt.se3 import Pose, interp_pose, rot_error, rot_exp

NX = 41  # base pos 3 + base rotvec 3 + body joints 29 + object dp 3 + object drotvec 3
IDX_BASE_P = slice(0, 3)
IDX_BASE_R = slice(3, 6)
IDX_QJ = slice(6, 35)
IDX_OBJ_P = slice(35, 38)
IDX_OBJ_R = slice(38, 41)
N_LEG_JOINTS = 12  # the first 12 body joints are the two legs
IDX_UPPER = slice(IDX_QJ.start + N_LEG_JOINTS, IDX_QJ.stop)  # waist + both arms


@dataclass
class Weights:
    """Residual weights (sqrt-scale: least_squares minimizes 0.5*||r||^2)."""

    contact_pos: float = 120.0
    contact_normal: float = 40.0
    contact_twist: float = 40.0
    grasp_align: float = 60.0
    feet: float = 200.0
    swing: float = 150.0
    chair_floor: float = 120.0
    balance: float = 200.0
    object_support: float = 25.0
    collision: float = 120.0
    hand_clearance: float = 120.0
    posture: float = 1.0
    base_reg: float = 2.0
    obj_reg: float = 8.0
    com_reg: float = 6.0
    swing_reg: float = 3.0
    step_reg: float = 1.5
    stance_reg: float = 3.0
    smooth: float = 2.0


@dataclass
class Tolerances:
    """Feasibility thresholds on the *unweighted* constraint violations."""

    contact_pos: float = 0.012  # m
    contact_normal: float = 0.20  # ~11.5 deg of palm/patch normal misalignment
    # Same units and meaning, about the patch normal: how far the hand's bearing on the
    # object may drift from the one its episode holds.
    contact_twist: float = 0.20
    # Hinged on GRASP_ALIGN_SIN, so a violation means the fingers would have to close
    # along the object's long side rather than wrap across it.
    grasp_align: float = 0.02
    # m and rad: how far a planted foot may sit from the placement its stance episode owns.
    # This is drift, not motion — but it is one norm over position and orientation of a
    # foot the solver is also asked to balance on, and tightening it much further starts
    # rejecting configurations for millimetres.
    feet: float = 0.008
    # m: a swing foot must clear the floor and stay flat enough to land on. Hinged on
    # SWING_CLEARANCE, so a violation here means the foot is dragging.
    swing: float = 0.010
    chair_floor: float = 0.010  # m
    # m the CoM may sit outside the *true* support polygon (see KinematicProblem.report:
    # the shaping margin is already subtracted). Not exactly zero only because the solver
    # converges to millimetres; the real gate on staying up through the motion is the ZMP
    # check in the dynamic pass.
    balance: float = 0.002
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
    "feet", "swing", "contact_pos", "contact_normal", "contact_twist", "grasp_align",
    "chair_floor", "collision", "hand_clearance", "balance",
)
ADVISORY_TERMS = CONSTRAINT_TERMS + ("object_support",)
# Terms whose residuals are 3-vectors, so violations are measured as vector norms.
_VECTOR_TERMS = frozenset({"feet", "contact_pos", "contact_normal", "contact_twist"})

FILTER_MAX_NFEV = 600  # a mode/edge filter that has not converged by here is not going to
COLLISION_MARGIN = 0.035  # m: keep this much clear of the chair and of itself
HAND_CLEARANCE_MARGIN = 0.015  # m: a free hand near the chair it is reaching for
# m: CoM must stay this far inside the support polygon. In double support the polygon is
# ~17 x 30 cm and 2.5 cm is a comfortable shaping target; a single foot is only 6 cm wide,
# so the same number would be unsatisfiable — a step is *supposed* to put the CoM near the
# edge of the stance foot, which is why quasi-static stepping is slow.
BALANCE_MARGIN = 0.025
BALANCE_MARGIN_SS = 0.012  # single support
CHAIR_CLEARANCE = 0.02  # m: minimum floor gap once the chair is airborne
GRASP_SPAN_MARGIN = 0.04  # m: chair CoM must stay this far inside the hand-to-hand span
SWING_CLEARANCE = 0.03  # m: how high a swinging foot must be off the floor
SWING_TILT = 0.25  # rad of roll/pitch a swinging foot may take (it has to land flat)
# sin of the angle a wrapped grasp's curl axis may make with the long side of its patch.
# 25 deg: enough freedom for the wrist to find a posture, not enough to let the fingers
# end up closing along the rail instead of around it.
GRASP_ALIGN_SIN = float(np.sin(np.deg2rad(25.0)))

# Where a step may put a foot, relative to the nominal stance placement (m, m, rad).
STEP_BOX_LOW = np.array([-0.30, -0.22, -0.6])
STEP_BOX_HIGH = np.array([0.45, 0.22, 0.6])
FOOT_HALF_GAP = 0.055  # m: |y| a stance placement must keep, so the legs cannot cross
# Base travel allowed around the nominal stance, standing still vs. free to step.
BASE_BOX_STATIC = (np.array([0.35, 0.35, 0.30]), np.array([0.35, 0.35, 0.12]))
BASE_BOX_STEPPING = (np.array([0.75, 0.55, 0.30]), np.array([0.75, 0.55, 0.12]))


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
    # Feet that may be balanced on here. ``None`` = every foot with a stance episode, which
    # is right for a single mode; a keyframe *between* two modes has to use the intersection
    # instead (see plans.frame_support).
    support_sides: Optional[Tuple[str, ...]] = None


@dataclass
class Episode:
    """A contact episode: one (hand, patch) pairing held over a set of frames.

    Its three variables are ``(u, v, theta)``: where on the patch the palm sits, and the
    bearing of the hand about the patch normal. All three are shared by every frame of the
    episode, so the hand can neither slide across the object nor *rotate* on it while it
    holds on — which a hand whose fingers are wrapped round something cannot do either.
    """

    side: str
    patch: str
    frames: List[int]
    var_index: int  # column offset of its three variables
    fixed: Optional[np.ndarray] = None


@dataclass
class StanceEpisode:
    """A foot planted at one place over a set of frames — the feet's version of ``Episode``.

    The placement ``(x, y, yaw)`` is a single decision variable triple shared by every frame
    of the episode, so a planted foot cannot slide, for the same structural reason a grasped
    patch point cannot: there is only one variable to slide. A step is then nothing but the
    boundary between two episodes of the same foot.
    """

    side: str
    frames: List[int]
    var_index: int  # column offset of its three placement variables
    fixed: Optional[np.ndarray] = None  # pinned placement (the stance the plan starts in)
    # Where this stance would ideally be — squared up to the object at a comfortable
    # working distance. Weakly weighted: the constraints decide what is *possible*, this
    # decides where inside that set the robot chooses to stand. Defaults to the spawn
    # stance, which is what a filter with nothing to square up to should keep.
    target: Optional[np.ndarray] = None


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
        stances: Sequence[StanceEpisode] = (),
    ):
        self.scene = scene
        self.specs = list(specs)
        self.episodes = list(episodes)
        self.stances = list(stances)
        # Copied, never aliased: solve() escalates these in place during continuation, and
        # a shared instance would carry the inflated penalties into every later problem.
        self.w = Weights(**vars(weights)) if weights is not None else Weights()
        self.couple = couple_frames and len(specs) > 1
        self.patches = scene.chair.patches()
        self.n_frames = len(self.specs)
        self.n_grasp = 3 * sum(1 for e in self.episodes if e.fixed is None)
        self.n_foot = 3 * sum(1 for e in self.stances if e.fixed is None)
        self.foot_offset = NX * self.n_frames + self.n_grasp
        self.n_z = self.foot_offset + self.n_foot

        # Episodes indexed by frame and side, for O(1) lookup in the inner loop.
        self._ep_at: Dict[Tuple[int, str], Episode] = {}
        for ep in self.episodes:
            for s in ep.frames:
                self._ep_at[(s, ep.side)] = ep
        self._stance_at: Dict[Tuple[int, str], StanceEpisode] = {}
        for ep in self.stances:
            for s in ep.frames:
                self._stance_at[(s, ep.side)] = ep
        # Where a swinging foot is headed: the stance episodes it sits between.
        self._swing_between: Dict[Tuple[int, str], Tuple[StanceEpisode, StanceEpisode]] = {}
        for side in FEET:
            eps = [e for e in self.stances if e.side == side]
            for s in range(self.n_frames):
                if (s, side) in self._stance_at or not eps:
                    continue
                before = [e for e in eps if e.frames[-1] < s] or [eps[0]]
                after = [e for e in eps if e.frames[0] > s] or [eps[-1]]
                self._swing_between[(s, side)] = (before[-1], after[0])

        # True unless both feet are planted, at the spawn stance, in every frame. Anything
        # else needs the base free to travel with the feet — including a stage path solved
        # *after* a step, whose placements are fixed but no longer nominal.
        self.stepping = any(
            ep is None
            or ep.fixed is None
            or not np.allclose(ep.fixed, scene.nominal_placement(side), atol=1e-6)
            for s in range(self.n_frames)
            for side in FEET
            for ep in [self._stance_at.get((s, side))]
        )
        self._steps = self._step_pairs()
        self._support_sides = [
            spec.support_sides
            if spec.support_sides is not None
            else tuple(s for s in FEET if (i, s) in self._stance_at)
            for i, spec in enumerate(self.specs)
        ]
        # How far inside its polygon the CoM is *asked* to stay. Single support is a much
        # smaller polygon (one 17 x 6 cm sole), so the same shaping target would be
        # unsatisfiable there; feasibility is judged against the true polygon either way.
        self._balance_margin = [
            BALANCE_MARGIN if len(sides) >= 2 else BALANCE_MARGIN_SS
            for sides in self._support_sides
        ]
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

    def grasp_of(self, z: np.ndarray, ep: Episode) -> np.ndarray:
        """The ``(u, v, theta)`` this contact episode holds."""
        if ep.fixed is not None:
            return ep.fixed
        base = NX * self.n_frames + ep.var_index
        return z[base : base + 3]

    def place_of(self, z: np.ndarray, ep: StanceEpisode) -> np.ndarray:
        """The ``(x, y, yaw)`` placement a stance episode holds."""
        if ep.fixed is not None:
            return ep.fixed
        base = self.foot_offset + ep.var_index
        return z[base : base + 3]

    def _stance_or_swing(self, s: int, side: str) -> Tuple[StanceEpisode, ...]:
        """The stance episodes frame ``s`` reads for ``side`` — one if planted, the two it
        is stepping between if swinging."""
        ep = self._stance_at.get((s, side))
        if ep is not None:
            return (ep,)
        return self._swing_between.get((s, side), ())

    def support(self, s: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(polygon, edges)`` of the frame's support: the soles of its planted feet.

        Built from the *commanded* placements rather than from the achieved foot poses.
        Those agree to within the foot tolerance, and taking the commanded ones makes the
        balance residual a direct function of the placement variables — which is what lets
        the solver move a foot to catch a centre of mass instead of only bending at the hip.
        """
        pts = [
            self.scene.sole_polygon(side, self.scene.foot_placement(side, self.place_of(z, ep)))
            for side in self._support_sides[s]
            for ep in [self._stance_at.get((s, side))]
            if ep is not None
        ]
        if not pts:  # no foot on the ground: not a configuration this generator produces
            pts = [self.scene.sole_polygon(side, self.scene.foot_nominal[side])
                   for side in FEET]
        poly = np.concatenate(pts)
        return poly, _convex_edges(poly)

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

        # Feet: a planted one sits on its episode's placement, a swinging one clears the
        # floor, stays flat enough to land on, and heads for where it is going.
        feet, swing, swing_reg = [], [], []
        for side in FEET:
            pose = scene.body_pose(FOOT_BODY[side])
            ep = self._stance_at.get((s, side))
            if ep is not None:
                target = scene.foot_placement(side, self.place_of(z, ep))
                feet.append(np.concatenate([pose.p - target.p, rot_error(pose.Rm, target.Rm)]))
                continue
            tilt = rot_error(pose.Rm, scene.foot_nominal[side].Rm)[:2]
            swing.append(max(0.0, SWING_CLEARANCE - scene.foot_height(side)))
            swing += list(np.maximum(0.0, np.abs(tilt) - SWING_TILT))
            between = self._swing_between.get((s, side))
            if between is not None:
                mid = 0.5 * sum(self.place_of(z, e)[:2] for e in between)
                swing_reg.append(pose.p[:2] - mid)
        out["feet"] = np.concatenate(feet) if feet else np.zeros(0)
        out["swing"] = np.asarray(swing)
        out["swing_reg"] = np.concatenate(swing_reg) if swing_reg else np.zeros(0)

        # Contacts. Position and normal pin five of the palm's six degrees of freedom; the
        # sixth — the bearing about the patch normal — is what ``theta`` fixes. Leaving it
        # free is not a harmless relaxation: the object then turns *under* a stationary
        # hand, which measured 108 deg of relative rotation inside a single grasp on the 90
        # deg task. Fingers wrapped round a rail cannot do that.
        cpos, cnrm, ctwist, calign = [], [], [], []
        for side in HANDS:
            patch_name = mode.patch(side)
            if patch_name is None:
                continue
            patch: ContactPatch = self.patches[patch_name]
            u, v, theta = self.grasp_of(z, self._ep_at[(s, side)])
            palm = scene.palm_pose(side)
            cpos.append(palm.p - obj.apply(patch.point(np.array([u, v]))))
            cnrm.append(palm.Rm[:, 2] + obj.rotate(patch.normal))
            # The hand's own curl axis, held at a constant bearing on the patch.
            bearing = patch.tangent * np.cos(theta) + patch.bitangent * np.sin(theta)
            ctwist.append(palm.Rm[:, 0] - obj.rotate(bearing))
            if patch.kind == "grasp":
                # ...and that bearing has to be one the fingers can actually close on: the
                # curl axis along the patch's long side, so they wrap the short one. Either
                # sign will do — that is only which way round the hand is.
                off = np.sin(theta) if patch.half_extents[0] >= patch.half_extents[1] \
                    else np.cos(theta)
                calign.append(max(0.0, abs(float(off)) - GRASP_ALIGN_SIN))
        out["contact_pos"] = np.concatenate(cpos) if cpos else np.zeros(0)
        out["contact_normal"] = np.concatenate(cnrm) if cnrm else np.zeros(0)
        out["contact_twist"] = np.concatenate(ctwist) if ctwist else np.zeros(0)
        out["grasp_align"] = np.asarray(calign)

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
        poly, edges = self.support(s, z)
        out["balance"] = np.array([_outside(edges, com[:2], self._balance_margin[s])])

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

        # Regularization toward the ready stance and the plan's object pose. The base is
        # pulled over its own feet rather than to a fixed spot, so walking forward costs
        # nothing while leaning out over the toes still does.
        center = poly.mean(0)
        out["posture"] = _POSTURE_W * spec.posture_scale * (x[IDX_QJ] - scene.nominal["qj"])
        out["base_reg"] = np.concatenate(
            [
                x[IDX_BASE_P][:2] - center,
                [x[IDX_BASE_P][2] - scene.nominal["base_p"][2]],
                x[IDX_BASE_R] - scene.nominal["base_rv"],
            ]
        )
        out["obj_reg"] = np.concatenate([x[IDX_OBJ_P], x[IDX_OBJ_R]])
        # Weak pull of the CoM toward the middle of the support. The balance hinge above is
        # flat wherever it is satisfied, so on its own it gives the solver nothing to
        # follow until the configuration is already unstable; this term is what actually
        # makes the robot sit back as it reaches forward.
        out["com_reg"] = com[:2] - center
        return out

    TERM_ORDER = (
        "feet", "swing", "contact_pos", "contact_normal", "contact_twist", "grasp_align",
        "chair_floor", "balance",
        "object_support", "collision", "hand_clearance", "posture", "base_reg", "obj_reg",
        "com_reg", "swing_reg",
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
            np.asarray(self.grasp_of(z, self._ep_at[(s, side)])).tobytes()
            for side in HANDS
            if (s, side) in self._ep_at
        ) + b"|" + b"".join(
            np.asarray(self.place_of(z, ep)).tobytes()
            for side in FEET
            for ep in self._stance_or_swing(s, side)
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

    def _step_pairs(self) -> List[Tuple[StanceEpisode, StanceEpisode]]:
        """Consecutive stance episodes of the same foot — i.e. the plan's actual steps."""
        out = []
        for side in FEET:
            eps = sorted((e for e in self.stances if e.side == side),
                         key=lambda e: e.frames[0])
            out += list(zip(eps, eps[1:]))
        return out

    def residuals(self, z: np.ndarray) -> np.ndarray:
        self.n_eval += 1
        parts = [self._frame_vector(s, z[self.frame_slice(s)], z) for s in range(self.n_frames)]
        if self.couple:
            for s in range(self.n_frames - 1):
                a = z[self.frame_slice(s)]
                b = z[self.frame_slice(s + 1)]
                parts.append(self.w.smooth * (b[: IDX_QJ.stop] - a[: IDX_QJ.stop]))
        # Keep steps short unless something is paying for them. Without these the placements
        # are underdetermined wherever the constraints do not bite — with the base and the
        # CoM both regularized *toward the feet*, a mode with no hand contact can translate
        # the whole robot for free — and the solver wanders off into that null direction.
        for a, b in self._steps:
            parts.append(self.w.step_reg * (self.place_of(z, b) - self.place_of(z, a)))
        for ep in self.stances:
            if ep.fixed is None:
                goal = (
                    ep.target if ep.target is not None
                    else self.scene.nominal_placement(ep.side)
                )
                parts.append(self.w.stance_reg * (self.place_of(z, ep) - goal))
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
                elif k == "balance":
                    # The residual is hinged on a shaping margin that differs between single
                    # and double support; what decides feasibility is the only thing that
                    # means the same in both — how far outside the real polygon the CoM is.
                    v = max(0.0, float(terms[k].max()) - self._balance_margin[s])
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
        base_box = BASE_BOX_STEPPING if self.stepping else BASE_BOX_STATIC
        for s, spec in enumerate(self.specs):
            sl = self.frame_slice(s)
            low = np.empty(NX)
            h = np.empty(NX)
            low[IDX_BASE_P] = scene.nominal["base_p"] - base_box[0]
            h[IDX_BASE_P] = scene.nominal["base_p"] + base_box[1]
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
            if ep.fixed is not None:
                continue
            half = np.asarray(self.patches[ep.patch].half_extents, dtype=float) * 0.9
            i = off + ep.var_index
            lo[i : i + 2] = -half
            hi[i : i + 2] = half
            lo[i + 2], hi[i + 2] = -np.pi, np.pi
        for ep in self.stances:
            if ep.fixed is not None:
                continue
            nom = scene.nominal_placement(ep.side)
            i = self.foot_offset + ep.var_index
            lo[i : i + 3] = nom + STEP_BOX_LOW
            hi[i : i + 3] = nom + STEP_BOX_HIGH
            # The legs cannot cross: each foot keeps to its own side of the pelvis.
            if ep.side == "left":
                lo[i + 1] = max(lo[i + 1], FOOT_HALF_GAP)
            else:
                hi[i + 1] = min(hi[i + 1], -FOOT_HALF_GAP)
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
        for ep in self.stances:
            if ep.fixed is None:
                i = self.foot_offset + ep.var_index
                z[i : i + 3] = self.scene.nominal_placement(ep.side)
        return z

    def sparsity(self) -> lil_matrix:
        rows_per_frame = [
            sum(self._term_sizes[s][k] for k in self.TERM_ORDER) for s in range(self.n_frames)
        ]
        n_rows = sum(rows_per_frame)
        edge_rows = IDX_QJ.stop
        if self.couple:
            n_rows += edge_rows * (self.n_frames - 1)
        n_rows += 3 * (len(self._steps) + self.n_foot // 3)
        S = lil_matrix((n_rows, self.n_z), dtype=np.int8)
        off = 0
        uv_off = NX * self.n_frames
        for s in range(self.n_frames):
            r = slice(off, off + rows_per_frame[s])
            S[r, self.frame_slice(s)] = 1
            for side in HANDS:
                ep = self._ep_at.get((s, side))
                if ep is not None and ep.fixed is None:
                    S[r, uv_off + ep.var_index : uv_off + ep.var_index + 3] = 1
                for st in self._stance_or_swing(s, side):
                    if st.fixed is None:
                        j = self.foot_offset + st.var_index
                        S[r, j : j + 3] = 1
            off += rows_per_frame[s]
        if self.couple:
            for s in range(self.n_frames - 1):
                r = slice(off, off + edge_rows)
                S[r, NX * s : NX * s + edge_rows] = 1
                S[r, NX * (s + 1) : NX * (s + 1) + edge_rows] = 1
                off += edge_rows
        for a, b in self._steps:
            r = slice(off, off + 3)
            for ep in (a, b):
                if ep.fixed is None:
                    j = self.foot_offset + ep.var_index
                    S[r, j : j + 3] = 1
            off += 3
        for ep in self.stances:
            if ep.fixed is None:
                j = self.foot_offset + ep.var_index
                S[off : off + 3, j : j + 3] = 1
                off += 3
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
        eps.append(Episode(side=side, patch=patch, frames=[0], var_index=idx))
        idx += 3
    return eps


def _stances_for_mode(
    scene: TrajOptScene, mode: ContactMode, free: bool
) -> List[StanceEpisode]:
    """Stance episodes of a single-configuration problem.

    ``free`` asks the filter the question a stepping plan actually poses — *is there any
    stance from which this mode works* — instead of only testing the spawn stance.
    """
    out, idx = [], 0
    for side in FEET:
        if not mode.planted(side):
            continue
        fixed = None if free else scene.nominal_placement(side)
        out.append(StanceEpisode(side=side, frames=[0], var_index=idx, fixed=fixed))
        if fixed is None:
            idx += 3
    return out


def _grasp_center(
    scene: TrajOptScene, mode: ContactMode, chair: Pose
) -> Optional[np.ndarray]:
    """World midpoint of the patches this mode has hold of (``None`` if the hands are free)."""
    patches = scene.chair.patches()
    pts = [
        chair.apply(patches[mode.patch(side)].center)
        for side in HANDS
        if mode.patch(side) is not None
    ]
    return None if not pts else np.mean(pts, axis=0)


def _step_onto_targets(
    scene: TrajOptScene, problem: KinematicProblem, guess: np.ndarray
) -> None:
    """Move a standing-still seed onto the stance the plan would like, in place.

    Each free placement is set to its target and the frames it supports are translated by
    the same amount, so the seed stays a *consistent* configuration — the same posture,
    carried forward on the feet. Starting the solve there and letting the constraints pull
    the feet back to whatever is actually reachable is far more robust than starting from a
    converged standing solution and hoping a soft preference can walk it out of that
    minimum: the feet are shared across many frames, so the pull on any one of them is
    weak, and the solve terminates before it moves.
    """
    delta = np.zeros((problem.n_frames, 2))
    count = np.zeros(problem.n_frames)
    for ep in problem.stances:
        if ep.fixed is not None or ep.target is None:
            continue
        i = problem.foot_offset + ep.var_index
        guess[i : i + 3] = ep.target
        shift = ep.target[:2] - scene.nominal_placement(ep.side)[:2]
        for s in ep.frames:
            delta[s] += shift
            count[s] += 1
    for s in range(problem.n_frames):
        planted = sum((s, side) in problem._stance_at for side in FEET)
        if count[s] == 0 or planted == 0:
            continue
        # Feet that stay put contribute no shift, which is what averaging over *all* of
        # them (not just the moving ones) expresses: mid-step the base is between stances.
        guess[problem.frame_slice(s)][IDX_BASE_P][:2] += delta[s] / planted


def stance_target(
    scene: TrajOptScene, side: str, chair: Pose, aim: Optional[np.ndarray] = None
) -> np.ndarray:
    """Where a foot would ideally be planted to work on the chair at ``chair``.

    Squared up to the object — feet abreast, both facing it — just outside the envelope it
    sweeps as it turns. This is the difference between a robot that steps because a
    constraint forced it to and one that walks up to what it is about to work on: the
    constraints only say what is *reachable*, and a chair 0.55 m away already is. Kept weak
    (``Weights.stance_reg``) so it never overrides feasibility.

    ``aim`` is the world point the robot should *face* — the middle of whatever its hands
    are holding. It sets the stance yaw only, not where the stance is: turning to square the
    shoulders onto a grasp that has swung round is a small motion that directly buys reach,
    whereas re-deriving the position from the grasp too would shuffle the robot a fifth of a
    metre sideways every time the chair turned, which is a much bigger claim than the weak
    preference this is meant to be.
    """
    # As close as the robot's own knees allow it to stand to the envelope the chair sweeps.
    # Both halves are measured, not chosen: a wider chair has to be stood further back from,
    # and a deeper crouch needs more room in front of it.
    standoff = scene.chair.sweep_radius + scene.body_reach
    nominal = {s: scene.nominal_placement(s) for s in FEET}
    home = np.mean([nominal[s][:2] for s in FEET], axis=0)
    to_chair = chair.p[:2] - home
    dist = float(np.linalg.norm(to_chair))
    if dist < 1e-6:
        return nominal[side]
    center = chair.p[:2] - (to_chair / dist) * standoff
    face = chair.p[:2] if aim is None else np.asarray(aim, dtype=float)[:2]
    yaw = float(np.arctan2(*(face - center)[::-1]))
    offset = nominal[side][:2] - home  # this foot's place in the stance, unrotated
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        center[0] + cy * offset[0] - sy * offset[1],
        center[1] + sy * offset[0] + cy * offset[1],
        yaw,
    ])


def mode_problem(
    scene: TrajOptScene, mode: ContactMode, chair_pose: Pose,
    weights: Optional[Weights] = None, free_stance: bool = False,
) -> KinematicProblem:
    """Single-configuration problem for one contact mode (FARO eq. 14)."""
    spec = FrameSpec(mode=mode, chair_ref=chair_pose, posture_scale=1.0)
    return KinematicProblem(
        scene, [spec], _episodes_for_mode(scene, mode), weights,
        stances=_stances_for_mode(scene, mode, free_stance),
    )


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
                 fallback: Optional[np.ndarray], free_stance: bool = False
                 ) -> Optional[np.ndarray]:
    """Warm start for the next filter, taken from a cached solution when there is one."""
    cfg = cache.config(mode, pose, free_stance)
    return fallback if cfg is None else cfg[:NX].copy()


def _grasp_slot(mode: ContactMode, side: str) -> int:
    """Index of a hand's contact variables inside a single-mode problem's variables."""
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
    def key(mode: ContactMode, pose: Pose, free_stance: bool = False) -> Tuple:
        p = tuple(np.round(pose.p, 3))
        r = tuple(np.round(pose.quat_wxyz(), 3))
        # Whether the feet may be placed freely is part of the question being asked: the
        # same mode can be infeasible from the spawn stance and fine from a step away.
        return (mode.key, p, r, free_stance)

    def get(self, mode: ContactMode, pose: Pose, free_stance: bool = False) -> Optional[bool]:
        v = self.entries.get(self.key(mode, pose, free_stance))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, mode: ContactMode, pose: Pose, ok: bool,
            config: Optional[np.ndarray] = None, free_stance: bool = False) -> None:
        k = self.key(mode, pose, free_stance)
        self.entries[k] = ok
        if ok and config is not None:
            self.configs[k] = np.asarray(config).copy()

    def config(self, mode: ContactMode, pose: Pose,
               free_stance: bool = False) -> Optional[np.ndarray]:
        return self.configs.get(self.key(mode, pose, free_stance))


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
    # A plan that steps gets to choose its stance, so its filters must be allowed to as
    # well — otherwise every mode is tested from a spawn stance the plan never uses.
    steps = any(not m.planted(f) for m in plan.modes for f in FEET)

    # --- M: every mode must admit at least one configuration ------------------
    t0 = time.perf_counter()
    for i, mode in enumerate(plan.modes):
        # Halfway through the stage the mode governs: evaluating "lift" at the keyframe
        # where the chair is still on the floor would contradict the mode itself.
        pose = interp_pose(plan.frames[i].chair_pose, plan.frames[i + 1].chair_pose, 0.5)
        cached = cache.get(mode, pose, steps)
        if cached is False:
            timings["mode"] = time.perf_counter() - t0
            return PlanEvaluation(plan, False, "mode", f"mode '{mode.name}' infeasible (cached)",
                                  timings=timings)
        if cached is True:
            warm_frame = _cached_warm(cache, mode, pose, warm_frame, steps)
            continue
        prob = mode_problem(scene, mode, pose, weights, free_stance=steps)
        res = _filter_solve(prob, warm_frame, tol)
        cache.put(mode, pose, res.success, res.z, steps)
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
        cached = cache.get(frame.mode, frame.chair_pose, steps)
        if cached is False:
            timings["edge"] = time.perf_counter() - t0
            return PlanEvaluation(plan, False, "edge", f"transition {s} infeasible (cached)",
                                  timings=timings)
        if cached is True:
            warm_frame = _cached_warm(cache, frame.mode, frame.chair_pose, warm_frame, steps)
            continue
        prob = mode_problem(scene, frame.mode, frame.chair_pose, weights, free_stance=steps)
        res = _filter_solve(prob, warm_frame, tol)
        cache.put(frame.mode, frame.chair_pose, res.success, res.z, steps)
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
        Episode(side=e["side"], patch=e["patch"], frames=e["frames"], var_index=3 * i)
        for i, e in enumerate(contact_episodes(plan))
    ]
    # The first stance of each foot is where the robot already is — that is the sim's own
    # spawn pose, and a reference that starts anywhere else cannot be played back. Every
    # stance after a step is a decision variable.
    stances: List[StanceEpisode] = []
    seen, idx = set(), 0
    for e in stance_episodes(plan):
        first = e["side"] not in seen
        seen.add(e["side"])
        fixed = scene.nominal_placement(e["side"]) if first else None
        # Square up to whatever this stance has to work on, halfway through it: the chair
        # where it will be, and the faces the hands will be holding there.
        mid = plan.frames[e["frames"][len(e["frames"]) // 2]]
        stances.append(
            StanceEpisode(
                side=e["side"], frames=e["frames"], var_index=idx, fixed=fixed,
                target=stance_target(
                    scene, e["side"], mid.chair_pose,
                    aim=_grasp_center(scene, mid.mode, mid.chair_pose),
                ),
            )
        )
        if fixed is None:
            idx += 3
    specs = []
    support = frame_support(plan)
    for s, frame in enumerate(plan.frames):
        free_hands = sum(frame.mode.patch(h) is None for h in HANDS)
        specs.append(
            FrameSpec(
                mode=frame.mode,
                chair_ref=frame.chair_pose,
                chair_pos_tol=0.0 if frame.chair_fixed else chair_pos_tol,
                chair_rot_tol=0.0 if frame.chair_fixed else chair_rot_tol,
                # With both hands free the arms should actually return to a natural pose.
                posture_scale=1.0 + 2.0 * free_hands,
                support_sides=support[s],
            )
        )
    problem = KinematicProblem(scene, specs, episodes, weights, stances=stances)
    # Seed every keyframe with the configuration its own edge filter already found. Without
    # this the sequence solve starts from the nominal stance at every frame and can settle
    # into a different branch of the arm's null space than the one that actually works —
    # the smoothness coupling then keeps it there, and a keyframe the edge filter proved
    # feasible comes back infeasible.
    guess = problem.initial_guess()
    for s, frame in enumerate(plan.frames):
        cfg = cache.config(frame.mode, frame.chair_pose, steps)
        if cfg is None:
            continue
        if steps:
            # Upper body only. Each single-frame filter picked its own stance, and the base
            # and legs that go with it contradict the *shared* placement of the episode
            # spanning this frame — transplanting them lands the sequence solve in a
            # compromise local minimum that satisfies neither the feet nor the contacts.
            # The arms are what the filter actually has to say, and they seed cleanly.
            guess[problem.frame_slice(s)][IDX_UPPER] = cfg[IDX_UPPER]
        else:
            guess[problem.frame_slice(s)][: IDX_QJ.stop] = cfg[: IDX_QJ.stop]
        for side in HANDS:
            ep = problem._ep_at.get((s, side))
            if ep is not None and ep.fixed is None and s == ep.frames[0]:
                off = NX * problem.n_frames + ep.var_index
                src = NX + 3 * _grasp_slot(frame.mode, side)
                guess[off : off + 3] = cfg[src : src + 3]
    t0 = time.perf_counter()
    if not steps:
        res = solve(problem, warm=guess, max_nfev=6000, tol=tol)
    else:
        # Continuation on the stance. The single-frame filters each answered from a stance
        # of their own choosing, so their configurations do not agree about where the robot
        # is standing — and a sequence solve started from that disagreement settles into a
        # compromise that satisfies neither the contacts nor the feet. Solving the same
        # sequence *standing still* first costs one extra solve and produces a seed whose
        # arms and whose feet tell the same story; stepping onto the target stance from
        # there is then a correction rather than a search.
        pinned = [
            replace(ep, fixed=scene.nominal_placement(ep.side), var_index=0)
            for ep in stances
        ]
        stand = KinematicProblem(scene, specs, episodes, weights, stances=pinned)

        def continuation(seed: np.ndarray) -> SolveResult:
            pre = solve(stand, warm=seed[: stand.n_z], max_nfev=6000, tol=tol)
            seed[: problem.foot_offset] = pre.z
            _step_onto_targets(scene, problem, seed)
            return solve(problem, warm=seed, max_nfev=6000, tol=tol)

        res = continuation(guess)
        if not res.success:
            # The filter seeds reach this plan through a cache shared with every candidate
            # tried before it, so which minimum the sequence lands in depends on what ran
            # earlier — a plan can pass in isolation and fail in a batch. One retry from the
            # nominal stance removes that dependence, and costs nothing when the first
            # attempt works.
            cold = continuation(problem.initial_guess())
            if cold.success or cold.cost < res.cost:
                res = cold
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


def _hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D ``points``, counter-clockwise (Andrew's monotone chain).

    scipy's ConvexHull is a Qhull call per invocation; this runs on every residual
    evaluation of a stepping problem, where the input is eight foot-sole corners.
    """
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def chain(seq: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 1e-12:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(pts), chain(pts[::-1])
    hull = np.asarray(lower[:-1] + upper[:-1])
    return hull if len(hull) >= 3 else pts


def _convex_edges(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Outward edge normals + offsets of the convex hull of 2D ``points``."""
    verts = _hull2d(np.asarray(points, dtype=float))
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
        grasp = {
            side: np.asarray(problem.grasp_of(z, problem._ep_at[(s, side)])).copy()
            for side in HANDS
            if (s, side) in problem._ep_at
        }
        place = {
            side: np.asarray(problem.place_of(z, problem._stance_at[(s, side)])).copy()
            for side in HANDS
            if (s, side) in problem._stance_at
        }
        out.append(
            {
                "base_p": x[IDX_BASE_P].copy(),
                "base_rv": x[IDX_BASE_R].copy(),
                "qj": x[IDX_QJ].copy(),
                "object": problem.object_pose(s, x),
                "grasp": grasp,
                "stance": place,
                "mode": problem.specs[s].mode,
            }
        )
    return out


assert len(_POSTURE_W) == len(BODY_JOINT_NAMES)
