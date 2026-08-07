"""Contact-mode sequences (FARO §II-A) and the hardcoded plan library for the chair task.

A *contact mode* assigns every interface a state: each palm takes a chair patch or "free",
each foot is planted or swinging, and the chair itself is on the floor or not. A *plan* is a
sequence of K modes plus the K+1 keyframe configurations between them, each keyframe
carrying the chair pose the plan wants there (FARO's kinematic sequence optimization solves
one configuration per transition, §II-E). Walking and stepping are therefore not a separate
mechanism — they are what the mode sequence says about the feet.

FARO discovers mode sequences with a feasibility-guided tree search. This module instead
enumerates a small, task-specific candidate set — which patch pair to pinch, whether to
lift at all — and hands it to the same filter cascade (``kso.py``), which keeps the part
that carries the practical value: infeasible plans are rejected by cheap kinematic tests
long before anything expensive runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gear_sonic.trajopt.chair import ChairSpec
from gear_sonic.trajopt.se3 import Pose, interp_pose

HANDS: Tuple[str, str] = ("left", "right")
FEET: Tuple[str, str] = ("left", "right")

# Opposing patch pairs a two-hand pinch can use, as (patch_for_+y_side, patch_for_-y_side)
# in *chair* coordinates. Which one goes to which hand is decided per plan from the chair's
# world orientation at grasp time.
PINCH_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("seat_py", "seat_ny"),
    ("seat_px", "seat_nx"),
    ("back_py", "back_ny"),
)


@dataclass(frozen=True)
class ContactMode:
    """One contact mode: an interface assignment for each palm *and* each foot.

    The palms take a chair patch or ``None`` (free); the feet are either planted on the
    floor or swinging. ``grounded`` records whether the chair itself rests on the floor.
    """

    name: str
    left: Optional[str]
    right: Optional[str]
    grounded: bool
    closure: float = 0.0  # finger closure during this mode, in [0, 1]
    stance: Tuple[bool, bool] = (True, True)  # (left foot, right foot) planted

    def patch(self, side: str) -> Optional[str]:
        return self.left if side == "left" else self.right

    def planted(self, side: str) -> bool:
        return self.stance[0] if side == "left" else self.stance[1]

    @property
    def key(self) -> Tuple:
        """Hashable identity used by the feasibility cache."""
        return (self.left, self.right, self.grounded, self.stance)

    def union(self, other: "ContactMode") -> "ContactMode":
        """Instantaneous transition mode c1 ∪ c2 (FARO §II-D edge test).

        A hand in contact on either side of the transition is in contact at the instant
        itself; the chair touches the floor if it does on either side (lift-off / touchdown
        happen exactly at that configuration). A foot is likewise planted at the instant it
        leaves or lands, so a step boundary belongs to the stance episodes on both sides of
        it. What may be *balanced* on there is a different question — see ``frame_support``.
        """
        for side in HANDS:
            a, b = self.patch(side), other.patch(side)
            if a is not None and b is not None and a != b:
                raise ValueError(
                    f"{side} hand switches patch {a}->{b} without releasing; malformed plan"
                )
        return ContactMode(
            name=f"{self.name}|{other.name}",
            left=self.left or other.left,
            right=self.right or other.right,
            grounded=self.grounded or other.grounded,
            closure=max(self.closure, other.closure),
            stance=(self.stance[0] or other.stance[0], self.stance[1] or other.stance[1]),
        )


@dataclass
class PlanFrame:
    """A keyframe: the mode(s) it must satisfy plus the chair pose the plan wants there."""

    mode: ContactMode
    chair_pose: Pose
    chair_fixed: bool  # True = hard-constrained (task endpoints), False = slack-bounded


@dataclass
class ContactPlan:
    name: str
    modes: List[ContactMode]
    frames: List[PlanFrame] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)  # seconds, one per mode

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def describe(self) -> str:
        return f"{self.name}: " + " -> ".join(m.name for m in self.modes)


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #


def _frames(modes: Sequence[ContactMode], chair_poses: Sequence[Pose],
            fixed: Sequence[bool]) -> List[PlanFrame]:
    """Build the K+1 keyframes of a K-mode plan (frame s spans modes s-1 and s)."""
    assert len(chair_poses) == len(modes) + 1 == len(fixed)
    out = []
    for s in range(len(modes) + 1):
        a = modes[max(s - 1, 0)]
        b = modes[min(s, len(modes) - 1)]
        out.append(PlanFrame(a.union(b), chair_poses[s], fixed[s]))
    return out


def _rebuild(plan: ContactPlan, modes, chair_poses, fixed, durations, name: str) -> ContactPlan:
    return ContactPlan(
        name=name, modes=list(modes), frames=_frames(modes, chair_poses, fixed),
        durations=list(durations),
    )


def step_modes(
    tag: str,
    feet: Sequence[str],
    template: ContactMode,
    step_time: float = 1.1,
    shift_time: float = 0.7,
) -> Tuple[List[ContactMode], List[float]]:
    """One single-support stage per entry of ``feet``: that foot swings to a new stance.

    Consecutive swings are separated by a double-support stage. That is not padding: the
    weight has to cross from one foot to the other, and back-to-back swing stages leave a
    transition instant whose two sides have *no foot in common* — nothing the robot could
    be standing on at that moment.

    Everything else — what the hands hold, whether the chair is on the floor — is inherited
    from ``template``, so the same construction serves a free-handed walk and a shuffle
    performed while carrying the chair.
    """
    def like(name: str, stance: Tuple[bool, bool]) -> ContactMode:
        return ContactMode(
            name=name, left=template.left, right=template.right,
            grounded=template.grounded, closure=template.closure, stance=stance,
        )

    modes: List[ContactMode] = []
    times: List[float] = []
    for i, foot in enumerate(feet):
        if modes:
            modes.append(like(f"{tag}_shift{i}", (True, True)))
            times.append(shift_time)
        modes.append(like(f"{tag}_{foot[0]}{i}", (foot != "left", foot != "right")))
        times.append(step_time)
    return modes, times


def frame_support(plan: ContactPlan) -> List[Tuple[str, ...]]:
    """Feet carrying weight at each keyframe: those planted in *both* adjacent stages.

    The intersection, not the union the transition mode takes. A foot about to leave the
    ground cannot be leaned on at the instant it goes, so the centre of mass has to be over
    the other one *already* — which is what makes a step a weight transfer rather than a
    lurch, and is exactly the condition a balance check at that keyframe has to impose.
    """
    last = len(plan.modes) - 1
    return [
        tuple(
            side for side in FEET
            if plan.modes[max(s - 1, 0)].planted(side)
            and plan.modes[min(s, last)].planted(side)
        )
        for s in range(len(plan.modes) + 1)
    ]


def _step_order(n: int, first: str = "right") -> List[str]:
    other = "left" if first == "right" else "right"
    return [first if i % 2 == 0 else other for i in range(n)]


def with_walk_in(plan: ContactPlan, n_steps: int = 2, first: str = "right") -> ContactPlan:
    """Prefix a plan with steps that carry the robot up to the object before it reaches.

    This is the single change that buys the most envelope: standing at the spawn stance and
    reaching for a chair half a metre away puts the CoM at the toe edge for the whole
    motion, and every margin downstream is spent paying for it. Two steps in, and the same
    grasp is made from a comfortable stance.
    """
    if n_steps <= 0:
        return plan
    free = ContactMode("stand", None, None, grounded=True, closure=0.0)
    steps, times = step_modes("walk", _step_order(n_steps, first), free)
    # A double-support stage in front, so keyframe 0 is the stance the sim actually starts
    # in rather than an instant with a foot already off the ground.
    modes = [free] + steps + plan.modes
    n_pre = len(modes) - len(plan.modes)
    poses = [plan.frames[0].chair_pose.copy() for _ in range(n_pre)] + [
        f.chair_pose for f in plan.frames
    ]
    fixed = [True] * n_pre + [f.chair_fixed for f in plan.frames]
    durations = [0.6] + times + list(plan.durations)
    return _rebuild(plan, modes, poses, fixed, durations, f"walk+{plan.name}")


def with_pivot_steps(
    plan: ContactPlan, n_steps: int = 2, first: str = "right"
) -> ContactPlan:
    """Insert a step pair in the middle of the object motion, chair held still.

    The robot lets its stance follow the chair instead of trying to cover the whole
    displacement from one spot. This is what a person does to turn a chair through a large
    angle, and — unlike a regrasp — it keeps the contacts, so the object never has to be
    released and re-acquired.
    """
    if n_steps <= 0:
        return plan
    # Split at the stage that moves the object furthest while both hands hold it: that is
    # the one whose reach the step actually relieves.
    movers = [
        s for s, m in enumerate(plan.modes)
        if all(m.patch(h) is not None for h in HANDS)
        and _pose_gap(plan.frames[s].chair_pose, plan.frames[s + 1].chair_pose) > 1e-6
    ]
    if not movers:
        return plan
    cut = movers[len(movers) // 2]  # after this stage
    template = plan.modes[cut]
    steps, times = step_modes("pivot", _step_order(n_steps, first), template)
    n_new = len(steps)  # not n_steps: consecutive swings carry a shift stage between them
    modes = plan.modes[: cut + 1] + steps + plan.modes[cut + 1 :]
    hold = plan.frames[cut + 1].chair_pose
    poses = (
        [f.chair_pose for f in plan.frames[: cut + 2]]
        + [hold.copy() for _ in range(n_new)]
        + [f.chair_pose for f in plan.frames[cut + 2 :]]
    )
    fixed = (
        [f.chair_fixed for f in plan.frames[: cut + 2]]
        + [plan.frames[cut + 1].chair_fixed] * n_new
        + [f.chair_fixed for f in plan.frames[cut + 2 :]]
    )
    durations = list(plan.durations[: cut + 1]) + times + list(plan.durations[cut + 1 :])
    return _rebuild(plan, modes, poses, fixed, durations, f"{plan.name}+pivot")


def _pose_gap(a: Pose, b: Pose) -> float:
    return float(np.linalg.norm(b.p - a.p)) + abs(float(b.yaw() - a.yaw()))


def _assign_hands(chair: ChairSpec, pair: Tuple[str, str], grasp_pose: Pose) -> Tuple[str, str]:
    """Map an opposing patch pair to (left_patch, right_patch) by world normal direction.

    The left hand takes the patch whose outward normal points to the robot's left (+y).
    """
    patches = chair.patches()
    n0 = grasp_pose.rotate(patches[pair[0]].normal)
    return (pair[0], pair[1]) if n0[1] >= 0 else (pair[1], pair[0])


# Approximate shoulder positions of the standing G1, used only to rank candidate grasps
# by how far the robot would have to reach (the real verdict comes from the IK filters).
SHOULDERS = np.array([[0.0, 0.17, 1.10], [0.0, -0.17, 1.10]])
COMFORT_REACH = 0.55  # m from shoulder to palm before a grasp starts costing balance


def _pair_score(chair: ChairSpec, pair: Tuple[str, str], grasp_pose: Pose) -> float:
    """Rank a candidate pinch pair: lateral squeeze is good, long reaches are not.

    A pair whose normals run along the robot's y can be squeezed between the palms with
    the arms in front of the body; one that has to be reached around costs the whole
    balance margin. The reach term then discounts pairs the robot would have to lean for
    (e.g. the seat of a chair standing half a metre away).
    """
    patches = chair.patches()
    n = grasp_pose.rotate(patches[pair[0]].normal)
    laterality = abs(float(n[1])) / max(float(np.linalg.norm(n[:2])), 1e-9)
    reach = max(
        float(np.linalg.norm(grasp_pose.apply(patches[p].center) - SHOULDERS, axis=1).min())
        for p in pair
    )
    return laterality - max(0.0, reach - COMFORT_REACH) / COMFORT_REACH


def lift_plan(
    chair: ChairSpec,
    start: Pose,
    goal: Pose,
    pair: Tuple[str, str],
    lift_height: float = 0.12,
    n_transport: int = 2,
    segments: int = 1,
    name: Optional[str] = None,
) -> ContactPlan:
    """Grasp two opposing patches, lift the chair clear of the floor, move it, set it down.

    With ``segments > 1`` the displacement is done in that many grasps: the chair is set
    down, the hands let go and reposition, and the next grasp continues. That is not a
    convenience — with both feet planted the robot's wrists simply run out of range past
    roughly half a turn, so a large rotation is *only* reachable as a sequence of contact
    modes. It is the same reason FARO searches over mode sequences instead of optimizing
    one long trajectory.
    """
    up = np.array([0.0, 0.0, lift_height])
    waypoints = [interp_pose(start, goal, k / segments) for k in range(segments + 1)]
    # One pair per segment, re-chosen at that segment's starting orientation: after the
    # chair has turned, a *different* pair of faces is the one the robot can squeeze
    # between its palms. Re-picking is what makes regrasping worth anything — keeping the
    # same faces would put the hands back into the pose that was infeasible to begin with.
    seg_pairs = [pair] + [
        max(PINCH_PAIRS, key=lambda p: _pair_score(chair, p, waypoints[k]))
        for k in range(1, segments)
    ]
    left, right = _assign_hands(chair, seg_pairs[0], waypoints[0])

    modes: List[ContactMode] = [ContactMode("approach", None, None, True, 0.0)]
    chair_poses: List[Pose] = [waypoints[0].copy(), waypoints[0].copy()]
    fixed: List[bool] = [True, True]
    durations: List[float] = [1.5]

    for k in range(segments):
        p0, p1 = waypoints[k], waypoints[k + 1]
        air0, air1 = Pose(p0.p + up, p0.Rm), Pose(p1.p + up, p1.Rm)
        tag = "" if segments == 1 else str(k)
        left, right = _assign_hands(chair, seg_pairs[k], p0)
        modes.append(ContactMode(f"grasp{tag}", left, right, True, 1.0))
        modes.append(ContactMode(f"lift{tag}", left, right, False, 1.0))
        modes += [
            ContactMode(f"transport{tag}_{i}", left, right, False, 1.0)
            for i in range(n_transport)
        ]
        # "place" is the *descent*: the chair is airborne all the way down and only touches
        # at the keyframe that ends the stage (where the union with the next, grounded mode
        # makes it grounded). Marking the stage itself grounded contradicts its geometry.
        modes.append(ContactMode(f"place{tag}", left, right, False, 1.0))

        chair_poses.append(p0.copy())  # after "grasp": still on the floor
        chair_poses.append(air0.copy())  # after "lift": clear of the floor
        chair_poses += [
            interp_pose(air0, air1, (i + 1) / n_transport) for i in range(n_transport)
        ]
        chair_poses.append(p1.copy())  # touchdown
        # Waypoints on the floor are exact; the airborne knots in between may drift a
        # little if that is what makes the arm work.
        fixed += [True] + [False] * (n_transport + 1) + [True]
        durations += [0.8, 1.0] + [2.4 / n_transport] * n_transport + [1.0]

        if k < segments - 1:
            # Regrasp: let go, move the hands back, take hold again. Two consecutive
            # free-hand stages are needed — one alone would have contact at both of its
            # keyframes and the hands would never actually leave.
            modes.append(ContactMode(f"release{k}", None, None, True, 0.0))
            modes.append(ContactMode(f"reposition{k}", None, None, True, 0.0))
            chair_poses += [p1.copy(), p1.copy()]
            fixed += [True, True]
            durations += [1.0, 1.2]

    modes.append(ContactMode("release", None, None, True, 0.0))
    chair_poses.append(waypoints[-1].copy())
    fixed.append(True)
    durations.append(1.2)

    suffix = "" if segments == 1 else f"x{segments}"
    first_left, first_right = _assign_hands(chair, seg_pairs[0], waypoints[0])
    return ContactPlan(
        name=name or f"lift{suffix}[{first_left}|{first_right}]",
        modes=modes,
        frames=_frames(modes, chair_poses, fixed),
        durations=durations,
    )


def rail_lift_plan(chair: ChairSpec, start: Pose, goal: Pose, **kwargs) -> ContactPlan:
    """Both palms on the backrest top rail (hands side by side), then lift and move."""
    plan = lift_plan(chair, start, goal, ("rail_top", "rail_top"), name="rail_lift", **kwargs)
    return plan


def slide_plan(
    chair: ChairSpec,
    start: Pose,
    goal: Pose,
    pair: Tuple[str, str],
    n_slide: int = 3,
) -> ContactPlan:
    """Pinch two opposing faces and swivel the chair *on the floor*, without lifting.

    The natural way to turn a chair through a large angle: the legs keep carrying the
    weight and slide, so the hands only have to overcome floor friction and reorient it.
    Cheaper than a lift for the robot, but it needs the grasp to hold through the whole
    turn — which is exactly what the filters decide.
    """
    left, right = _assign_hands(chair, pair, start)
    modes = [ContactMode("approach", None, None, True, 0.0),
             ContactMode("grasp", left, right, True, 1.0)]
    modes += [ContactMode(f"slide{i}", left, right, True, 1.0) for i in range(n_slide)]
    modes.append(ContactMode("release", None, None, True, 0.0))

    chair_poses = [start.copy(), start.copy(), start.copy()]
    chair_poses += [interp_pose(start, goal, (i + 1) / n_slide) for i in range(n_slide)]
    chair_poses.append(goal.copy())
    fixed = [True, True, True] + [False] * (n_slide - 1) + [True, True]
    durations = [1.5, 0.8] + [2.4 / n_slide] * n_slide + [1.2]
    return ContactPlan(
        name=f"slide[{left}|{right}]",
        modes=modes,
        frames=_frames(modes, chair_poses, fixed),
        durations=durations,
    )


def push_plan(
    chair: ChairSpec,
    start: Pose,
    goal: Pose,
    patch: str = "back_rear",
    n_push: int = 3,
) -> ContactPlan:
    """Keep the chair on the floor and slide/turn it with both palms on one face.

    Cheapest plan when the goal is a modest in-plane change: no lifting, so no balance
    problem — but it only works while both hands stay in reach, which the feasibility
    filters decide.
    """
    free_g = ContactMode("approach", None, None, grounded=True, closure=0.0)
    touch = ContactMode("contact", patch, patch, grounded=True, closure=0.3)
    pushes = [
        ContactMode(f"push{i}", patch, patch, grounded=True, closure=0.3) for i in range(n_push)
    ]
    release = ContactMode("release", None, None, grounded=True, closure=0.0)
    modes = [free_g, touch, *pushes, release]

    chair_poses = [start.copy(), start.copy()]
    for i in range(n_push + 1):
        chair_poses.append(interp_pose(start, goal, i / n_push))
    chair_poses.append(goal.copy())
    fixed = [True, True] + [False] * n_push + [True, True]
    durations = [1.5, 0.8] + [2.4 / n_push] * n_push + [1.2]
    return ContactPlan(
        name=f"push[{patch}]",
        modes=modes,
        frames=_frames(modes, chair_poses, fixed),
        durations=durations,
    )


def candidate_plans(
    chair: ChairSpec,
    start: Pose,
    goal: Pose,
    lift_height: float = 0.12,
    allow_push: bool = True,
    n_transport: int = 2,
    walk_steps: int = 2,
) -> List[ContactPlan]:
    """Candidate contact plans in try-order (cheapest / most likely first).

    ``walk_steps = 0`` turns stepping off entirely and leaves only the standing plans.

    Ordering heuristics, in the spirit of FARO's cost-guided expansion:
      * a floor-supported push first, but only for a modest in-plane change — it needs no
        lift and no balance margin;
      * then pinch-lifts, best patch pair first (a pair the robot can squeeze along its
        own lateral axis is far easier than one it has to reach around);
      * the backrest rail lift last: the chair hangs off the grasp, so it is the least
        forgiving of the three.

    Each family is offered *walking in* first, then shuffling mid-motion if the
    displacement is large, and only then standing still. Stepping is strictly more capable
    — it is the standing variants that are the fallback, kept because they are cheaper to
    filter and still perfectly valid for a small displacement.
    """
    dxy = float(np.linalg.norm((goal.p - start.p)[:2]))
    dz = abs(float(goal.p[2] - start.p[2]))
    dyaw = abs(float(np.degrees(goal.yaw() - start.yaw())))
    ranked = sorted(PINCH_PAIRS, key=lambda p: -_pair_score(chair, p, start))

    families: List[ContactPlan] = []
    if allow_push and dxy < 0.25 and dz < 0.02 and dyaw < 100.0:
        families.append(push_plan(chair, start, goal))
    if dz < 0.02:
        # Swivelling it on the floor beats lifting whenever the goal stays in the plane:
        # the legs keep carrying the weight, so nothing has to be balanced.
        for pair in ranked[:2]:
            families.append(slide_plan(chair, start, goal, pair))
    for pair in ranked:
        families.append(
            lift_plan(chair, start, goal, pair, lift_height=lift_height,
                      n_transport=n_transport)
        )
    families.append(rail_lift_plan(chair, start, goal, lift_height=lift_height,
                                   n_transport=n_transport))

    plans: List[ContactPlan] = [with_walk_in(p, walk_steps) for p in families]
    # Past roughly a third of a turn one stance cannot cover the whole displacement, so
    # offer variants that shuffle the feet halfway through while keeping the contacts.
    if walk_steps > 0 and _segments_needed(dyaw, dxy) > 1:
        plans += [with_pivot_steps(with_walk_in(p, walk_steps)) for p in families[:3]]
    if walk_steps > 0:
        plans += families
    # Last resort: release the object, reposition the hands and take a fresh grasp. A
    # regrasp buys reach that neither reaching nor stepping can, at the price of putting
    # the chair down mid-way and clearing another full set of filters.
    if _segments_needed(dyaw, dxy) > 1:
        for pair in ranked[:2]:
            plans.append(
                with_walk_in(
                    lift_plan(chair, start, goal, pair, lift_height=lift_height,
                              n_transport=max(1, n_transport - 1),
                              segments=_segments_needed(dyaw, dxy)),
                    walk_steps,
                )
            )
    return plans


def stance_episodes(plan: ContactPlan) -> List[Dict]:
    """Contiguous stance episodes per foot, over the plan's keyframes.

    A foot planted across a run of consecutive stages holds *one* placement for the whole
    run — the exact analogue of a hand's no-slip contact episode, and the reason a planted
    foot cannot slide: its placement is a single shared decision variable. The keyframes of
    a run of stages ``[i..j]`` are ``[i..j+1]``, so the frames on either side of a swing
    stage belong to the stance episodes it separates.
    """
    out: List[Dict] = []
    for side in FEET:
        planted = [m.planted(side) for m in plan.modes]
        s = 0
        while s < len(planted):
            if not planted[s]:
                s += 1
                continue
            j = s
            while j + 1 < len(planted) and planted[j + 1]:
                j += 1
            out.append({"side": side, "frames": list(range(s, j + 2))})
            s = j + 1
    return out


def _segments_needed(dyaw_deg: float, dxy: float) -> int:
    """How many grasps to split a displacement into (1 = a single grasp should do)."""
    return max(1, int(np.ceil(max(abs(dyaw_deg) / 55.0, dxy / 0.30))))


def plan_by_name(
    name: str, chair: ChairSpec, start: Pose, goal: Pose, **kwargs
) -> List[ContactPlan]:
    """Resolve ``--plan``: ``auto`` | ``push`` | ``slide`` | ``side_lift`` | ``rail_lift``.

    A named family is offered in the same stepping-first order ``auto`` uses.
    """
    if name == "auto":
        return candidate_plans(chair, start, goal, **kwargs)
    lift_height = kwargs.get("lift_height", 0.12)
    n_transport = kwargs.get("n_transport", 2)
    walk_steps = kwargs.get("walk_steps", 2)
    ranked = sorted(PINCH_PAIRS, key=lambda p: -_pair_score(chair, p, start))
    dyaw = float(np.degrees(goal.yaw() - start.yaw()))
    dxy = float(np.linalg.norm((goal.p - start.p)[:2]))
    if name == "push":
        families = [push_plan(chair, start, goal)]
    elif name == "slide":
        families = [slide_plan(chair, start, goal, pair) for pair in ranked]
    elif name == "side_lift":
        n_seg = _segments_needed(dyaw, dxy)
        families = [
            lift_plan(chair, start, goal, pair, lift_height=lift_height,
                      n_transport=n_transport, segments=seg)
            for seg in sorted({1, n_seg})
            for pair in ranked
        ]
    elif name == "rail_lift":
        families = [rail_lift_plan(chair, start, goal, lift_height=lift_height,
                                   n_transport=n_transport)]
    else:
        raise ValueError(f"unknown plan '{name}' (auto|push|slide|side_lift|rail_lift)")

    if walk_steps <= 0:
        return families
    plans = [with_walk_in(p, walk_steps) for p in families]
    if _segments_needed(dyaw, dxy) > 1:
        plans += [with_pivot_steps(with_walk_in(p, walk_steps)) for p in families]
    return plans + families


def contact_episodes(plan: ContactPlan) -> List[Dict]:
    """Contiguous (hand, patch) contact episodes over the plan's keyframes.

    Each episode owns one in-patch contact location (two decision variables) that stays
    constant while the contact persists — this is how FARO's no-slip condition (eq. 8) is
    imposed, by construction rather than as a constraint.
    """
    episodes: List[Dict] = []
    for side in HANDS:
        current: Optional[Dict] = None
        for s, frame in enumerate(plan.frames):
            patch = frame.mode.patch(side)
            if patch is None:
                current = None
                continue
            if current is None or current["patch"] != patch:
                current = {"side": side, "patch": patch, "frames": [s]}
                episodes.append(current)
            else:
                current["frames"].append(s)
    return episodes
