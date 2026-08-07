"""MuJoCo scene for chair trajectory optimization: G1 (BrainCo hands) + chair proxy.

Wraps one ``MjModel``/``MjData`` pair and exposes exactly what the optimizer needs:
set a configuration, run kinematics, read frames / CoM, query pairwise geom distances
(the SDF-style collision term of FARO eq. 9), and the joint position/velocity/torque
limits of eq. 13.

The chair enters the scene the same way ``base_sim.py`` injects it — a free body named
``object`` — so a generated trajectory can be replayed in a scene that matches the sim.
Its *collision* geometry is the box proxy from ``ChairSpec`` (convex, cheap, slightly
conservative); the staged visual mesh, when available, is drawn on top with collisions off.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gear_sonic.trajopt.chair import ChairSpec
from gear_sonic.trajopt.se3 import Pose, rot_exp

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/with_brainco"
SCENE_XML = MODEL_DIR / "scene_41dof.xml"
URDF_PATH = MODEL_DIR / "g1_29dof_with_hand.urdf"
OBJECT_BODY_NAME = "object"  # same name base_sim.py uses, so replay tooling matches

# The 29 actuated body joints, in MuJoCo/MJCF order. This is also the order the
# motion_lib exporter expects (see data_process/convert_soma_csv_to_motion_lib.py).
BODY_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The 6 actuated BrainCo joints per hand (distal joints are passive mimics).
HAND_JOINT_NAMES: Dict[str, Tuple[str, ...]] = {
    "left": ("left_thumb_metacarpal_joint", "left_thumb_proximal_joint",
             "left_index_proximal_joint", "left_middle_proximal_joint",
             "left_ring_proximal_joint", "left_pinky_proximal_joint"),
    "right": ("right_thumb_metacarpal_joint", "right_thumb_proximal_joint",
              "right_index_proximal_joint", "right_middle_proximal_joint",
              "right_ring_proximal_joint", "right_pinky_proximal_joint"),
}

# Ready stance used as the posture prior and as the frame-0 pose: a light crouch, arms
# down and slightly out. Straight knees are a kinematic singularity for the legs, so the
# nominal deliberately bends them.
NOMINAL_JOINTS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.25, "left_knee_joint": 0.5, "left_ankle_pitch_joint": -0.25,
    "right_hip_pitch_joint": -0.25, "right_knee_joint": 0.5, "right_ankle_pitch_joint": -0.25,
    "left_shoulder_pitch_joint": 0.25, "left_shoulder_roll_joint": 0.25, "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "right_elbow_joint": 0.6,
}

# BrainCo grasp postures, in HAND_JOINT_NAMES order (thumb metacarpal, thumb proximal,
# index, middle, ring, pinky). Ranges are [0, ~1.47] per joint.
HAND_OPEN = np.zeros(6)
HAND_CLOSED = np.array([1.0, 0.6, 1.1, 1.1, 1.1, 1.1])

# Smallest radius the spurious-zero probe descends to (see TrajOptScene._distance). Below a
# couple of millimetres a zero is taken at face value: at that separation the difference
# between touching and nearly touching does not change any verdict here.
_DISTANCE_PROBE_MIN = 0.002

FOOT_SIDES = ("left", "right")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_BODY = dict(zip(FOOT_SIDES, FOOT_BODIES))

# Robot bodies kept clear of the chair. Hands and wrists are excluded on purpose: they are
# *supposed* to touch, and their placement is governed by the contact constraints instead.
CHAIR_AVOID_BODIES = (
    "pelvis", "torso_link", "waist_yaw_link", "waist_roll_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_roll_link", "right_shoulder_yaw_link",
)

# The arm chain below the shoulder is exempt from chair avoidance *while that hand is
# grasping*: an arm holding an object is necessarily within centimetres of it, and the
# contact constraints already govern where it goes. It is checked normally when the hand
# is free (that is what stops the robot reaching through the chair on approach).
GRASP_ARM_BODIES = (
    "{side}_elbow_link", "{side}_wrist_roll_link", "{side}_wrist_pitch_link",
    "{side}_wrist_yaw_link", "{side}_base_link",
)

# The arm from the shoulder yaw joint down. A two-hand task brings both of these into the
# same small volume in front of the chest, so *every* left link has to be checked against
# *every* right one — listing only the hands lets the left palm sit inside the right wrist,
# which is exactly what happened before this was enumerated.
ARM_CHAIN = (
    "{side}_shoulder_yaw_link", "{side}_elbow_link", "{side}_wrist_roll_link",
    "{side}_wrist_pitch_link", "{side}_wrist_yaw_link", "{side}_base_link",
)
# Against the chest, the shoulder yaw link is excluded: it sits 3 cm from the torso in the
# nominal stance, inside COLLISION_MARGIN, so checking it would report a permanent
# violation of a clearance the robot's own geometry does not have.
TORSO_CHECKED_ARM = ARM_CHAIN[1:]

SELF_COLLISION_BODY_PAIRS = tuple(
    [(a.format(side="left"), b.format(side="right")) for a in ARM_CHAIN for b in ARM_CHAIN]
    + [
        (a.format(side=s), t)
        for s in ("left", "right")
        for a in TORSO_CHECKED_ARM
        for t in ("torso_link", "pelvis")
    ]
    # An arm swinging down past its own thigh, and the leg-on-leg pairs that keep a step
    # from swinging one foot through the other.
    + [(f"{s}_base_link", f"{s}_knee_link") for s in ("left", "right")]
    + [
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        ("left_knee_link", "right_knee_link"),
    ]
)


@dataclass
class JointLimits:
    """Per-joint limits for the 29 body joints (FARO eq. 13)."""

    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


def _parse_urdf_limits() -> Dict[str, Tuple[float, float, float, float]]:
    """``joint -> (lower, upper, velocity, effort)`` from the G1 URDF."""
    root = ET.parse(URDF_PATH).getroot()
    out = {}
    for joint in root.findall("joint"):
        lim = joint.find("limit")
        if lim is None:
            continue
        out[joint.get("name")] = (
            float(lim.get("lower", "-3.14")), float(lim.get("upper", "3.14")),
            float(lim.get("velocity", "10.0")), float(lim.get("effort", "50.0")),
        )
    return out


class TrajOptScene:
    """G1 + chair MuJoCo model with the kinematic queries the optimizer needs."""

    def __init__(self, chair: ChairSpec, chair_pose: Pose, dump_xml: Path | None = None):
        self.chair = chair
        xml = self._build_xml(chair, chair_pose)
        if dump_xml is not None:
            Path(dump_xml).write_text(xml)
        # Written next to the original so the relative <include> of the robot MJCF resolves.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", dir=str(MODEL_DIR), delete=False
        ) as f:
            f.write(xml)
            tmp = f.name
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.remove(tmp)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0

        m = self.model
        self.body_qadr = np.array(
            [m.jnt_qposadr[m.joint(n).id] for n in BODY_JOINT_NAMES], dtype=int
        )
        self.hand_qadr = {
            side: np.array([m.jnt_qposadr[m.joint(n).id] for n in names], dtype=int)
            for side, names in HAND_JOINT_NAMES.items()
        }
        obj_jnt = int(m.body_jntadr[m.body(OBJECT_BODY_NAME).id])
        self.object_qadr = int(m.jnt_qposadr[obj_jnt])
        self.pelvis_bid = m.body("pelvis").id
        self.object_bid = m.body(OBJECT_BODY_NAME).id
        self.robot_mass = float(sum(m.body_mass[b] for b in self._subtree_bodies(self.pelvis_bid)))

        self.couplings = self._joint_couplings()
        self.limits = self._joint_limits()
        self.hand_range = {
            side: (
                np.array([m.jnt_range[m.joint(n).id][0] for n in names]),
                np.array([m.jnt_range[m.joint(n).id][1] for n in names]),
            )
            for side, names in HAND_JOINT_NAMES.items()
        }
        self.palm_local = {side: self._palm_frame(side) for side in ("left", "right")}
        self.chair_pairs, self.self_pairs, self.hand_chair_pairs = self._collision_pairs()
        self.foot_bids = {s: m.body(n).id for s, n in zip(FOOT_SIDES, FOOT_BODIES)}
        self.foot_sole_geoms = {s: self._sole_geom(n) for s, n in zip(FOOT_SIDES, FOOT_BODIES)}
        self.sole_local = {s: self._sole_local(s) for s in FOOT_SIDES}

        self.nominal = self._nominal_stance()
        self.foot_nominal, self.ground_z = self._foot_nominal()
        self.body_reach = self._body_reach()

    # ------------------------------------------------------------------ #
    # model construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_xml(chair: ChairSpec, chair_pose: Pose, absolute_include: bool = False) -> str:
        """Scene XML as a string. ``absolute_include`` rewrites the robot MJCF include to an
        absolute path, so the result can be loaded from anywhere (that is what gets exported
        next to a trajectory); the in-process model is built in the model directory itself
        and keeps the relative include."""
        tree = ET.parse(SCENE_XML)
        root = tree.getroot()
        if absolute_include:
            for inc in root.iter("include"):
                inc.set("file", str((MODEL_DIR / inc.get("file")).resolve()))
            # meshdir in the included robot MJCF is relative to the *main* file's directory,
            # which is no longer the model directory — pin it. A compiler element after the
            # include wins, so the exported scene resolves its meshes from anywhere.
            comp = ET.SubElement(root, "compiler")
            comp.set("angle", "radian")
            comp.set("meshdir", str((MODEL_DIR / "meshes").resolve()))
        worldbody = root.find("worldbody")

        body = ET.SubElement(worldbody, "body")
        body.set("name", OBJECT_BODY_NAME)
        body.set("pos", " ".join(f"{v:.6f}" for v in chair_pose.p))
        body.set("quat", " ".join(f"{v:.6f}" for v in chair_pose.quat_wxyz()))
        ET.SubElement(body, "freejoint")

        inertial = ET.SubElement(body, "inertial")
        inertial.set("pos", " ".join(f"{v:.6f}" for v in chair.com))
        inertial.set("mass", f"{chair.mass:.6f}")
        diag = np.diag(chair.inertia)
        inertial.set("diaginertia", " ".join(f"{v:.8f}" for v in diag))

        for name, center, half in chair.boxes():
            g = ET.SubElement(body, "geom")
            g.set("name", f"{OBJECT_BODY_NAME}_{name}")
            g.set("type", "box")
            g.set("pos", " ".join(f"{v:.6f}" for v in center))
            g.set("size", " ".join(f"{v:.6f}" for v in half))
            g.set("mass", "0")  # inertia comes from <inertial> above
            g.set("rgba", "0.75 0.6 0.4 1" if chair.visual_mesh is None else "0.75 0.6 0.4 0.25")

        if chair.visual_mesh is not None:
            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            mesh = ET.SubElement(asset, "mesh")
            mesh.set("name", f"{OBJECT_BODY_NAME}_visual")
            mesh.set("file", str(Path(chair.visual_mesh).resolve()))
            g = ET.SubElement(body, "geom")
            g.set("type", "mesh")
            g.set("mesh", f"{OBJECT_BODY_NAME}_visual")
            # The mesh lives in the asset's own frame, a yaw offset from the canonical
            # proxy frame the optimizer works in (see ChairSpec.asset_yaw).
            g.set("quat", " ".join(
                f"{v:.6f}" for v in
                Pose(np.zeros(3), rot_exp([0.0, 0.0, chair.asset_yaw])).quat_wxyz()
            ))
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("mass", "0")
            g.set("rgba", "0.75 0.6 0.4 1")
        return ET.tostring(root, encoding="unicode")

    def _subtree_bodies(self, root_bid: int) -> List[int]:
        out, stack = [], [root_bid]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(
                i for i in range(self.model.nbody) if int(self.model.body_parentid[i]) == b and i != b
            )
        return out

    def _joint_couplings(self):
        """BrainCo distal<-proximal mimics, read from the model's joint equalities."""
        out = []
        for e in range(self.model.neq):
            if self.model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            dep = int(self.model.jnt_qposadr[int(self.model.eq_obj1id[e])])
            ref = int(self.model.jnt_qposadr[int(self.model.eq_obj2id[e])])
            poly = np.asarray(self.model.eq_data[e][:5], dtype=float)
            out.append((dep, ref, poly, float(self.model.qpos0[dep]), float(self.model.qpos0[ref])))
        return out

    def _joint_limits(self) -> JointLimits:
        urdf = _parse_urdf_limits()
        lo, hi, vel, eff = [], [], [], []
        for name in BODY_JOINT_NAMES:
            jr = self.model.jnt_range[self.model.joint(name).id]
            u = urdf.get(name, (jr[0], jr[1], 10.0, 50.0))
            lo.append(float(jr[0]))
            hi.append(float(jr[1]))
            vel.append(u[2])
            eff.append(u[3])
        return JointLimits(np.array(lo), np.array(hi), np.array(vel), np.array(eff))

    def _palm_frame(self, side: str) -> Pose:
        """Palm contact frame in the hand base link, derived from the model's fingers.

        +z of the returned frame is the palm's outward normal — the face the fingers curl
        *toward*, which is the one that may press on an object; +y is the finger direction;
        the origin sits on the palm surface. Derived rather than hardcoded so it stays right
        for either hand and survives MJCF changes.

        The normal comes from where the fingertips travel when the fingers flex. Taking it
        from the thumb instead (an obvious-looking shortcut) gives the hand's *lateral*
        axis on this model — the BrainCo thumb rests roughly in the plane of the palm — and
        pressing that on a patch puts the edge and back of the hand against the object.
        """
        m = self.model
        d = mujoco.MjData(m)
        fingers = ("index", "middle", "ring", "pinky")

        def tips(closure: float) -> np.ndarray:
            """(4, 3) fingertip positions in the hand base frame at a given flexion."""
            d.qpos[:] = m.qpos0
            for name in HAND_JOINT_NAMES[side]:
                j = m.joint(name).id
                lo, hi = m.jnt_range[j]
                d.qpos[m.jnt_qposadr[j]] = lo + closure * (hi - lo)
            for dep, ref, poly, dep_q0, ref_q0 in self.couplings:
                d.qpos[dep] = dep_q0 + np.polyval(poly[::-1], d.qpos[ref] - ref_q0)
            mujoco.mj_kinematics(m, d)
            base = m.body(f"{side}_base_link").id
            p0, R0 = d.xpos[base].copy(), d.xmat[base].reshape(3, 3).copy()
            return np.stack(
                [R0.T @ (d.xpos[m.body(f"{side}_{f}_tip_Link").id] - p0) for f in fingers]
            )

        open_tips = tips(0.0)
        flex = tips(0.75) - open_tips
        finger_dir = open_tips.mean(0)
        finger_dir /= np.linalg.norm(finger_dir)
        n = flex.mean(0)
        n -= finger_dir * float(n @ finger_dir)  # drop the part that is just the fingers curling up
        n /= np.linalg.norm(n)
        finger_dir -= n * float(finger_dir @ n)
        finger_dir /= np.linalg.norm(finger_dir)

        # Pad on the palm surface: the hand's collision mesh is bounded by a box in its own
        # frame, and the palm is the face of that box the normal points through.
        gid = self._palm_geom(side)
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[gid])
        Rg = Rg.reshape(3, 3)
        surface = float(m.geom_size[gid][int(np.argmax(np.abs(Rg.T @ n)))])
        origin = np.asarray(m.geom_pos[gid], dtype=float) + n * surface

        Rm = np.stack([np.cross(finger_dir, n), finger_dir, n], axis=1)  # x, y, z(=normal)
        return Pose(origin, Rm)

    def _palm_geom(self, side: str) -> int:
        """The hand base link's collision geom (the palm shell)."""
        bid = self.model.body(f"{side}_base_link").id
        return next(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == bid and int(self.model.geom_contype[g]) != 0
        )

    def _collision_pairs(self):
        """``(chair_pairs, self_pairs, hand_chair_pairs)`` as lists of (geom_a, geom_b) ids.

        ``hand_chair_pairs`` is kept separate and per side because a hand is only required
        to avoid the chair while it is *not* in contact with it — during a grasp its
        placement is governed by the contact constraints instead.
        """
        m = self.model

        def collision_geoms(body_name: str) -> List[int]:
            bid = m.body(body_name).id
            return [
                g for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == bid and int(m.geom_contype[g]) != 0
            ]

        chair_geoms = [
            g for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) == self.object_bid and int(m.geom_contype[g]) != 0
        ]
        chair_pairs = [
            (a, b) for name in CHAIR_AVOID_BODIES for a in collision_geoms(name) for b in chair_geoms
        ]
        self_pairs = [
            (a, b)
            for n1, n2 in SELF_COLLISION_BODY_PAIRS
            for a in collision_geoms(n1)
            for b in collision_geoms(n2)
        ]
        hand_chair_pairs = {}
        for side in ("left", "right"):
            geoms: List[int] = []
            for body in GRASP_ARM_BODIES:
                geoms += collision_geoms(body.format(side=side))
            hand_chair_pairs[side] = [(a, b) for a in geoms for b in chair_geoms]
        return chair_pairs, self_pairs, hand_chair_pairs

    def _sole_geom(self, body_name: str) -> int:
        """The flat box geom under a foot (the one that touches the floor)."""
        bid = self.model.body(body_name).id
        cands = [
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == bid and int(self.model.geom_contype[g]) != 0
        ]
        return min(cands, key=lambda g: self.model.geom_pos[g][2])

    def _sole_local(self, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sole geom's ``(offset, rotation, (hx, hy))`` in its ankle body's frame."""
        g = self.foot_sole_geoms[side]
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, self.model.geom_quat[g])
        return (
            np.asarray(self.model.geom_pos[g], dtype=float),
            Rm.reshape(3, 3),
            np.asarray(self.model.geom_size[g][:2], dtype=float),
        )

    # ------------------------------------------------------------------ #
    # configuration / kinematics
    # ------------------------------------------------------------------ #

    def set_state(
        self,
        base_p: np.ndarray,
        base_rv: np.ndarray,
        qj: np.ndarray,
        hand_q: Dict[str, np.ndarray] | None = None,
        obj_pose: Pose | None = None,
    ) -> None:
        """Write a full scene configuration into ``data.qpos`` and run kinematics."""
        q = self.data.qpos
        q[0:3] = base_p
        Rb = rot_exp(base_rv)
        q[3:7] = Pose(base_p, Rb).quat_wxyz()
        q[self.body_qadr] = qj
        if hand_q is not None:
            for side, vals in hand_q.items():
                q[self.hand_qadr[side]] = vals
        for dep, ref, poly, dep_q0, ref_q0 in self.couplings:
            q[dep] = dep_q0 + np.polyval(poly[::-1], q[ref] - ref_q0)
        if obj_pose is not None:
            q[self.object_qadr : self.object_qadr + 3] = obj_pose.p
            q[self.object_qadr + 3 : self.object_qadr + 7] = obj_pose.quat_wxyz()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def body_pose(self, name: str) -> Pose:
        bid = self.model.body(name).id
        return Pose(self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy())

    def palm_pose(self, side: str) -> Pose:
        return self.body_pose(f"{side}_base_link") * self.palm_local[side]

    def robot_com(self) -> np.ndarray:
        """Whole-robot CoM (pelvis subtree — i.e. everything but the chair)."""
        return self.data.subtree_com[self.pelvis_bid].copy()

    def sole_polygon(self, side: str, ankle: Pose) -> np.ndarray:
        """(4, 2) world xy corners of a foot sole, given that foot's ankle pose."""
        off, Rg, half = self.sole_local[side]
        c = ankle.apply(off)
        Rw = ankle.Rm @ Rg
        return np.array(
            [(c + Rw @ np.array([sx * half[0], sy * half[1], 0.0]))[:2]
             for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        )

    def foot_polygon(self, sides: Sequence[str] = FOOT_SIDES) -> np.ndarray:
        """(N, 2) world xy corners of the given foot soles (their hull is the support)."""
        pts = []
        for side in sides:
            g = self.foot_sole_geoms[side]
            c = self.data.geom_xpos[g]
            Rm = self.data.geom_xmat[g].reshape(3, 3)
            hx, hy = self.model.geom_size[g][:2]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append((c + Rm @ np.array([sx * hx, sy * hy, 0.0]))[:2])
        return np.asarray(pts)

    def distances(self, pairs: np.ndarray, distmax: float) -> np.ndarray:
        """Signed distance for each geom pair, saturated at ``distmax``.

        ``mj_geomDistance`` runs a convex solver per pair, which dominates the cost of a
        residual evaluation, so pairs whose bounding spheres are already further apart than
        ``distmax`` are skipped by a vectorized test first.

        Guarded against ``mj_geomDistance`` inventing contacts: for mesh pairs it sometimes
        returns exactly 0 for geoms that are centimetres apart, and an unguarded 0 reads as
        a collision the optimizer then flees. See ``_distance``.
        """
        a, b = pairs[:, 0], pairs[:, 1]
        centers = self.data.geom_xpos
        gap = (
            np.linalg.norm(centers[a] - centers[b], axis=1)
            - self.model.geom_rbound[a]
            - self.model.geom_rbound[b]
        )
        out = np.full(len(pairs), distmax)
        for i in np.flatnonzero(gap < distmax):
            # The bounding-sphere gap is itself a valid lower bound on the true distance.
            out[i] = max(self._distance(int(a[i]), int(b[i]), distmax), float(gap[i]))
        return out

    def _distance(self, ga: int, gb: int, distmax: float) -> float:
        """``mj_geomDistance`` with its spurious zeros filtered out.

        A *saturated* return — the query radius itself — is a proof that the true distance
        is at least that radius. The failure only ever goes one way (0 instead of a real,
        larger value) and the radius at which it starts depends on the pose, not just the
        pair, so it cannot be dodged by picking a good ``distmax``. Instead a zero is
        re-queried at successively smaller radii: the first probe that comes back non-zero
        is either the true distance or a proof it exceeds that probe, and both are safe to
        return. Only zeros pay for this, which is a few queries in a few thousand.
        """
        d = mujoco.mj_geomDistance(self.model, self.data, ga, gb, distmax, None)
        if d != 0.0:
            return d
        r = distmax
        while r > _DISTANCE_PROBE_MIN:
            r *= 0.5
            probe = mujoco.mj_geomDistance(self.model, self.data, ga, gb, r, None)
            if probe != 0.0:
                return probe
        return 0.0  # genuinely touching, as far as anything here can tell

    # ------------------------------------------------------------------ #
    # nominal stance
    # ------------------------------------------------------------------ #

    def _nominal_stance(self) -> Dict[str, np.ndarray]:
        """Ready stance with the feet resting exactly on the floor."""
        qj = np.zeros(len(BODY_JOINT_NAMES))
        for name, val in NOMINAL_JOINTS.items():
            qj[BODY_JOINT_NAMES.index(name)] = val
        base_p = np.array([0.0, 0.0, 0.8])
        base_rv = np.zeros(3)
        self.set_state(base_p, base_rv, qj, {"left": HAND_OPEN, "right": HAND_OPEN})
        floor = self.model.geom("floor").id
        gap = min(
            mujoco.mj_geomDistance(self.model, self.data, floor, g, 2.0, None)
            for g in self.foot_sole_geoms.values()
        )
        base_p[2] -= gap
        self.set_state(base_p, base_rv, qj, {"left": HAND_OPEN, "right": HAND_OPEN})
        return {"base_p": base_p, "base_rv": base_rv, "qj": qj}

    def _foot_nominal(self) -> Tuple[Dict[str, Pose], float]:
        """Ankle poses of the nominal stance, and the sole height of a foot on the ground.

        This is where the robot *starts*; a stepping plan is free to place later stances
        elsewhere (see ``kso.StanceEpisode``).
        """
        n = self.nominal
        self.set_state(n["base_p"], n["base_rv"], n["qj"])
        poses = {s: self.body_pose(FOOT_BODY[s]) for s in FOOT_SIDES}
        sole_z = float(
            np.mean([self.data.geom_xpos[g][2] for g in self.foot_sole_geoms.values()])
        )
        return poses, sole_z

    def _body_reach(self) -> float:
        """How far the standing robot's body sticks out in front of its ankles, meters.

        The knee, in the nominal crouch. Measured with the geoms' bounding spheres, so it
        is an over-estimate — which is the right side to err on for something used to decide
        how close to stand to an object.
        """
        n = self.nominal
        self.set_state(n["base_p"], n["base_rv"], n["qj"])
        bids = {self.model.body(b).id for b in CHAIR_AVOID_BODIES}
        front = max(
            float(self.data.geom_xpos[g][0] + self.model.geom_rbound[g])
            for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) in bids and int(self.model.geom_contype[g]) != 0
        )
        return front - float(np.mean([p.p[0] for p in self.foot_nominal.values()]))

    def foot_placement(self, side: str, xy_yaw: np.ndarray) -> Pose:
        """Ankle pose of a foot planted flat on the floor at ``(x, y, yaw)``.

        Height and tilt are those of the nominal stance: a planted foot lies on the ground,
        so a placement only has the three in-plane degrees of freedom.
        """
        nom = self.foot_nominal[side]
        return Pose(
            np.array([xy_yaw[0], xy_yaw[1], nom.p[2]]),
            rot_exp(np.array([0.0, 0.0, xy_yaw[2]])) @ nom.Rm,
        )

    def nominal_placement(self, side: str) -> np.ndarray:
        """``(x, y, yaw)`` of the nominal stance for one foot."""
        nom = self.foot_nominal[side]
        return np.array([nom.p[0], nom.p[1], float(Pose(nom.p, nom.Rm).yaw())])

    def foot_height(self, side: str) -> float:
        """Sole clearance above the floor of the current configuration."""
        return float(self.data.geom_xpos[self.foot_sole_geoms[side]][2]) - self.ground_z

    # ------------------------------------------------------------------ #

    def hand_posture(self, closed: float) -> Dict[str, np.ndarray]:
        """Finger joints for a grasp closure in [0, 1] (0 = open, 1 = closed).

        Clipped to the model's ranges: the BrainCo thumb metacarpal does not reach 0 (its
        range starts at ~35 deg), and an out-of-range value silently swings the thumb into
        the palm in FK, which is not where it is on the robot.
        """
        q = HAND_OPEN + float(np.clip(closed, 0.0, 1.0)) * (HAND_CLOSED - HAND_OPEN)
        return {side: np.clip(q, lo, hi) for side, (lo, hi) in self.hand_range.items()}
