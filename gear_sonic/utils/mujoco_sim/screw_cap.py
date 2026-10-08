"""Screw cap of a staged mesh object (``object.json["screw_cap"]``, make_primitive_asset.py).

The cap is its own free body, welded to a geometry-less seat body inside the object. The seat
carries the thread: a hinge about the cap axis (Coulomb friction + damping, can't tighten past 0)
and a slide coupled to it by the lead. Twisting the cap drives the seat through the weld, so the
friction is solved implicitly by the constraint solver like any joint friction. Once the hinge
passes ``release_angle`` the weld is switched off and the cap is a plain free body (it collides
with the bottle normally, being a separate top-level body).
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

CAP_BODY_NAME = "object_cap"
SEAT_BODY_NAME = "object_cap_seat"
TWIST_JOINT_NAME = "cap_twist"
LIFT_JOINT_NAME = "cap_lift"
WELD_NAME = "cap_thread"

# Same as the object grip contacts (run_sim_loop BOX_SOLREF): 2x the 5 ms sim timestep.
WELD_SOLREF = "0.01 1"
WELD_SOLIMP = "0.95 0.99 0.001 0.5 2"
HARD_SOLIMP = "0.99 0.999 0.001 0.5 2"
TWIST_ARMATURE = 2e-4  # kg.m2


def _cap_world_pose(obj_pos, obj_quat, cap_pos):
    """World (pos, wxyz quat) of the screwed-on cap, given the object pose."""
    rot = Rotation.from_quat(np.asarray(obj_quat, dtype=float), scalar_first=True)
    return np.asarray(obj_pos, dtype=float) + rot.apply(cap_pos), np.asarray(obj_quat, dtype=float)


def append_screw_cap(
    root: ET.Element,
    object_body: ET.Element,
    object_config: dict,
    collision_group: int,
) -> None:
    """Add the seat (inside ``object_body``), the free cap body and the thread equalities."""
    cap = object_config["meta"]["screw_cap"]
    asset_dir = Path(object_config["asset_dir"])
    axis = " ".join(f"{v:g}" for v in cap["axis"])
    cap_pos = np.asarray(cap["pos"], dtype=float)

    seat = ET.SubElement(object_body, "body", name=SEAT_BODY_NAME, pos=" ".join(map(str, cap_pos)))
    # Massless seat would make the hinge/slide mass matrix singular; the cap's mass reaches it
    # through the weld, so these only need to be small and positive.
    ET.SubElement(seat, "inertial", pos="0 0 0", mass="1e-3", diaginertia="1e-7 1e-7 1e-7")
    twist_max = cap["release_angle"] + 0.3
    # MuJoCo's soft friction/limit constraints creep under a sustained load at a rate ~ load /
    # inertia, and the real cap's ~3e-6 kg.m2 lets it slip at a few rad/s below friction_torque.
    # The armature and the near-hard solimp bring that down to mrad/s.
    ET.SubElement(
        seat,
        "joint",
        name=TWIST_JOINT_NAME,
        type="hinge",
        axis=axis,
        range=f"0 {twist_max}",
        limited="true",
        armature=str(TWIST_ARMATURE),
        frictionloss=str(cap["friction_torque"]),
        damping=str(cap["damping"]),
        solreffriction=WELD_SOLREF,
        solimpfriction=HARD_SOLIMP,
        solreflimit=WELD_SOLREF,
        solimplimit=HARD_SOLIMP,
    )
    lead_per_rad = cap["lead"] / (2.0 * np.pi)
    ET.SubElement(
        seat,
        "joint",
        name=LIFT_JOINT_NAME,
        type="slide",
        axis=axis,
        range=f"0 {lead_per_rad * twist_max}",
        limited="true",
    )

    pos, quat = _cap_world_pose(object_config["pos"], object_config["quat"], cap_pos)
    body = ET.SubElement(root.find("worldbody"), "body", name=CAP_BODY_NAME)
    body.set("pos", " ".join(map(str, pos)))
    body.set("quat", " ".join(map(str, quat)))
    ET.SubElement(body, "freejoint")
    scale = cap["mass"] / max(cap["unit_mass"], 1e-12)
    ET.SubElement(
        body,
        "inertial",
        pos=" ".join(map(str, cap["com"])),
        mass=str(cap["mass"]),
        fullinertia=" ".join(str(v * scale) for v in cap["fullinertia"]),
    )

    asset = root.find("asset")
    for i, part in enumerate(cap["parts"]):
        col, vis = f"{CAP_BODY_NAME}_col_{i}", f"{CAP_BODY_NAME}_visual_{i}"
        ET.SubElement(asset, "mesh", name=col, file=str((asset_dir / part["collision"]).resolve()))
        ET.SubElement(asset, "mesh", name=vis, file=str((asset_dir / part["mesh"]).resolve()))
        geom = ET.SubElement(
            body, "geom", type="mesh", mesh=col, group=str(collision_group), mass="0"
        )
        for key in ("friction", "condim", "priority", "solref", "solimp"):
            if object_config.get(key) is not None:
                geom.set(key, str(object_config[key]))
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=vis,
            contype="0",
            conaffinity="0",
            mass="0",
            rgba=part.get("rgba", "0.8 0.1 0.1 1"),
        )

    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name=WELD_NAME,
        body1=CAP_BODY_NAME,
        body2=SEAT_BODY_NAME,
        solref=WELD_SOLREF,
        solimp=WELD_SOLIMP,
    )
    ET.SubElement(
        equality,
        "joint",
        name="cap_lead",
        joint1=LIFT_JOINT_NAME,
        joint2=TWIST_JOINT_NAME,
        polycoef=f"0 {lead_per_rad} 0 0 0",
    )


class ScrewCap:
    """Runtime side: releases the cap past ``release_angle`` and re-caps on object reset."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, object_config: dict):
        cap = object_config["meta"]["screw_cap"]
        self.model, self.data = model, data
        self.cap_pos = np.asarray(cap["pos"], dtype=float)
        self.release_angle = float(cap["release_angle"])
        self.body_id = model.body(CAP_BODY_NAME).id
        self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
        cap_jnt = int(model.body_jntadr[self.body_id])
        self.cap_qadr = int(model.jnt_qposadr[cap_jnt])
        self.cap_dofadr = int(model.jnt_dofadr[cap_jnt])
        self.twist_qadr = int(model.joint(TWIST_JOINT_NAME).qposadr[0])
        self.twist_dofadr = int(model.joint(TWIST_JOINT_NAME).dofadr[0])
        self.lift_qadr = int(model.joint(LIFT_JOINT_NAME).qposadr[0])
        self.lift_dofadr = int(model.joint(LIFT_JOINT_NAME).dofadr[0])
        self.attached = True

    @property
    def angle(self) -> float:
        """Unscrewed angle [rad]; frozen at the release value once the cap is off."""
        return float(self.data.qpos[self.twist_qadr])

    @property
    def cap_vel(self) -> np.ndarray:
        """(6,) cap freejoint qvel: world lin, local ang."""
        return self.data.qvel[self.cap_dofadr : self.cap_dofadr + 6].copy()

    def update(self) -> None:
        """Call after every mj_step."""
        if self.attached and self.angle >= self.release_angle:
            self.data.eq_active[self.weld_id] = 0
            self.attached = False
            print(f"[Sim] bottle cap released after {self.angle:.2f} rad")

    def reset(self, obj_qpos7) -> None:
        """Screw the cap back on the object at ``obj_qpos7`` (pos + wxyz quat). Caller runs
        mj_forward."""
        d = self.data
        pos, quat = _cap_world_pose(obj_qpos7[:3], obj_qpos7[3:7], self.cap_pos)
        d.qpos[self.cap_qadr : self.cap_qadr + 3] = pos
        d.qpos[self.cap_qadr + 3 : self.cap_qadr + 7] = quat
        d.qvel[self.cap_dofadr : self.cap_dofadr + 6] = 0.0
        d.qpos[[self.twist_qadr, self.lift_qadr]] = 0.0
        d.qvel[[self.twist_dofadr, self.lift_dofadr]] = 0.0
        d.eq_active[self.weld_id] = 1
        self.attached = True
