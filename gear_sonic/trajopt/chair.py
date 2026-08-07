"""Parametric chair proxy and the contact patches a hand may press on.

Motion optimization needs *semantic* geometry — where can a palm push, which points touch
the floor — which the staged asset (a visual mesh plus anonymous convex hulls, see
``prepare_object_asset.py``) does not carry. ``ChairSpec`` is that semantic layer: a box
proxy (seat plate, backrest plate, four legs) sized either from a staged asset's bounding
box or from the hardcoded SANDSBERG defaults, plus a set of rectangular contact patches in
the spirit of FARO's patch-to-patch contact model (§II-C.1).

Chair local frame — same convention as the IKEA_interface assets the sim spawns: origin on
the floor under the seat centre, +z up, +x the direction a seated person faces (so the
backrest is at -x, which is why the sim spawns the chair with yaw = pi/2: backrest toward
the robot).

The optimizer collides against the proxy boxes only. That is intentional: they are convex,
cheap, and slightly conservative compared to the real hulls.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ContactPatch:
    """A rectangular contact interface on the chair, expressed in the chair frame.

    ``normal`` points *out of* the chair body: a palm in contact has its own outward
    normal anti-parallel to it. ``tangent``/``bitangent`` span the patch; a contact is
    located inside it by two in-plane coordinates bounded by ``half_extents`` — those
    coordinates are decision variables in the optimizer, held constant while the contact
    persists, which is how FARO's no-slip condition (eq. 8) is enforced here.
    """

    name: str
    center: np.ndarray  # (3,) chair frame
    normal: np.ndarray  # (3,) chair frame, unit, outward
    tangent: np.ndarray  # (3,) chair frame, unit, in-plane
    half_extents: Tuple[float, float]  # along (tangent, bitangent), meters
    kind: str = "grasp"  # "grasp" (can pull/lift) or "push" (unilateral only)

    @property
    def bitangent(self) -> np.ndarray:
        return np.cross(self.normal, self.tangent)

    def point(self, uv: np.ndarray) -> np.ndarray:
        """Chair-frame position of the in-patch coordinate ``uv`` (meters)."""
        return self.center + self.tangent * uv[0] + self.bitangent * uv[1]


# --------------------------------------------------------------------------- #
# Chair
# --------------------------------------------------------------------------- #

# SANDSBERG-like defaults (IKEA dining chair): used when no staged asset is given, and as
# the shape prior when only a bounding box is known. All meters.
DEFAULT_SEAT_WIDTH = 0.40  # along chair y
DEFAULT_SEAT_DEPTH = 0.40  # along chair x
DEFAULT_TOTAL_HEIGHT = 0.86
DEFAULT_SEAT_HEIGHT_FRAC = 0.52  # seat top / total height
DEFAULT_MASS = 0.2  # matches run_sim_loop.CHAIR_MASS (the sim chair is deliberately light)


@dataclass
class ChairSpec:
    """Box proxy of the manipulated chair plus its contact patches."""

    name: str = "sandsberg"
    seat_depth: float = DEFAULT_SEAT_DEPTH  # x extent
    seat_width: float = DEFAULT_SEAT_WIDTH  # y extent
    total_height: float = DEFAULT_TOTAL_HEIGHT
    seat_height: float = DEFAULT_TOTAL_HEIGHT * DEFAULT_SEAT_HEIGHT_FRAC  # top of the seat
    seat_thickness: float = 0.04
    back_thickness: float = 0.035
    leg_size: float = 0.035
    mass: float = DEFAULT_MASS
    # Staged asset dir (object.json + visual.obj); only used for the drawn mesh.
    asset_dir: Path | None = None
    visual_mesh: Path | None = field(default=None, repr=False)
    # Yaw of the staged asset's own frame relative to this canonical frame. The IKEA
    # assets do not share our convention: the sim spawns SANDSBERG with yaw = pi/2 to put
    # the backrest toward the robot, which in the canonical frame (backrest at -x) is
    # yaw = 0 — so the asset frame is the canonical one turned by +pi/2. Poses are
    # optimized in the canonical frame and converted on export (``to_asset_frame``).
    asset_yaw: float = 0.0

    # ---- derived geometry ------------------------------------------------- #

    @property
    def half_depth(self) -> float:
        return self.seat_depth / 2.0

    @property
    def half_width(self) -> float:
        return self.seat_width / 2.0

    @property
    def seat_center_z(self) -> float:
        return self.seat_height - self.seat_thickness / 2.0

    @property
    def back_center_x(self) -> float:
        """Backrest plate centre along x (at the rear edge of the seat)."""
        return -self.half_depth + self.back_thickness / 2.0

    @property
    def back_center_z(self) -> float:
        return (self.seat_height + self.total_height) / 2.0

    @property
    def back_half_height(self) -> float:
        return (self.total_height - self.seat_height) / 2.0

    @property
    def com(self) -> np.ndarray:
        """Chair-frame centre of mass (crude: seat + backrest + legs, uniform density)."""
        parts = [
            (self.seat_depth * self.seat_width * self.seat_thickness,
             np.array([0.0, 0.0, self.seat_center_z])),
            (self.back_thickness * self.seat_width * 2 * self.back_half_height,
             np.array([self.back_center_x, 0.0, self.back_center_z])),
            (4 * self.leg_size**2 * (self.seat_height - self.seat_thickness),
             np.array([0.0, 0.0, (self.seat_height - self.seat_thickness) / 2.0])),
        ]
        w = np.array([p[0] for p in parts])
        pts = np.stack([p[1] for p in parts])
        return (w[:, None] * pts).sum(0) / w.sum()

    @property
    def inertia(self) -> np.ndarray:
        """Diagonal inertia about the CoM, from a solid box of the chair's bounding box.

        Deliberately crude — it only feeds the object Newton-Euler check, where the chair
        is light and slow, so the inertial term is far below the gravity term.
        """
        lx, ly, lz = self.seat_depth, self.seat_width, self.total_height
        m = self.mass / 12.0
        return np.diag([m * (ly**2 + lz**2), m * (lx**2 + lz**2), m * (lx**2 + ly**2)])

    def leg_tips(self) -> np.ndarray:
        """(4, 3) chair-frame floor-contact points, one per leg."""
        dx = self.half_depth - self.leg_size / 2.0
        dy = self.half_width - self.leg_size / 2.0
        return np.array(
            [[dx, dy, 0.0], [dx, -dy, 0.0], [-dx, dy, 0.0], [-dx, -dy, 0.0]]
        )

    def boxes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Proxy collision boxes as ``(name, center, half_extents)`` in the chair frame."""
        leg_h = (self.seat_height - self.seat_thickness) / 2.0
        half_leg = self.leg_size / 2.0
        out = [
            ("seat", np.array([0.0, 0.0, self.seat_center_z]),
             np.array([self.half_depth, self.half_width, self.seat_thickness / 2.0])),
            ("back", np.array([self.back_center_x, 0.0, self.back_center_z]),
             np.array([self.back_thickness / 2.0, self.half_width, self.back_half_height])),
        ]
        for i, tip in enumerate(self.leg_tips()):
            out.append(
                (f"leg{i}", np.array([tip[0], tip[1], leg_h]),
                 np.array([half_leg, half_leg, leg_h]))
            )
        return out

    # ---- contact patches -------------------------------------------------- #

    def patches(self) -> Dict[str, ContactPatch]:
        """Named contact patches (see module docstring for the frame convention).

        Grasp patches come in opposing pairs so that two hands can pinch the chair with
        the object CoM between them — the configuration that makes a two-hand carry
        quasi-statically sound. ``push`` patches are unilateral (floor-supported moves).
        """
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        # Seat side faces: a palm presses on the seat edge. The usable strip is a bit
        # shorter than the face so the hand does not hang over a corner.
        seat_c = self.seat_center_z
        st = self.seat_thickness / 2.0
        p: Dict[str, ContactPatch] = {
            "seat_py": ContactPatch("seat_py", np.array([0.0, self.half_width, seat_c]),
                                    y, x, (0.7 * self.half_depth, st)),
            "seat_ny": ContactPatch("seat_ny", np.array([0.0, -self.half_width, seat_c]),
                                    -y, x, (0.7 * self.half_depth, st)),
            "seat_px": ContactPatch("seat_px", np.array([self.half_depth, 0.0, seat_c]),
                                    x, y, (0.7 * self.half_width, st)),
            "seat_nx": ContactPatch("seat_nx", np.array([-self.half_depth, 0.0, seat_c]),
                                    -x, y, (0.7 * self.half_width, st)),
        }
        # Backrest side faces (narrow edges) and the top rail, both graspable.
        bx, bz, bh = self.back_center_x, self.back_center_z, self.back_half_height
        bt = self.back_thickness / 2.0
        p.update(
            {
                "back_py": ContactPatch("back_py", np.array([bx, self.half_width, bz]),
                                        y, z, (0.6 * bh, bt)),
                "back_ny": ContactPatch("back_ny", np.array([bx, -self.half_width, bz]),
                                        -y, z, (0.6 * bh, bt)),
                "rail_top": ContactPatch("rail_top", np.array([bx, 0.0, self.total_height]),
                                         z, y, (0.75 * self.half_width, bt)),
            }
        )
        # Backrest faces, for floor-supported pushing (no pulling).
        p.update(
            {
                "back_rear": ContactPatch("back_rear", np.array([bx - bt, 0.0, bz]),
                                          -x, y, (0.75 * self.half_width, 0.6 * bh),
                                          kind="push"),
                "back_front": ContactPatch("back_front", np.array([bx + bt, 0.0, bz]),
                                           x, y, (0.75 * self.half_width, 0.6 * bh),
                                           kind="push"),
            }
        )
        return p

    # ---- construction ----------------------------------------------------- #

    @classmethod
    def from_asset(cls, asset_dir: Path, mass: float | None = None) -> "ChairSpec":
        """Size the proxy from a staged asset's ``object.json`` bounding box.

        Only the bounding box is available, so the *proportions* (seat height fraction,
        plate thicknesses) stay at the SANDSBERG defaults — override on the CLI if a
        candidate chair is shaped very differently.
        """
        asset_dir = Path(asset_dir)
        meta = json.loads((asset_dir / "object.json").read_text())
        lo = np.asarray(meta["bbox_min"], dtype=float)
        hi = np.asarray(meta["bbox_max"], dtype=float)
        extents = hi - lo
        visual = asset_dir / meta.get("visual_mesh", "visual.obj")
        total_h = float(extents[2])
        # The asset's spawn yaw is what makes its backrest face the robot; in the canonical
        # frame that orientation is yaw 0, so the difference is the frame offset.
        asset_yaw = float(meta.get("spawn_yaw", np.pi / 2))
        # The proxy is built around the seat centre; the asset's own origin is wherever
        # IKEA_interface put it, so swap the x/y extents when the frames are a quarter turn
        # apart (the canonical x is the seat depth).
        depth, width = float(extents[0]), float(extents[1])
        if abs(abs(np.sin(asset_yaw)) - 1.0) < 0.5:
            depth, width = width, depth
        return cls(
            name=str(meta.get("name", asset_dir.name)),
            seat_depth=depth,
            seat_width=width,
            total_height=total_h,
            seat_height=total_h * DEFAULT_SEAT_HEIGHT_FRAC,
            mass=DEFAULT_MASS if mass is None else mass,
            asset_dir=asset_dir,
            visual_mesh=visual if visual.is_file() else None,
            asset_yaw=asset_yaw,
        )

    @classmethod
    def resolve(cls, asset_dir: Path | None, mass: float | None = None) -> "ChairSpec":
        """Staged asset if it exists, hardcoded SANDSBERG proxy otherwise."""
        if asset_dir is not None and (Path(asset_dir) / "object.json").is_file():
            return cls.from_asset(Path(asset_dir), mass=mass)
        spec = cls()
        if mass is not None:
            spec.mass = mass
        return spec

    def to_asset_frame(self, pose_canonical) -> "object":
        """Canonical chair pose -> the staged asset's own frame (what the sim spawns).

        Takes and returns a ``se3.Pose``; imported lazily to keep this module dependency-free.
        """
        from gear_sonic.trajopt.se3 import Pose  # local import: avoids a circular import

        rz = Pose(np.zeros(3), np.array(
            [[np.cos(self.asset_yaw), -np.sin(self.asset_yaw), 0.0],
             [np.sin(self.asset_yaw), np.cos(self.asset_yaw), 0.0],
             [0.0, 0.0, 1.0]]
        ))
        return pose_canonical * rz

    def describe(self) -> str:
        src = f"staged asset {self.asset_dir}" if self.asset_dir else "hardcoded SANDSBERG proxy"
        return (
            f"chair '{self.name}' ({src}): seat {self.seat_depth:.2f} x {self.seat_width:.2f} m "
            f"at {self.seat_height:.2f} m, total height {self.total_height:.2f} m, "
            f"mass {self.mass:.2f} kg"
        )
