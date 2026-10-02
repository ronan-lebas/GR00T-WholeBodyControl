"""Generate a staged object asset from built-in primitive specs (no CAD download needed).

The primitive-shape counterpart to ``prepare_object_asset.py``: it writes the exact same
directory layout, so the sim (``run_sim_loop.py --object-asset``), the recorder, the replay
visualizer and ``process_contacts.py`` all consume these objects unchanged::

    <out>/visual.obj              # every part concatenated (the replay loads one .obj)
    <out>/part_000.obj ...        # per-part visual, so the sim can color parts separately
    <out>/collision_000.stl ...   # one convex hull per part
    <out>/fixture_000.obj ...     # (task objects with a fixture) static props, see LAYOUTS
    <out>/object.json

plate / bar / handled_box force *bimanual* manipulation: each is too long or too wide for one
hand to control. The task objects (laptop, hammer, carton, pitcher, bottle, peg, pan, book,
drill) come with a scene layout (``LAYOUTS``): a spawn offset on the table and, for most, a static
fixture that defines the goal (nail block, bowl, hole block, stove, bookshelf, target). Only the
object is tracked; the fixture never moves.

The bar also has a catalog of geometric variants (``VARIANTS``) for collecting the same task
on different shapes. Variant 0 is the plain ``bar``; variant N is staged as ``bar_vN``.

Usage (needs gear_sonic[sim] — trimesh):
    python gear_sonic/scripts/make_primitive_asset.py all
    python gear_sonic/scripts/make_primitive_asset.py bar --length 0.6 --force
    python gear_sonic/scripts/make_primitive_asset.py bar --list-variants
    python gear_sonic/scripts/make_primitive_asset.py bar --variant 7 --force
    python gear_sonic/scripts/make_primitive_asset.py bar --variant all --force
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_DIR = REPO_ROOT / "data" / "objects"

CYLINDER_SECTIONS = 24


def _extra(part, rpy, density):
    """``rpy``: static-frame xyz Euler angles applied about the part center after ``axis``/``roll``.
    ``density`` is relative (1 = the unit density every other part uses)."""
    if rpy is not None:
        part["rpy"] = tuple(float(a) for a in rpy)
    if density != 1.0:
        part["density"] = float(density)
    return part


def _box(half, pos, rgba, rpy=None, density=1.0):
    part = {"shape": "box", "half": tuple(half), "pos": tuple(pos), "rgba": tuple(rgba)}
    return _extra(part, rpy, density)


def _cylinder(
    radius,
    length,
    axis,
    pos,
    rgba,
    sections=CYLINDER_SECTIONS,
    flat_down=False,
    rpy=None,
    density=1.0,
):
    """``radius`` is the circumradius; ``flat_down`` turns an x/y-axis prism onto a face."""
    part = {
        "shape": "cylinder",
        "radius": float(radius),
        "length": float(length),
        "axis": axis,
        "pos": tuple(pos),
        "rgba": tuple(rgba),
        "sections": int(sections),
        "roll": float(np.pi / sections) if flat_down else 0.0,
    }
    return _extra(part, rpy, density)


def _inradius(radius, sections):
    return radius * np.cos(np.pi / sections)


def _staves(r_bottom, height, flare, n, thickness, z0, rgba, center=(0.0, 0.0)):
    """Hollow flared wall (bowl, pan) as ``n`` slabs leaning out by ``flare`` rad, inner face
    starting at ``r_bottom`` on z = ``z0``: a revolved wall is not convex, so it is built from
    convex pieces like every other part."""
    r_top = r_bottom + height * np.tan(flare)
    width = 2 * np.pi * r_top / n * 1.05
    r_mid = r_bottom + 0.5 * height * np.tan(flare) + thickness / 2
    half = (thickness / 2, width / 2, height / np.cos(flare) / 2)
    parts = []
    for i in range(n):
        th = 2 * np.pi * i / n
        pos = (center[0] + r_mid * np.cos(th), center[1] + r_mid * np.sin(th), z0 + height / 2)
        parts.append(_box(half, pos, rgba, rpy=(0.0, flare, th)))
    return parts


# --- object specs. Each returns a list of convex parts, built with the object's bottom at
# --- z=0; _recenter() then moves the origin to the bbox center.


def spec_plate(length=0.40, width=0.18, thickness=0.015, rail=0.02):
    """Long flat board, long axis +x, on two rails.

    The rails are structural, not decoration: a 15 mm slab flat on the table leaves no room to
    get a finger underneath, so the plate would be ungraspable without them.
    """
    slab = (0.85, 0.85, 0.88, 1.0)
    dark = (0.30, 0.30, 0.36, 1.0)
    rail_y = width / 2 - rail / 2 - 0.01
    return [
        _box((length / 2, width / 2, thickness / 2), (0, 0, rail + thickness / 2), slab),
        _box((length / 2, rail / 2, rail / 2), (0, +rail_y, rail / 2), dark),
        _box((length / 2, rail / 2, rail / 2), (0, -rail_y, rail / 2), dark),
    ]


def spec_bar(
    length=0.50,
    thickness=0.04,
    cap=0.06,
    shaft="box",
    width=None,
    sections=6,
    cap_shape="cube",
    cap_neg=None,
    cap_height=None,
    cap_width=0.03,
    center_block=None,
):
    """Long bar, long axis +x, with end caps. Defaults: square 40 mm shaft, 60 mm cube caps.

    Square section so it cannot roll; the caps hold the shaft ~1 cm clear of the table (finger
    clearance) and stop a grasp from sliding off the ends.

    shaft: "box" (``thickness`` tall x ``width`` wide), "round" or "prism" (``sections`` sides);
    for round/prism ``thickness`` is the circumscribed diameter and a face rests down.
    cap_shape: "cube" (edge ``cap``), "puck" (vertical cylinder, diameter ``cap``, height
    ``cap_height``), "disc" (coaxial cylinder, diameter ``cap``, ``cap_width`` along x; rolls) or
    "none". ``cap_neg`` sizes the -x cap separately. Every part's bottom sits at z >= 0 and the
    shaft axis is at the smaller cap's center height (or resting on the table with no caps).
    """
    shaft_c = (0.90, 0.45, 0.10, 1.0)
    dark = (0.20, 0.20, 0.25, 1.0)
    width = thickness if width is None else width
    cap_neg = cap if cap_neg is None else cap_neg
    cap_height = cap if cap_height is None else cap_height
    r = thickness / 2

    if shaft == "box":
        shaft_bottom = thickness / 2
    elif shaft in ("round", "prism"):
        shaft_bottom = _inradius(r, CYLINDER_SECTIONS if shaft == "round" else sections)
    else:
        raise ValueError(f"unknown shaft '{shaft}'")

    def cap_center_z(size):
        return {"cube": size / 2, "puck": cap_height / 2, "disc": size / 2}[cap_shape]

    if cap_shape == "none":
        z = shaft_bottom
    else:
        z = min(cap_center_z(cap), cap_center_z(cap_neg))
    assert z >= shaft_bottom - 1e-9, "shaft would sink below the table; enlarge the caps"

    if shaft == "box":
        parts = [_box((length / 2, width / 2, thickness / 2), (0, 0, z), shaft_c)]
    else:
        n = CYLINDER_SECTIONS if shaft == "round" else sections
        parts = [_cylinder(r, length, "x", (0, 0, z), shaft_c, sections=n, flat_down=True)]

    for sign, size in ((+1, cap), (-1, cap_neg)):
        if cap_shape == "cube":
            parts.append(_box((size / 2,) * 3, (sign * (length / 2 - size / 2), 0, size / 2), dark))
        elif cap_shape == "puck":
            x = sign * (length / 2 - size / 2)
            parts.append(_cylinder(size / 2, cap_height, "z", (x, 0, cap_height / 2), dark))
        elif cap_shape == "disc":
            x = sign * (length / 2 - cap_width / 2)
            parts.append(
                _cylinder(size / 2, cap_width, "x", (x, 0, size / 2), dark, flat_down=True)
            )
        elif cap_shape != "none":
            raise ValueError(f"unknown cap_shape '{cap_shape}'")

    if center_block:
        parts.append(_box((center_block / 2,) * 3, (0, 0, z), dark))
    return parts


def spec_handled_box(
    length=0.30,
    width=0.20,
    height=0.15,
    handle_radius=0.015,
    handle_length=0.10,
    handle_gap=0.04,
    rim_height=0.03,
    rim_thickness=0.01,
):
    """Crate with a vertical grab handle on each +-y face and a raised rim on top."""
    body_c = (0.75, 0.60, 0.40, 1.0)
    handle_c = (0.10, 0.55, 0.90, 1.0)
    dark = (0.20, 0.20, 0.25, 1.0)
    rim_c = (0.55, 0.42, 0.26, 1.0)

    hx, hy, hz = length / 2, width / 2, height / 2
    handle_y = hy + handle_gap + handle_radius
    bracket_y = (hy + handle_y) / 2
    bracket_hy = (handle_y - hy) / 2
    bracket_dz = handle_length / 2 - 0.005

    parts = [
        _box((hx, hy, hz), (0, 0, hz), body_c),
        _cylinder(handle_radius, handle_length, "z", (0, +handle_y, hz), handle_c),
        _cylinder(handle_radius, handle_length, "z", (0, -handle_y, hz), handle_c),
    ]
    for sy in (+1, -1):
        for sz in (+1, -1):
            parts.append(
                _box(
                    (0.015, bracket_hy, 0.008),
                    (0, sy * bracket_y, hz + sz * bracket_dz),
                    dark,
                )
            )
    rim_z = height + rim_height / 2
    rt = rim_thickness / 2
    parts += [
        _box((hx, rt, rim_height / 2), (0, +(hy - rt), rim_z), rim_c),
        _box((hx, rt, rim_height / 2), (0, -(hy - rt), rim_z), rim_c),
        _box((rt, hy - rim_thickness, rim_height / 2), (+(hx - rt), 0, rim_z), rim_c),
        _box((rt, hy - rim_thickness, rim_height / 2), (-(hx - rt), 0, rim_z), rim_c),
    ]
    return parts


# --- task objects. Same frame convention: +x away from the robot, +y to its left. Fixture specs
# --- are static props placed next to the object; they are built centered in x/y with their
# --- bottom at z=0 (the table surface) and are never recentered.

STEEL = (0.70, 0.71, 0.74, 1.0)
BLACK = (0.08, 0.08, 0.09, 1.0)
WOOD = (0.72, 0.52, 0.32, 1.0)


def spec_laptop(depth=0.22, width=0.31, base_t=0.018, lid_t=0.007, open_deg=110.0, feet=0.004):
    """Open laptop, screen facing the robot, hinge on the far (+x) edge. One rigid body: the hinge
    is frozen at ``open_deg``. Rubber feet lift the base a few mm so a fingertip can catch its edge.
    """
    shell = (0.74, 0.75, 0.78, 1.0)
    hx, hy = depth / 2, width / 2
    top = feet + base_t
    parts = [_box((hx, hy, base_t / 2), (0, 0, feet + base_t / 2), shell)]
    for sx in (+1, -1):
        for sy in (+1, -1):
            parts.append(
                _cylinder(0.006, feet, "z", (sx * (hx - 0.02), sy * (hy - 0.03), feet / 2), BLACK)
            )
    parts += [
        _box((0.06, 0.13, 0.0005), (-0.02, 0, top + 0.0005), BLACK),  # keyboard
        _box((0.028, 0.045, 0.0005), (-0.085, 0, top + 0.0005), (0.55, 0.56, 0.6, 1.0)),  # trackpad
    ]
    th = np.deg2rad(open_deg)
    hinge = np.array([hx - 0.004, 0.0, top])
    d = np.array([-np.cos(th), 0.0, np.sin(th)])  # hinge -> lid top edge
    n = np.array([-np.sin(th), 0.0, -np.cos(th)])  # screen normal, toward the robot
    lid_c = hinge + d * depth / 2 - n * lid_t / 2
    parts += [
        _box((hx, hy, lid_t / 2), tuple(lid_c), shell, rpy=(0, th, 0)),
        _box((hx - 0.012, hy - 0.012, 0.0005), tuple(lid_c + n * (lid_t / 2 + 0.0005)),
             (0.10, 0.20, 0.45, 1.0), rpy=(0, th, 0)),  # screen
        _box((0.025, 0.025, 0.0005), tuple(lid_c - n * (lid_t / 2 + 0.0005)),
             (0.95, 0.95, 0.95, 1.0), rpy=(0, th, 0)),  # logo, shows the lid's back in the ego view
        _cylinder(0.007, width - 0.06, "y", tuple(hinge + [0, 0, 0.002]), (0.3, 0.3, 0.32, 1.0)),
    ]
    return parts


def spec_hammer(handle_len=0.30, handle_d=0.03, head=0.05, head_len=0.10, head_density=8.0):
    """Claw hammer lying on its side, handle toward the robot. The head and the end knob hold the
    handle ~1 cm off the table. The head is dense, so the COM sits near it like a real hammer."""
    grip_c = (0.12, 0.12, 0.13, 1.0)
    handle_c = (0.80, 0.62, 0.38, 1.0)
    z = head / 2
    head_x = handle_len / 2
    parts = [
        _cylinder(handle_d / 2, handle_len, "x", (0, 0, z), handle_c),
        _cylinder(handle_d / 2 + 0.002, 0.12, "x", (-handle_len / 2 + 0.07, 0, z), grip_c),
        _cylinder(z, 0.015, "x", (-handle_len / 2 + 0.0075, 0, z), grip_c, flat_down=True),
        _box((head / 2, head_len / 2, head / 2), (head_x, 0, z), STEEL, density=head_density),
        _cylinder(0.02, 0.03, "y", (head_x, head_len / 2 + 0.015, z), STEEL, density=head_density),
    ]
    for sz in (+1, -1):  # the two claw prongs, curving back toward the handle
        parts.append(
            _box((0.007, 0.03, 0.007), (head_x - 0.008, -head_len / 2 - 0.025, z + sz * 0.011),
                 STEEL, rpy=(0, 0, -0.45), density=head_density)
        )
    return parts


def fixture_nail_block():
    return [
        _box((0.07, 0.06, 0.03), (0, 0, 0.03), WOOD),
        _cylinder(0.003, 0.04, "z", (0, 0, 0.08), STEEL),
        _cylinder(0.011, 0.004, "z", (0, 0, 0.102), STEEL),
    ]


def spec_carton(length=0.24, width=0.36, height=0.18):
    """Closed, taped cardboard box with no handles: too wide for one hand, so it is lifted by
    squeezing it between both palms. A label marks the robot-facing side."""
    card = (0.62, 0.45, 0.28, 1.0)
    tape = (0.82, 0.72, 0.52, 1.0)
    hx, hy, hz = length / 2, width / 2, height / 2
    return [
        _box((hx, hy, hz), (0, 0, hz), card),
        _box((0.025, hy + 0.0005, 0.0005), (0, 0, height + 0.0005), tape),
        _box((0.025, 0.0005, 0.04), (0, hy + 0.0005, height - 0.04), tape),
        _box((0.025, 0.0005, 0.04), (0, -hy - 0.0005, height - 0.04), tape),
        _box((0.0005, 0.05, 0.03), (-hx - 0.0005, 0.07, hz + 0.02), (0.85, 0.2, 0.2, 1.0)),
    ]


def spec_pitcher(radius=0.05, height=0.16, handle_gap=0.03):
    """Water pitcher: spout on +y, a vertical loop handle on -y (the robot's right hand) with a
    ``handle_gap`` finger gap. Solid body: nothing goes inside, it only has to be tilted."""
    body_c = (0.55, 0.75, 0.90, 1.0)
    dark = (0.12, 0.25, 0.45, 1.0)
    bar_y = -(radius + handle_gap + 0.01)
    bracket_hy = (handle_gap + 0.005) / 2
    parts = [
        _cylinder(radius, height, "z", (0, 0, height / 2), body_c),
        _cylinder(radius + 0.003, 0.012, "z", (0, 0, height - 0.006), body_c),
        _cylinder(radius - 0.005, 0.001, "z", (0, 0, height + 0.0005), (0.1, 0.2, 0.35, 1.0)),
        _box((0.018, 0.02, 0.008), (0, radius + 0.008, height - 0.008), body_c, rpy=(0.45, 0, 0)),
        _box((0.011, 0.01, 0.05), (0, bar_y, height / 2 + 0.005), dark),
    ]
    for z in (height / 2 + 0.047, height / 2 - 0.037):
        parts.append(_box((0.011, bracket_hy, 0.008), (0, -(radius - 0.005 + bracket_hy), z), dark))
    return parts


def fixture_bowl():
    ceramic = (0.93, 0.91, 0.86, 1.0)
    return [
        _cylinder(0.062, 0.008, "z", (0, 0, 0.004), ceramic),
        *_staves(0.06, 0.06, 0.45, 16, 0.006, 0.008, ceramic),
    ]


def spec_bottle(radius=0.0375, body_len=0.19, sections=16):
    """Bottle lying on its side (cap +x). The label ring is the lowest hull, so it rests on the
    label; the faceted section keeps it from rolling away on its own."""
    glass = (0.20, 0.50, 0.35, 1.0)
    label_r = radius + 0.0008
    z = _inradius(label_r, sections)
    x0 = -body_len / 2
    return [
        _cylinder(radius, body_len, "x", (0, 0, z), glass, sections=sections, flat_down=True),
        _cylinder(label_r, 0.09, "x", (x0 + 0.08, 0, z), (0.95, 0.90, 0.75, 1.0),
                  sections=sections, flat_down=True),
        _cylinder(0.029, 0.025, "x", (x0 + body_len + 0.0125, 0, z), glass),
        _cylinder(0.016, 0.05, "x", (x0 + body_len + 0.05, 0, z), glass),
        _cylinder(0.018, 0.018, "x", (x0 + body_len + 0.084, 0, z), (0.80, 0.10, 0.10, 1.0)),
    ]


def spec_peg(radius=0.02, length=0.12, knob=0.06):
    """Round peg lying on its side with a cube knob at +x. The knob is wider than the hole, so it
    is the insertion stop, and it holds the shaft ~1 cm off the table."""
    return [
        _cylinder(radius, length, "x", (-knob / 2, 0, knob / 2), (0.85, 0.70, 0.45, 1.0)),
        _box((knob / 2,) * 3, (length / 2, 0, knob / 2), (0.80, 0.20, 0.15, 1.0)),
    ]


def fixture_hole_block(outer=0.14, hole=0.052, height=0.08):
    """Through-hole block: the inserted peg stands on the table. Square hole, round peg, so yaw
    doesn't matter; 12 mm of total clearance."""
    blue = (0.20, 0.35, 0.70, 1.0)
    w = (outer - hole) / 2
    hz = height / 2
    return [
        _box((outer / 2, w / 2, hz), (0, +(hole + w) / 2, hz), blue),
        _box((outer / 2, w / 2, hz), (0, -(hole + w) / 2, hz), blue),
        _box((w / 2, hole / 2, hz), (+(hole + w) / 2, 0, hz), blue),
        _box((w / 2, hole / 2, hz), (-(hole + w) / 2, 0, hz), blue),
    ]


def spec_pan(radius=0.10, wall=0.045, handle_len=0.18, handle_tilt=0.17):
    """Frying pan, handle toward the robot and rising at ``handle_tilt`` rad, so fingers fit
    under it. Hollow: bottom disc + 16 flared wall slabs."""
    iron = (0.25, 0.25, 0.27, 1.0)
    bottom_t = 0.006
    start = np.array([-(radius + 0.008), 0.0, bottom_t + wall - 0.008])
    d = np.array([-np.cos(handle_tilt), 0.0, np.sin(handle_tilt)])
    return [
        _cylinder(radius, bottom_t, "z", (0, 0, bottom_t / 2), iron),
        *_staves(radius - 0.004, wall, 0.25, 16, 0.004, bottom_t, iron),
        _box((handle_len / 2, 0.0125, 0.007), tuple(start + d * handle_len / 2), BLACK,
             rpy=(0, handle_tilt, 0)),
    ]


def fixture_stove():
    return [
        _box((0.14, 0.14, 0.01), (0, 0, 0.01), (0.15, 0.15, 0.16, 1.0)),
        _cylinder(0.09, 0.003, "z", (0, 0, 0.0215), (0.80, 0.15, 0.10, 1.0)),
        _cylinder(0.075, 0.004, "z", (0, 0, 0.022), BLACK),
    ]


def spec_book(depth=0.17, height=0.24, thickness=0.036, cover=0.0015):
    """Hardcover book lying flat, spine toward the robot. The covers overhang the pages by 5 mm."""
    red = (0.55, 0.10, 0.12, 1.0)
    pages_t = thickness - 2 * cover
    hx, hy = depth / 2, height / 2
    return [
        _box((hx, hy, cover / 2), (0, 0, cover / 2), red),
        _box((hx, hy, cover / 2), (0, 0, thickness - cover / 2), red),
        _box((hx - 0.005, hy - 0.005, pages_t / 2), (0.003, 0, thickness / 2),
             (0.95, 0.93, 0.85, 1.0)),
        _box((cover / 2, hy, thickness / 2), (-hx + cover / 2, 0, thickness / 2),
             (0.40, 0.07, 0.09, 1.0)),
        _box((0.035, 0.06, 0.0003), (0.01, 0, thickness + 0.0003), (0.85, 0.70, 0.25, 1.0)),
    ]


def fixture_bookshelf():
    """Base board with an end wall at +y and two standing books; the free space on -y is where
    the book goes."""
    base_t = 0.006
    return [
        _box((0.08, 0.15, base_t / 2), (0, 0, base_t / 2), WOOD),
        _box((0.08, 0.004, 0.08), (0, 0.146, base_t + 0.08), WOOD),
        _box((0.08, 0.0175, 0.10), (0, 0.1245, base_t + 0.10), (0.15, 0.30, 0.60, 1.0)),
        _box((0.075, 0.015, 0.09), (0, 0.091, base_t + 0.09), (0.20, 0.50, 0.25, 1.0)),
    ]


def spec_drill(body_r=0.03, body_len=0.16, thick=0.035):
    """Cordless drill lying on its side, bit +x, pistol grip toward the robot's right hand (-y).
    The battery is the thickest part, so it and the motor housing hold the grip ~2 cm off the
    table. A red trigger marks where the index finger goes."""
    yellow = (0.95, 0.75, 0.10, 1.0)
    z = thick
    return [
        _cylinder(body_r, body_len, "x", (0, 0, z), yellow),
        _cylinder(0.018, 0.04, "x", (body_len / 2 + 0.02, 0, z), (0.3, 0.3, 0.32, 1.0)),
        _cylinder(0.004, 0.05, "x", (body_len / 2 + 0.065, 0, z), STEEL),
        _box((0.02, 0.05, 0.016), (-0.03, -0.075, z), BLACK, rpy=(0, 0, -0.25)),
        _box((0.006, 0.008, 0.008), (-0.003, -0.042, z), (0.85, 0.10, 0.10, 1.0)),
        _box((0.05, 0.03, thick), (-0.055, -0.14, z), BLACK, density=2.0),
    ]


def fixture_target():
    """Standing target board, bullseye facing the robot (-x)."""
    parts = [
        _box((0.06, 0.10, 0.005), (0, 0, 0.005), WOOD),
        _box((0.005, 0.10, 0.10), (0.03, 0, 0.11), (0.95, 0.95, 0.95, 1.0)),
    ]
    for r, c in ((0.07, (0.85, 0.1, 0.1, 1.0)), (0.045, (0.95, 0.95, 0.95, 1.0)),
                 (0.02, (0.85, 0.1, 0.1, 1.0))):
        parts.append(_cylinder(r, 0.002, "x", (0.024 - (0.07 - r) * 0.04, 0, 0.11), c))
    return parts


SPECS = {
    "plate": (spec_plate, 0.4),
    "bar": (spec_bar, 0.5),
    "handled_box": (spec_handled_box, 0.6),
    "laptop": (spec_laptop, 0.8),
    "hammer": (spec_hammer, 0.45),
    "carton": (spec_carton, 0.5),
    "pitcher": (spec_pitcher, 0.35),
    "bottle": (spec_bottle, 0.35),
    "peg": (spec_peg, 0.2),
    "pan": (spec_pan, 0.45),
    "book": (spec_book, 0.5),
    "drill": (spec_drill, 0.6),
}

# Per-object scene layout, written into object.json. ``spawn_offset`` shifts the default tabletop
# spawn (x, y); ``fixture`` is (spec, (dx, dy) from the object's spawn position in world axes);
# ``sliding_friction`` replaces the sim's grip-tuned 2.0 (the object's contact params win against
# both the fingers and the table, so a lower value is what makes sliding on the table possible).
LAYOUTS = {
    "laptop": dict(spawn_offset=(0.06, 0.0), sliding_friction=0.8),
    "hammer": dict(spawn_offset=(0.0, -0.12), fixture=(fixture_nail_block, (0.06, 0.26))),
    "pitcher": dict(spawn_offset=(0.12, -0.10), fixture=(fixture_bowl, (0.0, 0.25))),
    "bottle": dict(spawn_yaw=float(np.pi / 2)),
    "peg": dict(spawn_offset=(0.0, -0.10), fixture=(fixture_hole_block, (0.05, 0.24))),
    "pan": dict(spawn_offset=(0.02, -0.12), fixture=(fixture_stove, (0.09, 0.30))),
    "book": dict(spawn_offset=(-0.06, -0.10), fixture=(fixture_bookshelf, (0.08, 0.28))),
    "drill": dict(spawn_offset=(-0.02, 0.0), fixture=(fixture_target, (0.28, 0.22))),
}

# (description, spec kwargs). Index = variant id, 0 = the plain object. Keep every bar within the
# table's 0.60 m x-extent and well under the reset-pose hands (~0.15 m above the tabletop).
# Append new variants at the end: recorded datasets are labeled by index.
VARIANTS = {
    "bar": [
        ("baseline: 40 mm square shaft, 60 mm cube caps, 0.50 m", {}),
        ("round shaft dia 40, cube caps 60", dict(shaft="round")),
        ("plain cylinder dia 40, no caps (rolls)", dict(shaft="round", cap_shape="none")),
        (
            "plain thick cylinder dia 60, no caps (rolls)",
            dict(shaft="round", thickness=0.06, cap_shape="none"),
        ),
        ("thick square shaft 55 mm, cube caps 75", dict(thickness=0.055, cap=0.075)),
        ("thin square shaft 28 mm, cube caps 50", dict(thickness=0.028, cap=0.05)),
        ("thin round shaft dia 28, cube caps 50", dict(shaft="round", thickness=0.028, cap=0.05)),
        (
            "thick round shaft dia 55, cube caps 75",
            dict(shaft="round", thickness=0.055, cap=0.075),
        ),
        ("hexagonal shaft dia 45, cube caps 60", dict(shaft="prism", sections=6, thickness=0.045)),
        (
            "triangular shaft (60 mm circumdia), cube caps 65",
            dict(shaft="prism", sections=3, thickness=0.06, cap=0.065),
        ),
        ("flat shaft 60 wide x 25 tall, cube caps 70", dict(width=0.06, thickness=0.025, cap=0.07)),
        ("long: 0.58 m, baseline section", dict(length=0.58)),
        ("short: 0.40 m, baseline section", dict(length=0.40)),
        ("plain square rod 40 mm, no caps", dict(cap_shape="none")),
        (
            "round shaft dia 40, vertical pucks dia 70 x 60",
            dict(shaft="round", cap_shape="puck", cap=0.07, cap_height=0.06),
        ),
        (
            "barbell: round shaft dia 35, coaxial discs dia 90 (rolls)",
            dict(shaft="round", thickness=0.035, cap_shape="disc", cap=0.09),
        ),
        (
            "uneven cube caps 75 (+x) / 50 (-x), 35 mm shaft",
            dict(thickness=0.035, cap=0.075, cap_neg=0.05),
        ),
        ("baseline + 60 mm center block", dict(center_block=0.06)),
    ],
}


def variant_name(name: str, variant: int) -> str:
    return name if variant == 0 else f"{name}_v{variant}"


def _part_mesh(part):
    if part["shape"] == "box":
        mesh = trimesh.creation.box(extents=[2 * h for h in part["half"]])
    else:
        mesh = trimesh.creation.cylinder(
            radius=part["radius"], height=part["length"], sections=part["sections"]
        )
        if part["roll"]:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(part["roll"], [0, 0, 1]))
        if part["axis"] == "x":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif part["axis"] == "y":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if part.get("rpy"):
        mesh.apply_transform(trimesh.transformations.euler_matrix(*part["rpy"], "sxyz"))
    mesh.apply_translation(part["pos"])
    return mesh


def _part_inertia(part):
    """``(mass, com, 3x3 inertia about the part's own center)`` at the part's relative density."""
    rho = part.get("density", 1.0)
    if part["shape"] == "box":
        a, b, c = part["half"]
        m = 8 * a * b * c
        inertia = np.diag(m / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b]))
    elif part["sections"] != CYLINDER_SECTIONS:
        # Coarse prism: the circular-cylinder formula would be off, integrate the mesh instead.
        mesh = _part_mesh(part)
        return (
            rho * float(mesh.volume),
            np.asarray(mesh.center_mass),
            rho * np.asarray(mesh.moment_inertia),
        )
    else:
        r, ln = part["radius"], part["length"]
        m = np.pi * r * r * ln
        i_axial = 0.5 * m * r * r
        i_trans = m * (3 * r * r + ln * ln) / 12.0
        order = {"x": [i_axial, i_trans, i_trans], "y": [i_trans, i_axial, i_trans]}
        inertia = np.diag(order.get(part["axis"], [i_trans, i_trans, i_axial]))
    if part.get("rpy"):
        rot = trimesh.transformations.euler_matrix(*part["rpy"], "sxyz")[:3, :3]
        inertia = rot @ inertia @ rot.T
    return rho * m, np.asarray(part["pos"], dtype=float), rho * inertia


def _rigid_body_properties(parts):
    """Unit-density ``(mass, com, fullinertia)`` of the part union, by parallel-axis summation.

    Parts that overlap are double-counted, which is negligible here and irrelevant downstream:
    the sim rescales this tensor linearly to the requested spawn mass.
    """
    total_m = 0.0
    weighted = np.zeros(3)
    inertia_o = np.zeros((3, 3))
    for part in parts:
        m, d, i_c = _part_inertia(part)
        total_m += m
        weighted += m * d
        inertia_o += i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    com = weighted / total_m
    inertia_com = inertia_o - total_m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    fullinertia = [
        inertia_com[0, 0],
        inertia_com[1, 1],
        inertia_com[2, 2],
        inertia_com[0, 1],
        inertia_com[0, 2],
        inertia_com[1, 2],
    ]
    return total_m, com, fullinertia


def _recenter(parts):
    """Shift every part so the union's bbox center is the body origin."""
    meshes = [_part_mesh(p) for p in parts]
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    shift = -(lo + hi) / 2.0
    for p in parts:
        p["pos"] = tuple(np.asarray(p["pos"], dtype=float) + shift)
    return parts


def make(
    name: str,
    out: Path,
    mass: float | None = None,
    force: bool = False,
    variant: int | None = None,
    **kwargs,
) -> Path:
    """``variant`` picks a ``VARIANTS`` entry (None = 0 for objects that have a catalog);
    explicit dimension ``kwargs`` override the variant's."""
    if name not in SPECS:
        raise KeyError(f"unknown object '{name}' (have {sorted(SPECS)})")
    if variant is not None and name not in VARIANTS:
        raise KeyError(f"object '{name}' has no variants (only {sorted(VARIANTS)})")
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists (pass --force to overwrite)")

    spec_fn, default_mass = SPECS[name]
    variant_desc = None
    if name in VARIANTS:
        variant = 0 if variant is None else variant
        if not 0 <= variant < len(VARIANTS[name]):
            raise IndexError(f"'{name}' has variants 0..{len(VARIANTS[name]) - 1}, not {variant}")
        variant_desc, variant_kwargs = VARIANTS[name][variant]
        kwargs = {**variant_kwargs, **{k: v for k, v in kwargs.items() if v is not None}}
    spec_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    parts = _recenter(spec_fn(**spec_kwargs))
    meshes = [_part_mesh(p) for p in parts]

    out.mkdir(parents=True, exist_ok=True)
    collision_names, visual_parts = [], []
    for i, (part, mesh) in enumerate(zip(parts, meshes)):
        col = f"collision_{i:03d}.stl"
        vis = f"part_{i:03d}.obj"
        mesh.export(out / col)
        mesh.export(out / vis)
        collision_names.append(col)
        visual_parts.append({"mesh": vis, "rgba": " ".join(f"{v:g}" for v in part["rgba"])})

    combined = trimesh.util.concatenate(meshes)
    combined.export(out / "visual.obj")
    lo, hi = combined.bounds
    unit_mass, com, fullinertia = _rigid_body_properties(parts)

    label = name if variant is None else variant_name(name, variant)
    source = f"make_primitive_asset.py {name}" + (
        "" if variant is None else f" --variant {variant}"
    )
    meta = {
        "name": label,
        "source": source,
        "visual_mesh": "visual.obj",
        "visual_parts": visual_parts,
        "collision_meshes": collision_names,
        # Unit-density mass/inertia; the sim rescales both to spawn_mass (or --object-mass).
        "mass": float(unit_mass),
        "com": [float(v) for v in com],
        "fullinertia": [float(v) for v in fullinertia],
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "z_min": float(lo[2]),
        "spawn_yaw": 0.0,
        "spawn_mass": float(default_mass if mass is None else mass),
    }
    if variant is not None:
        defaults = {
            k: p.default
            for k, p in inspect.signature(spec_fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        meta["variant"] = variant
        meta["variant_desc"] = variant_desc
        meta["spec"] = {**defaults, **spec_kwargs}
    layout = LAYOUTS.get(name, {})
    for key in ("spawn_yaw", "spawn_offset", "sliding_friction"):
        if key in layout:
            meta[key] = layout[key]
    if "fixture" in layout:
        meta["fixture"] = _stage_fixture(out, *layout["fixture"])
    (out / "object.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[make_primitive_asset] {label} -> {out}")
    if variant_desc:
        print(f"  variant {variant}: {variant_desc}")
    print(f"  {len(parts)} convex parts, bbox extents {(hi - lo).round(3).tolist()} m")
    print(f"  spawn mass {meta['spawn_mass']} kg, z_min {meta['z_min']:.3f} m")
    if "fixture" in meta:
        fx = meta["fixture"]
        print(f"  fixture '{fx['name']}': {len(fx['parts'])} parts at offset {fx['offset']} m")
    return out


def _stage_fixture(out: Path, spec_fn, offset) -> dict:
    """Export a static fixture's parts (one convex mesh each, used for both collision and
    render) and return its ``object.json["fixture"]`` entry. The fixture origin is its footprint
    center on the table surface; ``offset`` is from the object's spawn position, in world axes."""
    parts = spec_fn()
    meshes = [_part_mesh(p) for p in parts]
    entries = []
    for i, (part, mesh) in enumerate(zip(parts, meshes)):
        fname = f"fixture_{i:03d}.obj"
        mesh.export(out / fname)
        entries.append({"mesh": fname, "rgba": " ".join(f"{v:g}" for v in part["rgba"])})
    lo, hi = trimesh.util.concatenate(meshes).bounds
    return {
        "name": spec_fn.__name__.removeprefix("fixture_"),
        "offset": [float(v) for v in offset],
        "yaw": 0.0,
        "parts": entries,
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("name", choices=sorted(SPECS) + ["all"])
    p.add_argument(
        "--variant",
        default=None,
        help=f"variant index or 'all' (objects with a catalog: {sorted(VARIANTS)})",
    )
    p.add_argument("--list-variants", action="store_true", help="print the variant catalog")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"staged output dir (default: {DEFAULT_STAGE_DIR}/<name>[_v<variant>])",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    p.add_argument("--mass", type=float, default=None, help="spawn mass in kg")
    p.add_argument("--length", type=float, default=None, help="x extent in meters")
    p.add_argument("--width", type=float, default=None, help="y extent (plate, handled_box)")
    p.add_argument("--height", type=float, default=None, help="z extent (handled_box)")
    p.add_argument("--thickness", type=float, default=None, help="slab / bar section thickness")
    args = p.parse_args()

    names = sorted(SPECS) if args.name == "all" else [args.name]
    if args.list_variants:
        for name in names:
            for i, (desc, _) in enumerate(VARIANTS.get(name, [])):
                print(f"{name} {i:2d}  {variant_name(name, i):<10s} {desc}")
        return

    variants = [None]
    if args.variant is not None:
        if args.name not in VARIANTS:
            p.error(f"--variant needs an object with a catalog: {sorted(VARIANTS)}")
        if args.variant == "all":
            variants = list(range(len(VARIANTS[args.name])))
        else:
            try:
                variants = [int(args.variant)]
            except ValueError:
                p.error(f"--variant must be an integer or 'all', got '{args.variant}'")
            if not 0 <= variants[0] < len(VARIANTS[args.name]):
                p.error(f"'{args.name}' has variants 0..{len(VARIANTS[args.name]) - 1}")
    if args.out is not None and len(names) * len(variants) > 1:
        p.error("--out takes a single object directory; drop it when generating several")
    for name, variant in ((n, v) for n in names for v in variants):
        spec_fn = SPECS[name][0]
        accepted = spec_fn.__code__.co_varnames[: spec_fn.__code__.co_argcount]
        dims = {
            k: v
            for k, v in (
                ("length", args.length),
                ("width", args.width),
                ("height", args.height),
                ("thickness", args.thickness),
            )
            if v is not None and k in accepted
        }
        label = name if variant is None else variant_name(name, variant)
        out = args.out if args.out is not None else DEFAULT_STAGE_DIR / label
        make(name, out.resolve(), mass=args.mass, force=args.force, variant=variant, **dims)


if __name__ == "__main__":
    main()
