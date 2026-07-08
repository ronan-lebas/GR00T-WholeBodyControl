"""Shared appearance for the manipulation cube: a distinct color per face.

A uniform-gray cube is rotationally ambiguous — both to the eye and, more importantly,
to FoundationPose, whose photometric term falls back to a flat gray when the mesh has no
colors (``third_party/FoundationPose/Utils.py``), giving it no signal to resolve
orientation. Coloring each face fixes that.

This module is the single source of truth for the face->color map so the two consumers
stay consistent:
  - the MuJoCo sim paints the box faces with thin colored slab geoms (``base_sim.py``);
  - the exported ``box.obj`` given to FoundationPose is written with matching per-face
    vertex colors (``foundation_pose_writer.py``).

Because the sim render and the CAD mesh use the *same* axis->color map, FoundationPose's
render-and-compare can lock onto the orientation.
"""

from __future__ import annotations

from pathlib import Path

# Face order: +x, -x, +y, -y, +z, -z. Each entry is (axis, sign).
_FACES: tuple[tuple[int, int], ...] = (
    (0, +1),
    (0, -1),
    (1, +1),
    (1, -1),
    (2, +1),
    (2, -1),
)

# One saturated, well-separated RGB color per face (values in [0, 1]), same order as
# ``_FACES``. Kept maximally distinct so each face is unambiguous under sim lighting.
BOX_FACE_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),  # +x  red
    (0.0, 1.0, 0.0),  # -x  green
    (0.0, 0.0, 1.0),  # +y  blue
    (1.0, 1.0, 0.0),  # -y  yellow
    (1.0, 0.0, 1.0),  # +z  magenta
    (0.0, 1.0, 1.0),  # -z  cyan
)


def _face_corners(axis: int, sign: int, half: tuple[float, float, float]):
    """Return the 4 corners of one axis-aligned face, wound so the normal points outward.

    ``axis`` is the face normal axis (0/1/2 = x/y/z), ``sign`` its direction (+1/-1).
    The two in-plane axes are traversed to give a counter-clockwise (outward) winding.
    """
    a = axis
    u, v = (a + 1) % 3, (a + 2) % 3
    hu, hv = half[u], half[v]
    # In-plane corner offsets; reverse order for the -axis face to keep outward winding.
    plane = [(-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv)]
    if sign < 0:
        plane = [(-du, dv) for du, dv in plane]  # mirror across u -> flips winding to outward
    corners = []
    for du, dv in plane:
        p = [0.0, 0.0, 0.0]
        p[a] = sign * half[a]
        p[u] = du
        p[v] = dv
        corners.append(tuple(p))
    return corners


def write_colored_box_obj(path, half_extents) -> None:
    """Write a centered axis-aligned box mesh (meters) with one flat color per face.

    Uses the ``v x y z r g b`` OBJ vertex-color extension (read by trimesh into
    ``mesh.visual.vertex_colors``). Each face gets its own 4 vertices so its color stays
    flat instead of blending into neighbors at shared corners — 24 vertices, 12 triangles.
    """
    hx, hy, hz = (float(s) for s in half_extents)
    half = (hx, hy, hz)

    vlines: list[str] = []
    flines: list[str] = []
    for fi, (axis, sign) in enumerate(_FACES):
        r, g, b = BOX_FACE_COLORS[fi]
        for x, y, z in _face_corners(axis, sign, half):
            vlines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}")
        base = fi * 4 + 1  # 1-indexed OBJ vertex ids for this face's 4 corners
        flines.append(f"f {base} {base + 1} {base + 2}")
        flines.append(f"f {base} {base + 2} {base + 3}")

    Path(path).write_text("\n".join(vlines + flines) + "\n")


def box_face_slabs(half_extents, thickness: float = 0.0005):
    """Thin colored slab geoms that paint each face of a box in the sim.

    Returns a list of ``(pos, size, rgba)`` for MuJoCo ``<geom type="box">`` visual-only
    slabs, one per face, colored to match :func:`write_colored_box_obj`. Each slab is
    inset by ``thickness`` so its outer surface sits flush with the box face (no
    protrusion) while fully covering it.
    """
    half = tuple(float(s) for s in half_extents)
    slabs = []
    for fi, (axis, sign) in enumerate(_FACES):
        size = list(half)
        size[axis] = thickness / 2.0
        pos = [0.0, 0.0, 0.0]
        pos[axis] = sign * (half[axis] - thickness / 2.0)
        r, g, b = BOX_FACE_COLORS[fi]
        slabs.append((tuple(pos), tuple(size), (r, g, b, 1.0)))
    return slabs
