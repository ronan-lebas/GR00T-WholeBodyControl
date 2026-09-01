"""Generate a staged object asset from built-in primitive specs (no CAD download needed).

The primitive-shape counterpart to ``prepare_object_asset.py``: it writes the exact same
directory layout, so the sim (``run_sim_loop.py --object-asset``), the recorder, the replay
visualizer and ``process_contacts.py`` all consume these objects unchanged::

    <out>/visual.obj              # every part concatenated (the replay loads one .obj)
    <out>/part_000.obj ...        # per-part visual, so the sim can color parts separately
    <out>/collision_000.stl ...   # one convex hull per part
    <out>/object.json

The three objects here exist to force *bimanual* manipulation: each is too long or too wide
for one hand to control, so the operator has to coordinate both BrainCo hands.

Usage (needs gear_sonic[sim] — trimesh):
    python gear_sonic/scripts/make_primitive_asset.py all
    python gear_sonic/scripts/make_primitive_asset.py bar --length 0.6 --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_DIR = REPO_ROOT / "data" / "objects"

CYLINDER_SECTIONS = 24


def _box(half, pos, rgba):
    return {"shape": "box", "half": tuple(half), "pos": tuple(pos), "rgba": tuple(rgba)}


def _cylinder(radius, length, axis, pos, rgba):
    return {
        "shape": "cylinder",
        "radius": float(radius),
        "length": float(length),
        "axis": axis,
        "pos": tuple(pos),
        "rgba": tuple(rgba),
    }


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


def spec_bar(length=0.50, thickness=0.04, cap=0.06):
    """Long bar, long axis +x, with cubic end caps.

    Square section so it cannot roll; the caps hold the shaft ~1 cm clear of the table (finger
    clearance) and stop a grasp from sliding off the ends.
    """
    shaft = (0.90, 0.45, 0.10, 1.0)
    dark = (0.20, 0.20, 0.25, 1.0)
    return [
        _box((length / 2, thickness / 2, thickness / 2), (0, 0, cap / 2), shaft),
        _box((cap / 2, cap / 2, cap / 2), (+(length / 2 - cap / 2), 0, cap / 2), dark),
        _box((cap / 2, cap / 2, cap / 2), (-(length / 2 - cap / 2), 0, cap / 2), dark),
    ]


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


SPECS = {
    "plate": (spec_plate, 0.4),
    "bar": (spec_bar, 0.5),
    "handled_box": (spec_handled_box, 0.6),
}


def _part_mesh(part):
    if part["shape"] == "box":
        mesh = trimesh.creation.box(extents=[2 * h for h in part["half"]])
    else:
        mesh = trimesh.creation.cylinder(
            radius=part["radius"], height=part["length"], sections=CYLINDER_SECTIONS
        )
        if part["axis"] == "x":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif part["axis"] == "y":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_translation(part["pos"])
    return mesh


def _part_inertia(part):
    """Unit-density ``(mass, com, 3x3 inertia about the part's own center)``."""
    if part["shape"] == "box":
        a, b, c = part["half"]
        m = 8 * a * b * c
        diag = m / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    else:
        r, ln = part["radius"], part["length"]
        m = np.pi * r * r * ln
        i_axial = 0.5 * m * r * r
        i_trans = m * (3 * r * r + ln * ln) / 12.0
        order = {"x": [i_axial, i_trans, i_trans], "y": [i_trans, i_axial, i_trans]}
        diag = np.array(order.get(part["axis"], [i_trans, i_trans, i_axial]))
    return m, np.asarray(part["pos"], dtype=float), np.diag(diag)


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


def make(name: str, out: Path, mass: float | None = None, force: bool = False, **kwargs) -> Path:
    if name not in SPECS:
        raise KeyError(f"unknown object '{name}' (have {sorted(SPECS)})")
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists (pass --force to overwrite)")

    spec_fn, default_mass = SPECS[name]
    parts = _recenter(spec_fn(**{k: v for k, v in kwargs.items() if v is not None}))
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

    meta = {
        "name": name,
        "source": f"make_primitive_asset.py {name}",
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
    (out / "object.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[make_primitive_asset] {name} -> {out}")
    print(f"  {len(parts)} convex parts, bbox extents {(hi - lo).round(3).tolist()} m")
    print(f"  spawn mass {meta['spawn_mass']} kg, z_min {meta['z_min']:.3f} m")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("name", choices=sorted(SPECS) + ["all"])
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"staged output dir (default: {DEFAULT_STAGE_DIR}/<name>)",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    p.add_argument("--mass", type=float, default=None, help="spawn mass in kg")
    p.add_argument("--length", type=float, default=None, help="x extent in meters")
    p.add_argument("--width", type=float, default=None, help="y extent (plate, handled_box)")
    p.add_argument("--height", type=float, default=None, help="z extent (handled_box)")
    p.add_argument("--thickness", type=float, default=None, help="slab / bar section thickness")
    args = p.parse_args()

    names = sorted(SPECS) if args.name == "all" else [args.name]
    if args.out is not None and len(names) > 1:
        p.error("--out takes a single object directory; drop it when generating 'all'")
    for name in names:
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
        out = args.out if args.out is not None else DEFAULT_STAGE_DIR / name
        make(name, out.resolve(), mass=args.mass, force=args.force, **dims)


if __name__ == "__main__":
    main()
