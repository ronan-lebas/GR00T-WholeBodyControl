"""Stage an IKEA_interface asset into a MuJoCo-ready object directory for the sim.

``IKEA_interface`` (https://github.com/leggedrobotics/IKEA_interface) downloads an IKEA product
and emits a single-link URDF plus ``visual.stl`` and CoACD convex ``collision_*.stl`` (meters,
Z-up, resting on z=0). This script repackages one such directory into the self-contained layout
``run_sim_loop.py --chair`` expects::

    <out>/visual.obj              # visual.stl re-exported as OBJ (the replay visualizer loads
                                  # a single .obj, so OBJ keeps sim and replay on one mesh)
    <out>/collision_000.stl ...   # copied verbatim; each is already convex
    <out>/object.json             # mass / com / inertia / bbox metadata (see below)

Doing the conversion here rather than at sim startup keeps ``run_sim_loop.py`` free of URDF
parsing and gives one inspectable directory that both the sim and the offline replay/contact
tooling point at.

Note: the IKEA API no longer returns product weights, so every generated URDF carries a bogus
5.0 kg. ``mass`` is kept in the metadata for reference only — the sim overrides it (``--chair-mass``
/ ``CHAIR_MASS``) and rescales the inertia tensor accordingly.

Usage (needs gear_sonic[sim] — trimesh):
    python gear_sonic/scripts/prepare_object_asset.py \
        /scratch/rlebas1/IKEA_interface/ikea_assets/SANDSBERG_10605424 --out data/objects/chair
"""

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_urdf_inertial(urdf_path: Path) -> dict:
    """Extract ``mass`` / ``com`` / ``fullinertia`` from a single-link URDF."""
    root = ET.parse(urdf_path).getroot()
    inertial = root.find("./link/inertial")
    if inertial is None:
        raise ValueError(f"{urdf_path} has no <link><inertial> block")
    mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    com = [
        float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
    ]
    i = inertial.find("inertia")
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    fullinertia = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    return {"mass": mass, "com": com, "fullinertia": fullinertia}


def prepare(src: Path, out: Path, force: bool = False) -> Path:
    src = src.resolve()
    visual_stl = src / "visual.stl"
    collisions = sorted(src.glob("collision_*.stl"))
    urdfs = sorted(src.glob("*.urdf"))
    if not visual_stl.exists():
        raise FileNotFoundError(
            f"{visual_stl} not found — run IKEA_interface with --collision first"
        )
    if not collisions:
        raise FileNotFoundError(
            f"no collision_*.stl in {src} — CoACD failed; sanitize the GLB with "
            "utils/sanitize_glb.py under blender, then re-run 'ikea_interface.py recompute'"
        )
    if not urdfs:
        raise FileNotFoundError(f"no .urdf in {src}")

    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists (pass --force to overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    visual = trimesh.load(visual_stl, force="mesh")
    visual.export(out / "visual.obj")

    collision_names = []
    hulls = []
    for i, c in enumerate(collisions):
        name = f"collision_{i:03d}.stl"
        shutil.copyfile(c, out / name)
        collision_names.append(name)
        hulls.append(trimesh.load(c, force="mesh"))

    combined = trimesh.util.concatenate(hulls)
    lo, hi = combined.bounds

    meta = {
        "name": src.name,
        "source": str(src),
        "visual_mesh": "visual.obj",
        "collision_meshes": collision_names,
        **_parse_urdf_inertial(urdfs[0]),
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "z_min": float(lo[2]),
    }
    (out / "object.json").write_text(json.dumps(meta, indent=2) + "\n")

    extents = np.asarray(hi) - np.asarray(lo)
    print(f"[prepare_object_asset] {src.name} -> {out}")
    print(f"  {len(collision_names)} convex hulls, visual.obj ({len(visual.faces)} faces)")
    print(f"  bbox extents {extents.round(3).tolist()} m, z_min {meta['z_min']:.3f} m")
    print(f"  urdf mass {meta['mass']} kg (unreliable — overridden by the sim)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "src", type=Path, help="IKEA_interface asset directory (contains visual.stl + *.urdf)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "objects" / "chair",
        help="staged output directory (default: data/objects/chair)",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = p.parse_args()
    prepare(args.src, args.out.resolve(), force=args.force)


if __name__ == "__main__":
    main()
