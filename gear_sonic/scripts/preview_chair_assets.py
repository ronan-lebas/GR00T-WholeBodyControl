"""Side-by-side previewer for staged object assets: the G1 standing next to a candidate.

Loads the robot scene plus every candidate asset, spawned on the floor exactly where
``run_sim_loop.py --object-asset`` would put it, and lets you cycle assets / spin them in place
from the keyboard so you can judge size and grasp affordance against the real robot. Works on
any staged asset — IKEA chairs and the make_primitive_asset.py bimanual objects alike.

    python gear_sonic/scripts/preview_chair_assets.py                    # everything staged
    python gear_sonic/scripts/preview_chair_assets.py /scratch/rlebas1/IKEA_interface/ikea_assets
    python gear_sonic/scripts/preview_chair_assets.py --objects          # every --object on its table, in a row
    python gear_sonic/scripts/preview_chair_assets.py --variants bar     # whole bar catalog, in a row
    python gear_sonic/scripts/preview_chair_assets.py --variants bar --only 0 3 7   # a subset

Arguments are staged asset dirs (object.json), raw IKEA_interface asset dirs (visual.stl +
*.urdf), or a parent dir of either — raw ones are staged on the fly into data/objects/.

Keys (viewer window):
    n / p      next / previous asset
    [ / ]      rotate the asset -15 deg / +15 deg about z
    a          toggle: a row of assets side by side (current one marked) vs. one at the spawn pose
    w          write the current yaw into the asset's object.json (spawn_yaw)
    i          print the asset's dimensions and the command that uses it
    l          show / hide the name labels above the assets

The staged dir of the asset you settle on goes straight into the stack:

    CHAIR_ASSET=data/objects/<name> ./scripts/launch_sim_setup.sh --task chair
    ./scripts/launch_sim_setup.sh --task tabletop --object <plate|bar|handled-box>
    ./scripts/launch_sim_setup.sh --task tabletop --object bar --variant <N>
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from gear_sonic.scripts.make_primitive_asset import SPECS, VARIANTS, make, variant_name
from gear_sonic.scripts.prepare_object_asset import prepare
from gear_sonic.scripts.run_sim_loop import (
    OBJECT_POS_X,
    OBJECT_SURFACE_CLEARANCE,
    OBJECT_YAW,
    TABLE_TOP_THICKNESS,
    ArgsConfig,
    build_mesh_object_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml"
STAGE_DIR = REPO_ROOT / "data" / "objects"
ROW_GAP = 0.15  # free space between neighbours in row mode, meters
ROW_WIDTH = 6.0  # row mode shows as many assets as fit in this width (at least ROW_MIN)
ROW_MIN = 8
LABEL_LIFT = 0.08  # label height above an asset's top, meters
YAW_STEP = math.radians(15.0)
# launch_sim_setup.sh lowers the table for these objects (too tall for the 0.74 default).
TABLE_HEIGHTS = {"handled_box": 0.58, "carton": 0.58}
TABLE_RGBA = ("0.6 0.4 0.25 1", "0.5 0.35 0.2 1")  # top, leg (as base_sim._append_table)
DEFAULT_RGBA = (0.75, 0.6, 0.4, 1.0)
SELECTED_RGBA = (0.9, 0.45, 0.15, 1.0)


def _is_staged(d: Path) -> bool:
    return (d / "object.json").is_file()


def _is_raw(d: Path) -> bool:
    return (d / "visual.stl").is_file() and any(d.glob("*.urdf"))


def resolve_assets(paths: list[Path]) -> list[Path]:
    """Turn user-given paths into staged asset dirs, staging raw IKEA downloads as needed."""
    candidates: list[Path] = []
    for p in paths:
        p = p.resolve()
        if not p.is_dir():
            raise NotADirectoryError(p)
        if _is_staged(p) or _is_raw(p):
            candidates.append(p)
        else:
            subs = sorted(c for c in p.iterdir() if c.is_dir() and (_is_staged(c) or _is_raw(c)))
            if not subs:
                raise FileNotFoundError(f"{p} holds no staged or IKEA_interface asset")
            candidates.extend(subs)

    # Name -> already-staged dir, so pointing at a raw dir that was staged earlier under some
    # other name (data/objects/chair, typically) reuses it instead of staging a second copy.
    by_name = {}
    if STAGE_DIR.is_dir():
        for d in sorted(STAGE_DIR.iterdir()):
            if _is_staged(d):
                by_name.setdefault(json.loads((d / "object.json").read_text())["name"], d)

    staged = []
    for c in candidates:
        if _is_staged(c):
            staged.append(c)
        elif c.name in by_name:
            staged.append(by_name[c.name])
        else:
            out = STAGE_DIR / c.name.lower()
            print(f"[preview] staging {c.name} -> {out}")
            prepare(c, out, force=True)
            staged.append(out)
    if not staged:
        raise SystemExit("no assets to preview")
    # De-duplicate while keeping order (the same dir can be reached by several paths).
    seen, unique = set(), []
    for s in staged:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _natural_key(path: Path):
    """bar_v2 before bar_v10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def stage_variants(name: str, only: list[int] | None = None) -> list[Path]:
    """(Re)generate the catalog variants of a make_primitive_asset.py object (all, or ``only``)."""
    if name not in VARIANTS:
        raise SystemExit(f"no variant catalog for '{name}' (have {sorted(VARIANTS)})")
    n = len(VARIANTS[name])
    bad = [v for v in only or [] if not 0 <= v < n]
    if bad:
        raise SystemExit(f"'{name}' has variants 0..{n - 1}, not {bad}")
    return [
        make(name, STAGE_DIR / variant_name(name, v), force=True, variant=v)
        for v in (sorted(set(only)) if only else range(n))
    ]


def stage_objects() -> list[Path]:
    """(Re)generate every make_primitive_asset.py object (variant 0): all tabletop --object choices."""
    return [make(name, STAGE_DIR / name, force=True) for name in SPECS]


def short_label(meta: dict) -> str:
    return f"v{meta['variant']}" if "variant" in meta else meta["name"]


def table_height_for(meta: dict) -> float:
    return TABLE_HEIGHTS.get(meta["name"], ArgsConfig.table_height)


def spawn_pose(
    meta: dict, adir: Path | None = None, tabletop: bool = False
) -> tuple[tuple[float, float, float], float]:
    """The (pos, yaw) run_sim_loop.py would use for this asset: on the floor, or on the table
    (``tabletop``, the real tabletop spawn including its clamping and per-asset offset)."""
    if tabletop:
        cfg = build_mesh_object_config(
            ArgsConfig(object_asset=str(adir), table=True, table_height=table_height_for(meta))
        )
        w, _, _, z = cfg["quat"]
        return tuple(float(v) for v in cfg["pos"]), 2 * math.atan2(z, w)
    pos = meta.get("spawn_pos") or (
        OBJECT_POS_X,
        0.0,
        -float(meta["z_min"]) + OBJECT_SURFACE_CLEARANCE,
    )
    yaw = meta.get("spawn_yaw", OBJECT_YAW)
    return tuple(float(v) for v in pos), float(yaw)


def build_scene(
    scene_path: Path, assets: list[Path], metas: list[dict], tabletop: bool = False
) -> str:
    """Inject one static, visual-only body per asset; returns a temp XML path.

    Assets with ``visual_parts`` (make_primitive_asset.py) get one geom per part in its own color,
    named ``preview_<i>_visual_<k>``; others get a single ``preview_<i>_visual_0``.
    """
    tree = ET.parse(scene_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset_el = root.find("asset")
    if asset_el is None:
        asset_el = ET.SubElement(root, "asset")

    for i, (adir, meta) in enumerate(zip(assets, metas)):
        pos, yaw = spawn_pose(meta, adir, tabletop)
        body = ET.SubElement(worldbody, "body")
        body.set("name", f"preview_{i}")
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}")

        parts = meta.get("visual_parts") or [
            {"mesh": meta["visual_mesh"], "rgba": " ".join(str(v) for v in DEFAULT_RGBA)}
        ]
        for k, part in enumerate(parts):
            mesh = ET.SubElement(asset_el, "mesh")
            mesh.set("name", f"preview_{i}_{k}")
            mesh.set("file", str((adir / part["mesh"]).resolve()))
            geom = ET.SubElement(body, "geom")
            geom.set("name", f"preview_{i}_visual_{k}")
            geom.set("type", "mesh")
            geom.set("mesh", f"preview_{i}_{k}")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("mass", "0")
            geom.set("rgba", part["rgba"])

        # Goal fixture (static prop, e.g. the nail block): placed at its world offset from the
        # spawn pose, expressed in the body frame so it follows the asset when spun. Its origin
        # is on the table surface, which is the object's z_min plane.
        # Surface height in the body frame: the table top (tabletop) or the floor.
        surface_z = (table_height_for(meta) if tabletop else 0.0) - pos[2]
        c, s_ = math.cos(-yaw), math.sin(-yaw)

        def world_xy(dx, dy):  # world-axes offset from the spawn pose -> body frame
            return c * dx - s_ * dy, s_ * dx + c * dy

        if tabletop:
            cfg = ArgsConfig()
            hx, hy = cfg.table_top_size
            ht = TABLE_TOP_THICKNESS
            top_z = table_height_for(meta)
            tx, ty = world_xy(cfg.table_pos[0] - pos[0], cfg.table_pos[1] - pos[1])
            leg_half = (top_z - 2 * ht) / 2
            for name, size, z, rgba in (
                ("table_top", (hx, hy, ht), surface_z - ht, TABLE_RGBA[0]),
                ("table_leg", (0.06, 0.06, leg_half), surface_z - top_z + leg_half, TABLE_RGBA[1]),
            ):
                geom = ET.SubElement(body, "geom")
                geom.set("name", f"preview_{i}_{name}")
                geom.set("type", "box")
                geom.set("size", " ".join(str(v) for v in size))
                geom.set("pos", f"{tx} {ty} {z}")
                geom.set("quat", f"{math.cos(-yaw / 2)} 0 0 {math.sin(-yaw / 2)}")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("mass", "0")
                geom.set("rgba", rgba)

        fixture = meta.get("fixture")
        if fixture:
            dx, dy = fixture["offset"]
            fyaw = fixture.get("yaw", 0.0) - yaw
            for k, part in enumerate(fixture["parts"]):
                mesh = ET.SubElement(asset_el, "mesh")
                mesh.set("name", f"preview_{i}_fx{k}")
                mesh.set("file", str((adir / part["mesh"]).resolve()))
                geom = ET.SubElement(body, "geom")
                geom.set("name", f"preview_{i}_fixture_{k}")
                geom.set("type", "mesh")
                geom.set("mesh", f"preview_{i}_fx{k}")
                fx, fy = world_xy(dx, dy)
                geom.set("pos", f"{fx} {fy} {surface_z}")
                geom.set("quat", f"{math.cos(fyaw / 2)} 0 0 {math.sin(fyaw / 2)}")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("mass", "0")
                geom.set("rgba", part["rgba"])

    # Written next to the original so its relative <include>/meshdir paths still resolve.
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(scene_path.parent))
    os.close(fd)
    tree.write(tmp)
    return tmp


class Previewer:
    def __init__(
        self,
        assets: list[Path],
        row_mode: bool = False,
        labels: bool = True,
        tabletop: bool = False,
    ):
        self.tabletop = tabletop
        self.assets = assets
        self.labels = labels
        self.metas = [json.loads((a / "object.json").read_text()) for a in assets]
        poses = [spawn_pose(m, a, tabletop) for m, a in zip(self.metas, assets)]
        self.positions = [p for p, _ in poses]
        self.yaws = [y for _, y in poses]
        self.index = 0
        self.row_mode = row_mode
        self.part_colored = ["visual_parts" in m for m in self.metas]

        # Row spacing from the widest footprint along y at the spawn yaw, so thin objects (the bar
        # variants) pack tightly while chairs keep a chair-sized gap.
        widths = []
        for meta, yaw in zip(self.metas, self.yaws):
            ex, ey = np.asarray(meta["bbox_max"][:2]) - np.asarray(meta["bbox_min"][:2])
            width = abs(math.sin(yaw)) * ex + abs(math.cos(yaw)) * ey
            fx = meta.get("fixture")
            if fx:  # keep neighbouring fixtures from overlapping: y span about the object center
                dy = fx["offset"][1]
                width = max(width, 2 * max(abs(dy + fx["bbox_min"][1]), abs(dy + fx["bbox_max"][1])))
            if tabletop:  # neighbouring tables must not overlap
                width = max(width, 2 * ArgsConfig().table_top_size[1])
            widths.append(width)
        self.row_spacing = max(widths) + ROW_GAP
        self.row_max = max(ROW_MIN, int(ROW_WIDTH // self.row_spacing) + 1)

        tmp = build_scene(DEFAULT_SCENE, assets, self.metas, tabletop)
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.remove(tmp)
        self.data = mujoco.MjData(self.model)
        self.body_ids = [self.model.body(f"preview_{i}").id for i in range(len(assets))]
        self.geom_ids = [
            [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
            for bid in self.body_ids
        ]
        self.base_rgba = [[self.model.geom_rgba[g].copy() for g in gids] for gids in self.geom_ids]
        self.apply()

    def row_window(self) -> range:
        """Indices shown in row mode: a window of row_max around the current asset, so a big
        asset list stays readable (n/p scrolls it)."""
        n = len(self.assets)
        if n <= self.row_max:
            return range(n)
        start = max(0, min(self.index - self.row_max // 2, n - self.row_max))
        return range(start, start + self.row_max)

    def visible(self) -> range | list[int]:
        return self.row_window() if self.row_mode else [self.index]

    def apply(self):
        window = self.row_window() if self.row_mode else range(0)
        for i, bid in enumerate(self.body_ids):
            x, y, z = self.positions[i]
            if self.row_mode:
                visible = i in window
                y = y + (i - window.start - (len(window) - 1) / 2.0) * self.row_spacing
            else:
                visible = i == self.index
            self.model.body_pos[bid] = (x, y, z)
            yaw = self.yaws[i]
            self.model.body_quat[bid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            for gid, rgba in zip(self.geom_ids[i], self.base_rgba[i]):
                if not visible:
                    rgba = (0, 0, 0, 0)
                elif not self.part_colored[i]:
                    # Single-mesh assets have no colors of their own: tint the current one.
                    rgba = SELECTED_RGBA if i == self.index else DEFAULT_RGBA
                self.model.geom_rgba[gid] = rgba
        mujoco.mj_forward(self.model, self.data)

    def draw_labels(self, scn):
        """A text label above every shown asset; the current one in brackets."""
        scn.ngeom = 0
        if not self.labels:
            return
        for i in self.visible():
            if scn.ngeom >= scn.maxgeom:
                break
            x, y, _ = self.model.body_pos[self.body_ids[i]]
            top = self.positions[i][2] + self.metas[i]["bbox_max"][2] + LABEL_LIFT
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                np.array([x, y, top]),
                np.eye(3).flatten(),
                np.ones(4, dtype=np.float32),
            )
            label = short_label(self.metas[i])
            g.label = f"[{label}]" if i == self.index and self.row_mode else label
            scn.ngeom += 1

    def info(self):
        meta = self.metas[self.index]
        adir = self.assets[self.index]
        extents = np.asarray(meta["bbox_max"]) - np.asarray(meta["bbox_min"])
        rel = adir.relative_to(REPO_ROOT) if adir.is_relative_to(REPO_ROOT) else adir
        if "variant" in meta:
            obj = meta["source"].split()[1].replace("_", "-")
            cmd = f"./scripts/launch_sim_setup.sh --object {obj} --variant {meta['variant']}"
        elif self.tabletop:
            cmd = f"./scripts/launch_sim_setup.sh --object {meta['name'].replace('_', '-')}"
        else:
            cmd = f"CHAIR_ASSET={rel} ./scripts/launch_sim_setup.sh --task chair"
        print(
            f"\n[{self.index + 1}/{len(self.assets)}] {meta['name']}  ({rel})\n"
            + (
                f"  variant {meta['variant']}: {meta['variant_desc']}\n"
                if "variant" in meta
                else ""
            )
            + f"  size (w x d x h): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m"
            f"   |  {len(meta['collision_meshes'])} convex hulls\n"
            f"  yaw {math.degrees(self.yaws[self.index]):+.0f} deg"
            f"   |  {cmd}"
        )

    def write_yaw(self):
        adir = self.assets[self.index]
        meta = self.metas[self.index]
        meta["spawn_yaw"] = self.yaws[self.index]
        (adir / "object.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[preview] spawn_yaw {math.degrees(meta['spawn_yaw']):+.0f} deg -> {adir}/object.json"
        )

    def key(self, code: int):
        char = chr(code) if 0 <= code < 0x110000 else ""
        if char in ("N", "n"):
            self.index = (self.index + 1) % len(self.assets)
        elif char in ("P", "p"):
            self.index = (self.index - 1) % len(self.assets)
        elif char == "[":
            self.yaws[self.index] -= YAW_STEP
        elif char == "]":
            self.yaws[self.index] += YAW_STEP
        elif char in ("A", "a"):
            self.row_mode = not self.row_mode
            print(f"[preview] row mode {'on' if self.row_mode else 'off'}")
        elif char in ("W", "w"):
            self.write_yaw()
            return
        elif char in ("I", "i"):
            self.info()
            return
        elif char in ("L", "l"):
            self.labels = not self.labels
            return
        else:
            return
        self.apply()
        if char in "NnPp":
            self.info()

    def run(self):
        self.info()
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.distance = 3.5
            viewer.cam.lookat[:] = (OBJECT_POS_X / 2, 0.0, 0.7)
            if self.row_mode:
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -35
                viewer.cam.distance = 0.6 * len(self.row_window()) * self.row_spacing + 1.0
                viewer.cam.lookat[:] = (OBJECT_POS_X, 0.0, 0.5 if self.tabletop else 0.1)
            while viewer.is_running():
                with viewer.lock():
                    self.draw_labels(viewer.user_scn)
                viewer.sync()
                time.sleep(1.0 / 60.0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[STAGE_DIR],
        help="staged asset dirs, raw IKEA_interface asset dirs, or a parent of either "
        f"(default: {STAGE_DIR})",
    )
    p.add_argument(
        "--variants",
        metavar="NAME",
        default=None,
        help=f"(re)stage every catalog variant of a primitive object and preview them in a row "
        f"(have: {sorted(VARIANTS)}); replaces the positional paths",
    )
    p.add_argument(
        "--objects",
        action="store_true",
        help="(re)stage every primitive --object and preview them in a row, each on its table "
        "with its goal fixture, as spawned by --task tabletop; replaces the positional paths",
    )
    p.add_argument(
        "--only",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help="with --variants: show only these variant indices",
    )
    p.add_argument("--row", action="store_true", help="start in row mode ('a' toggles)")
    p.add_argument(
        "--no-labels", action="store_true", help="start with the name labels hidden ('l' toggles)"
    )
    p.add_argument(
        "--stage-only",
        action="store_true",
        help="stage the assets and print them, without opening the viewer",
    )
    args = p.parse_args()
    if args.only and not args.variants:
        p.error("--only picks from a catalog: use it with --variants NAME")
    if args.objects and args.variants:
        p.error("--objects and --variants are mutually exclusive")
    if args.objects:
        assets = stage_objects()
    elif args.variants:
        assets = stage_variants(args.variants, args.only)
    else:
        assets = sorted(resolve_assets(args.paths), key=_natural_key)
    if args.stage_only:
        for a in assets:
            meta = json.loads((a / "object.json").read_text())
            ext = np.asarray(meta["bbox_max"]) - np.asarray(meta["bbox_min"])
            rel = a.relative_to(REPO_ROOT) if a.is_relative_to(REPO_ROOT) else a
            print(f"{rel}  {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m  {meta['name']}")
        return
    Previewer(
        assets,
        row_mode=args.row or args.objects or bool(args.variants),
        labels=not args.no_labels,
        tabletop=args.objects,
    ).run()


if __name__ == "__main__":
    main()
