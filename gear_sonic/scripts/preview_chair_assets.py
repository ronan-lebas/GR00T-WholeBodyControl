"""Side-by-side previewer for staged object assets: the G1 standing next to a candidate.

Loads the robot scene plus every candidate asset, spawned on the floor exactly where
``run_sim_loop.py --object-asset`` would put it, and lets you cycle assets / spin them in place
from the keyboard so you can judge size and grasp affordance against the real robot. Works on
any staged asset — IKEA chairs and the make_primitive_asset.py bimanual objects alike.

    python gear_sonic/scripts/preview_chair_assets.py                    # everything staged
    python gear_sonic/scripts/preview_chair_assets.py /scratch/rlebas1/IKEA_interface/ikea_assets

Arguments are staged asset dirs (object.json), raw IKEA_interface asset dirs (visual.stl +
*.urdf), or a parent dir of either — raw ones are staged on the fly into data/objects/.

Keys (viewer window):
    n / p      next / previous asset
    [ / ]      rotate the asset -15 deg / +15 deg about z
    a          toggle: a row of nearby assets (current one highlighted) vs. one at the spawn pose
    w          write the current yaw into the asset's object.json (spawn_yaw)
    i          print the asset's dimensions and the command that uses it

The staged dir of the asset you settle on goes straight into the stack:

    CHAIR_ASSET=data/objects/<name> ./scripts/launch_sim_setup.sh --task chair
    ./scripts/launch_sim_setup.sh --task tabletop --object <plate|bar|handled-box>
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from gear_sonic.scripts.prepare_object_asset import prepare
from gear_sonic.scripts.run_sim_loop import OBJECT_POS_X, OBJECT_SURFACE_CLEARANCE, OBJECT_YAW

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml"
STAGE_DIR = REPO_ROOT / "data" / "objects"
ROW_SPACING = 1.0  # lateral gap between assets in row mode, meters
ROW_MAX = 8  # assets shown at once in row mode
YAW_STEP = math.radians(15.0)
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


def spawn_pose(meta: dict) -> tuple[tuple[float, float, float], float]:
    """The (pos, yaw) run_sim_loop.py would use on the floor for this asset."""
    pos = meta.get("spawn_pos") or (
        OBJECT_POS_X,
        0.0,
        -float(meta["z_min"]) + OBJECT_SURFACE_CLEARANCE,
    )
    yaw = meta.get("spawn_yaw", OBJECT_YAW)
    return tuple(float(v) for v in pos), float(yaw)


def build_scene(scene_path: Path, assets: list[Path], metas: list[dict]) -> str:
    """Inject one static, visual-only body per asset; returns a temp XML path."""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset_el = root.find("asset")
    if asset_el is None:
        asset_el = ET.SubElement(root, "asset")

    for i, (adir, meta) in enumerate(zip(assets, metas)):
        mesh = ET.SubElement(asset_el, "mesh")
        mesh.set("name", f"preview_{i}")
        mesh.set("file", str((adir / meta["visual_mesh"]).resolve()))

        pos, yaw = spawn_pose(meta)
        body = ET.SubElement(worldbody, "body")
        body.set("name", f"preview_{i}")
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}")
        geom = ET.SubElement(body, "geom")
        geom.set("name", f"preview_{i}_visual")
        geom.set("type", "mesh")
        geom.set("mesh", f"preview_{i}")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("mass", "0")
        geom.set("rgba", " ".join(str(v) for v in DEFAULT_RGBA))

    # Written next to the original so its relative <include>/meshdir paths still resolve.
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(scene_path.parent))
    os.close(fd)
    tree.write(tmp)
    return tmp


class Previewer:
    def __init__(self, assets: list[Path]):
        self.assets = assets
        self.metas = [json.loads((a / "object.json").read_text()) for a in assets]
        poses = [spawn_pose(m) for m in self.metas]
        self.positions = [p for p, _ in poses]
        self.yaws = [y for _, y in poses]
        self.index = 0
        self.row_mode = False

        tmp = build_scene(DEFAULT_SCENE, assets, self.metas)
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.remove(tmp)
        self.data = mujoco.MjData(self.model)
        self.body_ids = [self.model.body(f"preview_{i}").id for i in range(len(assets))]
        self.geom_ids = [self.model.geom(f"preview_{i}_visual").id for i in range(len(assets))]
        self.apply()

    def row_window(self) -> range:
        """Indices shown in row mode: a window of ROW_MAX around the current asset, so a big
        asset list stays readable (n/p scrolls it)."""
        n = len(self.assets)
        if n <= ROW_MAX:
            return range(n)
        start = max(0, min(self.index - ROW_MAX // 2, n - ROW_MAX))
        return range(start, start + ROW_MAX)

    def apply(self):
        window = self.row_window() if self.row_mode else range(0)
        for i, (bid, gid) in enumerate(zip(self.body_ids, self.geom_ids)):
            x, y, z = self.positions[i]
            if self.row_mode:
                visible = i in window
                y = y + (i - window.start - (len(window) - 1) / 2.0) * ROW_SPACING
            else:
                visible = i == self.index
            self.model.body_pos[bid] = (x, y, z)
            yaw = self.yaws[i]
            self.model.body_quat[bid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            self.model.geom_rgba[gid] = (
                (SELECTED_RGBA if i == self.index else DEFAULT_RGBA) if visible else (0, 0, 0, 0)
            )
        mujoco.mj_forward(self.model, self.data)

    def info(self):
        meta = self.metas[self.index]
        adir = self.assets[self.index]
        extents = np.asarray(meta["bbox_max"]) - np.asarray(meta["bbox_min"])
        rel = adir.relative_to(REPO_ROOT) if adir.is_relative_to(REPO_ROOT) else adir
        print(
            f"\n[{self.index + 1}/{len(self.assets)}] {meta['name']}  ({rel})\n"
            f"  size (w x d x h): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m"
            f"   |  {len(meta['collision_meshes'])} convex hulls\n"
            f"  yaw {math.degrees(self.yaws[self.index]):+.0f} deg"
            f"   |  CHAIR_ASSET={rel} ./scripts/launch_sim_setup.sh --task chair"
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
        else:
            return
        self.apply()
        if char in "NnPp":
            self.info()

    def run(self):
        self.info()
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key, show_left_ui=False
        ) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.distance = 3.5
            viewer.cam.lookat[:] = (OBJECT_POS_X / 2, 0.0, 0.7)
            while viewer.is_running():
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
        "--stage-only",
        action="store_true",
        help="stage the assets and print them, without opening the viewer",
    )
    args = p.parse_args()
    assets = resolve_assets(args.paths)
    if args.stage_only:
        for a in assets:
            meta = json.loads((a / "object.json").read_text())
            ext = np.asarray(meta["bbox_max"]) - np.asarray(meta["bbox_min"])
            rel = a.relative_to(REPO_ROOT) if a.is_relative_to(REPO_ROOT) else a
            print(f"{rel}  {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m  {meta['name']}")
        return
    Previewer(assets).run()


if __name__ == "__main__":
    main()
