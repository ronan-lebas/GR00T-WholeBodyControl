#!/usr/bin/env python3
"""Interactive preview for tuning the "robot holding a box" demo.

Opens an onscreen MuJoCo viewer with the G1 (BrainCo hands) posed with both arms
in front of the chest and a box anchored to the two-hand midpoint exactly like
``run_sim_loop.py --held-box`` does. The arms hold the joint-space pose from
``arm_pose`` and a small sinusoid (``motion``) is added to one shoulder joint so
you can see the box move.

Edit ``hold_box_params.yaml`` and **save**; the viewer reloads (auto-reload on file change):

  * ``arm_pose.*`` , ``motion.*`` and ``box.anchor_offset`` reload live.
  * ``box.size`` is baked into the model at startup; change it and restart this
    script.

The same ``arm_pose`` + ``motion`` drive ``generate_hold_box_quest.py`` (via FK),
so what you frame here is what the generated trajectory reproduces.

Run it (in .venv_sim):
    python gear_sonic/scripts/iter_reset_pose.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from gear_sonic.scripts.generate_hold_box_quest import arm_pose_deg
from gear_sonic.scripts.quest_manager_thread_server import RobotRestReference

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE = (
    _REPO_ROOT
    / "gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml"
)
_PARAMS_DEFAULT = Path(__file__).resolve().parent / "hold_box_params.yaml"

_SIDES = ("left", "right")
_ARM_JOINTS = {
    side: [
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]
    for side in _SIDES
}


def _inject_box(scene_path: Path, size) -> str:
    """Inject a free, collision-disabled box body into the scene (mirrors base_sim)."""
    tree = ET.parse(scene_path)
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(worldbody, "body")
    body.set("name", "box")
    body.set("pos", "0.2 0 1.0")
    ET.SubElement(body, "freejoint")
    geom = ET.SubElement(body, "geom")
    geom.set("type", "box")
    geom.set("size", f"{size[0]} {size[1]} {size[2]}")
    geom.set("mass", "1.0")
    geom.set("rgba", "0.8 0.4 0.1 1")
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".xml", dir=str(scene_path.parent)
    ) as f:
        tree.write(f.name)
        return f.name


class Preview:
    def __init__(self, params_path: Path):
        self.params_path = params_path
        params = self._load()

        tmp = _inject_box(_SCENE, params["box"]["size"])
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.remove(tmp)
        self.data = mujoco.MjData(self.model)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.wrist_id = {
            s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link")
            for s in _SIDES
        }
        self.arm_qadr = {
            s: [
                int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in _ARM_JOINTS[s]
            ]
            for s in _SIDES
        }
        box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_qadr = int(self.model.jnt_qposadr[int(self.model.body_jntadr[box_body])])

        # Rest wrist LINK positions in the root frame (same reference the generator
        # uses) so the printed displacement maps onto the generated trajectory.
        fk = RobotRestReference().compute(None)
        self.link_ref = {s: np.asarray(fk[s][0], dtype=np.float64) for s in _SIDES}

    def _load(self) -> dict:
        with open(self.params_path) as f:
            return yaml.safe_load(f)

    def pose(self, params: dict, t: float, verbose: bool = False) -> None:
        """Set the arm joints (base pose + sine) and the box for media time ``t``."""
        pose_deg = arm_pose_deg(params)
        m = params["motion"]
        ji = int(m["joint_index"])
        offset = np.deg2rad(float(m["amplitude_deg"])) * np.sin(
            2.0 * np.pi * t / max(1e-3, float(m["period"]))
        )
        signs = {"left": 1.0, "right": -1.0 if m.get("mirror") else 1.0}
        for side in _SIDES:
            angles = np.deg2rad(pose_deg[side])
            angles[ji] += signs[side] * offset
            for qadr, ang in zip(self.arm_qadr[side], angles):
                self.data.qpos[qadr] = ang
        mujoco.mj_forward(self.model, self.data)

        offset_vec = np.asarray(params["box"]["anchor_offset"], dtype=np.float64)
        mid = 0.5 * (self.data.xpos[self.wrist_id["left"]] + self.data.xpos[self.wrist_id["right"]])
        root_rot = self.data.xmat[self.pelvis_id].reshape(3, 3)
        self.data.qpos[self.box_qadr : self.box_qadr + 3] = mid + root_rot @ offset_vec
        self.data.qpos[self.box_qadr + 3 : self.box_qadr + 7] = self.data.xquat[self.pelvis_id]
        mujoco.mj_forward(self.model, self.data)

        if verbose:
            pelvis = self.data.xpos[self.pelvis_id]
            print("[iter_reset] reloaded. Wrist displacement from rest (root frame, at sine=0):")
            for side in _SIDES:
                rel = root_rot.T @ (self.data.xpos[self.wrist_id[side]] - pelvis)
                disp = rel - self.link_ref[side]
                print(f"    {side:5s}: forward={disp[0]:+.3f}  up={disp[2]:+.3f}  y={disp[1]:+.3f} m")

    def run(self) -> None:
        params = self._load()
        last_mtime = self.params_path.stat().st_mtime
        self.pose(params, 0.0, verbose=True)
        t0 = time.monotonic()
        print(
            f"[iter_reset] Viewer open. Edit + save {self.params_path} to update "
            f"(box.size needs a restart). Ctrl-C or close the window to quit."
        )
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                try:
                    mtime = self.params_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        try:
                            params = self._load()
                            self.pose(params, time.monotonic() - t0, verbose=True)
                        except Exception as e:  # bad edit mid-save: keep the viewer alive
                            print(f"[iter_reset] reload failed ({e}); fix the YAML and save again")
                    else:
                        self.pose(params, time.monotonic() - t0)
                except FileNotFoundError:
                    pass
                viewer.sync()
                time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview/tune the held-box demo pose.")
    parser.add_argument("--params", default=None, help="Path to hold_box_params.yaml.")
    args = parser.parse_args()
    Preview(Path(args.params) if args.params else _PARAMS_DEFAULT).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
