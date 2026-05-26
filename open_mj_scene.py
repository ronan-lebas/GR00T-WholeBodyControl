#!/usr/bin/env python3
"""Open a MuJoCo XML scene in a viewer.

Tries the modern `mujoco` bindings first, then `mujoco_viewer`, then falls
back to `mujoco_py` if available. Usage:

  python3 open_mj_scene.py [path/to/scene.xml]

Default path is the requested scene under the repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_XML = Path("gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml")


def main() -> int:
    p = argparse.ArgumentParser(description="Open a MuJoCo XML scene in a viewer")
    p.add_argument("xml", nargs="?", default=str(DEFAULT_XML), help="path to mujoco XML file")
    args = p.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 2

    # Try modern `mujoco` python bindings (mujoco >= 2.3)
    try:
        import mujoco as mj

        # Load model and start viewer if available
        try:
            model = mj.MjModel.from_xml_path(str(xml_path))
            # remove gravity
            try:
                model.opt.gravity[:] = 0.0
                print("Set model gravity to [0.0, 0.0, 0.0]")
            except Exception:
                try:
                    # fallback assignment
                    model.opt.gravity = [0.0, 0.0, 0.0]
                    print("Set model gravity to [0.0, 0.0, 0.0]")
                except Exception:
                    print("Warning: failed to set model gravity to zero")

            data = mj.MjData(model)

            # try the integrated viewer launch if present
            try:
                # prefer mujoco.viewer.launch
                from mujoco.viewer import launch as _launch
                print(f"Launching viewer via mujoco.viewer for {xml_path}")
                _launch(model, data)
                return 0
            except Exception:
                pass

            # try mujoco_viewer package
            try:
                import mujoco_viewer
                print(f"Launching viewer via mujoco_viewer for {xml_path}")
                mujoco_viewer.launch(model, data)
                return 0
            except Exception:
                pass

        except Exception as e:
            print("mujoco bindings present but failed to load model:\n", e)
    except Exception:
        pass

    # Try mujoco_py as a fallback
    try:
        import mujoco_py

        try:
            model = mujoco_py.load_model_from_path(str(xml_path))
            sim = mujoco_py.MjSim(model)
            # remove gravity
            try:
                sim.model.opt.gravity[:] = 0.0
                print("Set sim model gravity to [0.0, 0.0, 0.0]")
            except Exception:
                try:
                    sim.model.opt.gravity = [0.0, 0.0, 0.0]
                    print("Set sim model gravity to [0.0, 0.0, 0.0]")
                except Exception:
                    print("Warning: failed to set sim model gravity to zero")

            viewer = mujoco_py.MjViewer(sim)
            print(f"Launching viewer via mujoco_py for {xml_path}")
            while True:
                sim.step()
                viewer.render()
        except Exception as e:
            print("mujoco_py present but failed to run the viewer:\n", e)
    except Exception:
        pass

    print("No supported MuJoCo Python viewer found. Install 'mujoco' (and mujoco_viewer) or 'mujoco_py'.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
