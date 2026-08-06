"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

import json
import math
from pathlib import Path
from typing import Dict

import tyro
import yaml

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel

ArgsConfig = SimLoopConfig

# Box parameters — edit these to change the (free, floor-resting) box in the simulation
BOX_SIZE = (0.2, 0.2, 0.2)   # half-extents in meters (x, y, z)
BOX_POS  = (1.5, 0.0, 0.1)   # position in meters (x, y, z); z = half-height to rest on floor
BOX_MASS = 1.0                # mass in kg

# Tabletop box defaults, used when --table and --box are combined (graspable cube)
TABLE_BOX_SIZE = (0.025, 0.025, 0.025)  # half-extents in meters (8 cm cube)
TABLE_BOX_MASS = 0.1                  # mass in kg

# Grip behaviour for the graspable box. The problem is twofold: the cube slips, AND the
# position-controlled fingers sink into it (default MuJoCo contacts are soft, so a finger
# whose closed target is inside the cube penetrates deeply before the contact force balances
# it). Cranking friction alone makes the sinking obvious and the grasp mushy. So instead we
# stiffen the box contact (less penetration -> the finger pads actually meet the faces) and
# use a MODERATE friction with torsional grip:
#   - solref timeconst 0.01s = 2*sim timestep (1/200s): the stiffest value that stays stable.
#   - solimp raised so the contact hardens quickly near the surface (less sink).
#   - friction moderate; condim=4 adds torsional friction so the cube can't twist out.
#   - priority=1 makes these win for every box contact (hands are all default), so the box's
#     stiff/grippy contact is used rather than a soft mix with the finger geoms.
BOX_FRICTION = "2 0.05 0.001"          # sliding / torsional / rolling (moderate)
BOX_CONDIM = 4                         # 3=slide only, 4=+torsional (grasp), 6=+rolling
BOX_PRIORITY = 1                       # box contact params win over the contacting geom's
BOX_SOLREF = "0.01 1"                  # stiffer contact (timeconst, dampratio); reduces sinking
BOX_SOLIMP = "0.95 0.99 0.001 0.5 2"   # harder impedance near contact; reduces sinking

TABLE_TOP_THICKNESS = 0.02  # tabletop half-thickness in meters

# Chair task (--chair): a mesh object staged by scripts/prepare_object_asset.py from an
# IKEA_interface download, spawned free-standing on the floor in front of the robot.
CHAIR_POS_X = 0.55            # spawn distance in front of the robot, meters
CHAIR_YAW = math.pi / 2       # backrest toward the robot for the SANDSBERG asset (see --chair-yaw)
CHAIR_MASS = 0.2              # kg — deliberately light; the recording needs no physical mass
CHAIR_FLOOR_CLEARANCE = 0.005  # spawn gap above the floor so it settles instead of interpenetrating

# Held-box parameters live in this shared YAML (single source of truth, also read
# by generate_hold_box_quest.py and iter_reset_pose.py). See load_held_box_params().
HELD_BOX_PARAMS_PATH = Path(__file__).resolve().parent / "hold_box_params.yaml"


def load_held_box_params() -> dict:
    """Load the shared held-box params YAML (box size + root-frame anchor offset)."""
    with open(HELD_BOX_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_chair_config(config: "ArgsConfig") -> dict:
    """Build the mesh ``object_config`` for --chair from the staged asset dir."""
    asset_dir = Path(config.chair_asset)
    if not asset_dir.is_absolute():
        asset_dir = Path(__file__).resolve().parents[2] / asset_dir
    meta_path = asset_dir / "object.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — stage a chair first, e.g.\n"
            "  python gear_sonic/scripts/prepare_object_asset.py "
            "<IKEA_interface>/ikea_assets/SANDSBERG_10605424 --out data/objects/chair"
        )
    meta = json.loads(meta_path.read_text())

    # Precedence: CLI flag > per-asset default in object.json (spawn_yaw / spawn_pos /
    # spawn_mass, e.g. written by preview_chair_assets.py) > the constants above.
    if config.chair_yaw is not None:
        yaw = config.chair_yaw
    else:
        yaw = float(meta.get("spawn_yaw", CHAIR_YAW))
    if config.chair_pos:
        pos = tuple(config.chair_pos)
    elif meta.get("spawn_pos"):
        pos = tuple(float(v) for v in meta["spawn_pos"])
    else:
        # z places the mesh's lowest collision point just above the floor, whatever the
        # asset's own origin height is.
        pos = (CHAIR_POS_X, 0.0, -float(meta["z_min"]) + CHAIR_FLOOR_CLEARANCE)
    if config.chair_mass is not None:
        mass = config.chair_mass
    else:
        mass = float(meta.get("spawn_mass", CHAIR_MASS))

    return {
        "kind": "mesh",
        "asset_dir": str(asset_dir),
        "meta": meta,
        "pos": pos,
        "quat": (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        "mass": mass,
        # Same grip/anti-sink tuning as the graspable cube (it is generic contact tuning).
        "friction": BOX_FRICTION,
        "condim": BOX_CONDIM,
        "priority": BOX_PRIORITY,
        "solref": BOX_SOLREF,
        "solimp": BOX_SOLIMP,
    }


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        # init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    # Start with the elastic-band gantry detached (same effect as pressing '9').
    wbc_config["detach_gantry"] = config.detach_gantry

    if config.dump_scene:
        wbc_config["dump_scene"] = config.dump_scene

    assert not (config.chair and (config.box or config.held_box or config.table)), (
        "--chair is its own scene (chair on the floor); it cannot be combined with "
        "--box/--held-box/--table"
    )

    if config.table:
        wbc_config["table_config"] = {
            "pos": tuple(config.table_pos),
            "top_size": tuple(config.table_top_size),
            "top_thickness": TABLE_TOP_THICKNESS,
            "height": config.table_height,
        }

    # --held-box implies --box (the held object is the same injected box body).
    spawn_box = config.box or config.held_box

    if spawn_box:
        if config.held_box:
            held = load_held_box_params()
            wbc_config["object_config"] = {
                "size": tuple(held["box"]["size"]),
                # Initial pos is irrelevant — _update_held_box() overrides it every
                # step — but a sensible spawn near the chest avoids a first-frame jump.
                "pos": (0.2, 0.0, 1.0),
                "mass": BOX_MASS,
                "held": True,
                "anchor_offset": tuple(held["box"]["anchor_offset"]),
            }
        else:
            if config.table:
                # Tabletop spawn: graspable cube slightly toward the robot, 5 mm above
                # the surface so it settles cleanly.
                box_size = tuple(config.box_size) if config.box_size else TABLE_BOX_SIZE
                if config.box_pos:
                    box_pos = tuple(config.box_pos)
                else:
                    box_pos = (
                        config.table_pos[0] - 0.1,
                        config.table_pos[1],
                        config.table_height + box_size[2] + 0.005,
                    )
                hx, hy = config.table_top_size
                assert (
                    abs(box_pos[0] - config.table_pos[0]) + box_size[0] <= hx
                    and abs(box_pos[1] - config.table_pos[1]) + box_size[1] <= hy
                ), f"box footprint at {box_pos} does not fit on the tabletop"
                box_mass = TABLE_BOX_MASS
            else:
                box_size = tuple(config.box_size) if config.box_size else BOX_SIZE
                box_pos = tuple(config.box_pos) if config.box_pos else BOX_POS
                box_mass = BOX_MASS
            wbc_config["object_config"] = {
                "size": box_size,
                "pos": box_pos,
                "mass": box_mass,
                "friction": BOX_FRICTION,
                "condim": BOX_CONDIM,
                "priority": BOX_PRIORITY,
                "solref": BOX_SOLREF,
                "solimp": BOX_SOLIMP,
            }

    if config.chair:
        wbc_config["object_config"] = build_chair_config(config)

    if config.scene_reset:
        wbc_config["scene_reset_config"] = {
            "host": config.manager_host,
            "port": config.manager_port,
        }

    if config.record_box_gt:
        assert (
            spawn_box or config.chair
        ), "record_box_gt requires --box, --held-box or --chair (there must be an object to track)"
        wbc_config["record_box_gt"] = True
        wbc_config["box_gt_port"] = config.box_gt_port

    if config.render_depth_seg:
        # Box-only: the mask covers a single geom, while a mesh object is a set of convex hulls.
        assert spawn_box, "render_depth_seg requires --box or --held-box (the segmented object is the box)"
        assert config.enable_image_publish and config.enable_offscreen, (
            "render_depth_seg requires --enable_image_publish and --enable_offscreen "
            "(depth/seg are rendered offscreen and shipped over the camera stream)"
        )
        wbc_config["render_depth_seg"] = True
        wbc_config["fp_render_scale"] = config.fp_render_scale
        wbc_config["fp_render_every"] = config.fp_render_every

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)