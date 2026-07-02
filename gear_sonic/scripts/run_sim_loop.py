"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

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

# Held-box parameters live in this shared YAML (single source of truth, also read
# by generate_hold_box_quest.py and iter_reset_pose.py). See load_held_box_params().
HELD_BOX_PARAMS_PATH = Path(__file__).resolve().parent / "hold_box_params.yaml"


def load_held_box_params() -> dict:
    """Load the shared held-box params YAML (box size + root-frame anchor offset)."""
    with open(HELD_BOX_PARAMS_PATH) as f:
        return yaml.safe_load(f)


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

    # --held-box implies --box (the held object is the same injected box body).
    spawn_box = config.box or config.held_box

    if spawn_box:
        if config.held_box:
            held = load_held_box_params()
            wbc_config["box_config"] = {
                "size": tuple(held["box"]["size"]),
                # Initial pos is irrelevant — _update_held_box() overrides it every
                # step — but a sensible spawn near the chest avoids a first-frame jump.
                "pos": (0.2, 0.0, 1.0),
                "mass": BOX_MASS,
                "held": True,
                "anchor_offset": tuple(held["box"]["anchor_offset"]),
            }
        else:
            wbc_config["box_config"] = {
                "size": BOX_SIZE,
                "pos": BOX_POS,
                "mass": BOX_MASS,
            }

    if config.render_depth_seg:
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