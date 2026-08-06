"""Dataclass-based configuration for MuJoCo simulation, WBC deployment, and teleop.

BaseConfig holds all tuneable knobs (interface, frequency, safety limits, etc.)
and can load/override values from a WBC YAML file. SimLoopConfig extends it
with multiprocessing and image-publish settings for the sim loop entry point.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
from typing import Any, Literal, Optional

import yaml

from gear_sonic.utils.network.network_utils import resolve_interface

WBC_VERSIONS = ["sonic_model12", "sonic_model12_brainco"]

@dataclass
class ArgsConfigTemplate:
    """Args Config for running the data collection loop."""

    def update(
        self,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] = [],
        allowed_keys: list[str] | None = None,
    ):
        for k, v in config_dict.items():
            if k in skip_keys:
                continue
            if allowed_keys is not None and k not in allowed_keys:
                continue
            if strict and not hasattr(self, k):
                raise ValueError(f"Config {k} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, k):
                continue
            setattr(self, k, v)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] = [],
        allowed_keys: list[str] | None = None,
    ):
        instance = cls()
        instance.update(
            config_dict=config_dict, strict=strict, skip_keys=skip_keys, allowed_keys=allowed_keys
        )
        return instance

    def to_dict(self):
        return asdict(self)

    def get(self, key: str, default: Any = None):
        return getattr(self, key) if hasattr(self, key) else default


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values."""
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "SIMULATE_DT": 1 / float(config.sim_frequency),
        "ENABLE_OFFSCREEN": config.enable_offscreen,
        "ENABLE_ONSCREEN": config.enable_onscreen,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
        "joint_safety_mode": config.joint_safety_mode,
        "arm_velocity_limit": config.arm_velocity_limit,
        "hand_velocity_limit": config.hand_velocity_limit,
        "lower_body_velocity_limit": config.lower_body_velocity_limit,
        "waist_pitch_limit": config.waist_pitch_limit,
        "hand_torque_limit": config.hand_torque_limit,
        "enable_natural_walk": config.enable_natural_walk,
    }

    if missed_keys_only:
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    # Sim-to-real KD gap: waist pitch (index 14) is over-damped in sim;
    # reduce KD by 10 on the real robot to avoid sluggish response
    if config.env_type == "real":
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class BaseConfig(ArgsConfigTemplate):
    """Base config inherited by all G1 control loops"""

    dataset_version: str = "sonic_model12"

    # WBC Configuration
    wbc_version: Literal[tuple(WBC_VERSIONS)] = "sonic_model12_brainco"
    """Version of the whole body controller."""

    wbc_model_path: str = "policy/stand.onnx,policy/walk.onnx"
    """Path to WBC model file (relative to GearCheckpoints/wbc)"""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration
    interface: str = "sim"
    """Interface to use for the control loop. [sim, real, lo, enxe8ea6a9c4e09]"""

    simulator: str = "mujoco"
    """Simulator to use."""

    sim_sync_mode: bool = False
    """Whether to run the control loop in sync mode."""

    control_frequency: int = 50
    """Frequency of the control loop."""

    sim_frequency: int = 200
    """Frequency of the simulation loop."""

    # Robot Configuration
    enable_waist: bool = True
    """Whether to include waist joints in IK."""

    with_hands: bool = True
    """Enable hand functionality."""

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration."""

    verbose: bool = True
    """Whether to print verbose output."""

    enable_offscreen: bool = False
    """Whether to enable offscreen rendering."""

    enable_onscreen: bool = True
    """Whether to enable onscreen rendering."""

    enable_teleop_evaluator: bool = False
    """Whether to enable teleop evaluator."""

    upper_body_joint_speed: float = 1000
    """Upper body joint speed."""

    env_name: str = "default"
    """Environment name."""

    ik_indicator: bool = False
    """Whether to draw IK indicators."""

    verbose_timing: bool = False
    """Enable verbose timing output every iteration."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher to use. [raw, ros]"""

    # Gravity Compensation Configuration
    enable_gravity_compensation: bool = False
    """Enable gravity compensation using pinocchio dynamics."""

    gravity_compensation_joints: Optional[list[str]] = None
    """Joint groups to apply gravity compensation to."""

    # Joint Safety Configuration
    joint_safety_mode: Literal["kill", "freeze"] = "kill"
    """Joint safety violation mode."""

    arm_velocity_limit: float = 25.0
    """Arm joint velocity limit in rad/s."""

    hand_velocity_limit: float = 1000.0
    """Hand/finger joint velocity limit in rad/s."""

    lower_body_velocity_limit: float = 20.0
    """Lower body joint velocity limit in rad/s."""

    waist_pitch_limit: float = 15.0
    """Waist pitch position limit in degrees."""

    hand_torque_limit: float = 0.1
    """Hand torque limit in radians."""

    enable_natural_walk: bool = False
    """Enable natural walk mode."""

    # Teleop/Device Configuration
    body_control_device: str = "dummy"
    """Device to use for body control."""

    hand_control_device: Optional[str] = "dummy"
    """Device to use for hand control."""

    body_streamer_ip: str = "10.112.210.229"
    """IP address for body streamer (vive only)."""

    body_streamer_keyword: str = "knee"
    """Body streamer keyword (vive only)."""

    enable_visualization: bool = False
    """Whether to enable visualization."""

    enable_real_device: bool = True
    """Whether to enable real device."""

    teleop_frequency: int = 20
    """Teleoperation frequency (Hz)."""

    teleop_replay_path: Optional[str] = None
    """Path to teleop replay data."""

    # Deployment/Camera Configuration
    robot_ip: str = "192.168.123.164"
    """Robot IP address"""

    data_collection: bool = True
    """Enable data collection"""

    data_collection_frequency: int = 20
    """Data collection frequency (Hz)"""

    root_output_dir: str = "outputs"
    """Root output directory"""

    offline_dc: bool = False
    """Offline data collection."""

    enable_upper_body_operation: bool = True
    """Enable upper body operation"""

    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    """Upper body operation mode"""

    inference_host: str = "localhost"
    """Inference server host"""

    inference_port: int = 5558
    """Inference server port (default 5558 to avoid conflict with camera_port 5555)"""

    inference_remote: bool = False
    """Whether to run inference on a remote server."""

    inference_prompt: str = "Pick up apple from table to plate"
    """Inference prompt"""

    inference_action_horizon: int = 16
    """Inference action horizon"""

    inference_control_freq: int = 20
    """Inference control frequency (Hz)"""

    inference_rate: float = 2.5
    """Inference rate (Hz)"""

    inference_plot_rerun: bool = False
    """Enable inference plot rerun"""

    inference_push_evals: bool = True
    """Whether to push evals."""

    inference_publish_single_action: bool = False
    """Whether to publish only a single action."""

    commit_id: str = ""
    """Commit ID for the current codebase"""

    enable_mode_switch: bool = False
    """Enable operation mode switching via ROS topic."""

    initial_mode: str = "idle"
    """Initial operation mode."""

    def __post_init__(self):
        # Resolve interface
        self.interface, self.env_type = resolve_interface(self.interface)

        # Set default gravity compensation joints if not specified
        if self.gravity_compensation_joints is None:
            self.gravity_compensation_joints = ["arms"]

        try:
            self.commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            )
        except Exception:
            self.commit_id = ""

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides"""
        import gear_sonic

        gear_sonic_path = Path(os.path.dirname(gear_sonic.__file__))
        configs_dir = gear_sonic_path / "utils" / "mujoco_sim" / "wbc_configs"

        if self.wbc_version == "sonic_model12":
            config_path = str(configs_dir / "g1_29dof_sonic_model12.yaml")
        elif self.wbc_version == "sonic_model12_brainco":
            config_path = str(configs_dir / "g1_29dof_sonic_model12_brainco.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: "
                f"sonic_model12"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config


@dataclass
class SimLoopConfig(BaseConfig):
    """Config for running the simulation loop."""

    mp_start_method: str = "spawn"
    """Multiprocessing start method"""

    enable_image_publish: bool = False
    """Enable image publishing in simulation"""

    camera_port: int = 5555
    """Camera port for image publishing"""

    verbose: bool = False
    """Verbose output, override the base config verbose"""

    detach_gantry: bool = False
    """Start with the elastic-band gantry detached, i.e. as if '9' were pressed in the
    viewer at startup (the robot is not held up by the virtual gantry). The band object
    still exists, so '9' continues to toggle it on/off during the run."""

    box: bool = False
    """Add a box object to the simulation scene"""

    held_box: bool = False
    """Spawn the box held in the robot's hands: it is kinematically anchored to the
    two-hand FK midpoint each sim step (collision disabled) and sways with the arms,
    instead of resting on the floor. Implies --box."""

    render_depth_seg: bool = False
    """Render ego-view depth + box segmentation and publish them (for FoundationPose export)."""

    fp_render_scale: float = 0.5
    """Resolution scale for depth/seg rendering relative to the ego camera (cuts render cost;
    the collector upscales back to the RGB resolution). 1.0 = full res."""

    fp_render_every: int = 2
    """Render depth/seg only every Nth image frame (1 = every frame). Lowers the FoundationPose
    sequence fps to reduce per-frame render cost."""

    record_box_gt: bool = False
    """Publish the box's exact pose from MuJoCo (object-in-head-camera) on a ZMQ PUB, in
    parallel to FoundationPose, so the data recorder can store ground truth for comparison.
    Sim-only (impossible on real hardware) — hence gated behind this flag."""

    box_gt_port: int = 5560
    """Port for the ground-truth box-pose publisher (see --record-box-gt)."""

    table: bool = False
    """Add a static table in front of the robot (manipulation setup). Combine with --box
    to spawn a graspable box on the tabletop."""

    table_pos: tuple[float, float] = (0.50, 0.0)
    """Table center (x, y) in meters. Robot spawns at the origin facing +x. The graspable
    box is placed at table_pos[0]-0.1, so both move together and stay within arm's reach."""

    table_top_size: tuple[float, float] = (0.3, 0.5)
    """Tabletop half-extents (x, y) in meters."""

    table_height: float = 0.74
    """Height of the tabletop top surface in meters."""

    box_pos: Optional[tuple[float, float, float]] = None
    """Explicit box spawn position override (x, y, z). Default: tabletop spawn with --table,
    else the floor-box constant in run_sim_loop.py."""

    box_size: Optional[tuple[float, float, float]] = None
    """Explicit box half-extents override (x, y, z). Default: tabletop cube with --table,
    else the floor-box constant in run_sim_loop.py."""

    chair: bool = False
    """Spawn a chair (mesh object) on the floor in front of the robot instead of the
    table+box tabletop scene. Mutually exclusive with --box/--held-box/--table."""

    chair_asset: str = "data/objects/chair"
    """Staged chair asset directory (as produced by scripts/prepare_object_asset.py from an
    IKEA_interface download): visual.obj + collision_*.stl + object.json. Relative paths are
    resolved against the repo root. Any staged dir works — compare candidates against the
    robot with scripts/preview_chair_assets.py."""

    chair_pos: Optional[tuple[float, float, float]] = None
    """Chair spawn position override (x, y, z) in meters. Default: the asset's ``spawn_pos``
    in object.json, else (CHAIR_POS_X, 0, z) with z placing the chair's lowest collision point
    5 mm above the floor."""

    chair_yaw: Optional[float] = None
    """Chair spawn yaw in radians about +z. Default: the asset's ``spawn_yaw`` in object.json
    (written by preview_chair_assets.py with 'w'), else CHAIR_YAW — which turns the backrest
    toward the robot for the shipped SANDSBERG asset only; the right yaw is per-asset."""

    chair_mass: Optional[float] = None
    """Chair mass in kg. Default: the asset's ``spawn_mass`` in object.json, else CHAIR_MASS in
    run_sim_loop.py. Deliberately light — the recordings don't need a physical mass, and IKEA's
    API reports a bogus constant anyway."""

    dump_scene: Optional[str] = None
    """Write the fully injected scene XML to this path at startup (debugging aid; open it with
    open_mj_scene.py). The sim continues to run normally."""

    scene_reset: bool = True
    """Listen for scene-reset commands from the teleop manager (manager_state topic)."""

    manager_host: str = "localhost"
    """Host of the teleop manager ZMQ publisher (for scene-reset commands)."""

    manager_port: int = 5556
    """Port of the teleop manager ZMQ publisher (for scene-reset commands)."""
