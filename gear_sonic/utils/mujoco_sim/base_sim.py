"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_MOTORS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        # FoundationPose export: render ego-view depth + box segmentation alongside RGB.
        # Depth/seg are rendered with a dedicated lower-resolution renderer (segmentation is
        # expensive at full res) and only every Nth image frame; the collector upscales them
        # back to the RGB resolution. update_scene is essentially free, so a 2nd renderer is fine.
        self.render_depth_seg = self.config.get("render_depth_seg", False)
        self.fp_render_scale = float(self.config.get("fp_render_scale", 0.5))
        self.fp_render_every = max(1, int(self.config.get("fp_render_every", 2)))
        # Background/invalid depth (in meters) above this is zeroed out before saving.
        self.fp_depth_max_m = 10.0
        self.fp_cam_id = -1
        self.fp_box_geom_id = -1
        self.fp_cam_K = None
        self.fp_box_half_extents = None
        self.fp_render_h = 0
        self.fp_render_w = 0
        self.fp_renderer = None
        self._fp_frame_counter = 0

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")

        extra_buffers = {}
        fp_meta = None
        if self.render_depth_seg and self.fp_cam_K is not None:
            h, w = self.fp_render_h, self.fp_render_w
            extra_buffers = {
                "ego_view_depth": {"shape": (h, w), "dtype": np.uint16},
                "ego_view_seg": {"shape": (h, w), "dtype": np.uint8},
            }
            fp_meta = {
                "cam_K": self.fp_cam_K.flatten().tolist(),
                "box_half_extents": self.fp_box_half_extents,
            }

        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
            extra_buffers=extra_buffers,
            fp_meta=fp_meta,
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def _inject_box(self, xml_path: str, box_config: dict) -> str:
        """Inject a free box body into the scene XML; returns path to temp file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        pos = box_config["pos"]
        size = box_config["size"]
        mass = box_config["mass"]

        box_body = ET.SubElement(worldbody, "body")
        box_body.set("name", "box")
        box_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        ET.SubElement(box_body, "freejoint")
        geom = ET.SubElement(box_body, "geom")
        geom.set("type", "box")
        geom.set("size", f"{size[0]} {size[1]} {size[2]}")
        geom.set("mass", str(mass))
        if box_config.get("held"):
            # Held box is kinematically scripted onto the hands every step
            # (_update_held_box); disabling collision keeps it purely visual so it
            # never perturbs the robot or fights the physics it is overridden by.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

        # Write next to the original so relative <include> paths remain valid
        xml_dir = os.path.dirname(xml_path)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".xml", dir=xml_dir
        ) as f:
            tree.write(f.name)
            return f.name

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])

        box_config = self.config.get("box_config", None)
        if box_config:
            tmp_xml = self._inject_box(xml_path, box_config)
            try:
                self.mj_model = mujoco.MjModel.from_xml_path(tmp_xml)
            finally:
                os.remove(tmp_xml)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        if self.render_depth_seg:
            self._setup_foundation_pose(box_config)

        self._setup_held_box(box_config)

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        # For hands we keep two lists when needed:
        #  - *_hand_motor_index: indices of actuated motor joints (NUM_HAND_MOTORS)
        #  - *_hand_index: indices of full hand joints (including mimic joints)
        left_hand_index = []
        right_hand_index = []
        left_hand_motor_index = []
        right_hand_motor_index = []

        # Decide filters based on scene
        if "brainco" in self.config["ROBOT_SCENE"]:
            motor_filter = ["proximal", "metacarpal"]
            full_filter = ["proximal", "metacarpal", "distal"]
        elif "dex3" in self.config["ROBOT_SCENE"]:
            motor_filter = ["hand"]
            full_filter = ["hand"]
        else:
            raise ValueError(f"Unknown robot scene: {self.config['ROBOT_SCENE']}")

        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)

            # collect full-hand joints (including mimic joints)
            if "left" in name and any(part in name for part in full_filter):
                left_hand_index.append(i)
            if "right" in name and any(part in name for part in full_filter):
                right_hand_index.append(i)

            # collect motor-only (actuated) joints
            if "left" in name and any(part in name for part in motor_filter):
                left_hand_motor_index.append(i)
            if "right" in name and any(part in name for part in motor_filter):
                right_hand_motor_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS

        # For hands with mimic joints: ensure at least actuated motors found
        assert len(left_hand_motor_index) >= self.robot.NUM_HAND_MOTORS
        assert len(right_hand_motor_index) >= self.robot.NUM_HAND_MOTORS

        # Store arrays: full joint lists and motor-only lists
        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(left_hand_index)
        self.right_hand_index = np.array(right_hand_index)
        self.left_hand_motor_index = np.array(left_hand_motor_index)
        self.right_hand_motor_index = np.array(right_hand_motor_index)

        # Build deterministic joint -> actuator index mapping
        # We assume actuator ordering follows joint ordering for actuated joints.
        joint_to_actuator = -np.ones(self.mj_model.njnt, dtype=int)
        actuated_joints = set()
        actuated_joints.update(self.body_joint_index.tolist())
        actuated_joints.update(self.left_hand_motor_index.tolist())
        actuated_joints.update(self.right_hand_motor_index.tolist())

        act_idx = 0
        for j in range(self.mj_model.njnt):
            if j in actuated_joints:
                joint_to_actuator[j] = act_idx
                act_idx += 1

        # Expose actuator-index arrays for body and hand motors
        self.body_actuator_index = joint_to_actuator[self.body_joint_index]
        self.left_hand_motor_actuator_index = joint_to_actuator[self.left_hand_motor_index]
        self.right_hand_motor_actuator_index = joint_to_actuator[self.right_hand_motor_index]

        # BrainCo: cache per-finger joint limits [rad] for normalized [0,1] <-> rad conversion.
        # BrainCo commands/states are normalized [0,1] (0 = open, 1 = closed). We map them
        # with an AFFINE transform over each joint's full [lower, upper] travel so that:
        #   - normalized 0 maps to the lower limit and 1 to the upper limit;
        #   - joints whose lower limit is non-zero (e.g. the thumb metacarpal) sweep their
        #     entire travel instead of dead-zoning against the lower limit (a plain
        #     `q = norm * upper` would clamp everything below `lower / upper` to the limit).
        # Limits are read per hand because the left/right joint ranges are not guaranteed
        # to match, and must be read here (mj_model available) since the bridge has no model.
        if "brainco" in self.config["ROBOT_SCENE"]:
            self.brainco_lower_limits_left = np.array([
                self.mj_model.jnt_range[j][0]
                for j in self.left_hand_motor_index[: self.num_hand_dof]
            ])
            self.brainco_upper_limits_left = np.array([
                self.mj_model.jnt_range[j][1]
                for j in self.left_hand_motor_index[: self.num_hand_dof]
            ])
            self.brainco_lower_limits_right = np.array([
                self.mj_model.jnt_range[j][0]
                for j in self.right_hand_motor_index[: self.num_hand_dof]
            ])
            self.brainco_upper_limits_right = np.array([
                self.mj_model.jnt_range[j][1]
                for j in self.right_hand_motor_index[: self.num_hand_dof]
            ])

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

        if self.render_depth_seg and self.fp_cam_K is not None:
            self.fp_renderer = mujoco.Renderer(
                self.mj_model, height=self.fp_render_h, width=self.fp_render_w
            )

    def _setup_foundation_pose(self, box_config: dict | None):
        """Cache the ego-camera intrinsics and box geom id for FoundationPose export."""
        ego_cfg = self.camera_configs.get("ego_view")
        if ego_cfg is None:
            print("[FoundationPose] No ego_view camera configured; depth/seg export disabled")
            self.render_depth_seg = False
            return
        if box_config is None:
            print("[FoundationPose] No box in scene; depth/seg export disabled")
            self.render_depth_seg = False
            return

        cam_name = ego_cfg.get("mjcf_name", "ego_view")
        self.fp_cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        box_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.fp_box_geom_id = int(self.mj_model.body_geomadr[box_body_id])

        # Pinhole intrinsics from MuJoCo's vertical FOV (square pixels, principal point centered).
        height = ego_cfg["height"]
        width = ego_cfg["width"]
        fovy_rad = np.deg2rad(float(self.mj_model.cam_fovy[self.fp_cam_id]))
        fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        self.fp_cam_K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.fp_box_half_extents = [float(s) for s in box_config["size"]]

        # Reduced render resolution for depth/seg (>=16px, even dims).
        self.fp_render_h = max(16, int(round(height * self.fp_render_scale)) & ~1)
        self.fp_render_w = max(16, int(round(width * self.fp_render_scale)) & ~1)
        print(
            f"[FoundationPose] ego depth/seg enabled (cam '{cam_name}', box geom "
            f"{self.fp_box_geom_id}, fovy {np.rad2deg(fovy_rad):.1f} deg, "
            f"render {self.fp_render_w}x{self.fp_render_h} every {self.fp_render_every} frame(s))"
        )

    def _render_ego_depth_seg(self) -> tuple[np.ndarray, np.ndarray]:
        """Render ego-view depth (uint16 mm) and box mask (uint8) at reduced resolution.

        Uses the dedicated low-res ``fp_renderer`` (segmentation is expensive at full res).
        The collector upscales both back to the RGB resolution before saving.
        """
        renderer = self.fp_renderer
        renderer.update_scene(self.mj_data, camera="head_camera")

        renderer.enable_depth_rendering()
        depth_m = renderer.render()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        invalid = (depth_m <= 0.0) | (depth_m > self.fp_depth_max_m) | ~np.isfinite(depth_m)
        depth_mm = np.where(invalid, 0.0, depth_m * 1000.0)
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        # Segmentation buffer is (H, W, 2): [..., 0] = object id, [..., 1] = object type.
        is_box = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == self.fp_box_geom_id)
        mask = (is_box.astype(np.uint8)) * 255
        return depth_mm, mask

    def _setup_held_box(self, box_config: dict | None):
        """Cache ids for a kinematically-held box.

        When ``box_config["held"]`` is set, the box is anchored every sim step to
        the midpoint of the two wrist links (see ``_update_held_box``) so it moves
        with the arms without any grasp physics. Sets ``self.held_box``.
        """
        self.held_box = False
        if not (box_config and box_config.get("held")):
            return
        box_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_jnt_id = int(self.mj_model.body_jntadr[box_body_id])
        self.held_box_qadr = int(self.mj_model.jnt_qposadr[box_jnt_id])
        self.held_box_dofadr = int(self.mj_model.jnt_dofadr[box_jnt_id])
        self.held_l_wrist_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.held_r_wrist_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        if self.held_l_wrist_id < 0 or self.held_r_wrist_id < 0:
            raise RuntimeError(
                "[HeldBox] left/right_wrist_yaw_link bodies not found; cannot anchor the held box"
            )
        self.held_anchor_offset = np.asarray(
            box_config.get("anchor_offset", (0.1, 0.0, 0.0)), dtype=np.float64
        )
        self.held_box = True
        print(
            f"[HeldBox] box anchored to wrist midpoint, root-frame offset "
            f"{self.held_anchor_offset.tolist()} (collision disabled)"
        )

    def _update_held_box(self):
        """Script the held box onto the live two-hand FK midpoint (called after mj_step).

        Box position = wrist midpoint + root-frame offset; orientation = robot root
        orientation. The freejoint velocity is zeroed and forward kinematics is
        re-run so the renderers see the updated box pose this frame. This pose is
        exact ground truth for the held object.
        """
        d = self.mj_data
        mid = 0.5 * (d.xpos[self.held_l_wrist_id] + d.xpos[self.held_r_wrist_id])
        root_quat = d.xquat[self.root_body_id]  # scalar-first [w, x, y, z]
        root_rot = d.xmat[self.root_body_id].reshape(3, 3)
        qadr = self.held_box_qadr
        d.qpos[qadr : qadr + 3] = mid + root_rot @ self.held_anchor_offset
        d.qpos[qadr + 3 : qadr + 7] = root_quat
        d.qvel[self.held_box_dofadr : self.held_box_dofadr + 6] = 0.0
        # Propagate the new box qpos into body/geom frames for the renderers.
        mujoco.mj_kinematics(self.mj_model, d)

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            if self.unitree_bridge.is_brainco:
                # BrainCo: cmd.cmds[i].q is normalized [0, 1]. Map it to a joint angle [rad]
                # with the per-hand affine transform q = lower + norm * (upper - lower) so the
                # full normalized command sweeps the joint's entire [lower, upper] travel
                # (see brainco_*_limits_* computation in the setup above), then apply PD control.
                # Gains chosen to stay within MuJoCo actuator ctrlrange limits:
                #   thumb_metacarpal ±0.5 Nm, thumb_proximal ±1.1 Nm, others ±2.0 Nm
                KP = 100 * np.array([0.3, 0.8, 1.5, 1.5, 1.5, 1.5][: self.num_hand_dof])
                KD = np.array([0.01, 0.02, 0.02, 0.02, 0.02, 0.02][: self.num_hand_dof])
                for i in range(self.unitree_bridge.num_hand_motor):
                    motor_idx_l = self.left_hand_motor_index[i]
                    motor_idx_r = self.right_hand_motor_index[i]
                    norm_l = float(self.unitree_bridge.left_hand_cmd.cmds[i].q)
                    q_des_l = self.brainco_lower_limits_left[i] + norm_l * (
                        self.brainco_upper_limits_left[i] - self.brainco_lower_limits_left[i]
                    )
                    q_cur_l = self.mj_data.qpos[motor_idx_l + self.qpos_offset - 1]
                    dq_cur_l = self.mj_data.qvel[motor_idx_l + self.qvel_offset - 1]
                    left_hand_torques[i] = KP[i] * (q_des_l - q_cur_l) + KD[i] * (0.0 - dq_cur_l)

                    norm_r = float(self.unitree_bridge.right_hand_cmd.cmds[i].q)
                    q_des_r = self.brainco_lower_limits_right[i] + norm_r * (
                        self.brainco_upper_limits_right[i] - self.brainco_lower_limits_right[i]
                    )
                    q_cur_r = self.mj_data.qpos[motor_idx_r + self.qpos_offset - 1]
                    dq_cur_r = self.mj_data.qvel[motor_idx_r + self.qvel_offset - 1]
                    right_hand_torques[i] = KP[i] * (q_des_r - q_cur_r) + KD[i] * (0.0 - dq_cur_r)
            else:
                # Dex3: cmd.motor_cmd[i].{q, dq, kp, kd, tau} are raw rad values
                for i in range(self.unitree_bridge.num_hand_motor):
                    motor_idx_l = self.left_hand_motor_index[i]
                    motor_idx_r = self.right_hand_motor_index[i]
                    left_hand_torques[i] = (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[motor_idx_l + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[motor_idx_l + self.qvel_offset - 1]
                        )
                    )
                    right_hand_torques[i] = (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[motor_idx_r + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[motor_idx_r + self.qvel_offset - 1]
                        )
                    )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]


        # Read actuator forces for body joints using actuator mapping
        body_tau = np.zeros(len(self.body_joint_index))
        valid_body = self.body_actuator_index >= 0
        if np.any(valid_body):
            body_tau[valid_body] = self.mj_data.actuator_force[self.body_actuator_index[valid_body]]
        obs["body_tau_est"] = body_tau


        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            # Estimated actuator torques exist only for actuated motor joints
            left_tau = np.zeros(len(self.left_hand_motor_index))
            valid_l = self.left_hand_motor_actuator_index >= 0
            if np.any(valid_l):
                left_tau[valid_l] = self.mj_data.actuator_force[self.left_hand_motor_actuator_index[valid_l]]
            obs["left_hand_tau_est"] = left_tau

            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]

            right_tau = np.zeros(len(self.right_hand_motor_index))
            valid_r = self.right_hand_motor_actuator_index >= 0
            if np.any(valid_r):
                right_tau[valid_r] = self.mj_data.actuator_force[self.right_hand_motor_actuator_index[valid_r]]
            obs["right_hand_tau_est"] = right_tau

            # BrainCo: also expose motor-only joint states (6 joints, excluding distal/mimic).
            # obs["left_hand_q"] contains all hand joints (11 for BrainCo: motor + distal),
            # so sequential indexing [0..5] would mix in distal joints. The bridge uses these
            # dedicated motor-only keys to publish normalized [0,1] state correctly.
            if "brainco" in self.config["ROBOT_SCENE"]:
                motor_l = self.left_hand_motor_index[: self.num_hand_dof]
                motor_r = self.right_hand_motor_index[: self.num_hand_dof]
                obs["left_hand_motor_q"]  = self.mj_data.qpos[motor_l + self.qpos_offset - 1]
                obs["left_hand_motor_dq"] = self.mj_data.qvel[motor_l + self.qvel_offset - 1]
                obs["right_hand_motor_q"]  = self.mj_data.qpos[motor_r + self.qpos_offset - 1]
                obs["right_hand_motor_dq"] = self.mj_data.qvel[motor_r + self.qvel_offset - 1]

        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        # Place torques into actuator-ordered `self.torques` using mapping
        # body
        b_valid = self.body_actuator_index >= 0
        if np.any(b_valid):
            self.torques[self.body_actuator_index[b_valid]] = body_torques[b_valid]

        if self.num_hand_dof > 0:
            # left hand motors
            l_act = self.left_hand_motor_actuator_index[: self.num_hand_dof]
            l_valid = l_act >= 0
            if np.any(l_valid):
                self.torques[l_act[l_valid]] = hand_torques[: self.num_hand_dof][l_valid]
            # right hand motors
            r_act = self.right_hand_motor_actuator_index[: self.num_hand_dof]
            r_valid = r_act >= 0
            if np.any(r_valid):
                self.torques[r_act[r_valid]] = hand_torques[self.num_hand_dof :][r_valid]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.held_box:
            self._update_held_box()

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge
        # Share the per-hand BrainCo joint limits (read from the MuJoCo model) with the
        # bridge so its normalized<->rad state feedback uses the exact same affine
        # [lower, upper] mapping as the command path in compute_hand_torques().
        if getattr(unitree_bridge, "is_brainco", False) and hasattr(self, "brainco_lower_limits_left"):
            unitree_bridge.set_brainco_limits(
                self.brainco_lower_limits_left,
                self.brainco_upper_limits_left,
                self.brainco_lower_limits_right,
                self.brainco_upper_limits_right,
            )

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        # Depth/seg are rendered separately (own low-res renderer) and only every Nth frame.
        if self.render_depth_seg and self.fp_renderer is not None:
            if self._fp_frame_counter % self.fp_render_every == 0:
                depth_mm, mask = self._render_ego_depth_seg()
                render_caches["ego_view_depth"] = depth_mm
                render_caches["ego_view_seg"] = mask
            self._fp_frame_counter += 1

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
