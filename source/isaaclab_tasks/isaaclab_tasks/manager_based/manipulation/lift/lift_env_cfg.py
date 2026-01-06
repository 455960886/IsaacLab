# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCameraCfg,CameraCfg, ContactSensorCfg

from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
# from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg

import torch
import math
import torch.nn as nn
# from .custom_ray_caster import FixedRayCaster

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    replicate_physics: bool = False
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame_1: FrameTransformerCfg = MISSING
    finger_frame_2: FrameTransformerCfg = MISSING
    ee_tip_probe_frame: FrameTransformerCfg = MISSING
    gripper_peak: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # FloorWithPanels
    FloorWithPanels = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorwithPanels",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[0, 0, 0, 1],
        ),
        spawn=UsdFileCfg(usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/FloorWithPanels.usd"),
    )

    # lights

    sphere_light_0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight_0",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[2.5, 0.0, 4.5],
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
            radius=0.4,
            enable_color_temperature=True,
            color_temperature=5000.0
        )
    )

    sphere_light_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[5.0, 0.0, 4.5],
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
            radius=0.4,
            enable_color_temperature=True,
            color_temperature=5000.0
        )
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3500.0),
    )

    sphere_light_0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight_0",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[2.5, 0.0, 4.5],
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
            radius=0.4,
            enable_color_temperature=True,
            color_temperature=5000.0
        )
    )

    sphere_light_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[5.0, 0.0, 4.5],
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
            radius=0.4,
            enable_color_temperature=True,
            color_temperature=5000.0
        )
    )

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        # prim_path="{ENV_REGEX_NS}/depth_camera",
        prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((0.0, -1.0, 0.0, 0.0)), convention="opengl"),
        data_types=["distance_to_image_plane"],  # Key change to depth
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40, focus_distance=400.0, horizontal_aperture=36, vertical_aperture=25.45,
        ),
        width=200,
        height=150,
        debug_vis=False,
        # update_period=0.2,
    )

    gripper_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M5_wrist_link/camera_Link/gripper_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.00009, -0.00402),
            rot=((0.04717, -0.99889, 0.0, 0.0)),
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.6,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,    
        ),
        width=640,
        height=480,
        debug_vis=False,
        update_period=0.2,
    )

    contact_forces_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",  # Left gripper finger link
        update_period=0.0,  # Update every step
        history_length=5,
        track_air_time=False,
        debug_vis=False,
    )

    contact_forces_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M6_2_rightfinger_link",  # Right gripper finger link
        update_period=0.0,
        history_length=5,
        track_air_time=False,
        debug_vis=False,
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.3),
            pos_y=(-0.01, 0.01),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # arm_action: mdp.RelativeJointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("gripper_camera"),
                    "data_type": "rgb",
                    "model_name": "resnet18",
                    "depth_cfg": SceneEntityCfg("depth_camera")},
        )

        # def __post_init__(self):
        #     self.enable_corruption = True
        #     self.concatenate_terms = True

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    initialize_cache = EventTerm(
        func=mdp.initialize_point_cloud_cache,
        mode="startup"
    )
    # randomize_bus_texture = EventTerm(
    #     func=mdp.randomize_bus_texture_event,
    #     mode="reset",
    #     params={
    #         "bus_name": "bus",
    #         "body_name": "Xform",
    #         "texture_paths": "/home/robo/code/IsaacLab/assets1/3D_assets_usd/car",
    #         "event_name": "randomize_bus_texture",
    #         "texture_rotation": (0.0, 2 * math.pi),
    #     },
    # )
    # randomize_floor = EventTerm(
    #     func=mdp.randomize_floor_texture,
    #     mode="reset",
    #     params={
    #         "texture_txt_path": "/home/robo/code/IsaacLab/assets1/3D_assets_usd/floor.txt"
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # set_rt_subframes = EventTerm(
    #     func=mdp.set_camera_rt_subframes, 
    #     mode="startup", 
    #     params={
    #         "subframes": 4,
    #     },
    # )

    object_pool_spawn = EventTerm(
        func=mdp.randomize_object_pool_selection,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object_pool")},
    )

    reset_object_position = EventTerm(
        func=mdp.reset_object_pool_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.07), 
                "y": (-0.03, 0.03), 
                # "y": (0.0, 0.0), 
                "z": (0.0, 0.0), 
                # "roll": (-0.1, 0.1),    
                # "pitch": (-0.1, 0.1),   
                # "yaw": (-0.3, 0.3),    
                "roll": (0.0, 0.0),    
                "pitch": (0.0, 0.0),   
                "yaw": (0.0, 0.0), 
            }, 
            "velocity_range": {}, 
            "asset_cfg": SceneEntityCfg("object_pool"), 
        }, 
    )

    # randomize_lighting_reset = EventTerm(
    #     func=mdp.randomize_multiple_sphere_lights,
    #     mode="startup",
    #     params={"num_lights": 2},
    # )

    # randomize_lighting_reset = EventTerm(
    #     func=mdp.randomize_sphere_light_intensity,
    #     mode="reset",
    #     params={
    #         "intensity_range": (2000.0, 50000.0),
    #     },
    # )

    # randomize_light_reset_2 = EventTerm(
    #     func=mdp.randomize_light_color_temperature,
    #     mode="reset",
    #     params={
    #         "temperature_range": (4500.0, 11000.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    # debug_contact = RewTerm(func=mdp.debug_contact_forces, weight=0.01)

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=5.0,
    )

    lifting_object_linear = RewTerm(
        func=mdp.object_is_lifted_linear,
        params={"minimal_height": 0.025, "max_height": 0.1},
        weight=100.0,   # 1500  150
    )

    # NEW: Lifting with contact verification
    lifting_object_linear_contact = RewTerm(
        func=mdp.object_is_lifted_with_contact,
        params={
            "minimal_height": 0.025,
            "max_height": 0.1,
            "contact_force_threshold": 1.5,  # 1.5N on Y-axis (based on your data)
            "require_both_contacts": True,  # Both fingers must contact
        },
        weight=1000.0,
    )

    # NEW: Point cloud density reward
    pcd_contain_object = RewTerm(
        func=mdp.pcd_contain_object,
        params={
            "density_scale": 1.0,
            "use_tanh": True,  # Set True for smoother gradients
            "min_ee_robot_distance": 0.26,
            "max_ee_height": 0.06,
        },
        weight=10.0,  # Tune this: 5.0-20.0 depending on importance
    )

    clamp_object_contact = RewTerm(
        func=mdp.contact_clamp_object,
        params={
            "contact_force_threshold": 1.5,
            "reward_value": 1.0,
            "gripper_closed_threshold": 0.2,
        },
        weight=300.0,
    )

    # action penalty
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    contain_object = RewTerm(
        func=mdp.contain_object,
        params={"std": 1},
        weight=30.0,  # 2.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.0},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=64, env_spacing=2.5)

    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()

    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):

        """Post initialization."""
        self.sim.dt = 0.01  # 100Hz
        self.decimation = 20  # 2 20 48
        self.episode_length_s = 6 * self.decimation * self.sim.dt
        # self.decimation = 1
        # self.episode_length_s = 10
        self.sim.render_interval = self.decimation
        # self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024          # 256 MB
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024   # 128 MB

        self.sim.physx.gpu_max_rigid_contact_count = 2_000_000        # 接触对上限
        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000          # 接触 patch 上限

        self.sim.physx.gpu_collision_stack_size = 96 * 1024 * 1024    # ≈ 100 MB

