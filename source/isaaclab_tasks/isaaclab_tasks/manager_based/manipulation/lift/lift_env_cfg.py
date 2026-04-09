# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg


# from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg

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

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame_1: FrameTransformerCfg = MISSING
    finger_frame_2: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # room
    FloorWithPanels = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorwithPanels",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[0, 0, 0, 1],
        ),
        spawn=UsdFileCfg(usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/FloorWithPanels.usd"),
    )

    # Global lights (3 sphere lights covering entire training area)
    # With 64 envs at 7m spacing (8x8 grid = ~56m x 56m), lights positioned high to cover all
    global_light_0 = AssetBaseCfg(
        prim_path="/World/GlobalLight_0",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 25.0],  # Center, high above
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=500000.0,  # Higher intensity for global coverage
            radius=5.0,  # Large radius for soft shadows
            enable_color_temperature=True,
            color_temperature=5000.0
        )
    )

    global_light_1 = AssetBaseCfg(
        prim_path="/World/GlobalLight_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-20.0, -20.0, 20.0],  # Front-left corner
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=400000.0,
            radius=5.0,
            enable_color_temperature=True,
            color_temperature=4500.0
        )
    )

    global_light_2 = AssetBaseCfg(
        prim_path="/World/GlobalLight_2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[20.0, 20.0, 20.0],  # Back-right corner
            rot=[0, 0, 0, 1],
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=400000.0,
            radius=5.0,
            enable_color_temperature=True,
            color_temperature=5500.0
        )
    )

    # depth_camera: TiledCameraCfg = TiledCameraCfg(
    #     # prim_path="{ENV_REGEX_NS}/depth_camera",
    #     prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.162, -0.0293681, 0.075), rot=((0.4912, 0.50865, -0.50865, -0.4912)), convention="opengl"),
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((1.0, 0.0, 0.0, 0.0)), convention="opengl"),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((0.0, -1.0, 0.0, 0.0)), convention="opengl"),
    #     data_types=["distance_to_image_plane"],  # Key change to depth
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=40, focus_distance=400.0, horizontal_aperture=36, vertical_aperture=25.45,
    #     ),
    #     width=200,
    #     height=150,
    #     debug_vis=False,
    #     # update_period=0.2,
    # )

    # depth_camera: TiledCameraCfg = TiledCameraCfg(
    #     # prim_path="{ENV_REGEX_NS}/depth_camera",
    #     prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.162, -0.0293681, 0.075), rot=((0.4912, 0.50865, -0.50865, -0.4912)), convention="opengl"),
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((1.0, 0.0, 0.0, 0.0)), convention="opengl"),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((0.0, -1.0, 0.0, 0.0)), convention="opengl"),
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((0.01309, -0.99991, 0.0, 0.0)), convention="opengl"),
    #     data_types=["distance_to_image_plane"],  # Key change to depth
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=40, focus_distance=400.0, horizontal_aperture=80, vertical_aperture=25.45,
    #     ),
    #     width=400,
    #     height=150,
    #     debug_vis=False,
    #     # update_period=0.2,
    # )

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((0.0, -1.0, 0.0, 0.0)), convention="opengl"),
        data_types=["distance_to_image_plane"],  # Key change to depth
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40, focus_distance=400.0, horizontal_aperture=93.5,
        ),
        width=243,
        height=90,
        debug_vis=False,
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
        update_period=0.15,
    )

    contact_forces_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",  # Left gripper finger link
        update_period=0.0,  # Update every step
        history_length=10,
        track_air_time=False,
        debug_vis=False,
    )

    contact_forces_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/M6_2_rightfinger_link",  # Right gripper finger link
        update_period=0.0,
        history_length=10,
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
            pos_y=(0.0, 0.0),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.RelativeJointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    wrist_action: mdp.JointPositionActionCfg | None = None
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "depth_cfg": SceneEntityCfg("depth_camera"),
            },
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_with_binary_m6_latched,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["M[345]", "M6_.*"]),
                "action_name": "gripper_action",
                "action_index": 0,
                "m6_open_value": 1,
                "m6_close_value": 0,
                "toggle_threshold": 0,
                "debug": False,
                "debug_every": 200,
            },
        )

    @configclass
    class CriticPrivilegedObsCfg(ObsGroup):
        """Low-dimensional privileged observations for critic only."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_with_binary_m6_latched,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["M[345]", "M6_.*"]),
                "action_name": "gripper_action",
                "action_index": 0,
                "m6_open_value": 1,
                "m6_close_value": 0,
                "toggle_threshold": 0,
                "debug": False,
                "debug_every": 200,
            },
        )
        object_yaw = ObsTerm(
            func=mdp.active_object_yaw,
            params={"object_cfg": SceneEntityCfg("object_pool")},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()
    critic: ObsGroup = CriticPrivilegedObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    initialize_cache = EventTerm(
        func=mdp.initialize_point_cloud_cache,
        mode="startup"
    )

    randomize_object_pool_scale = EventTerm(
        func=mdp.randomize_object_pool_scale_prestartup,
        mode="prestartup",
        params={
            "scale_factor_range": (0.95, 1.05),
            "asset_cfg": SceneEntityCfg("object_pool"),
            # 是否打印每个 env 的每个物体最终 scale。
            # 当前任务是 128 个 env * 8 个物体，会输出 1024 行。
            # 这里只做尺寸随机化和缓存，不在这里逐环境打印。
            "log_per_env_scales": False,

        },
    )

    randomize_floor = EventTerm(
        func=mdp.randomize_floor_texture,
        mode="reset",
        params={
            "texture_txt_path": "/home/robo/code/IsaacLab/assets/Floor/floor.txt"
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    object_pool_spawn = EventTerm(
        func=mdp.randomize_object_pool_selection,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("object_pool")},
    )

    # 注意这里要放在 object_pool_spawn 后面，确保 active_object_indices 已经确定
    # log_active_object_pool_scale = EventTerm(
    #     func=mdp.log_active_object_pool_scale,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("object_pool")},
    # )

    reset_object_position = EventTerm(
        func=mdp.reset_object_pool_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.03, 0.02),
                # "x": (-0.03, 0.1),
                "y": (-0.015, 0.015),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0, 0),
                "yaw": (-3.0, 0.015),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_pool"),
        },
    )

    # reset_object_position_slippers_m5_top1 = EventTerm(
    #     func=mdp.reset_object_pool_state_uniform_for_object1,
    #     mode="reset",
    #     params={
    #         "object_names": ["baisetuoxie"],
    #         "pose_range": {
    #             "x": (0.02, 0.05),
    #             "y": (-0.04, -0.01),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         # "yaw_ranges": [(-1.0, 0.0), (-6.28, -5.28)],  # Two separate yaw ranges to encourage top-down and side orientations
    #         "yaw_ranges": [(-1.0, 0.0)],
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )

    # reset_object_position_slippers_m5_top2 = EventTerm(
    #     func=mdp.reset_object_pool_state_uniform_for_object2,
    #     mode="reset",
    #     params={
    #         "object_names": ["fensemiantuo"],
    #         "pose_range": {
    #             "x": (0.02, 0.05),
    #             "y": (-0.02, 0.01),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "yaw_ranges": [(-1.0, 0.0), (-6.28, -5.28)],  # Two separate yaw ranges to encourage top-down and side orientations
    #         # "yaw_ranges": [(-1.0, 0.0)],
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )

    # reset_object_position_slippers_m5_top3 = EventTerm(
    #     func=mdp.reset_object_pool_state_uniform_for_object3,
    #     mode="reset",
    #     params={
    #         "object_names": ["slippers_m5_0"],
    #         "pose_range": {
    #             "x": (-0.01, 0.03),
    #             "y": (-0.01, 0.01),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "yaw_ranges": [(-0.5, 0.0), (-6.1, -5.6)],  # Two separate yaw ranges to encourage top-down and side orientations
    #         # "yaw_ranges": [(-1.1, 0.0)],
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )

    # reset_object_position_slippers_m5_top4 = EventTerm(
    #     func=mdp.reset_object_pool_state_uniform_for_object4,
    #     mode="reset",
    #     params={
    #         "object_names": ["slippers"],
    #         "pose_range": {
    #             "x": (-0.05, 0.01),
    #             "y": (-0.01, 0.01),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "yaw_ranges": [(-0.8, 0.0), (-6.1, -5.5)],  # Two separate yaw ranges to encourage top-down and side orientations
    #         # "yaw_ranges": [(-1.1, 0.0)],
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )

    randomize_lighting_interval = EventTerm(
        func=mdp.randomize_global_sphere_lights,
        mode="interval",
        interval_range_s=(1, 1),  # Randomize every 0.1 seconds
        is_global_time=True,
        params={
            "light_paths": ["/World/GlobalLight_0", "/World/GlobalLight_1", "/World/GlobalLight_2"],
            "intensity_range": (2000.0, 150000.0),
            "temperature_range": (1800.0, 12000.0),
            "color_variation": 0.6,
            "position_variation": (8.0, 8.0, 5.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # debug_contact = RewTerm(func=mdp.debug_contact_forces, weight=0.01)

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=5.0,
    )

    lifting_object_linear = RewTerm(
        func=mdp.object_is_lifted_linear,
        params={"minimal_height": 0.09, "max_height": 0.3},
        weight=50.0,   # 1500  150
    )

    # NEW: Lifting with contact verification
    lifting_object_linear_contact = RewTerm(
        func=mdp.object_is_lifted_with_contact,
        params={
            "minimal_height": 0.09,
            "max_height": 0.3,
            "contact_force_threshold": 1.5,  # 1.5N on Y-axis (based on your data)
            "require_both_contacts": True,  # Both fingers must contact
        },
        weight=100.0,
    )

    pcd_contain_object = RewTerm(
        func=mdp.pcd_contain_object,
        params={
            "density_scale": 1.0,
            "use_tanh": True,  # Set True for smoother gradients
            "min_ee_robot_distance": 0.26,
            "max_ee_height": 0.06,
            "excluded_objects": ["slipper"],
        },
        # weight=20.0,  # Tune this: 5.0-20.0 depending on importance
        weight=2.0,
    )

    # contain_object = RewTerm(
    #     func=mdp.contain_object,
    #     params={"std": 1},
    #     weight=2.0,  # 2.0
    # )

    clamp_object_contact = RewTerm(
        func=mdp.contact_clamp_object,
        params={
            "contact_force_threshold": 0.2,
            "reward_value": 1.0,
            "gripper_closed_threshold": 0.2,
        },
        weight=30.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    # M5 (wrist) alignment with object orientation
    wrist_alignment = RewTerm(
        func=mdp.wrist_object_orientation_alignment,
        params={
            "std": 0.5,  # Smaller = sharper reward peak (more precise alignment required)
            "peak_shift_object_names": ["baisetuoxie", "fensemiantuo"],  # Objects that benefit from a specific wrist orientation
            "peak_shift_value": math.pi / 2,
        },
        weight=5.0,  # Positive reward for good alignment
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if robot base orientation is tilted too much
    robot_base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.05,  # 0.5 rad ≈ 28.6° tilt limit
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    object_pushed = DoneTerm(
        func=mdp.object_pushed_away,
        params={
            "x_limits": (0.15, 0.6),
            "y_tolerance": 0.08,
            "object_cfg": SceneEntityCfg("object_pool"),
            "robot_cfg": SceneEntityCfg("robot")
        },
    )


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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=128, env_spacing=2)
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
        self.decimation = 40  # 2 20 48
        self.episode_length_s = 10 * self.decimation * self.sim.dt

        # self.decimation = 1
        # self.episode_length_s = 10

        self.sim.render_interval = self.decimation
        # self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024          # 256 MB
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024   # 128 MB

        self.sim.physx.gpu_max_rigid_contact_count = 2_000_000        # 接触对上限
        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000          # 接触 patch 上限

        self.sim.physx.gpu_collision_stack_size = 96 * 1024 * 1024    # ≈ 100 MB