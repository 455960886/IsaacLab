# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg

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
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame_1: FrameTransformerCfg = MISSING
    finger_frame_2: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING
    object_pool: RigidObjectCollectionCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    FloorWithPanels = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorwithPanels",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.04, 1.2, 0.775),
            rot=(0, 0, 0, 1),
        ),
        spawn=UsdFileCfg(usd_path="/home/robo/code/IsaacLab/assets/FloorWithPanels.usd"),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Domelight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.2, 0.0, 1.0),
            rot=(0, 0, 0, 1),
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
        prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=((0.0, -1.0, 0.0, 0.0)),
            convention="opengl"),
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        # data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=26.29,
            focus_distance=400.0,
            horizontal_aperture=36,
            vertical_aperture=25.45,
        ),
        width=200,
        height=150,
        debug_vis=False,
        update_period=0.05,
        colorize_semantic_segmentation=False,
    )

    # gripper_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/M5_wrist_link/camera_Link/gripper_camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.0, -0.00037, -0.00082),
    #         rot=(0, -1, 0.0, 0.0),
    #         convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         focal_length=18.14756,       # Focal Length
    #         focus_distance=400.0,        # Focus Distance
    #         f_stop=0.0,                  # fStop
    #         horizontal_aperture=20.955,  # Horizontal Aperture
    #         vertical_aperture=15.2908,   # Vertical Aperture

    #         projection_type="fisheyeRadTanThinPrism",  # Projection Type

    #         fisheye_nominal_width=1936.0,   # Nominal Width
    #         fisheye_nominal_height=1216.0,  # Nominal Height

    #         fisheye_optical_centre_x=970.94244,  # Optical Center X
    #         fisheye_optical_centre_y=600.37482,  # Optical Center Y

    #         fisheye_max_fov=200.0,  # Max FOV

    #         fisheye_polynomial_a=0.25,  # Poly k0
    #         fisheye_polynomial_b=0.0,   # Poly k1
    #         fisheye_polynomial_c=0.0,   # Poly k2
    #         fisheye_polynomial_d=-0.0,  # Poly k3
    #         fisheye_polynomial_e=0.0,   # Poly k4
    #         fisheye_polynomial_f=0.0,   # Poly k5
    #     ),
    #     width=640,
    #     height=480,
    #     debug_vis=False,
    #     update_period=0.15,
    # )
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

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # arm_action: mdp.RelativeJointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""
        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("gripper_camera"), "data_type": "rgb", "model_name": "resnet18", "depth_cfg": SceneEntityCfg("depth_camera")},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    initialize_cache = EventTerm(
        func=mdp.initialize_point_cloud_cache1,
        mode="startup"
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # lsm版本
    reset_object_or_paper_and_position = EventTerm(
        func=mdp.randomize_object_and_position,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.03, 0.02),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "rigid_asset_cfg": SceneEntityCfg("object_pool"),
        },
    )

    # debug_semantic_on_reset = EventTerm(
    #     func=mdp.debug_print_semantic_ids_on_reset,
    #     mode="reset",
    # )

    randomize_lighting_reset = EventTerm(
        func=mdp.randomize_multiple_sphere_lights,
        mode="reset",
        params={"num_lights": 1},
    )
    # randomize_floor = EventTerm(
    #     func=mdp.randomize_floor_texture,
    #     mode="reset",
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    ############################################### 1. reach ###############################################
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=2,
    )

    ############################################### 2. contain ###############################################
    # NEW: Point cloud density reward
    # contain_object = RewTerm(
    #     func=mdp.pcd_contain_object,
    #     params={
    #         "density_scale": 1.0,
    #         "use_tanh": True,  # Set True for smoother gradients
    #         "min_ee_robot_distance": 0.26,
    #     },
    #     weight=30.0,  # Tune this: 5.0-20.0 depending on importance
    # )
    contain_object = RewTerm(
        func=mdp.pcd_contain_object_semantic,
        params={
            "density_scale": 1.0,
            "use_tanh": True,  # Set True for smoother gradients
            "min_ee_robot_distance": 0.26,
            "valid_object_name": "eye_drops",
            "excluded_object_names": ["m6_1_leftfinger_link", "m6_2_rightfinger_link", "m5_wrist_link"],
        },
        weight=400.0,
    )
    # debug_semantic_pcd = RewTerm(
    #     func=mdp.debug_semantic_pcd_density,
    #     params={
    #         "sensor_cfg_name": "depth_camera",
    #         "valid_object_name": "eye_drops",
    #         "excluded_object_names": ["m6_1_leftfinger_link", "m6_2_rightfinger_link", "m5_wrist_link"],
    #         "log_interval": 50,           # 想每步打就改成 1
    #         "env_id_to_print": 0,
    #         "density_scale": 1.0,
    #         "use_tanh": True,
    #         "min_ee_robot_distance": 0.26,
    #         "contact_z_threshold": 0.7,
    #         "contact_force_threshold": 1.5,
    #         "require_both_contacts": True,
    #     },
    #     weight=0.00001,
    # )

    ############################################### 3. clamp ###############################################
    clamp_object_contact = RewTerm(
        func=mdp.contact_clamp_object,
        params={
            "contact_force_threshold": 1.5,
            "reward_value": 1.0,
            "gripper_closed_threshold": 0.2,
        },
        weight=150.0,
    )

    ################################################ 4. lift ###############################################
    lifting_object_linear = RewTerm(
        func=mdp.object_is_lifted_linear,
        params={"minimal_height": 0.01, "max_height": 0.06},
        weight=100.0,   # 1500  150
    )

    lifting_object_linear_contact = RewTerm(
        func=mdp.object_is_lifted_with_contact,
        params={
            "minimal_height": 0.01,
            "max_height": 0.1,
            "contact_force_threshold": 0.5,  # 1.5N on Y-axis (based on your data)
            "require_both_contacts": True,  # Both fingers must contact
        },
        weight=800.0,
    )

    ################################################ 5. penalty ###############################################
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # visualize_sphere = RewTerm(func=mdp.visualize_pcd_sphere, weight=0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.00},
    )

    object_pushed = DoneTerm(
        func=mdp.object_pushed_away,
        params={
            "x_limits": (0.24, 0.45),
            "y_tolerance": 0.06,
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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=5, env_spacing=7)
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
        self.decimation = 5  # 2 20 48
        self.episode_length_s = 0.35
        # self.decimation = 1
        # self.episode_length_s = 10
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        # self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024          # 256 MB
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024   # 128 MB

        self.sim.physx.gpu_max_rigid_contact_count = 2_000_000        # 接触对上限
        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000          # 接触 patch 上限

        self.sim.physx.gpu_collision_stack_size = 96 * 1024 * 1024    # ≈ 100 MB