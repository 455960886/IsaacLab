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
    ee_tip_probe_frame:FrameTransformerCfg = MISSING
    gripper_peak:FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # object_id :int=0

    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

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
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=5000.0),
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
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_1",
    #     # offset=TiledCameraCfg.OffsetCfg(pos=(0.16729, -0.02805, 0.055), rot=(( 0.49102 ,0.49805, -0.50194,   -0.50194)), convention="opengl"),
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.162, -0.02805, 0.0575158), rot=(( -0.48897357,-0.51078846,0.51078846,0.48897357)), convention="opengl"),

    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=17.5, focus_distance=400.0, horizontal_aperture=36,vertical_aperture=25.45
    #     ),
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=1.8, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     # ),
    #     width=800,
    #     height=600,
    # )

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        # prim_path="{ENV_REGEX_NS}/depth_camera",
        prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.162, -0.0293681, 0.075), rot=((0.4912, 0.50865, -0.50865, -0.4912)), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=((1.0, 0.0, 0.0, 0.0)), convention="opengl"),
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
        update_period=0.15,
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
    #         fisheye_polynomial_a=0.25,  # Poly k0False
    #         fisheye_polynomial_b=0.0,   # Poly k1
    #         fisheye_polynomial_c=0.0,   # Poly k2
    #         fisheye_polynomial_d=-0.0,  # Poly k3
    #         fisheye_polynomial_e=0.0,   # Poly k4
    #         fisheye_polynomial_f=0.0,   # Poly k5
    #     ),
    #     width=1936,
    #     height=1216,
    #     debug_vis=False,
    #     update_period=0.15,
    # )
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
            # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # pos_x=(0.25, 0.35),
            # pos_y=(-0.05, 0.05),
            # pos_z=(0.25, 0.5),
            # roll=(0.0, 0.0),
            # pitch=(0.0, 0.0),
            # yaw=(0.0, 0.0),
            
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
    '''
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    '''



    # observation groups
    policy: PolicyCfg = PolicyCfg()
    #rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image_features, params={"sensor_cfg": SceneEntityCfg("gripper_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()



@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""
        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("gripper_camera"), "data_type": "rgb","model_name": "resnet18","depth_cfg":SceneEntityCfg("depth_camera")},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         # "pose_range": {"x": (-0.05, 0.05), "y": (-0.25, 0.25), "z": (0.0, 0.0)},

    #         "pose_range": {
    #             "x": (-0.02, 0.02),
    #             "y": (0.00, 0.00),
    #             "z": (0.0, 0.0),
    #             # "yaw": (-0.5, 0.5),
    #         },

            
    #         "velocity_range": {},
    #     },
    # )
    # object_pool_spawn = EventTerm(
    #     func=mdp.randomize_object_pool_selection,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("object_pool")},
    # )
    reset_object_or_paper_and_position = EventTerm(
        func=mdp.randomize_object_and_position,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.00, 0.05),
                "y": (0.002, 0.006),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "rigid_asset_cfg": SceneEntityCfg("object_pool"),
        },
    )
    randomize_lighting_reset = EventTerm(
        func=mdp.randomize_multiple_sphere_lights,
        mode="reset",
        params={"num_lights": 2},
    )
    
    randomize_floor = EventTerm(
        func=mdp.randomize_floor_texture,
        mode="reset",
        params={
            "texture_txt_path" :"/home/robo/桌面/floor.txt"
        },
    )
    randomize_wall = EventTerm(
        func=mdp.randomize_wall_texture,
        mode="reset",
        params={
            "texture_txt_path" :"/home/robo/桌面/floor.txt"
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=40,
    )

    lifting_object_linear = RewTerm(
        func=mdp.object_is_lifted_linear,
        params={"minimal_height": 0.03, "max_height": 0.35},
        weight=100.0,   # 1500  150
    )

    lifting_object_linear_contact = RewTerm(
        func=mdp.object_is_lifted_with_contact,
        params={
            "minimal_height": 0.03,
            "max_height": 0.35,
            "contact_force_threshold": 0.5,  # 1.5N on Y-axis (based on your data)
            "require_both_contacts": True,  # Both fingers must contact
        },
        weight=500.0,
    )
    contain_object = RewTerm(
        func=mdp.contain_object,
        params={
            "std" : 1,
        },
        weight=30.0,  # Tune this: 5.0-20.0 depending on importance
    )
    # # NEW: Point cloud density reward
    # contain_object = RewTerm(
    #     func=mdp.pcd_contain_object,
    #     params={
    #         "density_scale": 1.0,
    #         "use_tanh": True,  # Set True for smoother gradients
    #         "min_ee_robot_distance": 0.26,
    #         "max_ee_height": 0.35,
    #     },
    #     weight=50.0,  # Tune this: 5.0-20.0 depending on importance
    # )
    # close_gripper = RewTerm(
    #     func=mdp.penalty_if_gripper_closed_far,
    #     params={
    #         "reach_threshold": 0.02,
    #         "open_threshold": 0.04,
    #         "penalty_value": 0.1,
    #     },
    #     weight=10
    # )
    # early_close = RewTerm(
    #     func=mdp.close_action_smooth_reward,
    #     weight=-10
    # )


    # clamp_object_contact = RewTerm(
    #     func=mdp.clamp_object,
    #     params={"std": 1},
    #     weight=20.0,
    # )

    # action penalty
    # action_rate = RewTerm(func=mdp.action_l2, weight=-0.0001)
    # visualize_sphere = RewTerm(func=mdp.visualize_pcd_sphere, weight=0.01)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # slide_penalty = RewTerm(
    #     func=mdp.penalize_xy_displacement,
    #     weight= -1.0,  # Negative weight since function returns negative values
    #     params={
    #         "penalty_scale": 1000,    # Adjust sensitivity
    #         "min_height": -0.05,     # Height where penalty is maximum
    #         "max_height": 0.1,       # Height where penalty becomes zero
    #         "object_cfg": SceneEntityCfg("object_pool")
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.0},
    )
    # gripper_z_force = DoneTerm(
    #     func=mdp.gripper_z_force_limit,
    #     params={
    #       "z_threshold": 30.0,  # Maximum Z-force in Newtons
    #       "left_sensor_cfg": SceneEntityCfg("contact_forces_left"),
    #       "right_sensor_cfg": SceneEntityCfg("contact_forces_right"),
    #       "check_either": True,  # Terminate if either finger exceeds
    #     },
    # )

    # NEW: Terminate if robot orientation is too tilted
    # bad_object_orientation = DoneTerm(
    #     func=mdp.bad_object_orientation,  # Use custom function, not bad_orientation
    #     params={
    #         "limit_angle": 1.15,  # 60 degrees
    #         "object_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )


    # object_pushed = DoneTerm(
    #     func=mdp.object_pushed_away,
    #     params={
    #         "x_limits": (0.24, 0.44),
    #         "y_tolerance": 0.08,
    #         "object_cfg": SceneEntityCfg("object_pool"),
    #         "robot_cfg": SceneEntityCfg("robot")
    #     },
    # )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #action_rate = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    #)

    #joint_vel = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    #)


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=32, env_spacing=2.5)
    # Basic settings
    # observations: ObservationsCfg = ObservationsCfg()
    #observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
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

        # general settings
        self.decimation = 20  # 2 20 48
        # self.decimation = 20
        

        # simulation settings
        self.sim.dt = 0.01# 100Hz
        self.sim.render_interval =20

        self.episode_length_s = 6*self.decimation* self.sim.dt 
        # self.episode_length_s = 20

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # self.sim.physx.contact_offset = 0.04
        # self.sim.physx.rest_offset = 0.0
        # self.sim.physx.solver_type    = "TGS"
