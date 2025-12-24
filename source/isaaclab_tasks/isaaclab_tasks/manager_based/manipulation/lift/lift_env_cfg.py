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

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame_1: FrameTransformerCfg = MISSING
    finger_frame_2: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # object_id :int=0

    # bear_frame: FrameTransformerCfg = MISSING

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
            pos=[-0.04, 1.2, 0.775],
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

    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Domelight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    # )


    # sphere_light = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/SphereLight",
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=[-0.2, 0.0, 1.0],
    #         rot=[0, 0, 0, 1],
    #     ),
    #     spawn=sim_utils.SphereLightCfg(
    #         color=(1.0, 1.0, 1.0),
    #         intensity=30000.0,
    #         radius=0.4,
    #         enable_color_temperature=True,
    #         color_temperature=5000.0
    #     )
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

    # NEW: Contact sensors on gripper fingers
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


    # prim_path="{ENV_REGEX_NS}/Robot/base_link/M0_chassis_link/tof_link",

    # ray_caster: RayCasterCfg = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link/M0_chassis_link/tof_link",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 1.0)),
    #     # mesh_prim_paths=["/World/GroundPlane"],
    #     mesh_prim_paths=[
    #         "/World/GroundPlane",
    #     ],
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=50, vertical_fov_range=[-90, 90], horizontal_fov_range=[-60, 60], horizontal_res=1.0
    #     ),
    #     debug_vis=True,
    # )


    # tiled_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(1.3, 0.0, 0.9), rot=((0.63281, 0.31551, 0.31551, 0.63281)), convention="opengl"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=38.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=1000,
    #     height=800,
    # )

    # contact_forces: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/M6_2_.*finger_link",  # Adjust to your gripper parts
    #     update_period=0.0,
    #     history_length=1,
    #     # filter_prim_paths_expr=[
    #     #     "{ENV_REGEX_NS}/FloorwithPanels",  # Match your actual ground name
    #     # ],
    #     force_threshold=0.1,
    # )    




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


def point_cloud_from_depth_camera(env, sensor_cfg: SceneEntityCfg, data_type: str = "distance_to_image_plane") -> torch.Tensor:
    """Generate point cloud from depth camera."""
    camera = env.scene[sensor_cfg.name]

    # Get camera data - use data_type parameter instead of sensor_cfg.data_type
    depth_data = camera.data.output[data_type]  # ✓ Use the parameter
    intrinsic_matrices = camera.data.intrinsic_matrices
    camera_positions = camera.data.pos_w
    camera_orientations = camera.data.quat_w_world

    num_envs = depth_data.shape[0]
    point_clouds = []

    for env_idx in range(num_envs):
        if depth_data.dim() == 4:
            depth = depth_data[env_idx, :, :, 0]
        else:
            depth = depth_data[env_idx]

        pointcloud = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsic_matrices[env_idx],
            depth=depth,
            keep_invalid=True,
            position=camera_positions[env_idx],
            orientation=camera_orientations[env_idx],
            device=env.device,
        )
        point_clouds.append(pointcloud)

    point_clouds = torch.stack(point_clouds, dim=0)

    # ============================================================
    # ⭐ ADD THIS: Normalize point clouds to reasonable range
    # ============================================================
    # Point clouds are currently in world coordinates (meters)
    # Step 1: From your Step 1 output, range was [-10.94, 11.23]
    # This suggests workspace is about 22 meters wide (unrealistic)
    # Let's normalize to [-1, 1] assuming 2m workspace (typical for manipulation)

    max_workspace_range = 2.0  # meters from robot base

    # Clip outliers and normalize
    point_clouds = torch.clamp(point_clouds, -max_workspace_range, max_workspace_range)
    point_clouds = point_clouds / max_workspace_range  # Now in [-1, 1]

    # Optional: Print normalization stats (first time only)
    if not hasattr(point_cloud_from_depth_camera, '_debug_printed'):
        print("\n" + "="*80)
        print("🔍 POINT CLOUD NORMALIZATION")
        print("="*80)
        print(f"Original range (from Step 1): [-10.94, 11.23]")
        print(f"Clipping to workspace: ±{max_workspace_range}m")
        print(f"After normalization:")
        print(f"  Range: [{point_clouds.min().item():.4f}, {point_clouds.max().item():.4f}]")
        print(f"  Mean: {point_clouds.mean().item():.4f}")
        print(f"  Std: {point_clouds.std().item():.4f}")
        print("="*80 + "\n")
        point_cloud_from_depth_camera._debug_printed = True

    return point_clouds


@configclass
class RgbPcdObservationCfg:
    
    @configclass
    class RgbPcdPolicyCfg(ObsGroup):

        point_cloud = ObsTerm(
            func = point_cloud_from_depth_camera,
            params = {
                "sensor_cfg": SceneEntityCfg("depth_camera"), 
                "data_type": "distance_to_image_plane",
            }
        )

        # RGB image from gripper camera
        rgb_image = ObsTerm(
            func=mdp.image,
            # func = rgb_image_with_check,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_camera"), 
                "data_type": "rgb",
                "normalize": True  # Normalize to [0,1]
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # This will flatten all observations        


    policy: ObsGroup = RgbPcdPolicyCfg()


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

    # @configclass
    # class SqueezeNetSA1FeaturesCameraPolicyCfg(ObsGroup):
    #     """Observations for policy group with SqueezeNet (RGB) + PointNet SA1-only (Point Cloud)."""

    #     image = ObsTerm(
    #         func=mdp.image_features,
    #         params={"sensor_cfg": SceneEntityCfg("gripper_camera"), "data_type": "rgb","model_name": "squeezenet1_1","depth_cfg":SceneEntityCfg("depth_camera")},
    #     )

    # policy: ObsGroup = SqueezeNetSA1FeaturesCameraPolicyCfg()
    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class GripperCameraObservationCfg:
    """Observation specifications for gripper camera only."""

    @configclass
    class GripperCameraPolicyCfg(ObsGroup):
        """Observations for policy group with gripper RGB camera."""

        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("gripper_camera"), 
                    "data_type": "rgb",
                    "normalize": True,
                    }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = GripperCameraPolicyCfg()



@configclass
class PcdObservationCfg:

    @configclass
    class PcdPolicyCfg(ObsGroup):

        pcd_features = ObsTerm(
            func = mdp.image_features,
            params={
                "depth_cfg": SceneEntityCfg("depth_camera"),
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: ObsGroup = PcdPolicyCfg()



@configclass
class EventCfg:
    """Configuration for events."""

    initialize_cache = EventTerm(
        func=mdp.initialize_point_cloud_cache,
        mode="startup"
    )

    randomize_floor = EventTerm(
        func=mdp.randomize_floor_texture,
        mode="reset",
        params={
            "texture_txt_path": "/home/robo/code/IsaacLab/assets1/3D_assets_usd/floor.txt"
        },
    )
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
                "y": (-0.0, 0.0), 
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

    randomize_lighting_reset = EventTerm(
        func=mdp.randomize_multiple_sphere_lights,
        mode="startup",
        params={"num_lights": 2},
    )

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

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # debug_contact = RewTerm(func=mdp.debug_contact_forces, weight=0.01)

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        # weight=20.0,
        weight=1.0,
    )

    lifting_object_linear = RewTerm(
        func=mdp.object_is_lifted_linear,
        params={"minimal_height": 0.01, "max_height": 0.1},
        weight=5.0,   # 1500  150
        # weight=100.0,   # 1500  150
    )

    # NEW: Lifting with contact verification
    lifting_object_linear_contact = RewTerm(
        func=mdp.object_is_lifted_with_contact,
        params={
            "minimal_height": 0.01,
            "max_height": 0.1,
            "contact_force_threshold": 1.5,  # 1.5N on Y-axis (based on your data)
            "require_both_contacts": True,  # Both fingers must contact
        },
        # weight=500.0,
        weight=15.0,
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
        # weight=20.0,  # Tune this: 5.0-20.0 depending on importance
        weight=2.0,
    )

    clamp_object_contact = RewTerm(
        func=mdp.contact_clamp_object,
        params={
            "contact_force_threshold": 1.5,
            "reward_value": 1.0,
            "gripper_closed_threshold": 0.2,
        },
        # weight=300.0,
        weight=10.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    # # Stage 2: Close gripper when object is inside
    # clamp_object = RewTerm(
    #     func=mdp.pcd_clamp_object,
    #     params={
    #         "density_threshold": 0.05,  # Object detected when 10%+ points in sphere
    #         "density_scale": 1.0,
    #         "gripper_closed_threshold": 0.2,
    #         "min_ee_robot_distance": 0.26,
    #     },
    #     weight=50.0,  # Higher weight since this is the actual grasp
    # )

    # open_gripper = RewTerm(
    #     func=mdp.penalty_if_gripper_closed_far,
    #     params={
    #         "reach_threshold": 0.02,
    #         "open_threshold": 0.04,
    #         "penalty_value": 0.1,
    #     },
    #     weight=-50
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

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.028, "command_name": "object_pose"},
    #     weight=10.0,  # 16.0
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     # params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     params={"std": 0.05, "minimal_height": 0.028, "command_name": "object_pose"},
    #     weight=0.5,  # 5.0
    # )

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # contain_object = RewTerm(
    #     func=mdp.contain_object,
    #     params={"std": 1},
    #     weight=1.0,  # 2.0
    # )

    # clamp_object = RewTerm(
    #     func=mdp.clamp_object,
    #     params={"std": 1},
    #     weight=0.0,  # 2.0
    # )

    # penalize_m0_after_lift = RewTerm(
    #     func=mdp.penalize_m0_after_lift,
    #     weight= 0.0,  # Adjust this weight as needed
    #     params={
    #         "minimal_height": 0.03,  
    #         "m0_movement_penalty_scale": 10.0,  # Higher = stronger penalty
    #     }
    # )

    # debug_density = RewTerm(func=mdp.debug_pcd_density, weight=0.1)
    # visualize_sphere = RewTerm(func=mdp.visualize_pcd_sphere, weight=0.01)

    # ee_to_cluster_centroid = RewTerm(
    #     func=mdp.ee_to_object_cluster_centroid_distance,
    #     params={"std": 0.1},
    #     weight=5.0,  # Adjust this weight based on your needs
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.00},
    )

    # NEW: Terminate if robot orientation is too tilted
    # bad_object_orientation = DoneTerm(
    #     func=mdp.bad_object_orientation,  # Use custom function, not bad_orientation
    #     params={
    #         "limit_angle": 1.15,  # 60 degrees
    #         "object_cfg": SceneEntityCfg("object_pool"),
    #     },
    # )

    gripper_z_force = DoneTerm(
        func=mdp.gripper_z_force_limit,
        params={
          "z_threshold": 30.0,  # Maximum Z-force in Newtons
          "left_sensor_cfg": SceneEntityCfg("contact_forces_left"),
          "right_sensor_cfg": SceneEntityCfg("contact_forces_right"),
          "check_either": True,  # Terminate if either finger exceeds
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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=128, env_spacing=2.5)
    
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()

    # observations: PcdObservationCfg = PcdObservationCfg()

    # observations: RGBObservationsCfg = RGBObservationsCfg()

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
        # self.scene.replicate_physics=False
        # self.decimation = 1  # 2 20 48
        self.decimation = 20  # 2 20 48
        # self.episode_length_s = 6 * self.decimation * self.sim.dt
        # self.decimation = 30  # 2 20 48
        self.episode_length_s = 2
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        # self.sim.render_interval = self.decimation
        self.sim.render_interval = 20
        # self.sim.render_interval = 1

        # self.sim.wait_for_textures = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

