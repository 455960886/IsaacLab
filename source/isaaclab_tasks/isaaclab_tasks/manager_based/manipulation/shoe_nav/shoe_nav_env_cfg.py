# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment config for the wheeled-robot shoe-navigation task.

Each env contains:
  * one shoe (rigid body) at the env origin,
  * one wheeled R50 robot spawned at a random pose around the shoe,
  * a depth camera mounted on the robot's tof_link.

The robot observes a point cloud derived from its depth camera and outputs
left/right wheel velocities. As is standard for IsaacLab, every env is its
own clone — robots in different envs neither see nor collide with each other.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp


SHOE_USD = "/home/roborock/gitlab5/drl_manipulation/assets/baisetuoxie/baisetuoxie.usdc"


##
# Scene
##


@configclass
class ShoeNavSceneCfg(InteractiveSceneCfg):
    """Per-env scene: ground, lights, shoe in the middle, wheeled robot."""

    # populated by derived (robot-specific) config
    robot: ArticulationCfg = MISSING

    # the shoe sits at the env origin; reset events nudge its yaw / position
    shoe: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shoe",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHOE_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(size=(500.0, 500.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )

    # populated by derived robot cfg (its prim_path depends on the robot's link names)
    depth_camera: TiledCameraCfg = MISSING

    # contact sensor over all robot bodies, filtered to only count contacts with the shoe
    robot_shoe_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Shoe"],
        debug_vis=False,
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """Two-wheel differential drive + 3-DoF arm position + binary gripper."""

    wheel_action: mdp.JointVelocityActionCfg = MISSING
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """PointNet features extracted from the on-board depth camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        pcd = ObsTerm(
            func=mdp.pointnet_features,
            params={"depth_cfg": SceneEntityCfg("depth_camera")},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()


@configclass
class EventCfg:
    """Reset events. Reward / curriculum tuning is left for the user."""

    initialize_pcd_cache = EventTerm(func=mdp.initialize_point_cloud_cache, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # nudge the shoe's yaw so the "back" direction varies
    reset_shoe = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-3.1416, 3.1416),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("shoe"),
        },
    )

    # spawn the robot somewhere around the shoe
    reset_robot = EventTerm(
        func=mdp.reset_robot_around_object,
        mode="reset",
        params={
            "radius_range": (0.3, 0.6),
            "yaw_noise": 3.1416,  # full random yaw initially
            "asset_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("shoe"),
        },
    )


@configclass
class RewardsCfg:
    """Placeholder rewards — user will iterate."""

    reach_back = RewTerm(
        func=mdp.distance_to_back_of_shoe,
        params={"behind_offset": 0.25, "std": 0.5},
        weight=2.0,
    )

    face_shoe = RewTerm(
        func=mdp.heading_alignment_to_shoe,
        weight=0.5,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # robot tipped over
    base_tilted = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )

    # any robot link bumped the shoe
    touched_shoe = DoneTerm(
        func=mdp.robot_object_contact,
        params={
            "threshold": 0.001,
            "sensor_cfg": SceneEntityCfg("robot_shoe_contact"),
        },
    )


##
# Env
##


@configclass
class ShoeNavEnvCfg(ManagerBasedRLEnvCfg):
    """Base config; the robot articulation, depth-camera prim path, and
    wheel-action joint names are filled in by the robot-specific subclass."""

    scene: ShoeNavSceneCfg = ShoeNavSceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 100 Hz sim, 10 Hz control
        self.sim.dt = 0.01
        self.decimation = 80
        # self.sim.render_interval = self.decimation
        self.sim.render_interval = 1
        self.episode_length_s = 40
        self.rerender_on_reset = True

        self.sim.physx.bounce_threshold_velocity = 0.01
