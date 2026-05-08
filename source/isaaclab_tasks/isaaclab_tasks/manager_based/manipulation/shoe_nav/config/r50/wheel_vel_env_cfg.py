# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R50 two-wheeled robot specialization of the shoe-navigation env."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.shoe_nav import mdp
from isaaclab_tasks.manager_based.manipulation.shoe_nav.shoe_nav_env_cfg import ShoeNavEnvCfg


R50_WHEELED_USD = (
    "/home/roborock/gitlab5/drl_manipulation/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/"
    "r50_v6_wheeled/r50_v6_wheeled_1.usd"
)


# The arm joints (M0..M6_*) are locked in place by stiff position actuators.
# The two wheel joints are torque/velocity-driven (stiffness=0, only damping),
# so JointVelocityActionCfg can target their joint_vel directly.
R50_WHEELED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R50_WHEELED_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),
        joint_pos={
            "M0": 0.0,
            "M1": 1.57,
            "M3": 3.8,
            "M4": 1.3,
            "M5": 0.0,
            "M6_1": 0.5,
            "M6_2": -0.5,
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    actuators={
        # Base/waist: locked stiffly; M2 is a fixed joint in the URDF.
        "base": ImplicitActuatorCfg(
            joint_names_expr=["M0", "M1"],
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness=400.0,
            damping=20.0,
        ),
        # Arm shoulder/elbow — driven by JointPositionActionCfg on M3, M4.
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["M3", "M4"],
            effort_limit=200.0,
            velocity_limit=10.175,
            stiffness=200.0,
            damping=4.0,
        ),
        # Wrist — driven by JointPositionActionCfg on M5.
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["M5"],
            effort_limit=300.0,
            velocity_limit=2.175,
            stiffness=500.0,
            damping=5.0,
        ),
        # Gripper — driven by BinaryJointPositionActionCfg on M6_*.
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["M6_.*"],
            effort_limit=50.0,
            velocity_limit=40.0,
            stiffness=500.0,
            damping=1.0,
        ),
        # Wheels: velocity-controlled (stiffness=0, damping > 0).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit=10.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class R50ShoeNavEnvCfg(ShoeNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # robot
        self.scene.robot = R50_WHEELED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # depth camera mounted on the on-board ToF link
        self.scene.depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/M0_chassis_link/tof_link/depth_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, -1.0, 0.0, 0.0),
                convention="opengl",
            ),
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=40.0,
                focus_distance=400.0,
                horizontal_aperture=93.5,
            ),
            width=243,
            height=90,
            debug_vis=False,
        )

        # wheel-velocity action: policy outputs are scaled rad/s targets
        self.actions.wheel_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel_joint", "right_wheel_joint"],
            scale=10.0,
            use_default_offset=False,
        )

        # arm position action on M3, M4, M5 (offsets from default joint pos)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["M[345]"],
            scale=0.3,
            use_default_offset=True,
        )

        # binary gripper action on M6_1 / M6_2
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": 1.57, "M6_2": -1.57},
            close_command_expr={"M6_1": 0.02, "M6_2": -0.02},
        )


@configclass
class R50ShoeNavEnvCfg_PLAY(R50ShoeNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
