# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np
import random

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/roborock/data/private/shengmei/IsaacLab/assets1/r50_v6_rev/r50_v6_rev.usd",
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False  ,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=32,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.005),
        joint_pos={
            # "M0": 0,    # 锁死
            # "M1": 1.57,  # 锁死
            # "M2": 1.57,  # 锁死
            "M3": 3.8,
            "M4": 1.4,
            # "M5": 0.0,
            "M6_1": 0,
            "M6_2": 0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["M[0-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,  # 2.175  0.17  0.5
            stiffness=60,
            damping=4,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["M5"],
            effort_limit=12.0,
            velocity_limit=0.5,  # 2.61  0.17  0.5
            stiffness=80.0,
            damping=4.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["M6_.*"],
            effort_limit=0.5,
            velocity_limit=15,
            stiffness=80,
            damping=4,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    debug_vis=False,
)


@configclass
class CoarseArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set CoarseArm as robot
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot", joint_names=["M[0345]"]
        # )
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["M[34]"],
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": 0.65, "M6_2": -0.65},  
            close_command_expr={"M6_1": 0.03, "M6_2": -0.03},
        )

        self.commands.object_pose.body_name = "M6_1_leftfinger_link"
        # self.commands.object_pose.body_name = "M6_1_rightfinger_link"

        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object_Dummy",
            spawn=sim_utils.CuboidCfg(
                size=(0.001, 0.001, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, -100.0]),
        )
        from isaaclab.assets import RigidObjectCollectionCfg

        self.scene.object_pool = RigidObjectCollectionCfg(
            rigid_objects={
                # "bus": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/bus",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/data/private/shengmei/IsaacLab/assets1/3D_assets_usd/bus_2.usd",
                      
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=64,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.00, 0.018), rot=(0.7071, 0.0, 0.0, 0.7071)),
                # ),
                "gongzixing_lego": RigidObjectCfg(
                    prim_path="/World/envs/env_.*/gongzixing_lego",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/roborock/data/private/shengmei/IsaacLab/assets1/3D_assets_usd/lego_1.usd",
                        scale=(1.0, 2.0, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=64,
                            solver_velocity_iteration_count=32,
                            disable_gravity=False,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.005, 0.0], rot=[0.5, 0.5, 0.5, 0.5]),
                ),
                # "eye_drops": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/eye_drops",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source_lsm/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/eyesdrop.usdc",
                #         # scale=(0.0002, 0.0002, 0.0002),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=64,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.005, 0.00), rot=(0.09990482, 0.0436194, 0, 0)),
                # ),
            }
        )
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            # prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    # prim_path="{ENV_REGEX_NS}/Robot/gripper_finger_link2",
                    # prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",
                    prim_path="{ENV_REGEX_NS}/Robot/M5_wrist_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.1008, 0.003, 0.01],
                    ),
                ),
            ],
        )
        self.scene.finger_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",
                    name="end_effector_1",
                    offset=OffsetCfg(
                        pos=[0.028, -0.001, 0.0],
                    ),
                ),
            ],
        )

        self.scene.finger_frame_2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/M6_2_rightfinger_link",
                    name="end_effector_2",
                    offset=OffsetCfg(
                        pos=[0.028, 0.003, 0.0],
                    ),
                ),
            ],
        )
        self.scene.ee_tip_probe_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/M5_wrist_link",
                    name="ee_probe_tip",
                    offset=OffsetCfg(
                        pos=[0.11, 0.0, -0.0015],
                    ),
                ),
            ],
        )
        self.scene.gripper_peak = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",
                    name="gripper_peak",
                    offset=OffsetCfg(
                        pos=[0.02349, -0.00603, 0.0027],
                    ),
                ),
            ],
        )

@configclass
class CoarseArmCubeLiftEnvCfg_PLAY(CoarseArmCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
