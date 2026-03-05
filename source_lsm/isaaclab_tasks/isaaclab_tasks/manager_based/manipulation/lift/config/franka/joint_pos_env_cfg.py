# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math


MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/r50_v6_rev/r50_v6_rev_flat_finger.usd",
        # usd_path=f"/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/r50_v6_rev/r50_v6_rev.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # max_depenetration_velocity=5.0,  # Balanced to prevent penetration without instability
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=32,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.005,  # Increased to 5mm for earlier collision detection
        #     rest_offset=0.001,     # 1mm air gap to prevent penetration
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "M0": 0,
            "M1": math.pi / 2,
            # "M2": 1.57,
            "M3": 3.8,
            "M4": 1.4,
            "M5": 0.0,
            "M6_1": math.pi / 2,
            "M6_2": -math.pi / 2,
        },
    ),
    actuators={

        "base": ImplicitActuatorCfg(
            joint_names_expr=["M[0]"],
            effort_limit=87.0,
            velocity_limit=2.175,  # 2.175  0.17  0.5
            stiffness=80.0,
            damping=4.0,
        ),

        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["M[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,  # 2.175  0.17  0.5
            stiffness=100,
            damping=4.0,
        ),

        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["M5"],
            effort_limit=300.0,
            velocity_limit=2.175,  # 2.61  0.17  0.5
            stiffness=500.0,
            damping=5.0,
        ),

        "hand": ImplicitActuatorCfg(
            joint_names_expr=["M6_.*"],
            effort_limit=50,      # Reduced to prevent excessive force
            velocity_limit=2.0,     # Keep same
            stiffness=1000,          # Increased from 15 to reduce penetration
            damping=10,           # Increased from 0.001 for better stability
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    debug_vis=True,
)


@configclass
class CoarseArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set CoarseArm as robot
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M[0345]"],
            scale={
                # "M0": 0.08,
                "M3": 0.3,
                "M4": 0.3,
                # "M5": 0.08
            }
        )
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["M[345]"],
        #     use_default_offset=True,
        # )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": math.pi / 2, "M6_2": -math.pi / 2},
            close_command_expr={"M6_1": 0.02, "M6_2": -0.02},
        )

        self.commands.object_pose.body_name = "M6_1_leftfinger_link"

        # Create dummy object to satisfy parent class validation
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_Dummy",
            spawn=sim_utils.CuboidCfg(
                size=(0.001, 0.001, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(200.0, 200.0, -100.0)),
        )

        # Object pool - all objects spawned, but only one active per env
        from isaaclab.assets import RigidObjectCollectionCfg

        self.scene.object_pool = RigidObjectCollectionCfg(
            rigid_objects={
                # "lego": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/lego",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/lego_gongzixing.usdc",
                #         scale=(4.0, 4.0, 8.0),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #             mass=0.01,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.005, 0.01)),
                # ),
                # "paperball": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_13",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/cloth/paperball.usdc",
                #         scale=(0.8, 0.8, 1.2),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #         mass=0.01,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.005, -0.01]),
                # ),

                # "bus": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/bus",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/bus_new_usd/bus.usdc",
                #         scale=(0.08, 0.08, 0.08),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #             mass=0.01,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.00, 0.02),rot = (1.0, 0.0 ,0.0 , 0.0)),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.00, 0.0)),
                # ),

                # "slippers": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/slippers",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/slipper/slipper.usdc",
                #         scale=(1.0, 1.1, 1.3),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             # max_depenetration_velocity=10.0,  # CRITICAL: Limit depenetration speed
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #             mass=0.001,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.01, 0.06)),
                # ),

                "slippers_m5_0": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/slippers_m5_0",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/robo/code/IsaacLab/assets/slipper_top/slipper.usdc",
                        scale=(1.0, 1.0, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=64,
                            solver_velocity_iteration_count=32,
                            disable_gravity=False,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=0.001,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.0, 0.08), rot=(0.7071, 0, 0, -0.7071)),
                ),
                # "renzituo": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/renzituo",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/renzituo/renzituo.usdc",
                #         scale=(1.0, 1.0, 1.3),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #             mass=0.001,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.01, 0.06)),
                # ),
                # "eye_drops": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/eye_drops",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/eyedrops/2.usdc",
                #         scale=(0.0002, 0.0002, 0.0002),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=64,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.012, 0.00)),
                # ),

                # "vans": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/vans",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/vans_black/vans_black.usdc",
                #         scale=(0.25, 0.25, 0.25),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, 0, 0.05)),
                # ),

                # "cube": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/cube",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/cube1.usd",
                #         # scale=(0.25, 0.25, 0.25),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0, 0.0)),
                # ),

                # "slippers_m5": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/slippers_m5",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/slipper_top/slipper.usdc",
                #         scale=(1.0, 0.8, 1.1),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         mass_props=sim_utils.MassPropertiesCfg(
                #             mass=0.001,  
                #         ),
                #         # collision_props=sim_utils.CollisionPropertiesCfg(
                #         #     contact_offset=0.002,  # Start collision detection at 2mm
                #         #     rest_offset=0.0,       # Rest at surface contact
                #         # ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, 0.01, 0.1), rot=(0.7071, 0, 0, -0.7071)),
                # ),
            }
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/M5_wrist_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.10, 0, -0.0015),
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
                        pos=(0.028, -0.001, 0.0),
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
                        pos=(0.028, 0.003, 0.0),
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
        self.scene.env_spacing = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False