# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import math
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim import RigidBodyMaterialCfg

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
            "M3": 3.8,
            "M4": 1.4,
            "M5": 0.0,
            "M6_1": 1.57,
            "M6_2": -1.57,
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
            effort_limit=200.0,
            velocity_limit=10.175,  # 2.175  0.17  0.5
            stiffness=200,
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
            effort_limit=5,      # Reduced to prevent excessive force
            velocity_limit=4.0,     # Keep same
            stiffness=300,          # Increased from 15 to reduce penetration
            damping=1.0,           # Increased from 0.001 for better stability
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

        # self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["M[0345]"],
        #     scale={
        #         # "M0": 0.08,
        #         "M3": 0.1,
        #         "M4": 0.1,
        #         "M5": 0.1
        #     }
        # )
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["M[34]"],
        #     use_default_offset=True,
        # )
        # # Keep the M5 action interface aligned with the real robot: the policy's M5 output
        # # is interpreted directly as an absolute wrist angle in [0, pi] radians.
        # self.actions.wrist_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["M5"],
        #     scale=1.0,  # Scale up from policy output to encourage more wrist movement
        #     offset=0.0,
        #     use_default_offset=False,
        #     clip={"M5": (0.0, math.pi)},
        # )

        ################################### 统一 M3/M4/M5 的连续动作语义 ###################################
        # policy 先输出归一化动作 u in [-1, 1]，再在环境侧线性映射到各自目标范围。
        #
        # 映射公式：
        #   target = clip(u * scale + offset, min, max)
        #
        # 其中：
        #   offset = (min + max) / 2
        #   scale  = (max - min) / 2
        #
        # 这样就得到：
        #   u = -1 -> 目标下界
        #   u =  0 -> 目标中点
        #   u = +1 -> 目标上界
        self.actions.arm_action = mdp.EMAJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M3", "M4", "M5"],
            # alpha=0.9,  # 对映射后的真实关节目标做轻度 EMA 平滑，降低抓取时的关节抖动。
            preserve_order=True,
            scale={
                "M3": (math.radians(256) - math.radians(180.0)) / 2.0,
                "M4": (math.radians(160.0) - math.radians(0)) / 2.0,
                "M5": (math.pi - 0.0) / 2.0,
            },
            offset={
                "M3": math.radians(218),
                "M4": math.radians(80.0),
                "M5": (math.pi + 0.0) / 2.0,                                 # 中点: 90.00°

            },
            use_default_offset=False,
            clip={
                "M3": (math.radians(180.0), math.radians(286.02)),
                "M4": (math.radians(5.0), math.radians(180.0)),
                "M5": (0.0, math.pi),
            },
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": 1.57, "M6_2": -1.57},
            close_command_expr={"M6_1": 0.0, "M6_2": -0.0},
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=[200.0, 200.0, -100.0]),
        )

        # Object pool - all objects spawned, but only one active per env
        from isaaclab.assets import RigidObjectCollectionCfg

        self.scene.object_pool = RigidObjectCollectionCfg(
            rigid_objects={
                # "paper": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/paper",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/cloth/paperball.usdc",
                #         scale=(1.5, 1.0, 1.5),
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
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/bus_new_usd/bus.usdc",
                #         scale=(0.08, 0.08, 0.08),
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.00, 0.0)),
                # ),

                # "lego": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/lego",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/lego.usdc",
                #         scale=(4.0, 4.0, 8.0),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.005, 0.01), rot=(1, 0, 0, 0)),
                # ),
                # "cylinder": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/cylinder",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/cylinder/cylinder.usdc",
                #         scale=(0.8, 1.1, 1.1),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.005, 0.01), rot=(1, 0, 0, 0)),
                # ),
                # "ur10_wrist_3": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/ur10_wrist_3",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/ur10_wrist_3/ur10_wrist_3.usd",
                #         scale=(0.8, 0.8, 0.8),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.005, 0.01), rot=(1, 0, 0, 0)),
                # ),
                # "bear": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/bear",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/teddybear1/bear1.usdc",
                #         scale=(0.4, 0.4, 0.4),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, -0.005, 0.01)),
                # ),
                # "toy1": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/toy1",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/toy1/toy1.usdc",
                #         scale=(0.08, 0.05, 0.05),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, -0.005, 0.01)),
                # ),
                # "car": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/car",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/car/car1.usdc",
                #         scale=(1, 1, 1),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.26, -0.005, 0.01)),
                # ),

                # "slippers": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/slippers_zuo",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/slipper/slipper.usdc",
                #         scale=(1.0, 1.1, 1.3),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         # mass_props=sim_utils.MassPropertiesCfg(
                #         #     mass=0.001,  
                #         # ),
                #         # collision_props=sim_utils.CollisionPropertiesCfg(
                #         #     contact_offset=0.002,  # Start collision detection at 2mm
                #         #     rest_offset=0.0,       # Rest at surface contact
                #         # ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.01, 0.06)),
                # ),
                # "slippers_m5_0": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/slippers_m5_0_you_shang",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/slipper_top/slipper.usdc",
                #         scale=(1.0, 1.0, 1.0),
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.08), rot=(0.7071, 0, 0, -0.7071)),
                # ),
                "baisetuoxie": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/baisetuoxie",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/robo/code/IsaacLab/assets/baisetuoxie/baisetuoxie.usdc",
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
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.31, 0.02, 0.06)),
                ),
                "fensemiantuo": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/fensemiantuo",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/robo/code/IsaacLab/assets/fensemiantuo/fensemiantuo.usdc",
                        scale=(1.0, 1.0, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=64,
                            solver_velocity_iteration_count=32,
                            disable_gravity=False,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.03, 0.06)),
                ),
                # ########################################################## 自己的 ###################################################################
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
                #         # mass_props=sim_utils.MassPropertiesCfg(
                #         #     mass=0.001,  
                #         # ),
                #         # collision_props=sim_utils.CollisionPropertiesCfg(
                #         #     contact_offset=0.002,  # Start collision detection at 2mm
                #         #     rest_offset=0.0,       # Rest at surface contact
                #         # ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.01, 0.06)),
                # ),
                # "baiseyundongxie": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/baiseyundongxie",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/baiseyundongxie/baiseyundongxie.usdc",
                #         scale=(1.0, 1.0, 1.0),
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.04, 0.06)),
                # ),
                # "zongsepixie": RigidObjectCfg(
                #     prim_path="{ENV_REGEX_NS}/fensemiantuo",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/assets/zongsepixie/zongsepixie.usdc",
                #         scale=(1.0, 1.0, 1.0),
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
                # "cube": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/cube",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/cube1.usd",
                #         scale=(2, 2, 2),
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
            },
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
                        pos=[0.10, 0, -0.0015],
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


@configclass
class CoarseArmCubeLiftEnvCfg_PLAY(CoarseArmCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.sim.wait_for_textures = True