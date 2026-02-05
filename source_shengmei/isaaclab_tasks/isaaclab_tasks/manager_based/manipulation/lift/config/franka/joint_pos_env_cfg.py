# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
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


MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/roborock/data/private/shengmei/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/r50_v6_rev/r50_v6_rev_flat_finger.usd",
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
            "M1": 1.57,
            # "M2": 1.57,
            "M3": 3.8,      
            "M4": 1.4,
            "M5": 0.0,
            "M6_1": 0.65,
            "M6_2": -0.65,
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
            stiffness=200,
            damping=4.0,
        ),
        
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["M5"],
            effort_limit=100.0,
            velocity_limit=2.175,  # 2.61  0.17  0.5
            stiffness=200.0,
            damping=5.0,
        ),

        "hand": ImplicitActuatorCfg(
            joint_names_expr=["M6_.*"],
            effort_limit=50,      # Reduced to prevent excessive force
            velocity_limit=4.0,     # Keep same
            stiffness=30,          # Increased from 15 to reduce penetration
            damping=0.001,           # Increased from 0.001 for better stability
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    debug_vis=True,
)


def generate_random_cube_configs(num_configs=64, base_size=0.022):
    """Generate random cube configurations with different sizes and colors."""
    assets_cfg = []
    
    for i in range(num_configs):

        scale_x = np.random.uniform(1.0, 1.0)
        scale_y = np.random.uniform(1.0, 1.0)
        scale_z = np.random.uniform(1.0, 1.0)
        
        size = (base_size * scale_x, base_size * scale_y, base_size * scale_z)
        
        color = (np.random.uniform(1.0, 1.0), np.random.uniform(0.0, 0.0), np.random.uniform(0.0, 0.0))
        
        assets_cfg.append(
            sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, 
                    metallic=0.2,
                ),
            # physics_material=high_friction_material,
            )
        )
    
    return assets_cfg



@configclass
class CoarseArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set CoarseArm as robot
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        #     #asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        #     asset_name = "robot", 
        #     joint_names = ["M[0345]"],
        #     scale={
        #         # "M0": 0.08,
        #         "M3": 0.3,
        #         "M4": 0.3,
        #         # "M5": 0.08
        #     }
        # )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            #asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
            asset_name = "robot", 
            joint_names = ["M[34]"],
            # scale={
            #     # "M0": 0.2,
            #     "M3": 2,
            #     "M4": 2,
            #     # "M5": 0.08,
            # },
            # scale = 0.5,
            use_default_offset=True,
        )


        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            #joint_names=["panda_finger.*"],
            #open_command_expr={"panda_finger_.*": 0.04},
            #close_command_expr={"panda_finger_.*": 0.0},
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": 0.65, "M6_2": -0.65},
            close_command_expr={"M6_1": 0.02, "M6_2": -0.02},
        )

        self.commands.object_pose.body_name = "M6_1_leftfinger_link"
        # self.commands.object_pose.body_name = "M6_1_rightfinger_link"

        # assets_cfg = generate_random_cube_configs(num_configs=64)

        # self.scene.object = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/Object",
        #     spawn=sim_utils.MultiAssetSpawnerCfg(
        #         assets_cfg=assets_cfg,
        #         random_choice=True,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=32,
        #             solver_velocity_iteration_count=16,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),   
        #     ),
        #     debug_vis=False,
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.00, 0], rot=[1, 0, 0, 0]),
        # )


        # Generate USD object configurations
        # assets_cfg = generate_random_cube_configs()

        # Configure the object with MultiAssetSpawnerCfg
        # self.scene.object = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/Object",
        #     spawn=sim_utils.MultiAssetSpawnerCfg(
        #         assets_cfg=assets_cfg,
        #         random_choice=True,  # Randomly select one object per environment
        #         # Note: rigid_props, mass_props, collision_props are now defined 
        #         # per-object in the UsdFileCfg above, not here at the spawner level
        #     ),
        #     debug_vis=False,
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.28, 0.00, 0.02],  # Slightly raised to avoid penetration
        #         rot=[1, 0, 0, 0]
        #     ),
        # )

        # self.scene.object = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/Object",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/toy_bear.usd",
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=32,
        #             solver_velocity_iteration_count=16,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        #     debug_vis=False,
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.34, 0.0, 0.0], rot=[1, 0, 0, 0]),
        # )


        # Create dummy object to satisfy parent class validation
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


        # Object pool - all objects spawned, but only one active per env
        from isaaclab.assets import RigidObjectCollectionCfg

        self.scene.object_pool = RigidObjectCollectionCfg(
            rigid_objects={
                # "object_1": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_1",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/toy_bear.usd",
                #         scale=(1.2, 1.2, 1.2),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.0], rot=(0.8192, 0.0, 0.0, -0.5736)),
                # ),

                # "object_2": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_2",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/marker.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
                #             solver_velocity_iteration_count=16,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.0], rot=[0.7071, 0, 0, 0.7071]),
                # ),

                # "Object_3": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_3",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/nut.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, -0.02]),
                # ),


                # "Object_4": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_4",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/mug.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
                #             solver_velocity_iteration_count=16,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.0]),
                # ),

                # "Object_5": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_5",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/drill.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
                #             solver_velocity_iteration_count=16,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.01], rot=[0.7071, 0, 0, 0.7071]),
                # ),

                # "Object_6": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_6",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/bolt.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
                #             solver_velocity_iteration_count=16,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.0]),
                # ),

                # "Object_7": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_7",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/plush_duck.usdc",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #         articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.00, 0.02), rot=(0.7071, 0.0, 0.0, -0.7071),),
                # ),

        
                # "Object_9": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_9",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/tape.usd",
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=64,
                #             solver_velocity_iteration_count=32,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, 0.0]),
                # ),

                # "Object_10": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_10",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/lego_plane_front.usdc",
                #         scale=(0.0004, 0.0004, 0.0004),
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, -0.015, 0.0]),
                # ),
        
                # "Object_11": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_11",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/lego_plane_back.usdc",
                #         scale=(0.0004, 0.0004, 0.0005),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, -0.022, 0.0]),
                # ),
        
                # "Object_12": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/Object_12",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/lego_plane_side.usdc",
                #         scale=(0.0004, 0.0004, 0.0004),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=32,
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
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, -0.025, 0.0]),
                # ),

                # "Object_13": RigidObjectCfg(
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

                "bus": RigidObjectCfg(
                    prim_path="/World/envs/env_.*/bus",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/roborock/data/private/shengmei/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/bus_2.usd",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=64,
                            solver_velocity_iteration_count=32,
                            disable_gravity=False,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.27, 0.00, 0.02),rot = (0.7071 , 0.0 ,0.0 , 0.7071)),
                ),

                "Object_8": RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Object_8",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/roborock/data/private/shengmei/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/lego_1.usd",
                        scale=(1.0, 2.0, 1.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=64,
                            solver_velocity_iteration_count=16,
                            disable_gravity=False,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.27, 0.005, 0.0], rot=[0.5, 0.5, 0.5, 0.5]),
                ),


                "slippers": RigidObjectCfg(
                    prim_path="/World/envs/env_.*/slippers",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/roborock/data/private/shengmei/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/assets/slipper/slipper.usdc",
                        scale=(1.0, 1.1, 1.3),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=32,
                            solver_velocity_iteration_count=16,
                             # max_depenetration_velocity=10.0,  # CRITICAL: Limit depenetration speed
                            disable_gravity=False,
                        ),
                        # mass_props=sim_utils.MassPropertiesCfg(
                        #     mass=0.001,  # Adjust based on actual slipper weight
                        # ),
                        # collision_props=sim_utils.CollisionPropertiesCfg(
                        #     contact_offset=0.002,  # Start collision detection at 2mm
                        #     rest_offset=0.0,       # Rest at surface contact
                        # ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.01, 0.06)),
                ),

                # "vans": RigidObjectCfg(
                #     prim_path="/World/envs/env_.*/vans",
                #     spawn=sim_utils.UsdFileCfg(
                #         usd_path="/home/roborock/IsaacLab/assets/vans_black/vans_black1.usdc",
                #         scale=(0.25, 0.25, 0.25),
                #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #             solver_position_iteration_count=128,
                #             solver_velocity_iteration_count=64,
                #             disable_gravity=False,
                #         ),
                #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #             articulation_enabled=False,  # CRITICAL: Disable articulation
                #         ),
                #     ),
                #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, 0.025, 0.001)),
                # ),
            },
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            #prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    #prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    # prim_path="{ENV_REGEX_NS}/Robot/gripper_finger_link2",
                    # prim_path="{ENV_REGEX_NS}/Robot/M6_1_leftfinger_link",
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


        # self.scene.bear_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Object/geometry/bear",
        #     debug_vis=False,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Object/geometry/bear",
        #             name="bear_grasp_point",
        #             offset=OffsetCfg(
        #                 pos=[-0.0341, -0.0188, 0.0172]
        #             ),
        #         ),
        #     ],
            
        # )



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
        self.sim.wait_for_textures = True
