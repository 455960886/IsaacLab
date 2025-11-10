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
        usd_path=f"/home/robo/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/R50/r50_v6_rev/r50_v6_rev_cont.usd",
        # usd_path=f"/home/xuyang/xuyang_ws/DRL/isaac/IsaacLab-2.0.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/robot_model/arm_description/urdf/marm_backup/marm_backup.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=32,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "M0": 0,   
            "M1": 1.57, 
            # "M2": 1.57,
            "M3": 3.9,
            "M4": 1.4,
            "M5": 0.0,
            "M6_1": 0.0,
            "M6_2": 0.0,
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
            stiffness=80.0,
            damping=4.0,
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
            effort_limit=2.5,
            velocity_limit=2.5,
            stiffness=5.0,
            damping=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
    debug_vis=False,
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

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            #asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
            asset_name = "robot", joint_names = ["M[134]"],scale=0.25
        )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            #joint_names=["panda_finger.*"],
            #open_command_expr={"panda_finger_.*": 0.04},
            #close_command_expr={"panda_finger_.*": 0.0},
            joint_names=["M6_.*"],
            open_command_expr={"M6_1": 0.55, "M6_2": -0.55},
            close_command_expr={"M6_1": 0.095, "M6_2": -0.095},
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

                "Object_3": RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Object_3",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/home/robo/code/IsaacLab/assets/nut.usd",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=128,
                            solver_velocity_iteration_count=32,
                            disable_gravity=False,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # CRITICAL: Disable articulation
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, 0.0, -0.02]),
                ),


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
            }
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
                        pos=[0.035, -0.005, 0.0],
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
                        pos=[0.035, 0.009, 0.0],
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
