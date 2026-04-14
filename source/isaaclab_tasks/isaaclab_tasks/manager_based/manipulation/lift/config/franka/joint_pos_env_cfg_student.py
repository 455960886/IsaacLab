# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student training environment for student-teacher distillation.

The student observes the same image-based pipeline as the original PPO actor
(ResNet18 + PointNet2 features + joint positions = 1541 dims).  The frozen
teacher is queried with the 21-dim privileged state exposed via the "teacher"
observation group.

Observation layout:
  policy group  → student input (1541 dims) — identical to old PPO actor
    image       : ResNet18 (512) + PointNet2 (1024) = 1536 dims
    joint_pos   : M3, M4, M5, latched binary M6     =    5 dims

  teacher group → frozen teacher input (21 dims)
    object_pos          (3)  – active object position in robot root frame
    object_orientation  (4)  – active object quaternion (w, x, y, z)
    ee_pos              (3)  – end-effector position in robot root frame
    joint_pos           (5)  – M3, M4, M5 + latched binary M6
    contact_left        (3)  – net contact force on left finger
    contact_right       (3)  – net contact force on right finger
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    CoarseArmCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ResNet18ObservationCfg


@configclass
class StudentDistillationObservationCfg:
    """Observation config for student distillation training."""

    @configclass
    class TeacherPrivilegedObsCfg(ObsGroup):
        """21-dim privileged state fed to the frozen teacher network at rollout time."""

        # Active object position in robot root frame (3 dims)
        object_pos = ObsTerm(
            func=mdp.active_object_pos_in_robot_frame,
            params={
                "object_cfg": SceneEntityCfg("object_pool"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        # Active object orientation as quaternion (w, x, y, z) in world frame (4 dims)
        object_orientation = ObsTerm(
            func=mdp.active_object_orientation,
            params={"object_cfg": SceneEntityCfg("object_pool")},
        )

        # End-effector position in robot root frame (3 dims)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_in_robot_frame,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        # Joint positions: M3, M4, M5 + latched binary M6 state (5 dims)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_with_binary_m6_latched,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["M[345]", "M6_.*"]),
                "action_name": "gripper_action",
                "action_index": 0,
                "m6_open_value": 1,
                "m6_close_value": 0,
                "toggle_threshold": 0,
                "debug": False,
                "debug_every": 200,
            },
        )

        # Net contact force on left gripper finger (3 dims)
        contact_left = ObsTerm(
            func=mdp.gripper_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces_left")},
        )

        # Net contact force on right gripper finger (3 dims)
        contact_right = ObsTerm(
            func=mdp.gripper_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces_right")},
        )

    # "policy" → student network input — identical instance to the old PPO actor obs group
    # "teacher" → frozen teacher network input (looked up by the distillation runner)
    policy: ObsGroup = ResNet18ObservationCfg.ResNet18FeaturesCameraPolicyCfg()
    teacher: ObsGroup = TeacherPrivilegedObsCfg()


@configclass
class CoarseArmCubeLiftStudentEnvCfg(CoarseArmCubeLiftEnvCfg):
    """Student distillation variant: same scene as PPO, obs swapped to distillation layout."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = StudentDistillationObservationCfg()


@configclass
class CoarseArmCubeLiftStudentEnvCfg_PLAY(CoarseArmCubeLiftStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.sim.wait_for_textures = True
