# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher training environment for the CoarseArm lift task.

The teacher observes full privileged state (object pose, EE pose, joint positions,
contact forces) and is trained with PPO. Its checkpoint will later be used as the
frozen teacher in student-teacher distillation.

Observation layout (actor = critic, both privileged):
  active_object_pos          (3)  – object position in robot root frame
  active_object_orientation  (4)  – object quaternion (w, x, y, z) in world frame
  ee_pos                     (3)  – end-effector position in robot root frame
  joint_pos                  (5)  – M3, M4, M5 + latched binary M6 gripper state
  contact_left               (3)  – net contact force on left finger (x, y, z)
  contact_right              (3)  – net contact force on right finger (x, y, z)
  ──────────────────────────────
  Total                     (21)
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    CoarseArmCubeLiftEnvCfg,
    CoarseArmCubeLiftEnvCfg_PLAY,
)


@configclass
class TeacherObservationCfg:
    """Observation config for teacher training — pure privileged state, no cameras."""

    @configclass
    class TeacherPolicyCfg(ObsGroup):
        """Full privileged state seen by both actor and critic during teacher training."""

        # Object position in robot root frame (3 dims)
        object_pos = ObsTerm(
            func=mdp.active_object_pos_in_robot_frame,
            params={
                "object_cfg": SceneEntityCfg("object_pool"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        # Object orientation as quaternion in world frame (4 dims)
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

    # Teacher uses the same state for actor and critic — both see full privileged info.
    policy: ObsGroup = TeacherPolicyCfg()
    critic: ObsGroup = TeacherPolicyCfg()


@configclass
class CoarseArmCubeLiftTeacherEnvCfg(CoarseArmCubeLiftEnvCfg):
    """Teacher training variant: replaces image-based observations with privileged state."""

    def __post_init__(self):
        super().__post_init__()

        # Swap to privileged-state observations — no cameras needed for inference,
        # but the scene still has cameras for the future student training.
        self.observations = TeacherObservationCfg()


@configclass
class CoarseArmCubeLiftTeacherEnvCfg_PLAY(CoarseArmCubeLiftTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.sim.wait_for_textures = True
