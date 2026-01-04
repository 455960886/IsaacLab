# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # return (asset.data.root_pos_w[:, 2] < minimum_height) | (ee_frame.data.target_pos_w[..., 0, 2] <0.001)
    return (ee_frame.data.target_pos_w[..., 0, 2] <0.001)
# def object_dropped_after_lifted(env, lift_threshold=0.065, drop_threshold=0.018):
#     obj = env.scene["object"]
#     heights = obj.data.root_pos_w[:, 2]
#     print("heights: ", heights[0].item())
#     # 初始化lifted状态，只在第一次调用时创建
#     if not hasattr(env, "lifted_flags"):
#         env.lifted_flags = torch.zeros_like(heights, dtype=torch.bool)
#     else:
#         # reset those environments where episode just ended
#         env.lifted_flags[env.reset_buf] = False
    
#     # 更新 lifted 状态
#     just_lifted = heights > lift_threshold
#     env.lifted_flags = env.lifted_flags | just_lifted

#     # 判断是否 dropped（仅在 lifted 过后才触发）
#     dropped = (heights < drop_threshold) & env.lifted_flags

#     # 打印调试信息
#     print("heights: ", heights[0].item())
#     print("lifted_flags: ", env.lifted_flags.nonzero(as_tuple=False).squeeze(-1).tolist())
#     if dropped.any():
#         print("掉落的环境编号:", dropped.nonzero(as_tuple=False).squeeze(-1).tolist())

#     return dropped
       
def gripper_tip_below_ground(
    env: ManagerBasedRLEnv,
    threshold: float,
    ee_probe_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_probe_frame"),
) -> torch.Tensor:
    """Terminate the episode if the gripper tip (probe frame) drops below the ground height."""
    ee_probe_frame: FrameTransformer = env.scene[ee_probe_cfg.name]
    z_pos = ee_probe_frame.data.target_pos_w[..., 0, 2]
    # print(z_pos[0])
    return z_pos < threshold
