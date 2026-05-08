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
from .rewards import get_active_object_states

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


def object_lifted_during_grasp(
    env: ManagerBasedRLEnv,
    excess_z_limit: float = 0.08,
    grasp_action_idx: int = 3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Terminate when a grasp attempt lifts the object beyond excess_z_limit."""
    if not hasattr(env, '_grasp_pen_prev_z'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    grasp_attempt = (
        (env.action_manager.prev_action[:, grasp_action_idx] >= 0) &
        (env.action_manager.action[:, grasp_action_idx] < 0)
    )

    active_pos_w, _ = get_active_object_states(env, object_cfg)
    obj_z = active_pos_w[:, 2]
    excess_lift = obj_z - env._grasp_pen_prev_z - excess_z_limit
    return grasp_attempt & (excess_lift > 0.0)
