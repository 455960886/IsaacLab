# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def active_object_yaw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return the active object's yaw as a temporary low-dimensional orientation signal."""
    from isaaclab.assets import RigidObjectCollection

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]
    all_quat_w = object_collection.data.object_quat_w

    if hasattr(env, "active_object_indices"):
        active_indices = env.active_object_indices
    else:
        # Observation terms are shape-checked before startup events initialize the active object selection.
        active_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env_indices = torch.arange(env.num_envs, device=env.device)
    active_quat_w = all_quat_w[env_indices, active_indices]

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return yaw.unsqueeze(-1)
