# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Placeholder reward terms for the shoe-navigation task.

These are intentionally minimal — the user asked for a reasonable scaffold and
will iterate on the reward design separately.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def distance_to_back_of_shoe(
    env: ManagerBasedRLEnv,
    behind_offset: float = 0.25,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("shoe"),
) -> torch.Tensor:
    """exp(-d^2/std^2) shaping reward — robot's planar distance to a target point
    a short distance behind the shoe (along the shoe's -x body axis)."""
    robot: Articulation = env.scene[robot_cfg.name]
    shoe: RigidObject = env.scene[object_cfg.name]

    shoe_pos_w = shoe.data.root_pos_w
    shoe_quat_w = shoe.data.root_quat_w

    # behind = -x axis of the shoe in world frame
    back_local = torch.tensor([-behind_offset, 0.0, 0.0], device=env.device).expand(shoe_pos_w.shape[0], -1)
    back_w = shoe_pos_w + math_utils.quat_apply(shoe_quat_w, back_local)

    robot_pos_w = robot.data.root_pos_w
    delta_xy = robot_pos_w[:, :2] - back_w[:, :2]
    dist = torch.norm(delta_xy, dim=-1)
    return torch.exp(-(dist ** 2) / (std ** 2))


def heading_alignment_to_shoe(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("shoe"),
) -> torch.Tensor:
    """Cosine alignment between the robot's forward (+x body) axis and the
    direction from robot to shoe. 1 when facing the shoe, -1 when facing away."""
    robot: Articulation = env.scene[robot_cfg.name]
    shoe: RigidObject = env.scene[object_cfg.name]

    robot_pos_w = robot.data.root_pos_w
    shoe_pos_w = shoe.data.root_pos_w

    to_shoe = shoe_pos_w[:, :2] - robot_pos_w[:, :2]
    to_shoe = to_shoe / (torch.norm(to_shoe, dim=-1, keepdim=True) + 1e-6)

    forward_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(robot_pos_w.shape[0], -1)
    forward_w = math_utils.quat_apply(robot.data.root_quat_w, forward_local)[:, :2]
    forward_w = forward_w / (torch.norm(forward_w, dim=-1, keepdim=True) + 1e-6)

    return (forward_w * to_shoe).sum(dim=-1)
