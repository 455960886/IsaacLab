# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for spawning the wheeled robot on a ring around the shoe."""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_robot_around_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (0.6, 1.5),
    yaw_noise: float = math.pi,
    z_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("shoe"),
):
    """Spawn the robot at a random point on a ring around the shoe, facing it (with noise).

    The robot is placed at a uniformly-sampled (radius, angle) around the shoe's
    current world position. Its base yaw is set to point toward the shoe, plus
    a uniform perturbation in ``[-yaw_noise, +yaw_noise]``. Joint state is reset
    to defaults so the arm stays in its tucked pose.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    shoe: RigidObject = env.scene[object_cfg.name]

    num = len(env_ids)
    device = robot.device

    # sample polar coordinates on a ring
    r = math_utils.sample_uniform(radius_range[0], radius_range[1], (num,), device=device)
    theta = math_utils.sample_uniform(-math.pi, math.pi, (num,), device=device)

    shoe_pos_w = shoe.data.root_pos_w[env_ids]
    dx = r * torch.cos(theta)
    dy = r * torch.sin(theta)

    positions = shoe_pos_w.clone()
    positions[:, 0] += dx
    positions[:, 1] += dy
    positions[:, 2] = robot.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2] + z_offset

    # face shoe (vector from robot -> shoe), then add noise
    facing_yaw = torch.atan2(-dy, -dx)
    yaw_jitter = math_utils.sample_uniform(-yaw_noise, yaw_noise, (num,), device=device)
    yaw = facing_yaw + yaw_jitter

    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    pose = torch.cat([positions, orientations], dim=-1)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)

    zero_vel = torch.zeros((num, 6), device=device)
    robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    # reset joints to defaults so the arm stays tucked and wheels stop spinning
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
