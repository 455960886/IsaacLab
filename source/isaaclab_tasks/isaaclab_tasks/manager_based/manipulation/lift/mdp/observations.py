# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
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


def _zeros(env: ManagerBasedRLEnv, dim: int) -> torch.Tensor:
    """Create a batched zero observation tensor on the environment device."""
    return torch.zeros((env.num_envs, dim), device=env.device)


def _get_active_object_indices(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return active object indices if they have been initialized."""
    active_indices = getattr(env, "active_object_indices", None)
    if not torch.is_tensor(active_indices):
        return None
    if active_indices.shape != (env.num_envs,) or active_indices.device != env.device:
        return None
    return active_indices


def _get_active_object_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Fetch the active object state from the object pool."""
    active_indices = _get_active_object_indices(env)
    if active_indices is None:
        return None, None, None

    object_collection = env.scene[object_cfg.name]
    env_indices = torch.arange(env.num_envs, device=env.device)

    active_pos_w = object_collection.data.object_pos_w[env_indices, active_indices]
    active_quat_w = object_collection.data.object_quat_w[env_indices, active_indices]
    active_vel_w = object_collection.data.object_vel_w[env_indices, active_indices]
    return active_pos_w, active_quat_w, active_vel_w


def active_object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Pose of the active object in the robot root frame as [pos, quat]."""
    active_pos_w, active_quat_w, _ = _get_active_object_states(env, object_cfg)
    if active_pos_w is None or active_quat_w is None:
        return _zeros(env, 7)

    robot: RigidObject = env.scene[robot_cfg.name]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        active_pos_w,
        active_quat_w,
    )
    object_quat_b = math_utils.quat_unique(object_quat_b)
    return torch.cat((object_pos_b, object_quat_b), dim=-1)


def active_object_velocity_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Linear and angular velocity of the active object in world frame."""
    _, _, active_vel_w = _get_active_object_states(env, object_cfg)
    if active_vel_w is None:
        return _zeros(env, 6)
    return active_vel_w


def active_object_position_in_ee_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Position of the active object expressed in the end-effector frame."""
    active_pos_w, _, _ = _get_active_object_states(env, object_cfg)
    if active_pos_w is None:
        return _zeros(env, 3)

    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    object_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, active_pos_w)
    return object_pos_ee


def active_object_yaw_sin_cos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Sine and cosine of the active object's world-frame yaw."""
    active_quat_w = _get_active_object_states(env, object_cfg)[1]
    if active_quat_w is None:
        return _zeros(env, 2)

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]

    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def active_object_identity_one_hot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """One-hot encoding of which object in the pool is active."""
    object_cfg_value = getattr(env.cfg.scene, object_cfg.name)
    num_objects = len(object_cfg_value.rigid_objects)
    active_indices = _get_active_object_indices(env)
    if active_indices is None:
        return _zeros(env, num_objects)

    one_hot = torch.zeros((env.num_envs, num_objects), device=env.device)
    one_hot.scatter_(1, active_indices.unsqueeze(-1), 1.0)
    return one_hot


def fingertip_contact_forces(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
) -> torch.Tensor:
    """Concatenated left/right fingertip contact forces in world frame."""
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        return _zeros(env, 6)

    left_forces = left_sensor.data.net_forces_w[:, 0, :]
    right_forces = right_sensor.data.net_forces_w[:, 0, :]
    return torch.cat((left_forces, right_forces), dim=-1)


def active_object_yaw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Yaw of the currently active object in world frame, shape (num_envs, 1)."""
    active_quat_w = _get_active_object_states(env, object_cfg)[1]
    if active_quat_w is None:
        return _zeros(env, 1)

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]

    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return yaw.unsqueeze(-1)
