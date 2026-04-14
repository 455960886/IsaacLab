# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
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


##
# Teacher privileged observation functions
##


def _get_active_object_states(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg):
    """Return (pos_w, quat_w) of the active object in each env. Shape: (N,3), (N,4)."""
    from isaaclab.assets import RigidObjectCollection

    collection: RigidObjectCollection = env.scene[object_cfg.name]
    if hasattr(env, "active_object_indices"):
        active_indices = env.active_object_indices
    else:
        active_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env_ids = torch.arange(env.num_envs, device=env.device)
    pos_w = collection.data.object_pos_w[env_ids, active_indices]   # (N, 3)
    quat_w = collection.data.object_quat_w[env_ids, active_indices]  # (N, 4)
    return pos_w, quat_w


def active_object_pos_in_robot_frame(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Active object position expressed in the robot root frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    obj_pos_w, _ = _get_active_object_states(env, object_cfg)
    obj_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)
    return obj_pos_b


def active_object_orientation(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Active object orientation as quaternion (w, x, y, z) in world frame. Shape: (N, 4)."""
    _, quat_w = _get_active_object_states(env, object_cfg)
    return quat_w


def ee_pos_in_robot_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector position expressed in the robot root frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)

    ee_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w)
    return ee_pos_b


def gripper_contact_forces(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
) -> torch.Tensor:
    """Net contact force (x, y, z) on a gripper finger sensor. Shape: (N, 3).

    Returns zeros if the sensor has not yet produced data.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    # net_forces_w shape: (N, num_bodies, 3) — each sensor tracks one finger link
    return sensor.data.net_forces_w[:, 0, :]  # (N, 3)


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
