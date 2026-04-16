# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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


def _get_active_object_indices(env: ManagerBasedRLEnv, num_objects: int) -> torch.Tensor:
    """Return active object indices with a safe zero fallback during startup shape checks."""
    if hasattr(env, "active_object_indices"):
        return env.active_object_indices
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _get_active_object_collection_data(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Fetch active object pool data tensors for the current environment batch."""
    from isaaclab.assets import RigidObjectCollection

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]
    active_indices = _get_active_object_indices(env, len(object_collection.object_names))
    env_indices = torch.arange(env.num_envs, device=env.device)
    return object_collection, active_indices, env_indices


def active_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return the active object's position in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object_collection, active_indices, env_indices = _get_active_object_collection_data(env, object_cfg)
    active_object_pos_w = object_collection.data.object_pos_w[env_indices, active_indices]
    active_object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], active_object_pos_w
    )
    return active_object_pos_b


def active_object_yaw_sincos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return sin/cos of the active object's yaw to avoid the discontinuity at +/-pi."""
    object_collection, active_indices, env_indices = _get_active_object_collection_data(env, object_cfg)
    active_quat_w = object_collection.data.object_quat_w[env_indices, active_indices]

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def active_object_id_onehot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return a one-hot encoding of the active object identity for heterogeneous object pools."""
    object_collection, active_indices, _ = _get_active_object_collection_data(env, object_cfg)
    num_objects = len(object_collection.object_names)
    return torch.nn.functional.one_hot(active_indices, num_classes=num_objects).to(dtype=torch.float32)


def active_object_scale(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return the active object's randomized scale from the prestartup cache."""
    object_collection, active_indices, env_indices = _get_active_object_collection_data(env, object_cfg)
    object_names = list(object_collection.object_names)

    if not hasattr(env, "object_pool_randomized_scales"):
        return torch.ones((env.num_envs, 3), dtype=torch.float32, device=env.device)

    cached_scales = getattr(env, "_object_pool_randomized_scales_device", None)
    cached_names = getattr(env, "_object_pool_randomized_scale_object_names_device", None)
    source_scales = env.object_pool_randomized_scales
    source_names = list(getattr(env, "object_pool_randomized_scale_object_names", object_names))

    need_refresh = (
        not torch.is_tensor(cached_scales)
        or cached_scales.device != env.device
        or tuple(cached_scales.shape) != tuple(source_scales.shape)
        or cached_names != object_names
    )
    if need_refresh:
        reordered_scales = source_scales
        if source_names != object_names:
            source_name_to_idx = {name: idx for idx, name in enumerate(source_names)}
            reorder_indices = [source_name_to_idx[name] for name in object_names]
            reordered_scales = source_scales[:, reorder_indices, :]
        cached_scales = reordered_scales.to(env.device)
        env._object_pool_randomized_scales_device = cached_scales
        env._object_pool_randomized_scale_object_names_device = object_names

    return cached_scales[env_indices, active_indices]


def gripper_contact_y_force(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    max_force: float = 10.0,
) -> torch.Tensor:
    """Return clipped, normalized Y-axis contact magnitudes for the two gripper fingers."""
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)

    left_y_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
    right_y_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])
    contact_force = torch.stack((left_y_force, right_y_force), dim=-1)

    if max_force > 0.0:
        contact_force = torch.clamp(contact_force, min=0.0, max=max_force) / max_force

    return contact_force


def log_observation_group_summary(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    group_name: str = "critic",
    sample_env_id: int = 0,
    once: bool = True,
    max_elements_per_term: int = 16,
) -> None:
    """Print a compact summary of one observation group for sanity-checking term wiring."""
    if not hasattr(env, "observation_manager"):
        print(f"[obs-log] observation_manager not available, skip group={group_name}.", flush=True)
        return

    printed_attr = f"_logged_observation_group_{group_name}"
    if once and getattr(env, printed_attr, False):
        return

    obs_manager = env.observation_manager
    if group_name not in obs_manager.active_terms:
        print(f"[obs-log] group '{group_name}' not found. Available: {list(obs_manager.active_terms.keys())}", flush=True)
        return

    obs_tensor = obs_manager.compute_group(group_name)
    if not torch.is_tensor(obs_tensor):
        print(f"[obs-log] group '{group_name}' is not concatenated; skip compact summary.", flush=True)
        if once:
            setattr(env, printed_attr, True)
        return

    env_index = max(0, min(int(sample_env_id), env.num_envs - 1))
    total_dim = obs_manager.group_obs_dim[group_name]
    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]

    print(
        f"[obs-log] group={group_name} env={env_index} total_dim={total_dim} num_terms={len(term_names)}",
        flush=True,
    )

    start = 0
    for term_name, term_dim in zip(term_names, term_dims):
        flat_dim = int(math.prod(term_dim))
        term_value = obs_tensor[env_index, start : start + flat_dim]
        start += flat_dim
        preview = term_value[:max_elements_per_term].detach().cpu().tolist()
        preview_suffix = ""
        if flat_dim > max_elements_per_term:
            preview_suffix = f" ... ({flat_dim} values)"
        print(
            f"[obs-log] {group_name}.{term_name} shape={term_dim} sample={preview}{preview_suffix}",
            flush=True,
        )

    if once:
        setattr(env, printed_attr, True)


def active_object_yaw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return the active object's yaw as a temporary low-dimensional orientation signal."""
    object_collection, active_indices, env_indices = _get_active_object_collection_data(env, object_cfg)
    active_quat_w = object_collection.data.object_quat_w[env_indices, active_indices]

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return yaw.unsqueeze(-1)
