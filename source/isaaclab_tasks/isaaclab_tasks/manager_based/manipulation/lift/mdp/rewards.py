# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#######################################################################
# Help Function to get the states of active object from the object pool
#######################################################################
def get_active_object_states(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")):
    """
    Helper function to get states of active objects from object pool.

    Returns:
        pos_w: (num_envs, 3) - World positions of active objects
        quat_w: (num_envs, 4) - World orientations of active objects [w, x, y, z]
    """
    from isaaclab.assets import RigidObjectCollection

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]

    # Get active object indices for each environment
    if not hasattr(env, 'active_object_indices'):
        raise RuntimeError("active_object_indices not found. Ensure randomize_object_pool_selection has been called.")

    active_indices = env.active_object_indices  # (num_envs,)

    # Get all object states: (num_envs, num_objects, state_dim)
    all_pos_w = object_collection.data.object_pos_w  # (num_envs, num_objects, 3)
    all_quat_w = object_collection.data.object_quat_w  # (num_envs, num_objects, 4)

    # Index to get only active objects
    # Use advanced indexing: env_indices = [0, 1, 2, ...], object_indices = active_indices
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_pos_w = all_pos_w[env_indices, active_indices]  # (num_envs, 3)
    active_quat_w = all_quat_w[env_indices, active_indices]  # (num_envs, 4)

    return active_pos_w, active_quat_w


def get_active_object_lin_vel_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")):
    """
    Returns:
        vel_w: (num_envs, 3) - World linear velocity of active objects
    """
    from isaaclab.assets import RigidObjectCollection

    obj_col: RigidObjectCollection = env.scene[object_cfg.name]
    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    active = env.active_object_indices.to(dtype=torch.long)
    env_ids = torch.arange(env.num_envs, device=env.device)

    # 1) best effort: read from sim buffers (name may differ by version)
    for attr in ["object_lin_vel_w", "object_vel_w", "root_lin_vel_w"]:
        if hasattr(obj_col.data, attr):
            all_vel = getattr(obj_col.data, attr)  # expected (num_envs, num_objects, 3)
            if all_vel is not None and all_vel.ndim == 3:
                return all_vel[env_ids, active]

    # 2) fallback: finite difference from position
    pos_w, _ = get_active_object_states(env, object_cfg)

    # dt ≈ sim_dt * decimation (best effort)
    sim_dt = float(getattr(getattr(env, "sim", None), "cfg", None).dt) if hasattr(env, "sim") else 0.01
    decim = int(getattr(getattr(env, "cfg", None), "decimation", 1))
    dt = max(sim_dt * decim, 1e-6)

    key = "_prev_active_obj_pos_w_for_vt"
    if not hasattr(env, key):
        setattr(env, key, pos_w.clone())
        return torch.zeros((env.num_envs, 3), device=env.device)

    prev = getattr(env, key)
    vel = (pos_w - prev) / dt
    setattr(env, key, pos_w.clone())
    return vel


def get_stable_grasp_mask(
    env,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
    stable_steps: int = 8,
    release_steps: int = 2,
) -> torch.Tensor:
    """Return (num_envs,) bool: stable grasp detected, with hysteresis & per-step cache."""

    step = int(getattr(env, "common_step_counter", 0))

    # --- init buffers ---
    if not hasattr(env, "_grasp_cache_step"):
        env._grasp_cache_step = -1
        env._grasp_cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        env._grasp_rel_cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        env._grasp_stable = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # --- cache: update only once per sim step ---
    if env._grasp_cache_step == step:
        return env._grasp_stable

    env._grasp_cache_step = step

    left = env.scene.sensors[left_sensor_cfg.name]
    right = env.scene.sensors[right_sensor_cfg.name]

    if left.data.net_forces_w is None or right.data.net_forces_w is None:
        env._grasp_stable[:] = False
        env._grasp_cnt[:] = 0
        env._grasp_rel_cnt[:] = 0
        return env._grasp_stable

    left_y = torch.abs(left.data.net_forces_w[:, 0, 1])
    right_y = torch.abs(right.data.net_forces_w[:, 0, 1])

    if require_both_contacts:
        contact = (left_y > contact_force_threshold) & (right_y > contact_force_threshold)
    else:
        contact = (left_y > contact_force_threshold) | (right_y > contact_force_threshold)

    # update hold counter
    env._grasp_cnt = torch.where(contact, env._grasp_cnt + 1, torch.zeros_like(env._grasp_cnt))
    newly_stable = env._grasp_cnt >= int(stable_steps)

    # hysteresis: once stable, only drop after release_steps of no-contact
    env._grasp_rel_cnt = torch.where(~contact, env._grasp_rel_cnt + 1, torch.zeros_like(env._grasp_rel_cnt))
    drop = env._grasp_rel_cnt >= int(release_steps)

    env._grasp_stable = (env._grasp_stable | newly_stable) & (~drop)

    return env._grasp_stable


#######################################################################
# reward function
#######################################################################
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),    
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the active object using tanh-kernel."""
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)

    # Extract the end-effector frame
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Calculate distance between EE and active object
    object_ee_distance = torch.norm(active_pos_w - ee_w, dim=1)

    # joint_pos = env.scene["robot"].data.joint_pos  # (envs, joints)
    # gripper_status = torch.abs(joint_pos[:, -1])
    # gripper_open = gripper_status >0.4
    # mask = 0.2 + 0.8*gripper_open
    # # print(object_ee_distance)
    # return (1 - torch.tanh(object_ee_distance/std)) *mask

    return 1 - torch.tanh(object_ee_distance/std)


def object_is_lifted_linear(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    max_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")
):
    """Linearly reward the agent for lifting the active object above the minimal height."""
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    
    current_height = active_pos_w[:, 2]
    
    # Clip height between [minimal_height, max_height]
    clipped_height = torch.clamp(current_height, minimal_height, max_height)
    # Normalize linearly to [0, 1]
    normalized = (clipped_height - minimal_height) / (max_height - minimal_height)
    # Square to increase reward for higher lifts
    reward = torch.square(normalized)
    
    return reward


def object_is_lifted_with_contact(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_height: float,
    contact_force_threshold: float = 0.7,
    require_both_contacts: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
) -> torch.Tensor:
    """
    Reward for lifting object ONLY when gripper fingers are in proper contact.

    Uses Y-axis forces (grasp axis) for robust contact detection.
    Prevents reward exploitation by hitting/throwing objects.

    Args:
        env: Environment
        minimal_height: Minimum height for reward (e.g., -0.01)
        max_height: Height where reward saturates (e.g., 0.045)
        contact_force_threshold: Minimum Y-axis force to consider contact (Newtons)
        require_both_contacts: If True, both fingers must contact. If False, at least one.
        object_cfg: Object pool configuration
        left_sensor_cfg: Left gripper contact sensor
        right_sensor_cfg: Right gripper contact sensor

    Returns:
        Reward tensor (num_envs,) - squared normalized height when contact detected, 0 otherwise
    """
    # 1. Get object height
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    current_height = active_pos_w[:, 2]  # Z coordinate

    # 2. Calculate height-based reward component
    clipped_height = torch.clamp(current_height, minimal_height, max_height)
    normalized = (clipped_height - minimal_height) / (max_height - minimal_height)
    height_reward = torch.square(normalized)  # Squared for exponential growth

    # 3. Get contact forces from both sensors
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        # No contact data available - return zero reward
        return torch.zeros(env.num_envs, device=env.device)

    left_forces = left_sensor.data.net_forces_w  # (num_envs, 1, 3)
    right_forces = right_sensor.data.net_forces_w  # (num_envs, 1, 3)

    # 4. Extract Y-axis forces (grasp axis - most reliable based on your data)
    left_y_force = torch.abs(left_forces[:, 0, 1])  # Y component, absolute value
    right_y_force = torch.abs(right_forces[:, 0, 1])  # Y component, absolute value

    # 5. Check if contact threshold exceeded
    left_contact = left_y_force > contact_force_threshold
    right_contact = right_y_force > contact_force_threshold

    # 6. Determine if proper contact condition is met
    if require_both_contacts:
        # Both fingers must be in contact (bilateral grasp - more robust)
        proper_contact = left_contact & right_contact
    else:
        # At least one finger in contact (more lenient)
        proper_contact = left_contact | right_contact

    # 7. Only reward lifting when proper contact is detected
    reward = torch.where(
        proper_contact,
        height_reward,  # Give height reward when contact verified
        torch.zeros_like(height_reward)  # Zero reward without contact
    )

    return reward


def pcd_contain_object(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    density_scale: float = 1.0,
    use_tanh: bool = False,
    min_ee_robot_distance: float = 0.15,
    max_ee_height: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_z_threshold: float = 0.7,
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
) -> torch.Tensor:
    """
    Reward based on point cloud density in the gripper sphere.
    If Y-axis contact forces exceed threshold, always returns maximum reward.
    Otherwise, gives reward based on density when other conditions are met.

    Args:
        env: Environment
        sensor_cfg_name: Name of the depth camera sensor
        density_scale: Scaling factor for reward sensitivity (also max reward when contact satisfied)
        use_tanh: If True, apply tanh for smoother gradients
        min_ee_robot_distance: Minimum distance (meters) between EE and robot base to give reward
        ee_frame_cfg: Configuration for end-effector frame
        robot_cfg: Configuration for robot entity
        left_sensor_cfg: Left contact sensor configuration
        right_sensor_cfg: Right contact sensor configuration
        contact_z_threshold: Maximum Z-component value for contact sensors (default: 0.7)
        contact_force_threshold: Minimum Y-axis force to consider contact (Newtons)
        require_both_contacts: If True, both fingers must contact. If False, at least one.

    Returns:
        Reward tensor (num_envs,)
    """
    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    # 1. Check contact sensors first - if Y threshold satisfied, return max reward immediately
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    contact_force_satisfied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if left_sensor.data.net_forces_w is not None and right_sensor.data.net_forces_w is not None:
        left_y_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
        right_y_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])

        left_contact = left_y_force > contact_force_threshold
        right_contact = right_y_force > contact_force_threshold

        if require_both_contacts:
            contact_force_satisfied = left_contact & right_contact
        else:
            contact_force_satisfied = left_contact | right_contact

    # If contact force satisfied, return maximum reward immediately
    max_reward = density_scale if not use_tanh else torch.ones(env.num_envs, device=env.device)

    if contact_force_satisfied.any():
        reward = torch.where(
            contact_force_satisfied,
            max_reward if isinstance(max_reward, torch.Tensor) else torch.full((env.num_envs,), max_reward, device=env.device),
            torch.zeros(env.num_envs, device=env.device)
        )
        
        # For envs where contact not satisfied, compute density-based reward
        if not contact_force_satisfied.all():
            # Continue with normal logic for non-contact-satisfied envs
            pointcloud, valid = get_cached_pointcloud(env)
            if valid and pointcloud is not None:
                left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
                right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

                left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
                right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)

                sphere_center = (left_finger_cam + right_finger_cam) / 2.0
                finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
                sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003)

                density, num_points = calculate_pointcloud_density_in_sphere(
                    pointcloud, sphere_center, sphere_radius
                )

                if use_tanh:
                    density_reward = torch.tanh(density * density_scale * 3.0)
                else:
                    density_reward = density * density_scale

                ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
                ee_w = ee_frame.data.target_pos_w[..., 0, :]
                robot = env.scene[robot_cfg.name]
                robot_base_pos = robot.data.root_pos_w
                ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
                ee_z_valid = ee_w[:, 2] < max_ee_height
                distance_mask = ee_robot_distance >= min_ee_robot_distance

                left_z = left_sensor.data.net_forces_w[:, 0, 2]
                right_z = right_sensor.data.net_forces_w[:, 0, 2]
                contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

                # joint_pos = env.scene["robot"].data.joint_pos
                # gripper_status = torch.abs(joint_pos[:, -1])
                # gripper_open = gripper_status > 0.4

                all_conditions_met = distance_mask & contact_z_valid & ee_z_valid

                # Update reward for non-contact-satisfied envs
                reward = torch.where(
                    contact_force_satisfied,
                    reward,  # Keep max reward
                    torch.where(all_conditions_met, density_reward, torch.zeros_like(density_reward))
                )

        return reward

    # 2. If no contact force satisfied, continue with normal density-based logic
    pointcloud, valid = get_cached_pointcloud(env)
    if not valid or pointcloud is None:
        return torch.zeros(env.num_envs, device=env.device)

    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)

    sphere_center = (left_finger_cam + right_finger_cam) / 2.0
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003)

    density, num_points = calculate_pointcloud_density_in_sphere(
        pointcloud, sphere_center, sphere_radius
    )

    if use_tanh:
        density_reward = torch.tanh(density * density_scale * 3.0)
    else:
        density_reward = density * density_scale

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    robot = env.scene[robot_cfg.name]
    robot_base_pos = robot.data.root_pos_w
    ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
    ee_z_valid = ee_w[:, 2] < max_ee_height
    distance_mask = ee_robot_distance >= min_ee_robot_distance

    left_z = left_sensor.data.net_forces_w[:, 0, 2]
    right_z = right_sensor.data.net_forces_w[:, 0, 2]
    contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

    joint_pos = env.scene["robot"].data.joint_pos
    gripper_status = torch.abs(joint_pos[:, -1])
    gripper_open = gripper_status > 0.4

    all_conditions_met = distance_mask & contact_z_valid & ee_z_valid & gripper_open

    reward = torch.where(
        all_conditions_met,
        density_reward,
        torch.zeros_like(density_reward)
    )

    return reward


def contact_clamp_object(
    env: ManagerBasedRLEnv,
    contact_force_threshold: float = 1.5,
    reward_value: float = 1.0,
    gripper_closed_threshold: float = 0.2,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
) -> torch.Tensor:
    """
    Binary reward for clamping - either 1.0 or 0.0.
    Simpler version that just checks if grasp is good enough.
    """
    # Check gripper closed
    joint_positions = env.scene['robot'].data.joint_pos_target
    gripper_closed = (torch.abs(joint_positions[:, 5]) < gripper_closed_threshold) & \
                     (torch.abs(joint_positions[:, 6]) < gripper_closed_threshold)

    # Get contact forces
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Y-axis forces
    left_y = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
    right_y = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])

    # Average force check
    avg_force = (left_y + right_y) / 2.0
    good_grasp = avg_force > contact_force_threshold

    # Binary reward
    reward = torch.where(
        gripper_closed & good_grasp,
        torch.ones(env.num_envs, device=env.device) * reward_value,
        torch.zeros(env.num_envs, device=env.device)
    )

    return reward


def contain_object(env, std, object_cfg=SceneEntityCfg("object_pool"),
                   finger_frame_1_cfg=SceneEntityCfg("finger_frame_1"),
                   finger_frame_2_cfg=SceneEntityCfg("finger_frame_2")):
    active_pos_w, _ = get_active_object_states(env, object_cfg)

    finger_frame_1 = env.scene[finger_frame_1_cfg.name]
    finger_frame_2 = env.scene[finger_frame_2_cfg.name]
    finger_w_1 = finger_frame_1.data.target_pos_w[..., 0, :]
    finger_w_2 = finger_frame_2.data.target_pos_w[..., 0, :]

    v1 = finger_w_1 - active_pos_w
    v2 = finger_w_2 - active_pos_w
    dot = torch.sum(v1 * v2, dim=1)
    n1 = torch.norm(v1, dim=1)
    n2 = torch.norm(v2, dim=1)
    cos_theta = dot / (n1 * n2 + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    x = theta / np.pi  # [0,1]

    gamma = 2.0  # 推荐先从 2 开始试
    # 或者用 std 控制：gamma = 1.0 + 1.0/(std+1e-6)
    reward = x.pow(gamma)
    return reward


def m0_turn_toward_object(
    env,
    std: float = 0.35,                       # 越小越“严格”
    in_range_deg: float | None = 20.0,       # 只奖励“转到范围内”；None 表示用连续高斯奖励
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    center_from_robot: bool = True,
    debug: bool = False,
    debug_every_steps: int = 1,
    debug_env: int = 0,
):
    """
    让 M0 朝向物体所在方向：
    - target_yaw = atan2(obj_y - center_y, obj_x - center_x)
    - err = wrap_to_pi(target_yaw - m0_yaw)
    - reward: 1) 如果 in_range_deg 不为 None：在范围内给 1，否则给 0（容易学“转到位”）
              2) 如果 in_range_deg 为 None：用 exp(-0.5*(err/std)^2)（更平滑）
    """
    robot = env.scene[robot_cfg.name]

    # Get M0 关节索引作为缓存
    if not hasattr(env, '_m0_joint_id'):
        # find_joints 返回 (list[int], list[str])
        joint_ids, joint_names = robot.find_joints("^M0$", preserve_order=True)
        if len(joint_ids) == 0:
            # 容错：如果你的关节命名不是严格 M0，可以先用 "M0" 模糊匹配一次
            joint_ids, joint_names = robot.find_joints("M0", preserve_order=True)
        if len(joint_ids) == 0:
            raise RuntimeError(
                f"Cannot resolve M0 joint id. Available joints: {robot.joint_names}"
            )
        if len(joint_ids) > 1:
            # 一般不会发生，但如果发生，打印一下你匹配到了哪些
            raise RuntimeError(f"Multiple joints matched M0: {list(zip(joint_ids, joint_names))}")
        env._m0_joint_id = int(joint_ids[0])

    m0_id = env._m0_joint_id
    m0_yaw = robot.data.joint_pos[:, m0_id]  # (num_envs,)

    # --- active object world position（你原来 rewards.py 里已经有 helper 就用它；没有就按下面方式取）---
    if "get_active_object_states" in globals():
        obj_pos_w, _ = get_active_object_states(env, object_cfg)
    else:
        # fallback: 尽量从 RigidObjectCollection 里取（不同版本字段名可能不同）
        obj_asset = env.scene[object_cfg.name]
        if hasattr(obj_asset.data, "root_pos_w"):
            # (num_envs, num_objects, 3) + active index
            active = env.active_object_indices.to(dtype=torch.long)
            obj_pos_w = obj_asset.data.root_pos_w[torch.arange(active.shape[0], device=active.device), active]
        elif hasattr(obj_asset.data, "object_pos_w"):
            active = env.active_object_indices.to(dtype=torch.long)
            obj_pos_w = obj_asset.data.object_pos_w[torch.arange(active.shape[0], device=active.device), active]
        else:
            raise RuntimeError("Cannot access active object position. Please ensure get_active_object_states exists.")

    # 圆心：用 robot root 或者 env_origin
    if center_from_robot:
        center_w = robot.data.root_pos_w  # (num_envs, 3)
    else:
        center_w = env.scene.env_origins  # (num_envs, 3)

    d = obj_pos_w - center_w
    target_yaw = torch.atan2(d[:, 1], d[:, 0])  # (num_envs,)

    # 封装到 [-pi, pi]
    err = torch.atan2(torch.sin(target_yaw - m0_yaw), torch.cos(target_yaw - m0_yaw))

    # ===== debug 打印：每步输出 target_yaw / m0_yaw / err(deg) =====
    if debug:
        step = int(getattr(env, "common_step_counter", -1))
        last = int(getattr(env, "_dbg_m0_turn_last_print_step", -10**9))
        every = max(int(debug_every_steps), 1)
        if step - last >= every:
            setattr(env, "_dbg_m0_turn_last_print_step", step)
            e = int(debug_env)
            if 0 <= e < err.shape[0]:
                rad2deg = 180.0 / math.pi
                t_deg = float(target_yaw[e].item() * rad2deg)
                m_deg = float(m0_yaw[e].item() * rad2deg)
                err_deg = float(err[e].item() * rad2deg)
                print(
                    f"[m0_turn_dbg] step={step} env={e} "
                    f"target_yaw={t_deg:+.2f}deg  m0_yaw={m_deg:+.2f}deg  err={err_deg:+.2f}deg",
                    flush=True,
                )

    if in_range_deg is not None:
        thr = float(in_range_deg) * math.pi / 180.0
        return (err.abs() <= thr).to(dtype=torch.float32)
    else:
        # smooth gaussian shaping
        std = max(float(std), 1e-6)
        return torch.exp(-0.5 * (err / std) ** 2)


def m0_turn_toward_object_until_grasp(
    env,
    post_grasp_scale: float = 0.0,
    # ---- grasp gating params ----
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
    stable_steps: int = 8,
    release_steps: int = 2,

    # ---- 原 m0_turn_toward_object 的参数（显式列出）----
    std: float = 0.35,
    in_range_deg: float | None = 20.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    center_from_robot: bool = True,
    debug: bool = False,
    debug_every_steps: int = 1,
    debug_env: int = 0,
):
    base = m0_turn_toward_object(
        env,
        std=std,
        in_range_deg=in_range_deg,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        center_from_robot=center_from_robot,
        debug=debug,
        debug_every_steps=debug_every_steps,
        debug_env=debug_env,
    )
    stable = get_stable_grasp_mask(
        env,
        left_sensor_cfg=left_sensor_cfg,
        right_sensor_cfg=right_sensor_cfg,
        contact_force_threshold=contact_force_threshold,
        require_both_contacts=require_both_contacts,
        stable_steps=stable_steps,
        release_steps=release_steps,
    )
    scale = torch.where(stable, torch.full_like(base, post_grasp_scale), torch.ones_like(base))
    return base * scale


#######################################################################
# penalty function
#######################################################################
def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def penalize_active_object_tangential_speed(
    env,
    penalty_scale: float = 1.0,
    v_deadzone: float = 0.01,
    center_from_robot: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # --- gating: contact-based grasp detection ---
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
    # optional hysteresis: require stable contact for N steps
    stable_steps: int = 5,  # 0 = disable; e.g. 3 or 5 to reduce noise
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize tangential speed v_t of the ACTIVE object around M0 rotation center.

    - v_t is computed in XY plane: v_t = dot(v_xy, t_hat), where t_hat is tangential unit vector.
    - penalty = relu(|v_t| - v_deadzone)^2 * penalty_scale
    - gating: only apply when grasp contact is detected (Y-force on both fingers)
    """

    # 1) active object pos & vel
    obj_pos_w, _ = get_active_object_states(env, object_cfg)            # (N, 3)
    obj_vel_w = get_active_object_lin_vel_w(env, object_cfg)            # (N, 3)

    obj_xy = obj_pos_w[:, :2]
    vel_xy = obj_vel_w[:, :2]

    # 2) center of rotation
    if center_from_robot:
        robot = env.scene[robot_cfg.name]
        center_xy = robot.data.root_pos_w[:, :2]
    else:
        center_xy = env.scene.env_origins[:, :2]

    # 3) tangential unit vector
    r = obj_xy - center_xy                                 # (N, 2)
    r_norm = torch.norm(r, dim=1, keepdim=True).clamp_min(eps)
    t_hat = torch.cat([-r[:, 1:2], r[:, 0:1]], dim=1) / r_norm  # (N, 2)

    # 4) tangential speed (signed)
    v_t = torch.sum(vel_xy * t_hat, dim=1)                 # (N,)

    # 5) deadzone penalty
    excess = torch.relu(torch.abs(v_t) - float(v_deadzone))
    pen = (excess * excess) * float(penalty_scale)         # (N,)

    # 6) gating from contact sensors (Y force)
    left = env.scene.sensors[left_sensor_cfg.name]
    right = env.scene.sensors[right_sensor_cfg.name]

    if left.data.net_forces_w is None or right.data.net_forces_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    left_y = torch.abs(left.data.net_forces_w[:, 0, 1])
    right_y = torch.abs(right.data.net_forces_w[:, 0, 1])

    if require_both_contacts:
        contact_mask = (left_y > contact_force_threshold) & (right_y > contact_force_threshold)
    else:
        contact_mask = (left_y > contact_force_threshold) | (right_y > contact_force_threshold)

    # 7) optional hysteresis (stable_steps)
    if stable_steps > 0:
        if not hasattr(env, "_vt_contact_streak"):
            env._vt_contact_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._vt_contact_streak = torch.where(
            contact_mask,
            (env._vt_contact_streak + 1).clamp_max(10**9),
            torch.zeros_like(env._vt_contact_streak),
        )
        contact_mask = env._vt_contact_streak >= int(stable_steps)

    # 8) apply gating
    pen = torch.where(contact_mask, pen, torch.zeros_like(pen))
    return pen


#######################################################################
# OPTIONAL: Add this helper function to track gripper state for debugging
#######################################################################
def debug_gripper_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Debug function to log gripper state.
    Add this to your rewards with weight=0.0 to enable logging without affecting training.
    """
    if env.common_step_counter % 100 == 0:
        joint_positions = env.scene['robot'].data.joint_pos_target
        gripper_opening = (torch.abs(joint_positions[:, 5]) + torch.abs(joint_positions[:, 6])) / 2.0

        object = env.scene['object']
        ee_frame = env.scene['ee_frame']
        distance = torch.norm(
            object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :],
            dim=1
        )

        print(f"\n=== Step {env.common_step_counter} ===")
        print(f"Gripper opening: {gripper_opening.mean():.3f} (0=closed, 0.5=open)")
        print(f"Object-EE distance: {distance.mean():.3f}")
        print(f"Object height: {object.data.root_pos_w[:, 2].mean():.3f}")

    return torch.zeros(env.num_envs, device=env.device)


def penalize_m0_after_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    m0_movement_penalty_scale: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize M0 joint movement after the active object is lifted.

    Args:
        env: The environment
        minimal_height: Height threshold to consider object as lifted
        m0_movement_penalty_scale: Scale factor for the penalty (higher = more penalty)
        object_cfg: Configuration for the object entity
        robot_cfg: Configuration for the robot entity

    Returns:
        Penalty for M0 movement (0 when object not lifted, penalty when lifted and M0 moves)
    """
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)

    # Get robot
    robot: RigidObject = env.scene[robot_cfg.name]

    # Check if active object is lifted
    object_lifted = active_pos_w[:, 2] > minimal_height

    # Get current M0 joint position (first joint, index 0)
    current_m0_pos = robot.data.joint_pos[:, 0]

    # Initialize previous M0 position storage if it doesn't exist
    if not hasattr(env, '_prev_m0_pos'):
        env._prev_m0_pos = current_m0_pos.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Calculate M0 movement (absolute change in position)
    m0_movement = torch.abs(current_m0_pos - env._prev_m0_pos)

    # Update previous position for next step
    env._prev_m0_pos = current_m0_pos.clone()

    # Apply penalty only when object is lifted
    # Negative reward (penalty) scaled by movement magnitude
    penalty = torch.where(
        object_lifted,
        -m0_movement * m0_movement_penalty_scale,
        torch.zeros_like(m0_movement)
    )

    return penalty


def penalize_xy_displacement(
    env: ManagerBasedRLEnv,
    penalty_scale: float = 1.0,
    min_height: float = -0.05,
    max_height: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:

    active_pos_w, _ = get_active_object_states(env, object_cfg)
    if not hasattr(env, '_prev_object_xy_pos'):
        env._prev_object_xy_pos = active_pos_w[:, :2].clone()  # Store only x,y
        return torch.zeros(env.num_envs, device=env.device)

    # Calculate XY displacement (L2 norm)
    current_xy = active_pos_w[:, :2]
    xy_displacement = torch.norm(current_xy - env._prev_object_xy_pos, dim=1)
    # print(xy_displacement)
    env._prev_object_xy_pos = current_xy.clone()

    current_height = active_pos_w[:, 2]

    # Normalize height
    normalized_height = torch.clamp(
        (current_height - min_height) / (max_height - min_height),
        0.0, 1.0
    )
    height_weight = 1.0 - normalized_height

    # Calculate penalty
    penalty = xy_displacement * height_weight * penalty_scale

    return penalty


def get_cached_pointcloud(env: ManagerBasedRLEnv) -> tuple[torch.Tensor | None, bool]:
    """Safely retrieve cached point cloud."""
    if not hasattr(env, 'point_cloud_cache') or env.point_cloud_cache is None:
        return None, False
    return env.point_cloud_cache, True


def debug_pcd_cache(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug cache access."""
    # Remove the modulo check - print EVERY step
    pcd, valid = get_cached_pointcloud(env)
    print(f"[DEBUG] Step={env.common_step_counter}, Valid={valid}, Shape={pcd.shape if pcd is not None else None}", flush=True)
    return torch.zeros(env.num_envs, device=env.device)


def penalty_if_gripper_closed_far(
    env: ManagerBasedRLEnv,
    reach_threshold: float = 0.03 ,  # 10cm 外必须张开  
    open_threshold: float = 0.04,   # gripper opening 小于这个算“关闭”
    penalty_value: float = -0.2,    # 惩罚
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    # 1. 获取 EE 和物体距离
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(active_pos_w - ee_w, dim=1)

    # 2. 获取 gripper 的两个关节角
    joint_pos = env.scene["robot"].data.joint_pos  # (envs, joints)
    gripper_opening = torch.abs(joint_pos[:, -1])

    # 3. 若距离远且没张开 → 惩罚
    need_open_mask = dist > reach_threshold        # True = 还没靠近
    is_closed_mask = gripper_opening < open_threshold

    penalty_mask = need_open_mask & is_closed_mask

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[penalty_mask] = penalty_value

    return penalty


def pcd_clamp_object(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    density_threshold: float = 0.09,
    density_scale: float = 1.0,
    min_ee_robot_distance: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_closed_threshold: float = 0.02,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_z_threshold: float = 0.7,
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
) -> torch.Tensor:
    """
    Combined reward for clamping with gripper closed and object detected.
    If Y-axis contact forces exceed threshold, always returns maximum reward.
    Otherwise, gives reward when all other conditions are met.

    Args:
        env: Environment
        sensor_cfg_name: Name of depth camera sensor
        density_threshold: Minimum density to consider object detected
        density_scale: Reward value when all conditions met (also max reward when contact satisfied)
        min_ee_robot_distance: Minimum EE-robot distance
        ee_frame_cfg: End-effector frame configuration
        robot_cfg: Robot configuration
        gripper_closed_threshold: Maximum joint position to consider gripper closed
        left_sensor_cfg: Left contact sensor configuration
        right_sensor_cfg: Right contact sensor configuration
        contact_z_threshold: Maximum Z-component value for contact sensors (default: 0.7)
        contact_force_threshold: Minimum Y-axis force to consider contact (Newtons)
        require_both_contacts: If True, both fingers must contact. If False, at least one.

    Returns:
        Reward tensor (num_envs,)
    """

    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    # 1. Check contact sensors first - if Y threshold satisfied, return max reward immediately
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    contact_force_satisfied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if left_sensor.data.net_forces_w is not None and right_sensor.data.net_forces_w is not None:
        left_y_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
        right_y_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])

        left_contact = left_y_force > contact_force_threshold
        right_contact = right_y_force > contact_force_threshold

        if require_both_contacts:
            contact_force_satisfied = left_contact & right_contact
        else:
            contact_force_satisfied = left_contact | right_contact

    # If contact force satisfied, return maximum reward
    if contact_force_satisfied.any():
        reward = torch.where(
            contact_force_satisfied,
            torch.full((env.num_envs,), density_scale, device=env.device),
            torch.zeros(env.num_envs, device=env.device)
        )

        # For envs where contact not satisfied, compute normal logic
        if not contact_force_satisfied.all():
            pointcloud, valid = get_cached_pointcloud(env)
            if valid and pointcloud is not None:
                joint_positions = env.scene['robot'].data.joint_pos_target
                finger_joint_1 = joint_positions[:, 5]
                finger_joint_2 = joint_positions[:, 6]
                gripper_closed = (torch.abs(finger_joint_1) < gripper_closed_threshold) & \
                                 (torch.abs(finger_joint_2) < gripper_closed_threshold)

                left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
                right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

                left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
                right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)
                sphere_center = (left_finger_cam + right_finger_cam) / 2.0
                finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
                sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003)

                density, num_points = calculate_pointcloud_density_in_sphere(
                    pointcloud, sphere_center, sphere_radius
                )
                object_detected = density > density_threshold

                ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
                ee_w = ee_frame.data.target_pos_w[..., 0, :]
                robot = env.scene[robot_cfg.name]
                robot_base_pos = robot.data.root_pos_w
                ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
                distance_mask = ee_robot_distance >= min_ee_robot_distance

                left_z = left_sensor.data.net_forces_w[:, 0, 2]
                right_z = right_sensor.data.net_forces_w[:, 0, 2]
                contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

                all_conditions_met = gripper_closed & object_detected & distance_mask & contact_z_valid

                # Update reward for non-contact-satisfied envs
                reward = torch.where(
                    contact_force_satisfied,
                    reward,  # Keep max reward
                    torch.where(all_conditions_met, torch.full((env.num_envs,), density_scale, device=env.device), torch.zeros(env.num_envs, device=env.device))
                )

        return reward

    # 2. If no contact force satisfied, continue with normal logic
    pointcloud, valid = get_cached_pointcloud(env)
    if not valid or pointcloud is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_positions = env.scene['robot'].data.joint_pos_target
    finger_joint_1 = joint_positions[:, 5]
    finger_joint_2 = joint_positions[:, 6]
    gripper_closed = (torch.abs(finger_joint_1) < gripper_closed_threshold) & \
                     (torch.abs(finger_joint_2) < gripper_closed_threshold)

    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)
    sphere_center = (left_finger_cam + right_finger_cam) / 2.0
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003)

    density, num_points = calculate_pointcloud_density_in_sphere(
        pointcloud, sphere_center, sphere_radius
    )
    object_detected = density > density_threshold

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    robot = env.scene[robot_cfg.name]
    robot_base_pos = robot.data.root_pos_w
    ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
    distance_mask = ee_robot_distance >= min_ee_robot_distance

    left_z = left_sensor.data.net_forces_w[:, 0, 2]
    right_z = right_sensor.data.net_forces_w[:, 0, 2]
    contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

    all_conditions_met = gripper_closed & object_detected & distance_mask & contact_z_valid

    reward = torch.where(
        all_conditions_met,
        torch.full((env.num_envs,), density_scale, device=env.device),
        torch.zeros(env.num_envs, device=env.device)
    )

    return reward


def debug_pcd_density(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug version - prints density info."""
    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    if env.common_step_counter % 3 == 0:
        pointcloud, valid = get_cached_pointcloud(env)
        if valid and pointcloud is not None:
            # Get gripper positions
            left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
            right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003)

            # Calculate density
            density, num_points = calculate_pointcloud_density_in_sphere(
                pointcloud, sphere_center, sphere_radius
            )

            print(f"\n=== PCD Density Debug (Step {env.common_step_counter}) ===")
            print(f"Sphere radius: {sphere_radius[0].item():.4f}")
            print(f"Points in sphere: {num_points[0].item():.0f} / 2048")
            print(f"Density: {density[0].item():.4f}")
            print("=" * 50)

    return torch.zeros(env.num_envs, device=env.device)


def visualize_pcd_sphere(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Save point cloud visualization with sphere region highlighted."""
    from .gripper_transform import transform_world_to_camera
    import os

    if env.common_step_counter % 1 == 0:
        pointcloud, valid = get_cached_pointcloud(env)
        if valid and pointcloud is not None:
            # Get gripper positions
            left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
            right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]
            active_pos_w, _ = get_active_object_states(env)

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            object_cam = transform_world_to_camera(active_pos_w, env, "depth_camera")
            
            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance *0.22, min=0.003)

            # Get points inside sphere
            env_id = 0
            sphere_center_expanded = sphere_center[env_id].unsqueeze(0)
            distances = torch.norm(pointcloud[env_id] - sphere_center_expanded, dim=-1)
            inside_mask = distances < sphere_radius[env_id]

            points_inside = pointcloud[env_id][inside_mask].cpu().numpy()
            points_outside = pointcloud[env_id][~inside_mask].cpu().numpy()

            left_finger = left_finger_cam[env_id].cpu().numpy()
            right_finger = right_finger_cam[env_id].cpu().numpy()
            center = sphere_center[env_id].cpu().numpy()
            obj_pos = object_cam[env_id].cpu().numpy()

            # Create ABSOLUTE path
            save_dir = os.path.abspath(os.path.expanduser("~/pcd_sphere_debug"))
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(save_dir, f"step_{env.common_step_counter:06d}.ply")

            # Print BEFORE writing
            print(f"\n{'='*80}")
            print(f"🎨 SAVING VISUALIZATION")
            print(f"{'='*80}")
            print(f"Absolute path: {filename}")
            print(f"Directory exists: {os.path.exists(save_dir)}")
            print(f"Points inside sphere: {len(points_inside)}")
            print(f"Points outside sphere: {len(points_outside)}")

            with open(filename, 'w') as f:
                total_points = len(points_outside) + len(points_inside) + 104

                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {total_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                # Points OUTSIDE sphere (gray)
                for pt in points_outside:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 150 150 150\n")

                # Points INSIDE sphere (bright red)
                for pt in points_inside:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 255 0 0\n")

                # Markers
                f.write(f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 255 255 0\n")  # Yellow
                f.write(f"{obj_pos[0]:.6f} {obj_pos[1]:.6f} {obj_pos[2]:.6f} 0 255 0\n")  # Green
                f.write(f"{left_finger[0]:.6f} {left_finger[1]:.6f} {left_finger[2]:.6f} 0 0 255\n")  # Blue
                f.write(f"{right_finger[0]:.6f} {right_finger[1]:.6f} {right_finger[2]:.6f} 0 255 255\n")  # Cyan

                # Sphere surface (magenta)
                import numpy as np
                phi = np.linspace(0, np.pi, 10)
                theta = np.linspace(0, 2*np.pi, 10)
                for p in phi:
                    for t in theta:
                        x = center[0] + sphere_radius[env_id].item() * np.sin(p) * np.cos(t)
                        y = center[1] + sphere_radius[env_id].item() * np.sin(p) * np.sin(t)
                        z = center[2] + sphere_radius[env_id].item() * np.cos(p)
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 0 255\n")

            # Print AFTER writing
            print(f"✅ File written successfully!")
            print(f"File size: {os.path.getsize(filename)} bytes")
            print(f"\nTo view:")
            print(f"  meshlab {filename}")
            print(f"  # OR list all files:")
            print(f"  ls -lh {save_dir}/")
            print(f"{'='*80}\n")

    return torch.zeros(env.num_envs, device=env.device)


def debug_contact_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Minimal contact force debug - just XYZ components."""

    if env.common_step_counter % 1 == 0:
        left_sensor = env.scene.sensors["contact_forces_left"]
        right_sensor = env.scene.sensors["contact_forces_right"]

        if left_sensor.data.net_forces_w is not None:
            left_force = left_sensor.data.net_forces_w[0].cpu().numpy().flatten()
        else:
            left_force = [0, 0, 0]

        if right_sensor.data.net_forces_w is not None:
            right_force = right_sensor.data.net_forces_w[0].cpu().numpy().flatten()
        else:
            right_force = [0, 0, 0]

        print(f"[Step {env.common_step_counter}] Left: [{left_force[0]:.3f}, {left_force[1]:.3f}, {left_force[2]:.3f}] | Right: [{right_force[0]:.3f}, {right_force[1]:.3f}, {right_force[2]:.3f}]")

    return torch.zeros(env.num_envs, device=env.device)
