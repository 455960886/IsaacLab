# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import os
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere, calculate_pointcloud_density_in_sphere1

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Help Function to get the states of active object from the object pool
def get_active_object_states(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")):
    """
    Helper function to get states of active objects from object pool.
    
    Returns:
        pos_w: (num_envs, 3) - World positions of active objects
        quat_w: (num_envs, 4) - World orientations of active objects [w, x, y, z]
    """
    from isaaclab.assets import RigidObjectCollection
    
    object_pool: RigidObjectCollection = env.scene[object_cfg.name]
    
    # Get active object indices for each environment
    if not hasattr(env, 'active_object_indices'):
        raise RuntimeError("active_object_indices not found. Ensure randomize_object_pool_selection has been called.")
    
    active_indices = env.active_object_indices  # (num_envs,)
    
    # Get all object states: (num_envs, num_objects, state_dim)
    all_pos_w = object_pool.data.object_pos_w  # (num_envs, num_objects, 3)
    all_quat_w = object_pool.data.object_quat_w  # (num_envs, num_objects, 4)
    
    # Index to get only active objects
    # Use advanced indexing: env_indices = [0, 1, 2, ...], object_indices = active_indices
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_pos_w = all_pos_w[env_indices, active_indices]  # (num_envs, 3)
    active_quat_w = all_quat_w[env_indices, active_indices]  # (num_envs, 4)
    
    return active_pos_w, active_quat_w


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")
):
    """Reward the agent for lifting the active object above the minimal height."""
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    
    # Check if active objects are above minimal height
    return torch.where(active_pos_w[:, 2] > minimal_height, 1.0, 0.0)


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
    dist = torch.norm(active_pos_w - ee_w, dim=1)
    raw_rew = torch.exp(-dist / std)
    mask = (dist < 0.085).float()
    rew = raw_rew * mask

    return rew


def contain_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    finger_frame_1_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_1"),
    finger_frame_2_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_2"),
) -> torch.Tensor:
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    
    finger_frame_1: FrameTransformer = env.scene[finger_frame_1_cfg.name]
    finger_frame_2: FrameTransformer = env.scene[finger_frame_2_cfg.name]

    finger_w_1 = finger_frame_1.data.target_pos_w[..., 0, :]
    finger_w_2 = finger_frame_2.data.target_pos_w[..., 0, :]
    
    vector_1 = finger_w_1 - active_pos_w
    vector_2 = finger_w_2 - active_pos_w
    dot_products = torch.sum(vector_1 * vector_2, dim=1)
    norm_1 = torch.norm(vector_1, dim=1)
    norm_2 = torch.norm(vector_2, dim=1)
    cos_theta = dot_products / (norm_1 * norm_2 + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    theta = torch.acos(cos_theta)
    reward = theta / np.pi

    return reward


def clamp_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    finger_frame_1_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_1"),
    finger_frame_2_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_2"),
) -> torch.Tensor:
    # Get active object positions
    active_pos_w, _ = get_active_object_states(env, object_cfg)
    
    finger_frame_1: FrameTransformer = env.scene[finger_frame_1_cfg.name]
    finger_frame_2: FrameTransformer = env.scene[finger_frame_2_cfg.name]

    finger_w_1 = finger_frame_1.data.target_pos_w[..., 0, :]
    finger_w_2 = finger_frame_2.data.target_pos_w[..., 0, :]

    vector_1 = finger_w_1 - active_pos_w
    vector_2 = finger_w_2 - active_pos_w
    dot_products = torch.sum(vector_1 * vector_2, dim=1)
    norm_1 = torch.norm(vector_1, dim=1)
    norm_2 = torch.norm(vector_2, dim=1)
    cos_theta = dot_products / (norm_1 * norm_2 + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)

    # Get gripper joint positions - use correct indices (5 and 6)
    joint_positions = env.scene['robot'].data.joint_pos_target
    finger_joint_1 = joint_positions[:, 5]  # M6_1
    finger_joint_2 = joint_positions[:, 6]  # M6_2
    
    # Closed state: both joints are near 0.0 (with tolerance)
    gripper_closed = (torch.abs(finger_joint_1) < 0.3) & (torch.abs(finger_joint_2) < 0.3)
    
    # Object is between fingers: angle between vectors should be large (> 4.5π/6 ≈ 135°)
    object_between_fingers = theta > (np.pi * 4/6) # 120 deg
    
    # Give reward only when gripper is closed AND object is between fingers
    mask = gripper_closed & object_between_fingers
    reward = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))

    return reward


# OPTIONAL: Add this helper function to track gripper state for debugging
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
    current_m0_pos = robot.data.joint_pos[:, 1]

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


def get_cached_pointcloud(env: ManagerBasedRLEnv) -> tuple[torch.Tensor | None, bool]:
    """Safely retrieve cached point cloud."""
    if not hasattr(env, 'point_cloud_cache') or env.point_cloud_cache is None:
        return None, False
    return env.point_cloud_cache, True


def get_cached_pointcloud_with_semantics(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """
    同时返回点云 (B,N,3) 和每个点的语义 ID (B,N)。
    """
    if (
        not hasattr(env, "point_cloud_cache")
        or env.point_cloud_cache is None
        or not hasattr(env, "point_cloud_semantic_cache")
        or env.point_cloud_semantic_cache is None
    ):
        return None, None, False
    return env.point_cloud_cache, env.point_cloud_semantic_cache, True


def debug_pcd_cache(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug cache access."""
    # Remove the modulo check - print EVERY step
    pcd, valid = get_cached_pointcloud(env)
    print(f"[DEBUG] Step={env.common_step_counter}, Valid={valid}, Shape={pcd.shape if pcd is not None else None}", flush=True)
    return torch.zeros(env.num_envs, device=env.device)


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

    all_conditions_met = distance_mask & contact_z_valid & ee_z_valid

    reward = torch.where(
        all_conditions_met,
        density_reward,
        torch.zeros_like(density_reward)
    )

    return reward


def pcd_contain_object1(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    density_scale: float = 1.0,
    use_tanh: bool = False,
    min_ee_robot_distance: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_z_threshold: float = 0.7,
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
    valid_object_name: str = "eye_drops",
    excluded_object_names: list = ["m6_1_leftfinger_link", "m6_2_rightfinger_link", "m5_wrist_link"],
) -> torch.Tensor:
    # ------------- 语义信息：class 名 -> id -------------
    sensor = env.scene.sensors[sensor_cfg_name]
    id_to_labels = sensor.data.info["semantic_segmentation"]["idToLabels"]
    step = getattr(env, "common_step_counter", 0)
    rank = getattr(env, "rank", 0)

    def _parse_id(k):
        try:
            return int(k)
        except (TypeError, ValueError):
            return None

    semantic_class_names = [f"{k}: {v.get('class', '')}" for k, v in id_to_labels.items()]

    raw_valid_object_id = next(
        (k for k, obj in id_to_labels.items() if obj.get("class") == valid_object_name),
        None,
    )
    valid_object_id = _parse_id(raw_valid_object_id) if raw_valid_object_id is not None else None

    excluded_object_ids: list[int] = []
    for k, obj in id_to_labels.items():
        if obj.get("class") in excluded_object_names:
            pid = _parse_id(k)
            if pid is not None:
                excluded_object_ids.append(pid)

    if not hasattr(env, "_debug_sem_last_print_step"):
        env._debug_sem_last_print_step = -1
    if not hasattr(env, "_debug_sem_last_valid_object_id"):
        env._debug_sem_last_valid_object_id = None

    should_print_semantic_table = False
    warmup_steps = 10          # 前 10 步视为“预热期”
    big_interval = 500         # 之后每隔这么多步打一回大表
    warn_interval = 50         # 找不到 eye_drops 时，隔多少步再提醒一次

    if valid_object_id is None:
        # 预热期：第 0 步和第 warmup_steps-1 步各打一回，确认语义表
        if step in (0, warmup_steps - 1):
            should_print_semantic_table = True
        # 预热期之后，如果还是找不到，就每 warn_interval 步提示一次
        elif step >= warmup_steps and (step % warn_interval == 0):
            should_print_semantic_table = True
    else:
        # 第一次成功找到 eye_drops：打印一次完整语义表
        if env._debug_sem_last_valid_object_id is None:
            should_print_semantic_table = True
        # 如果 id 发生变化（理论上很少），也打印
        elif valid_object_id != env._debug_sem_last_valid_object_id:
            should_print_semantic_table = True
        # 同一个 id，隔 big_interval 步再打印一次
        elif (step - env._debug_sem_last_print_step) >= big_interval:
            should_print_semantic_table = True

    if should_print_semantic_table and rank == 0:
        print("\n[SEMANTIC] [pcd_contain_object1]")
        print(f"  step              = {step}")
        print(f"  valid_object_name = {valid_object_name}")
        print(f"  valid_object_id   = {valid_object_id}")
        print(f"  excluded_names    = {excluded_object_names}")
        print(f"  excluded_ids      = {excluded_object_ids}")
        print("  idToLabels:")
        for line in semantic_class_names:
            print("    ", line)
        print("")

        env._debug_sem_last_print_step = step
        env._debug_sem_last_valid_object_id = valid_object_id

    # ------------- 目标物体 ID 缺失时的容错逻辑 -------------
    if valid_object_id is None:
        # 把当前 object_pool 名字也打印出来（只在 rank 0 打）
        object_pool_names = []
        scene_cfg = getattr(env, "cfg", None)
        if scene_cfg is not None and hasattr(scene_cfg.scene, "object_pool"):
            try:
                object_pool_names = list(scene_cfg.scene.object_pool.rigid_objects.keys())
            except Exception:
                object_pool_names = []

        if rank == 0:
            print("[WARN] [pcd_contain_object1] Target object name NOT found in semantic segmentation.")
            print(f"  step = {step}")
            print(f"  valid_object_name = {valid_object_name}")
            print(f"  Object pool names (from cfg.scene.object_pool): {object_pool_names}")
            print("  -> 目前还找不到目标物体语义 ID，先返回 0 reward。\n")

        # 前 warmup_steps 步：认为语义系统还在“预热”，不报错，只返回 0
        if step < warmup_steps:
            return torch.zeros(env.num_envs, device=env.device)

        # 如果已经超过 warmup_steps 还找不到 eye_drops，说明很可能是真出问题了，抛异常
        raise RuntimeError(
            f"Target object '{valid_object_name}' not found in idToLabels after {warmup_steps} steps. "
            "Check that object_pool names and semantic class names match."
        )

    # ------------- 1. 接触力优先：Y 超阈值就直接给满分 -------------
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    contact_force_satisfied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if left_sensor.data.net_forces_w is not None and right_sensor.data.net_forces_w is not None:
        left_y_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
        right_y_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])

        left_contact = left_y_force > contact_force_threshold
        right_contact = right_y_force > contact_force_threshold

        contact_force_satisfied = (
            left_contact & right_contact if require_both_contacts else left_contact | right_contact
        )

    max_reward = density_scale if not use_tanh else torch.ones(env.num_envs, device=env.device)

    # ------------- 2. 点云与几何条件的公共部分封装成小函数 -------------
    def compute_density_reward() -> torch.Tensor:
        # 1) 同时取点云 + 点级语义 ID
        pointcloud, semantic_ids, valid = get_cached_pointcloud_with_semantics(env)
        if not valid or pointcloud is None or semantic_ids is None:
            return torch.zeros(env.num_envs, device=env.device)

        B, N, _ = pointcloud.shape

        # ---- 处理 semantic_ids 形状，保证是 (B, N) ----
        semantic_ids = semantic_ids.to(env.device)
        if semantic_ids.ndim == 3 and semantic_ids.shape[-1] == 1:
            semantic_ids = semantic_ids[..., 0]                 # (B,N,1) -> (B,N)
        elif semantic_ids.ndim != 2:
            raise RuntimeError(
                f"[pcd_contain_object1] semantic_ids must be (B,N) or (B,N,1), "
                f"got {semantic_ids.shape}"
            )
        if semantic_ids.shape != (B, N):
            raise RuntimeError(
                f"[pcd_contain_object1] semantic_ids shape {semantic_ids.shape} "
                f"!= pointcloud (B,N) {(B, N)}"
            )

        # ---- dtype 对齐 ----
        if torch.is_floating_point(semantic_ids):
            valid_id_tensor = torch.as_tensor(float(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
            excluded_ids_tensor = (
                torch.as_tensor([float(x) for x in excluded_object_ids], device=env.device, dtype=semantic_ids.dtype)
                if excluded_object_ids else None
            )
        else:
            valid_id_tensor = torch.as_tensor(int(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
            excluded_ids_tensor = (
                torch.as_tensor(excluded_object_ids, device=env.device, dtype=semantic_ids.dtype)
                if excluded_object_ids else None
            )

        # 3) 点级语义 mask
        is_target = (semantic_ids == valid_id_tensor)          # (B,N)
        if excluded_ids_tensor is not None and excluded_ids_tensor.numel() > 0:
            is_excluded = torch.isin(semantic_ids, excluded_ids_tensor)
        else:
            is_excluded = torch.zeros_like(semantic_ids, dtype=torch.bool)

        point_semantic_mask = is_target & (~is_excluded)       # (B,N)

        # 4) 指尖位姿 -> 相机坐标，计算球心和半径
        left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
        right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

        left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
        right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)

        sphere_center = (left_finger_cam + right_finger_cam) / 2.0   # (B,3)
        finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
        sphere_radius = torch.clamp(finger_distance * 0.5, min=0.003)  # (B,)

        # 5) 用点级 mask 计算密度
        density, num_points = calculate_pointcloud_density_in_sphere1(
            pointcloud, sphere_center, sphere_radius, mask=point_semantic_mask
        )

        # 6) tanh / 线性 + 几何约束
        if use_tanh:
            density_reward = torch.tanh(density * density_scale * 3.0)
        else:
            density_reward = density * density_scale

        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        ee_w = ee_frame.data.target_pos_w[..., 0, :]
        robot = env.scene[robot_cfg.name]
        robot_base_pos = robot.data.root_pos_w
        ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
        distance_mask = ee_robot_distance >= min_ee_robot_distance

        left_z = left_sensor.data.net_forces_w[:, 0, 2]
        right_z = right_sensor.data.net_forces_w[:, 0, 2]
        contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

        all_conditions_met = distance_mask & contact_z_valid
        return torch.where(all_conditions_met, density_reward, torch.zeros_like(density_reward))

    # ------------- 总 reward 组合 -------------
    if contact_force_satisfied.any():
        reward_if_contact = (
            max_reward
            if isinstance(max_reward, torch.Tensor)
            else torch.full((env.num_envs,), max_reward, device=env.device)
        )
        density_reward = compute_density_reward()
        reward = torch.where(contact_force_satisfied, reward_if_contact, density_reward)
        return reward

    return compute_density_reward()


def debug_semantic_pcd_density(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    valid_object_name: str = "eye_drops",
    excluded_object_names: list = ["m6_1_leftfinger_link", "m6_2_rightfinger_link", "m5_wrist_link"],
    log_interval: int = 50,      # 每多少步打印一次
    env_id_to_print: int = 0,    # 打印第几个 env 的详细信息

    # 下面这些参数用来复现 pcd_contain_object1 的 reward 逻辑
    density_scale: float = 1.0,
    use_tanh: bool = True,
    min_ee_robot_distance: float = 0.26,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    contact_z_threshold: float = 0.7,
    contact_force_threshold: float = 1.5,
    require_both_contacts: bool = True,
) -> torch.Tensor:
    """
    调试函数：对齐「点云 + 点级语义」，并打印球体内三种情况：

    - ALL    : 不筛选，球内所有点
    - EX     : 只看“排除类”（比如手指 / 手腕）
    - TARGET : 只看目标物体（eye_drops）

    同时打印：
    - 每步总点数 N
    - 三种 mask 对应的“球内点数”和“密度”
    - 用于 reward 的有效 mask（TARGET 去掉 EX 后）的密度
    - 与 pcd_contain_object1 一致的几何 / 接触条件
    - 最终 contain_object1 的 reward 数值
    """
    step = getattr(env, "common_step_counter", 0)
    rank = getattr(env, "rank", 0)

    # 不到指定步长就直接返回
    if step % log_interval != 0:
        return torch.zeros(env.num_envs, device=env.device)

    # 1. 语义表 idToLabels
    sensor = env.scene.sensors[sensor_cfg_name]
    id_to_labels = sensor.data.info["semantic_segmentation"]["idToLabels"]

    # 解析出：目标 id、排除类 id、打印用的语义表字符串
    valid_object_id, excluded_object_ids, semantic_class_lines = \
        _parse_semantic_ids_from_labels(id_to_labels, valid_object_name, excluded_object_names)

    if rank == 0:
        print("\n========== [点云+语义调试 debug_semantic_pcd_density] ==========")
        print(f"当前步数 step = {step}")
        print(f"目标类别名称 = '{valid_object_name}', 对应语义 id = {valid_object_id}")
        print(f"排除的类别名称列表 = {excluded_object_names}")
        print(f"排除的语义 id 列表   = {excluded_object_ids}")
        print("当前语义 idToLabels 映射：")
        for line in semantic_class_lines:
            print("  ", line)
        print("==============================================================\n")

    if valid_object_id is None:
        # 语义表里还没有 eye_drops
        if rank == 0:
            print(f"[警告] 当前语义表中还没有目标类别 '{valid_object_name}'，本步点云奖励记为 0。（step={step}）\n")
        return torch.zeros(env.num_envs, device=env.device)

    # 2. 从缓存拿点云 + 点级语义 id
    pointcloud, semantic_ids, valid = get_cached_pointcloud_with_semantics(env)
    if (not valid) or pointcloud is None or semantic_ids is None:
        if rank == 0:
            print(f"[debug_semantic_pcd_density] step={step}：pointcloud_with_semantics 无效，本步奖励记为 0。")
        return torch.zeros(env.num_envs, device=env.device)

    B, N, _ = pointcloud.shape
    semantic_ids = semantic_ids.to(env.device)

    # 保证 semantic_ids 形状是 (B, N)
    if semantic_ids.ndim == 3 and semantic_ids.shape[-1] == 1:
        semantic_ids = semantic_ids[..., 0]        # (B,N,1) -> (B,N)
    elif semantic_ids.ndim != 2:
        raise RuntimeError(
            f"[debug_semantic_pcd_density] semantic_ids 必须是 (B,N) 或 (B,N,1)，"
            f"当前形状为 {semantic_ids.shape}"
        )
    if semantic_ids.shape != (B, N):
        raise RuntimeError(
            f"[debug_semantic_pcd_density] semantic_ids 形状 {semantic_ids.shape} "
            f"与 pointcloud (B,N) {(B, N)} 不一致"
        )

    # 3. 计算与 pcd_contain_object1 一致的“夹爪中点球体”
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)

    sphere_center = (left_finger_cam + right_finger_cam) / 2.0   # (B,3)
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.3, min=0.003)  # (B,)

    # 4. 构造三种 mask：ALL / EX / TARGET
    #    以及一个“用于 reward 的有效 mask”（TARGET 去掉 EX）
    if torch.is_floating_point(semantic_ids):
        valid_id_tensor = torch.as_tensor(float(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
        excluded_ids_tensor = (
            torch.as_tensor([float(x) for x in excluded_object_ids], device=env.device, dtype=semantic_ids.dtype)
            if excluded_object_ids else None
        )
    else:
        valid_id_tensor = torch.as_tensor(int(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
        excluded_ids_tensor = (
            torch.as_tensor(excluded_object_ids, device=env.device, dtype=semantic_ids.dtype)
            if excluded_object_ids else None
        )

    # ALL：不筛选
    mask_all = torch.ones((B, N), dtype=torch.bool, device=env.device)

    # EX：只看排除的类别（手指 / 手腕）
    if excluded_ids_tensor is not None and excluded_ids_tensor.numel() > 0:
        mask_ex = torch.isin(semantic_ids, excluded_ids_tensor)
    else:
        mask_ex = torch.zeros_like(semantic_ids, dtype=torch.bool)

    # TARGET：所有 eye_drops 点
    mask_target = (semantic_ids == valid_id_tensor)

    # 用于 reward 的“有效 mask”：eye_drops 且不是排除类
    mask_target_clean = mask_target & (~mask_ex)

    # 5. 计算四种情况的“球内点数”和“密度”
    density_all, num_all = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_all
    )
    density_ex, num_ex = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_ex
    )
    density_target, num_target = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_target
    )
    density_target_clean, num_target_clean = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_target_clean
    )

    # 6. 计算与 pcd_contain_object1 完全一致的几何 / 接触条件
    #    以及最终 reward 数值（只打印，不参与梯度）
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]

    # 预先定义，防止没有数据时报变量未定义
    left_y_force = torch.zeros(env.num_envs, device=env.device)
    right_y_force = torch.zeros(env.num_envs, device=env.device)
    left_z = torch.zeros(env.num_envs, device=env.device)
    right_z = torch.zeros(env.num_envs, device=env.device)

    contact_force_satisfied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contact_z_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if left_sensor.data.net_forces_w is not None and right_sensor.data.net_forces_w is not None:
        left_y_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 1])
        right_y_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 1])
        left_contact = left_y_force > contact_force_threshold
        right_contact = right_y_force > contact_force_threshold
        contact_force_satisfied = (
            left_contact & right_contact if require_both_contacts else left_contact | right_contact
        )

        left_z = left_sensor.data.net_forces_w[:, 0, 2]
        right_z = right_sensor.data.net_forces_w[:, 0, 2]
        contact_z_valid = (left_z < contact_z_threshold) & (right_z < contact_z_threshold)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    robot = env.scene[robot_cfg.name]
    robot_base_pos = robot.data.root_pos_w
    ee_robot_distance = torch.norm(ee_w - robot_base_pos, dim=1)
    distance_mask = ee_robot_distance >= min_ee_robot_distance

    # 用真正的 pcd_contain_object1 算一次 reward，用来对比
    with torch.no_grad():
        contain_reward_vec = pcd_contain_object1(
            env=env,
            sensor_cfg_name=sensor_cfg_name,
            density_scale=density_scale,
            use_tanh=use_tanh,
            min_ee_robot_distance=min_ee_robot_distance,
            ee_frame_cfg=ee_frame_cfg,
            robot_cfg=robot_cfg,
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            contact_z_threshold=contact_z_threshold,
            contact_force_threshold=contact_force_threshold,
            require_both_contacts=require_both_contacts,
            valid_object_name=valid_object_name,
            excluded_object_names=excluded_object_names,
        )

    # 7. 只打印一个 env 的详细信息
    if rank == 0:
        eid = min(max(env_id_to_print, 0), B - 1)
        print("\n========== [debug_semantic_pcd_density - 详细] ==========")
        print(f"步数 step = {step} | 打印的 env 索引 = {eid}")
        print(f"每帧总点数 N = {N}")
        print(f"目标类别 = '{valid_object_name}' (id = {valid_object_id})")
        print(f"排除的类别名称 = {excluded_object_names}")
        print(f"排除的类别 id = {excluded_object_ids}")

        print("\n—— 球体信息 ——")
        print(f"该 env 球心（相机坐标） = {sphere_center[eid].tolist()}")
        print(f"该 env 球半径 r = {sphere_radius[eid].item():.6f}")

        print("\n—— 球内点云统计（只看 env {}）——".format(eid))
        print(f"ALL：      球内点数 = {int(num_all[eid].item())}，密度 = {density_all[eid].item():.6f}")
        print(f"EX：       球内点数 = {int(num_ex[eid].item())}，密度 = {density_ex[eid].item():.6f}  （只包含被排除的类别，例如手指/手腕）")
        print(f"TARGET：   球内点数 = {int(num_target[eid].item())}，密度 = {density_target[eid].item():.6f}  （所有 eye_drops 点）")
        print(f"TARGET*：  球内点数 = {int(num_target_clean[eid].item())}，密度 = {density_target_clean[eid].item():.6f}  （eye_drops 且排除 EX，用于 reward）")

        print("\n—— 几何 / 接触条件（与 pcd_contain_object1 完全一致）——")
        print(f"末端-底座距离 = {ee_robot_distance[eid].item():.4f}，是否满足 >= {min_ee_robot_distance} ：{bool(distance_mask[eid].item())}")
        if left_sensor.data.net_forces_w is not None:
            print(f"左指 Y 向力 = {left_y_force[eid].item():.4f}，右指 Y 向力 = {right_y_force[eid].item():.4f}")
            print(f"接触力是否超过阈值 {contact_force_threshold} ：{bool(contact_force_satisfied[eid].item())}")
            print(f"左指 Z 向力 = {left_z[eid].item():.4f}，右指 Z 向力 = {right_z[eid].item():.4f}")
            print(f"Z 向接触条件是否满足（< {contact_z_threshold}） ：{bool(contact_z_valid[eid].item())}")
        else:
            print("当前步没有有效的接触力数据。")

        print("\n—— 最终 contain_object1 reward ——")
        print(f"该 env 的 contain_object1 reward = {contain_reward_vec[eid].item():.6f}")
        print("（注意：reward 实际上使用的是 TARGET* 对应的密度 + 上面的几何/接触条件）")
        print("===========================================================\n")

    # 纯调试函数，不返回任何实际奖励
    return torch.zeros(env.num_envs, device=env.device)


def debug_semantic_pcd_density1(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    valid_object_name: str = "eye_drops",
    excluded_object_names: list = ["m6_1_leftfinger_link", "m6_2_rightfinger_link", "m5_wrist_link"],
    log_interval: int = 1,      # 每多少步打印一次
    env_id_to_print: int = 0,    # 打印第几个 env 的详细信息
) -> torch.Tensor:
    """
    调试函数：对齐「点云 + 点级语义」，并打印球体内
    - ALL  所有点
    - TARGET 只 eye_drops
    - TARGET-EX eye_drops 且排除 m5/m6

    用法：在 RewardsCfg 里加一个权重为 0 的项，不影响训练，只打印信息：
        debug_semantic_pcd = RewTerm(
            func=mdp.debug_semantic_pcd_density,
            params={},
            weight=0.0,
        )
    """
    step = getattr(env, "common_step_counter", 0)
    rank = getattr(env, "rank", 0)

    if step % log_interval != 0:
        return torch.zeros(env.num_envs, device=env.device)

    # 1. 拿语义表 idToLabels
    sensor = env.scene.sensors[sensor_cfg_name]
    id_to_labels = sensor.data.info["semantic_segmentation"]["idToLabels"]

    # ---- 解析出 valid_object_id 和 excluded_object_ids，并返回语义表文本 ----
    valid_object_id, excluded_object_ids, semantic_class_lines = \
        _parse_semantic_ids_from_labels(id_to_labels, valid_object_name, excluded_object_names)

    if rank == 0:
        print("\n========== [debug_semantic_pcd_density] ==========")
        print(f"Step: {step}")
        print(f"valid_object_name : {valid_object_name}")
        print(f"valid_object_id   : {valid_object_id}")
        print(f"excluded_names    : {excluded_object_names}")
        print(f"excluded_ids      : {excluded_object_ids}")
        print("idToLabels:")
        for line in semantic_class_lines:
            print("  ", line)
        print("===================================================\n")

    if valid_object_id is None:
        # 语义表里还没有 eye_drops，说明物体可能还没 spawn / 语义还没更新
        if rank == 0:
            print(f"[debug_semantic_pcd_density] WARNING: '{valid_object_name}' not in idToLabels yet, "
                  f"return 0 (step={step}).\n")
        return torch.zeros(env.num_envs, device=env.device)

    # 2. 从缓存里拿点云 + 点级语义 ID
    pointcloud, semantic_ids, valid = get_cached_pointcloud_with_semantics(env)
    if (not valid) or pointcloud is None or semantic_ids is None:
        if rank == 0:
            print(f"[debug_semantic_pcd_density] step={step}: pointcloud_with_semantics invalid.")
        return torch.zeros(env.num_envs, device=env.device)

    B, N, _ = pointcloud.shape
    semantic_ids = semantic_ids.to(env.device)

    # 保证 semantic_ids 是 (B, N)
    if semantic_ids.ndim == 3 and semantic_ids.shape[-1] == 1:
        semantic_ids = semantic_ids[..., 0]        # (B,N,1) -> (B,N)
    elif semantic_ids.ndim != 2:
        raise RuntimeError(
            f"[debug_semantic_pcd_density] semantic_ids must be (B,N) or (B,N,1), "
            f"got {semantic_ids.shape}"
        )
    if semantic_ids.shape != (B, N):
        raise RuntimeError(
            f"[debug_semantic_pcd_density] semantic_ids shape {semantic_ids.shape} "
            f"!= pointcloud (B,N) {(B, N)}"
        )

    # 3. 计算与 pcd_contain_object1 相同的「夹爪中点球体」
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)

    sphere_center = (left_finger_cam + right_finger_cam) / 2.0   # (B,3)
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.4, min=0.003)  # (B,)

    # 4. 构造三种点级 mask
    # ---- dtype 对齐 ----
    if torch.is_floating_point(semantic_ids):
        valid_id_tensor = torch.as_tensor(float(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
        excluded_ids_tensor = (
            torch.as_tensor([float(x) for x in excluded_object_ids], device=env.device, dtype=semantic_ids.dtype)
            if excluded_object_ids else None
        )
    else:
        valid_id_tensor = torch.as_tensor(int(valid_object_id), device=env.device, dtype=semantic_ids.dtype)
        excluded_ids_tensor = (
            torch.as_tensor(excluded_object_ids, device=env.device, dtype=semantic_ids.dtype)
            if excluded_object_ids else None
        )

    # A: 所有点
    mask_all = torch.ones((B, N), dtype=torch.bool, device=env.device)
    # B: 只看目标物体
    mask_target = (semantic_ids == valid_id_tensor)
    # C: 目标物体 & 排除 finger / wrist 类
    if excluded_ids_tensor is not None and excluded_ids_tensor.numel() > 0:
        is_excluded = torch.isin(semantic_ids, excluded_ids_tensor)
    else:
        is_excluded = torch.zeros_like(semantic_ids, dtype=torch.bool)
    mask_target_ex = mask_target & (~is_excluded)

    # 5. 分别计算三种情况的密度 / 点数
    density_all, num_all = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_all
    )
    density_target, num_target = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_target
    )
    density_target_ex, num_target_ex = calculate_pointcloud_density_in_sphere1(
        pointcloud, sphere_center, sphere_radius, mask=mask_target_ex
    )

    # 6. 打印某个 env 的详细信息（默认 env 0）
    if rank == 0:
        eid = min(max(env_id_to_print, 0), B - 1)
        print("\n========== [debug_semantic_pcd_density - DETAIL] ==========")
        print(f"Step: {step} | Env: {eid}")
        print(f"Total points: N = {N}")
        # 点级语义分布
        unique_ids, counts = torch.unique(semantic_ids[eid], return_counts=True)
        print("--- Point-level semantic_ids distribution (env {}) ---".format(eid))
        for uid, cnt in zip(unique_ids[:20], counts[:20]):  # 只打印前 20 个
            uid_int = int(uid.item())
            cls_name = id_to_labels.get(str(uid_int), {}).get("class", "UNKNOWN")
            print(f"  id={uid_int:3d} | class={cls_name:25s} | count={int(cnt.item())}")
        print("--- Sphere stats (env {}) ---".format(eid))
        print(f"  ALL       : num={int(num_all[eid].item())} | density={density_all[eid].item():.4f}")
        print(f"  TARGET    : num={int(num_target[eid].item())} | density={density_target[eid].item():.4f}")
        print(f"  TARGET-EX : num={int(num_target_ex[eid].item())} | density={density_target_ex[eid].item():.4f}")
        print("  (TARGET 是只看 eye_drops; TARGET-EX 是 eye_drops 且排除 m5/m6 手指类之后)")
        print("===========================================================\n")

    # 纯调试，不给 reward
    return torch.zeros(env.num_envs, device=env.device)


def _parse_semantic_ids_from_labels(
    id_to_labels: dict,
    valid_object_name: str,
    excluded_object_names: list[str],
):
    """从 idToLabels 里解析：
    - valid_object_id: 目标物体的 id（int 或 None）
    - excluded_object_ids: 需要排除的若干 id 列表
    - semantic_class_lines: 用于打印的 ['id: class'] 文本列表
    """
    def _parse_id(k):
        try:
            return int(k)
        except (TypeError, ValueError):
            return None

    # 整张语义表作成字符串，方便打印
    semantic_class_lines = [f"{k}: {v.get('class', '')}" for k, v in id_to_labels.items()]

    # 目标物体 ID
    raw_valid_object_id = next(
        (k for k, obj in id_to_labels.items() if obj.get("class") == valid_object_name),
        None,
    )
    valid_object_id = _parse_id(raw_valid_object_id) if raw_valid_object_id is not None else None

    # 需要排除的 ID
    excluded_object_ids: list[int] = []
    for k, obj in id_to_labels.items():
        if obj.get("class") in excluded_object_names:
            pid = _parse_id(k)
            if pid is not None:
                excluded_object_ids.append(pid)

    return valid_object_id, excluded_object_ids, semantic_class_lines


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
                sphere_radius = torch.clamp(finger_distance * 0.23, min=0.003)

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
    sphere_radius = torch.clamp(finger_distance * 0.23, min=0.003)

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


def debug_pcd_density(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Debug version - prints density info."""

    if env.common_step_counter % 3 == 0:
        pointcloud, valid = get_cached_pointcloud(env)
        if valid and pointcloud is not None:
            # Get gripper positions
            left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
            right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

            # Transform to camera frame
            camera = env.scene.sensors["depth_camera"]

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.23, min=0.003)

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
    if env.common_step_counter % 1 == 0:
        pointcloud, valid = get_cached_pointcloud(env)
        if valid and pointcloud is not None:
            # Get gripper positions
            left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
            right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]
            active_pos_w, _ = get_active_object_states(env)

            # Transform to camera frame
            camera = env.scene.sensors["depth_camera"]

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            object_cam = transform_world_to_camera(active_pos_w, env, "depth_camera")
            
            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance *0.23, min=0.003)

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
