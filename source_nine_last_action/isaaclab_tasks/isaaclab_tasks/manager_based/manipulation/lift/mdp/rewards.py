# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import math

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


## Help Function to get the states of active object from the object pool
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


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")):
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
    object_ee_distance = torch.norm(active_pos_w - ee_w, dim=1)
    # print(object_ee_distance)
    
    # joint_pos = env.scene["robot"].data.joint_pos  # (envs, joints)
    # gripper_status = torch.abs(joint_pos[:, -1])
    # gripper_open = gripper_status >0.4
    # mask = 0.2 + 0.8*gripper_open
    # # print(object_ee_distance)
    # return (1 - torch.tanh(object_ee_distance/std)) *mask
    
    return 1 - torch.tanh(object_ee_distance/std)


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


def get_excluded_object_mask(
    env: ManagerBasedRLEnv,
    excluded_objects: list[str],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """
    Get a boolean mask indicating which environments have excluded objects active.

    Args:
        env: Environment
        excluded_objects: List of object name patterns to exclude (e.g., ["slipper", "shoe"])
                         Uses substring matching - "slipper" matches "slippers", "slippers_m5_0", etc.
        object_cfg: Object pool configuration

    Returns:
        Boolean tensor (num_envs,) - True for environments with excluded objects
    """
    if not excluded_objects:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Get object pool and names
    object_collection = env.scene[object_cfg.name]
    object_names = object_collection.object_names  # List of object names

    # Get active object indices
    active_indices = env.active_object_indices  # (num_envs,)

    # Build exclusion mask
    excluded_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for env_idx in range(env.num_envs):
        obj_idx = active_indices[env_idx].item()
        obj_name = object_names[obj_idx]

        # Check if object name matches any exclusion pattern (substring match)
        for pattern in excluded_objects:
            if pattern.lower() in obj_name.lower():
                excluded_mask[env_idx] = True
                break

    return excluded_mask


def get_active_object_name_mask(
    env: ManagerBasedRLEnv,
    object_names: list[str] | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return a mask for environments whose active object matches any exact name in ``object_names``."""
    if not object_names:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_collection = env.scene[object_cfg.name]
    active_indices = env.active_object_indices

    matched_ids = [
        object_collection.object_names.index(name)
        for name in object_names
        if name in object_collection.object_names
    ]
    if not matched_ids:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    matched_ids_tensor = torch.tensor(matched_ids, dtype=torch.long, device=env.device)
    return (active_indices[:, None] == matched_ids_tensor[None, :]).any(dim=1)


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
    excluded_objects: list[str] | None = None,
    max_sphere_radius: float = 0.015,
    sphere_z_offset: float = 0.01,
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
        excluded_objects: List of object name patterns to exclude (e.g., ["slipper", "shoe"]).
                         Uses substring matching. Returns zero reward for these objects.

    Returns:
        Reward tensor (num_envs,)
    """
    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    # Check for excluded objects - return zero for those environments
    if excluded_objects:
        excluded_mask = get_excluded_object_mask(env, excluded_objects)
        if excluded_mask.all():
            # All environments have excluded objects
            return torch.zeros(env.num_envs, device=env.device)

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
                sphere_center[:, 2] -= sphere_z_offset
                finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
                sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003, max=max_sphere_radius)

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

        # Zero out reward for excluded objects
        if excluded_objects:
            excluded_mask = get_excluded_object_mask(env, excluded_objects)
            reward = torch.where(excluded_mask, torch.zeros_like(reward), reward)

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
    sphere_center[:, 2] -= sphere_z_offset
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003, max=max_sphere_radius)

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

    # Zero out reward for excluded objects
    if excluded_objects:
        excluded_mask = get_excluded_object_mask(env, excluded_objects)
        reward = torch.where(excluded_mask, torch.zeros_like(reward), reward)

    return reward


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

    # Y-axis forces (raw values)
    left_y_raw = left_sensor.data.net_forces_w[:, 0, 1]
    right_y_raw = right_sensor.data.net_forces_w[:, 0, 1]
    
    # Check opposite signs (clamping from opposite directions)
    opposite_forces = (left_y_raw * right_y_raw) < 0

    # Average force magnitude check
    left_y = torch.abs(left_y_raw)
    right_y = torch.abs(right_y_raw)
    avg_force = (left_y + right_y) / 2.0
    good_grasp = avg_force > contact_force_threshold

    # Binary reward - all conditions must be met
    reward = torch.where(
        gripper_closed & good_grasp & opposite_forces,
        torch.ones(env.num_envs, device=env.device) * reward_value,
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

            # Transform to camera frame
            camera = env.scene.sensors["depth_camera"]
            camera_pos_w = camera.data.pos_w
            camera_quat_w = camera.data.quat_w_ros
            camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003, max=max_sphere_radius)

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
    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere
    import os

    if env.common_step_counter % 1 == 0:
        pointcloud, valid = get_cached_pointcloud(env)
        if valid and pointcloud is not None:
            # Get gripper positions
            left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
            right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]
            active_pos_w, _ = get_active_object_states(env)

            # Transform to camera frame
            camera = env.scene.sensors["depth_camera"]
            camera_pos_w = camera.data.pos_w
            camera_quat_w = camera.data.quat_w_ros
            camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, "depth_camera")
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, "depth_camera")

            object_cam = transform_world_to_camera(active_pos_w, env, "depth_camera")
            
            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            sphere_center[:, 2] -= 0.01  # TODO: match sphere_z_offset from pcd_contain_object
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.22, min=0.003, max=0.015)

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


def base_orientation_penalty_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation using exponential kernel for better sensitivity.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    projected_gravity = asset.data.projected_gravity_b[:, :2]  # (num_envs, 2)

    tilt_magnitude = torch.sum(torch.square(projected_gravity), dim=1)  # (num_envs,)

    penalty = 1.0 - torch.exp(-tilt_magnitude / (std ** 2))

    return penalty


def penalize_m5_movement_after_alignment(
    env: ManagerBasedRLEnv,
    alignment_steps: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]

    # Get M5 joint index
    m5_idx = robot.joint_names.index("M5")

    # Get M5 joint velocity (absolute value)
    m5_velocity = torch.abs(robot.data.joint_vel[:, m5_idx])

    # Only penalize after alignment phase
    after_alignment = env.episode_length_buf > alignment_steps

    penalty = torch.where(
        after_alignment,
        m5_velocity,  # Penalize velocity after alignment
        torch.zeros_like(m5_velocity)  # No penalty during alignment phase
    )

    # print(f"M5 stability penalty: {penalty}")

    return penalty


def wrist_object_orientation_alignment(
    env: ManagerBasedRLEnv,
    std: float = 0.2,  # 越小越严格（单位：弧度），0.2rad≈11.5°
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peak_shift_object_names: list[str] | None = None,
    peak_shift_value: float = math.pi / 2,
    debug: bool = False,
) -> torch.Tensor:
    _, active_quat_w = get_active_object_states(env, object_cfg)

    # object yaw in world
    w, x, y, z = active_quat_w[:, 0], active_quat_w[:, 1], active_quat_w[:, 2], active_quat_w[:, 3]
    object_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    robot = env.scene[asset_cfg.name]

    m5_idx = robot.joint_names.index("M5")
    wrist_yaw = robot.data.joint_pos[:, m5_idx]

    # M5 only moves in [0, pi]. Use the active object yaw as the nominal wrist target,
    # then optionally shift the Gaussian peak for a configured subset of objects.
    target_wrist_yaw = -object_yaw

    raw_diff = wrist_yaw - target_wrist_yaw
    peak_shift = torch.where(
        get_active_object_name_mask(env, peak_shift_object_names, object_cfg),
        torch.full_like(raw_diff, peak_shift_value),
        torch.zeros_like(raw_diff),
    )

    slippers_mask = get_active_object_name_mask(env, ["slippers"], object_cfg)
    slippers_peak_shift = torch.where(
        target_wrist_yaw < 0,
        torch.full_like(raw_diff, math.pi),
        torch.zeros_like(raw_diff),
    )
    peak_shift = torch.where(slippers_mask, slippers_peak_shift, peak_shift)

    centered_diff = torch.atan2(torch.sin(raw_diff - peak_shift), torch.cos(raw_diff - peak_shift))  # wrap to [-pi, pi]
    angle_diff = torch.abs(centered_diff)

    # 高斯奖励：默认在 diff=0 处峰值为 1；特殊物体可通过 peak_shift_value 将峰值平移。
    reward = torch.exp(-0.5 * (angle_diff / std) ** 2)

    if debug:
        peak_raw_diff = peak_shift
        peak_raw_diff_deg = peak_raw_diff * (180.0 / math.pi)
        print(
            f"[腕部朝向奖励调试]\n"
            f"   target_wrist_yaw = {target_wrist_yaw}\n"
            f"   wrist_yaw        = {wrist_yaw}\n"
            f"   raw_diff         = {raw_diff}\n"
            f"   peak_shift       = {peak_raw_diff} rad ({peak_raw_diff_deg} deg)\n",
            flush=True,
        )

    return reward


def debug_m5_episode_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Debug helper: print per-episode M5 min/max and runtime limits."""
    robot = env.scene[asset_cfg.name]
    m5_idx = robot.joint_names.index("M5")
    m5 = robot.data.joint_pos[:, m5_idx]

    min_attr = "_debug_m5_ep_min"
    max_attr = "_debug_m5_ep_max"

    ep_min = getattr(env, min_attr, None)
    ep_max = getattr(env, max_attr, None)

    need_init = (
        (not torch.is_tensor(ep_min))
        or (not torch.is_tensor(ep_max))
        or (ep_min.shape != (env.num_envs,))
        or (ep_max.shape != (env.num_envs,))
        or (ep_min.device != env.device)
        or (ep_max.device != env.device)
    )

    if need_init:
        ep_min = m5.clone()
        ep_max = m5.clone()
    else:
        first_step_mask = env.episode_length_buf <= 1
        ep_min = torch.where(first_step_mask, m5, torch.minimum(ep_min, m5))
        ep_max = torch.where(first_step_mask, m5, torch.maximum(ep_max, m5))

    reset_mask = getattr(env, "reset_buf", None)
    if torch.is_tensor(reset_mask) and reset_mask.any():
        soft_limits = robot.data.soft_joint_pos_limits[:, m5_idx, :]
        done_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            m5_min = ep_min[env_id].item()
            m5_max = ep_max[env_id].item()
            low = soft_limits[env_id, 0].item()
            high = soft_limits[env_id, 1].item()
            ep_len = int(env.episode_length_buf[env_id].item())
            print(
                f"[M5 DEBUG] env={env_id} ep_len={ep_len} "
                f"min={m5_min:.4f} rad ({m5_min * 57.2958:.1f} deg) "
            )

        ep_min = ep_min.clone()
        ep_max = ep_max.clone()
        ep_min[done_ids] = m5[done_ids]
        ep_max[done_ids] = m5[done_ids]

    setattr(env, min_attr, ep_min)
    setattr(env, max_attr, ep_max)
    return torch.zeros(env.num_envs, device=env.device)
