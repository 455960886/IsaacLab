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
) -> torch.Tensor:
    """
    Reward based on point cloud density in the gripper sphere.
    Encourages the robot to position grippers around the object.

    Args:
        env: Environment
        sensor_cfg_name: Name of the depth camera sensor
        density_scale: Scaling factor for reward sensitivity
        use_tanh: If True, apply tanh for smoother gradients

    Returns:
        Reward tensor (num_envs,)
    """
    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    # 1. Get cached point cloud
    pointcloud, valid = get_cached_pointcloud(env)
    if not valid or pointcloud is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 2. Get gripper positions in world frame
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    # 3. Get camera pose and transform grippers to camera frame
    camera = env.scene.sensors[sensor_cfg_name]
    camera_pos_w = camera.data.pos_w
    camera_quat_w = camera.data.quat_w_ros

    # Convert ROS quaternion (x,y,z,w) to IsaacLab convention (w,x,y,z)
    camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)

    # Transform gripper positions to camera frame
    left_finger_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)

    # 4. Calculate sphere parameters (matches create_gripper_pointclouds logic)
    sphere_center = (left_finger_cam + right_finger_cam) / 2.0  # (B, 3)
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)  # (B,)
    sphere_radius = torch.clamp(finger_distance * 0.3, min=0.005)  # (B,)

    # 5. Calculate point cloud density in sphere
    density, num_points = calculate_pointcloud_density_in_sphere(
        pointcloud, sphere_center, sphere_radius
    )

    # 6. Convert density to reward
    if use_tanh:
        # Smooth saturation - good for stable gradients
        reward = torch.tanh(density * density_scale * 3.0)
    else:
        # Linear scaling - direct proportional to density
        reward = density * density_scale

    return reward



def pcd_clamp_object(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "depth_camera",
    density_threshold: float = 0.09,
    density_scale: float = 1.0,
    gripper_closed_threshold: float = 0.02,
) -> torch.Tensor:

    from .gripper_transform import transform_world_to_camera, calculate_pointcloud_density_in_sphere

    # 1. Get cached point cloud
    pointcloud, valid = get_cached_pointcloud(env)
    if not valid or pointcloud is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 2. Check if gripper is closed
    joint_positions = env.scene['robot'].data.joint_pos_target
    finger_joint_1 = joint_positions[:, 5]  # M6_1
    finger_joint_2 = joint_positions[:, 6]  # M6_2

    # Both fingers must be closed (near 0.0)
    gripper_closed = (torch.abs(finger_joint_1) < gripper_closed_threshold) & \
                     (torch.abs(finger_joint_2) < gripper_closed_threshold)

    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    camera = env.scene.sensors[sensor_cfg_name]
    camera_pos_w = camera.data.pos_w
    camera_quat_w = camera.data.quat_w_ros
    camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)

    left_finger_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)

    sphere_center = (left_finger_cam + right_finger_cam) / 2.0
    finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
    sphere_radius = torch.clamp(finger_distance * 0.3, min=0.005)

    density, num_points = calculate_pointcloud_density_in_sphere(
        pointcloud, sphere_center, sphere_radius
    )

    object_detected = density > density_threshold

    both_conditions_met = gripper_closed & object_detected

    reward = torch.where(
        both_conditions_met,
        density_scale,  # Higher density = higher reward
        torch.zeros_like(density) 
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

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)

            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.3, min=0.005)

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

            left_finger_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
            right_finger_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
            object_cam = transform_world_to_camera(active_pos_w, camera_pos_w, camera_quat_w_isaac)

            # Calculate sphere
            sphere_center = (left_finger_cam + right_finger_cam) / 2.0
            finger_distance = torch.norm(left_finger_cam - right_finger_cam, dim=-1)
            sphere_radius = torch.clamp(finger_distance * 0.3, min=0.005)

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
