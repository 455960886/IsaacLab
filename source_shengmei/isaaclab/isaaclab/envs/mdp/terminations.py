# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle



def object_pushed_away(
    env: ManagerBasedRLEnv, 
    x_limits: tuple[float, float] = (0.22, 0.42),
    y_tolerance: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Terminate when the active object is pushed too far from its spawn area.
    Works with RigidObjectCollection (object pools).

    Object base position relative to robot: (0.28, 0.0, 0.0)
    Randomization: x(-0.01, 0.09), y(0.0, 0.0), z(0.0, 0.0)
    """
    from isaaclab.assets import RigidObjectCollection, Articulation

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Get active object indices
    if not hasattr(env, 'active_object_indices'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    active_indices = env.active_object_indices

    # Get positions: (num_envs, 3)
    all_positions = object_collection.data.object_link_pos_w
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_positions_w = all_positions[env_indices, active_indices]

    robot_base_pos_w = robot.data.root_pos_w

    # Convert to robot frame
    active_positions_robot = active_positions_w - robot_base_pos_w

    # Check if outside legal range
    outside_x = (active_positions_robot[:, 0] < x_limits[0]) | (active_positions_robot[:, 0] > x_limits[1])
    outside_y = torch.abs(active_positions_robot[:, 1] - 0.0) > y_tolerance

    return outside_x | outside_y



def bad_object_orientation(
    env: ManagerBasedRLEnv, 
    limit_angle: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")
) -> torch.Tensor:
    """
    Terminate when the active object's orientation is too tilted.
    Works with RigidObjectCollection (object pools).
    """
    from isaaclab.assets import RigidObjectCollection

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]

    # Get active object indices
    if not hasattr(env, 'active_object_indices'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    active_indices = env.active_object_indices

    # Get projected gravity for all objects: (num_envs, num_objects, 3)
    all_projected_gravity = object_collection.data.projected_gravity_b

    # Index to get only active objects: (num_envs, 3)
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_projected_gravity = all_projected_gravity[env_indices, active_indices]

    # Calculate tilt angle
    tilt_angle = torch.acos(-active_projected_gravity[:, 2].clamp(-1.0, 1.0)).abs()

    # Terminate if angle exceeds limit
    return tilt_angle > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    return (asset.data.root_pos_w[:, 2] < minimum_height) | (ee_frame.data.target_pos_w[..., 0, 2] < -0.01)


"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


def object_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # with open('output_formres5.txt', 'a') as f:
    #     f.write(f"dis: {torch.mean(object_ee_distance).item()}\n")
        
    # print("reaching: ",torch.mean(1 - torch.tanh(object_ee_distance / std)))
    return torch.any(
        object.data.root_pos_w[:, 2] > 0.1
    )  # Returns True if the distance is less than 0.05 meters


# def gripper_floor_contact(
#     env: ManagerBasedRLEnv, 
#     threshold: float = 1.0, 
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
# ) -> torch.Tensor:
#     """Terminate when claw contacts filtered objects (ground/table) above threshold."""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # Get filtered contact forces (only contacts with ground/table)
#     # This is None if no filtering is configured
#     if contact_sensor.data.force_matrix_w is None:
#         return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

#     # Shape: (num_envs, num_bodies, num_filtered_objects, 3)
#     filtered_forces = contact_sensor.data.force_matrix_w

#     # Check if any filtered contact exceeds threshold
#     contact_magnitudes = torch.norm(filtered_forces, dim=-1)  # (N, B, M)
#     max_contact_force = torch.max(contact_magnitudes.view(env.num_envs, -1), dim=1)[0]

#     return max_contact_force > threshold


def gripper_floor_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Terminate when claw contacts ground (using unfiltered contacts)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Use unfiltered contact forces
    if contact_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Get all contact forces
    net_forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)  # Shape: (num_envs, num_bodies)
    max_contact_force = torch.max(net_forces, dim=1)[0]  # Max across all bodies

    # Simple heuristic: if contact force is high and object is not being lifted,
    # assume it's ground contact
    object: RigidObject = env.scene["object"]
    object_height = object.data.root_pos_w[:, 2]
    object_not_lifted = object_height < 0.025  # Object still on ground

    # Terminate if high contact force while object is still on ground
    violations = (max_contact_force > threshold) & object_not_lifted

    print(f"[DEBUG] Max contact: {max_contact_force.max().item():.4f}, Object heights: {object_height.cpu().numpy()}")
    print(f"[DEBUG] Violations: {violations.sum().item()}/{env.num_envs}")

    return violations


def gripper_z_force_limit(
    env: ManagerBasedRLEnv,
    z_threshold: float = 5.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    check_either: bool = True,
) -> torch.Tensor:

    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Extract Z-component (index 2) of contact forces
    left_z_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 2])
    right_z_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 2])

    # Check if forces exceed threshold
    left_exceeds = left_z_force > z_threshold
    right_exceeds = right_z_force > z_threshold

    if check_either:
        # Terminate if either finger exceeds threshold
        terminate = left_exceeds | right_exceeds 
        if terminate.any():
            print("termination due to z force limit")
    else:
        # Terminate only if both fingers exceed threshold
        terminate = left_exceeds & right_exceeds 
        if terminate.any():
            print("termination due to z force limit")

    return terminate


def hand_z_force_limit(
    env: ManagerBasedRLEnv,
    z_threshold: float = 50.0,
    middle_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_middle"),
) -> torch.Tensor:
    """Terminate when the middle contact sensor (wrist/hand) Z-force exceeds threshold.

    This prevents excessive downward force on the wrist link that could damage the robot
    or indicate the gripper is pushing too hard against a surface.

    Args:
        env: The RL environment.
        z_threshold: Maximum allowed Z-axis force (in Newtons). Default 50.0N.
        middle_sensor_cfg: Configuration for the middle/wrist contact sensor.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    middle_sensor: ContactSensor = env.scene.sensors[middle_sensor_cfg.name]

    if middle_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Extract Z-component (index 2) of contact forces
    middle_z_force = torch.abs(middle_sensor.data.net_forces_w[:, 0, 2])

    # Check if force exceeds threshold
    middle_exceeds = middle_z_force > z_threshold

    if middle_exceeds.any():
        print(f"[TERMINATION] Hand Z-force limit exceeded: max={middle_z_force.max().item():.2f}N (threshold={z_threshold}N)")

    return middle_exceeds


def base_orientation_out_of_limits(
    env: ManagerBasedRLEnv,
    threshold_rad: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot base orientation deviates too much from upright.

    This checks if the robot base has tilted beyond the specified threshold, which could
    indicate the robot has tipped over or is in an unstable configuration.

    Args:
        env: The environment.
        threshold_rad: Maximum allowed deviation from upright orientation in radians.
            Defaults to 0.5 rad (≈28.6°).
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Extract the asset (robot)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get current base orientation (quaternion: w, x, y, z)
    base_quat = asset.data.root_quat_w

    # Upright orientation quaternion (identity: w=1, x=0, y=0, z=0)
    upright_quat = torch.zeros_like(base_quat)
    upright_quat[:, 0] = 1.0  # w component

    # Compute quaternion difference
    # For small angles, we can use: angle ≈ 2 * arccos(|q1 · q2|)
    dot_product = torch.abs((base_quat * upright_quat).sum(dim=-1))
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability

    # Angle between quaternions (in radians)
    angle_diff = 2.0 * torch.acos(dot_product)

    # Check if angle exceeds threshold
    out_of_limits = angle_diff > threshold_rad

    # Log when termination occurs
    if out_of_limits.any():
        max_angle = angle_diff.max().item()
        print(f"[TERMINATION] Base orientation limit exceeded: max={max_angle:.3f} rad ({max_angle*57.3:.1f}°), threshold={threshold_rad:.3f} rad ({threshold_rad*57.3:.1f}°)")

    return out_of_limits
