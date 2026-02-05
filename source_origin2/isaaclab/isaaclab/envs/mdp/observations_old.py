# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import numpy as np
import cv2
import os
import time
# from source.isaaclab.isaaclab.pointnet.models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from isaaclab.pointnet.models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import importlib
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
import open3d as o3d

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

# torch.serialization.add_safe_globals([np.core.multiarray.scalar])

"""
Root state.
"""

def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs1 m: {(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).mean().item()}\n")
    #     f.write(f"obs1 s: {(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).std().item()}\n")
    # print("obs1 m: ",(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).mean().item())
    # print("obs1 s: ",(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).std().item())
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs2 m: {(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).mean().item()}\n")
    #     f.write(f"obs2 s: {(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).std().item()}\n")
    # print("obs2 m:",(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).mean().item())
    # print("obs2 s:",(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).std().item())
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    # cnt: int = 0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    depth_cfg : SceneEntityCfg = SceneEntityCfg("depth_camera"),
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]
    # print("depth shape:",depth.shape)
    # depth_np = depth.squeeze(0).squeeze(-1).cpu().numpy()  # shape [H, W]

    # # 归一化到 0~255
    # depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    # depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # os.makedirs("depth_images", exist_ok=True)
    # timestamp = time.time()
    # cv2.imwrite(f"depth_images/depth_{timestamp}.png", depth_uint8)
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)
    # obs_np = rgb_image_tensor.squeeze(0).cpu().numpy() 
    # # act_np = actions.cpu().numpy() 
    # # os.makedirs(act_log_dir, exist_ok=True)
    # # np.save(os.path.join(act_log_dir, f"act_step_{t}.npy"), act_np)
    # if obs_np.dtype == np.float32 or obs_np.max() <= 1.0:
    #     obs_np = (obs_np * 255).astype(np.uint8)

    # # RGB 转 BGR 再保存
    # obs_bgr = cv2.cvtColor(obs_np, cv2.COLOR_RGB2BGR)
    # os.makedirs("IMAGES1", exist_ok=True)
    # cv2.imwrite(f"./IMAGES1/observation_{cnt}.png", obs_bgr)
    # print(f"./IMAGES1/observation_{cnt}.png")
    # rgb/depth image normalization
    if normalize:
        # print(f"Normalizing images of type: {data_type}")
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
            
            # images = images.float()

            # obs_np2 = images.squeeze(0).cpu().numpy() 
            # # act_np = actions.cpu().numpy() 
            # # os.makedirs(act_log_dir, exist_ok=True)
            # # np.save(os.path.join(act_log_dir, f"act_step_{t}.npy"), act_np)
            # if obs_np2.dtype == np.float32 or obs_np2.max() <= 1.0:
            #     obs_np2 = (obs_np2 * 255).astype(np.uint8)

            # # RGB 转 BGR 再保存
            # obs_bgr2 = cv2.cvtColor(obs_np2, cv2.COLOR_RGB2BGR)
            # os.makedirs("IMAGES13", exist_ok=True)
            # cv2.imwrite(f"./IMAGES13/observation_{time.time()}.png", obs_np2)
            
            pass
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
    # print("image shape11:",images.shape)
    #深度图与RGB图拼接
    images = torch.cat((images,depth),dim=-1)
    # print("image shape22:",images.shape)
    return images.clone()

# # # 纯rgb版本
# class image_features(ManagerTermBase):
#     """RGB-only feature extraction from gripper camera using ResNet-style architecture."""

#     def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#         # Step counter
#         self.step_count = 0

#         # Debug
#         self.debug_mode = True
#         self.debug_interval = 50

#         # ResNet-style RGB encoder with skip connections
#         self.rgb_block1 = torch.nn.Sequential(
#             torch.nn.Linear(400 * 300 * 3, 512),
#             torch.nn.LayerNorm(512),
#             torch.nn.ReLU()
#         ).to(env.device)

#         self.rgb_block2 = torch.nn.Sequential(
#             torch.nn.Linear(512, 512),
#             torch.nn.LayerNorm(512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 512),
#             torch.nn.LayerNorm(512)
#         ).to(env.device)

#         self.rgb_block3 = torch.nn.Sequential(
#             torch.nn.Linear(512, 256),
#             torch.nn.LayerNorm(256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128)
#         ).to(env.device)

#     def reset(self, env_ids: torch.Tensor | None = None):
#         pass

#     def debug_tensor(self, name: str, tensor: torch.Tensor):
#         """Print tensor statistics for debugging."""
#         if not self.debug_mode or self.step_count % self.debug_interval != 0:
#             return

#         print(f"\n[{name}]")
#         print(f"  Shape: {tensor.shape}")
#         print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
#         print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

#         if torch.isnan(tensor).any():
#             print(f"  ⚠️ WARNING: Contains NaN!")
#         if torch.isinf(tensor).any():
#             print(f"  ⚠️ WARNING: Contains Inf!")
#         if tensor.std().item() > 10:
#             print(f"  ⚠️ WARNING: High variance!")
#         if tensor.abs().max().item() > 50:
#             print(f"  ⚠️ WARNING: Large values!")

#     def __call__(
#         self,
#         env: ManagerBasedEnv,
#         sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
#         data_type: str = "rgb",
#     ) -> torch.Tensor:

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"\n{'='*60}")
#             print(f"DEBUG RGB-ONLY: Step {self.step_count}")
#             print(f"{'='*60}")

#         # Get RGB sensor (gripper camera only)
#         rgb_sensor = env.scene.sensors[sensor_cfg.name]
#         rgb_data = rgb_sensor.data.output[data_type].float() / 255.0
#         batch_size = rgb_data.shape[0]
#         rgb_flat = rgb_data.reshape(batch_size, -1)

#         self.debug_tensor("RGB Input (normalized)", rgb_flat)

#         # ResNet-style processing with skip connection
#         x = self.rgb_block1(rgb_flat)  # [B, 512]
#         self.debug_tensor("RGB Block1", x)

#         residual = x
#         x = self.rgb_block2(x)  # [B, 512]
#         self.debug_tensor("RGB Block2 (before skip)", x)

#         x = torch.nn.functional.relu(x + residual)  # Skip connection
#         self.debug_tensor("RGB Block2 (after skip)", x)

#         rgb_features = self.rgb_block3(x)  # [B, 128]
#         self.debug_tensor("RGB Features (final)", rgb_features)

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"{'='*60}\n")

#         self.step_count += 1

#         return rgb_features

# # # 纯pcd版本
# class image_features(ManagerTermBase):
#     """PCD-only feature extraction from base camera using PointNet."""

#     def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#         # Camera intrinsics
#         self.fx, self.fy = 140.0, 140.0
#         self.cx, self.cy = 200.0, 150.0

#         # Downsampling parameters
#         self.num_points = 2048

#         # PCD saving
#         self.step_count = 0
#         self.save_interval = 100
#         self.save_pcd = True
#         self.pcd_save_dir = "./pcd_logs_only"
#         os.makedirs(self.pcd_save_dir, exist_ok=True)

#         # Debug
#         self.debug_mode = True
#         self.debug_interval = 50

#         # PointNet-style PCD encoder
#         self.point_mlp = torch.nn.Sequential(
#             torch.nn.Linear(3, 64),
#             torch.nn.LayerNorm(64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 256)
#         ).to(env.device)

#         self.global_mlp = torch.nn.Sequential(
#             torch.nn.Linear(256, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128)
#         ).to(env.device)

#     def reset(self, env_ids: torch.Tensor | None = None):
#         pass

#     def debug_tensor(self, name: str, tensor: torch.Tensor):
#         """Print tensor statistics for debugging."""
#         if not self.debug_mode or self.step_count % self.debug_interval != 0:
#             return

#         print(f"\n[{name}]")
#         print(f"  Shape: {tensor.shape}")
#         print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
#         print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

#         if torch.isnan(tensor).any():
#             print(f"  ⚠️ WARNING: Contains NaN!")
#         if torch.isinf(tensor).any():
#             print(f"  ⚠️ WARNING: Contains Inf!")

#     def voxel_downsample_gpu(self, points: torch.Tensor) -> torch.Tensor:
#         """Voxel downsampling (same as your existing method)."""
#         device = points.device
#         N = points.shape[0]

#         if N == 0:
#             return torch.zeros((self.num_points, 3), device=device)

#         depths = points[:, 2:3]
#         base_voxel_size = 0.001
#         power = 2.3
#         depth_scale = 3.3

#         adaptive_voxel_sizes = base_voxel_size * torch.pow(1 + depths * depth_scale, power)
#         adaptive_voxel_sizes = torch.clamp(adaptive_voxel_sizes, min=0.002, max=0.15)

#         normalized_points = points / adaptive_voxel_sizes
#         voxel_indices = torch.floor(normalized_points).to(torch.int64)

#         min_idx = voxel_indices.min(dim=0)[0]
#         shifted_indices = voxel_indices - min_idx

#         voxel_hash = (shifted_indices[:, 0] * 1000000 + 
#                     shifted_indices[:, 1] * 1000 + 
#                     shifted_indices[:, 2])

#         unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)

#         M = len(unique_hashes)
#         first_occurrence = torch.full((M,), N, dtype=torch.long, device=device)
#         first_occurrence.scatter_reduce_(0, inverse_indices, 
#                                         torch.arange(N, device=device), 
#                                         reduce='min')

#         down_points = points[first_occurrence]

#         current_size = down_points.shape[0]
#         if current_size >= self.num_points:
#             down_depths = down_points[:, 2]
#             weights = 1.0 / (down_depths + 0.1)
#             weights = weights / weights.sum()
#             indices = torch.multinomial(weights, self.num_points, replacement=False)
#             down_points = down_points[indices]
#         else:
#             extra_indices = torch.randint(0, current_size, (self.num_points - current_size,), device=device)
#             down_points = torch.cat([down_points, down_points[extra_indices]], dim=0)

#         return down_points

#     def save_point_cloud(self, points: torch.Tensor, filename: str):
#         """Save point cloud using Open3D."""
#         points_np = points.cpu().numpy()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_np)
#         o3d.io.write_point_cloud(filename, pcd)

#     def __call__(
#         self,
#         env: ManagerBasedEnv,
#         depth_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
#     ) -> torch.Tensor:

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"\n{'='*60}")
#             print(f"DEBUG PCD-ONLY: Step {self.step_count}")
#             print(f"{'='*60}")

#         # Get depth sensor (base camera only)
#         depth_sensor = env.scene.sensors[depth_cfg.name]
#         depth_data = depth_sensor.data.output["distance_to_image_plane"]
#         depth_data = depth_data.squeeze(-1)

#         batch_size = depth_data.shape[0]
#         H, W = depth_data.shape[1], depth_data.shape[2]

#         v, u = torch.meshgrid(
#             torch.arange(H, device=depth_data.device, dtype=torch.float32),
#             torch.arange(W, device=depth_data.device, dtype=torch.float32),
#             indexing='ij'
#         )

#         pcd_features_list = []
#         for i in range(batch_size):
#             depth = depth_data[i]

#             depth = depth.clone()
#             depth[~torch.isfinite(depth)] = 0.0
#             depth = depth.clamp(min=0.0)

#             valid_mask = (depth > 0) & (depth < 5.0)

#             if valid_mask.sum() > 0:
#                 Z = depth[valid_mask]
#                 X = (u[valid_mask] - self.cx) * Z / self.fx
#                 Y = (v[valid_mask] - self.cy) * Z / self.fy

#                 points_3d = torch.stack([X, -Y, Z], dim=-1)

#                 if i == 0:
#                     self.debug_tensor(f"Points3D (batch {i})", points_3d)

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     filename_orig = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_original.ply"
#                     )
#                     self.save_point_cloud(points_3d, filename_orig)

#                 downsampled_points = self.voxel_downsample_gpu(points_3d)

#                 if i == 0:
#                     self.debug_tensor(f"Downsampled Points (batch {i})", downsampled_points)

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     filename_down = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_downsampled.ply"
#                     )
#                     self.save_point_cloud(downsampled_points, filename_down)

#                 # PointNet encoding
#                 point_features = self.point_mlp(downsampled_points)  # [2048, 256]

#                 if i == 0:
#                     self.debug_tensor(f"Point Features (batch {i})", point_features)

#                 global_feature = torch.max(point_features, dim=0)[0]  # [256]

#                 if i == 0:
#                     self.debug_tensor(f"Global Feature (batch {i})", global_feature)

#                 pcd_feat = self.global_mlp(global_feature)  # [128]

#                 if i == 0:
#                     self.debug_tensor(f"PCD Features (batch {i})", pcd_feat)
#             else:
#                 pcd_feat = torch.zeros(128, device=depth.device)

#             pcd_features_list.append(pcd_feat)

#         pcd_features = torch.stack(pcd_features_list)  # [B, 128]
#         self.debug_tensor("Final PCD Features", pcd_features)

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"{'='*60}\n")

#         self.step_count += 1

#         return pcd_features

# 预训练模型版本 (速度慢, sim卡死)
class image_features(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]
        self._prepare_pointnet_model()
        self.fx, self.fy = 140.0, 140.0
        self.cx, self.cy = 200.5, 150.5


    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def depth_to_pointcloud(self,depth_image, fx, fy, cx, cy, rgb_image=None, output_path="pointcloud.ply"):
        assert len(depth_image.shape) == 2, "深度图必须是单通道 (H, W)"
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # 深度图中无效值置0（避免NaN）
        depth = np.nan_to_num(depth_image, nan=0.0)
        depth = (depth.max() - depth)

        mask = depth > 0  # 有效深度
        
        # 反投影到3D空间
        Z = depth[mask]
        X = (u[mask] - cx) * Z / fx
        Y = (v[mask] - cy) * Z / fy
        points = np.stack((X, -Y, Z), axis=-1)

        def voxel_down_sample_fixed(points, voxel_size=0.01, num_points=4096, seed=None):
            
            if len(points) == 0:
                raise ValueError("Input point cloud is empty!")

            if seed is not None:
                np.random.seed(seed)

            voxel_indices = np.floor(points / voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            down_points = points[unique_indices]

            N = down_points.shape[0]

            if N >= num_points:
                indices = np.random.choice(N, num_points, replace=False)
                down_points = down_points[indices]
            else:
                extra_indices = np.random.choice(N, num_points - N, replace=True)
                down_points = np.concatenate([down_points, down_points[extra_indices]], axis=0)

            return down_points
        
        points = voxel_down_sample_fixed(points, voxel_size=0.01)
        
        return points

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
        depth_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        # image_data = image(
        #     env=env,
        #     sensor_cfg=sensor_cfg,
        #     data_type=data_type,
        #     convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        #     normalize=False,  # we pre-process based on model
        # )
        sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

        # obtain the input image
        images = sensor.data.output[data_type]
        # store the device of the image
        image_device = images.device
        # forward the images through the model
        features = self._inference_fn(self._model, images, **(inference_kwargs or {}))
        # print("features shape before cat:",features.shape)

        depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]
        # print("depth shape:",depth.shape)
        depth_np = depth.squeeze(0).squeeze(-1).cpu().numpy()  # shape [H, W]

        batch_points = []  

        for i in range(depth_np.shape[0]): 

            depth_img = depth_np[i, :, :]

            # 归一化到 0~255
            depth_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            # print("depth_uint8 shape:",depth_uint8.shape)
            # 转点云
            points_i = self.depth_to_pointcloud(depth_uint8, self.fx, self.fy, self.cx, self.cy)
            # print("points_i shape:",points_i.shape)
            # 转成 tensor 并放到 device
            points_i = torch.tensor(points_i, dtype=torch.float32).to(image_device)  # [num_points,3]

            batch_points.append(points_i)

        # 现在 batch_points 是长度 num_envs 的列表，每个 [num_points_i,3]
        # print("num_envs:",len(batch_points))

        depth_features_list = []

        # import pdb
        # pdb.set_trace()

        with torch.no_grad():
            for points_i in batch_points:
                # PointNet 期望输入 [B, 3, N]

                pts_input = points_i.unsqueeze(0).permute(0,2,1).contiguous()  # [1,3,N]
                # print("pts_input shape:",pts_input.shape)
                dfeatures = self._point_encoder(pts_input)  # [1, feature_dim]
                depth_features_list.append(dfeatures.squeeze(0))  # [feature_dim]

        img_feat_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        depth_features_batch = torch.stack(depth_features_list, dim=0)
        pc_feat_norm = torch.nn.functional.normalize(depth_features_batch, p=2, dim=1) 
        features = torch.cat((img_feat_norm,pc_feat_norm),dim=-1)
        
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std
            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}
    
    def _prepare_pointnet_model(self) :
        import torch.nn as nn

        experiment_dir = '/home/roborock/IsaacLab'
        ckpt_path = f"{experiment_dir}/best_model.pth"

        classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()  

        # checkpoint = torch.load(ckpt_path, map_location='cuda')
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

        state_dict = checkpoint['model_state_dict']
       
        classifier.load_state_dict(state_dict, strict=False)
        # print("[INFO] Missing keys:", missing)
        # print("[INFO] Unexpected keys:", unexpected)

        classifier.eval()

        
        class PointNet2Encoder(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.normal_channel = True  
                self.sa1 = base_model.sa1
                self.sa2 = base_model.sa2
                self.sa3 = base_model.sa3

            def forward(self, xyz):
                B, _, _ = xyz.shape
                norm = None
                l1_xyz, l1_points = self.sa1(xyz, norm)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
                features = l3_points.view(B, 1024)
                return features

        self._point_encoder = PointNet2Encoder(classifier).cuda().eval()


# # 最基础版本 (处理速度快,效果差)
# class image_features(ManagerTermBase):
#     """RGB + Point Cloud feature extraction with ResNet and PointNet architecture."""

#     def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#         # Camera intrinsics
#         self.fx, self.fy = 140.0, 140.0
#         self.cx, self.cy = 200.0, 150.0

#         # Downsampling parameters
#         self.voxel_size = 0.01
#         self.num_points = 2048

#         ## PCD saving configuration
#         self.step_count = 0
#         self.save_interval = 100
#         self.save_pcd = True
#         self.pcd_save_dir = "./pcd_logs"
#         os.makedirs(self.pcd_save_dir, exist_ok=True)

#         # Debug configuration
#         self.debug_mode = True  # Set to False to disable debug prints
#         self.debug_interval = 50  # Print every N steps

#         # ResNet-style RGB encoder with skip connections
#         self.rgb_block1 = torch.nn.Sequential(
#             torch.nn.Linear(400 * 300 * 3, 512),
#             torch.nn.LayerNorm(512),
#             torch.nn.ReLU()
#         ).to(env.device)

#         self.rgb_block2 = torch.nn.Sequential(
#             torch.nn.Linear(512, 512),
#             torch.nn.LayerNorm(512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 512),
#             torch.nn.LayerNorm(512)
#         ).to(env.device)

#         self.rgb_block3 = torch.nn.Sequential(
#             torch.nn.Linear(512, 256),
#             torch.nn.LayerNorm(256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128)
#         ).to(env.device)

#         # PointNet-style PCD encoder
#         self.point_mlp = torch.nn.Sequential(
#             torch.nn.Linear(3, 64),
#             torch.nn.LayerNorm(64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 256)
#         ).to(env.device)

#         self.global_mlp = torch.nn.Sequential(
#             torch.nn.Linear(256, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128)
#         ).to(env.device)

#     def reset(self, env_ids: torch.Tensor | None = None):
#         pass

#     def debug_tensor(self, name: str, tensor: torch.Tensor):
#         """Print tensor statistics for debugging."""
#         if not self.debug_mode or self.step_count % self.debug_interval != 0:
#             return

#         print(f"\n[{name}]")
#         print(f"  Shape: {tensor.shape}")
#         print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
#         print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

#         # Check for issues
#         if torch.isnan(tensor).any():
#             print(f"  WARNING: Contains NaN!")
#         if torch.isinf(tensor).any():
#             print(f"  WARNING: Contains Inf!")
#         if tensor.std().item() > 10:
#             print(f"  WARNING: High variance (std > 10)!")
#         if tensor.abs().max().item() > 50:
#             print(f"  WARNING: Large values (|max| > 50)!")

#     def voxel_downsample_gpu(self, points: torch.Tensor) -> torch.Tensor:
#         """Voxel size grows exponentially with DEPTH (Z-coordinate)."""
#         device = points.device
#         N = points.shape[0]

#         if N == 0:
#             return torch.zeros((self.num_points, 3), device=device)

#         depths = points[:, 2:3]
#         base_voxel_size = 0.001
#         power = 2.3
#         depth_scale = 3.3

#         adaptive_voxel_sizes = base_voxel_size * torch.pow(1 + depths * depth_scale, power)
#         adaptive_voxel_sizes = torch.clamp(adaptive_voxel_sizes, min=0.002, max=0.15)

#         normalized_points = points / adaptive_voxel_sizes
#         voxel_indices = torch.floor(normalized_points).to(torch.int64)

#         min_idx = voxel_indices.min(dim=0)[0]
#         shifted_indices = voxel_indices - min_idx

#         voxel_hash = (shifted_indices[:, 0] * 1000000 + 
#                     shifted_indices[:, 1] * 1000 + 
#                     shifted_indices[:, 2])

#         unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)

#         M = len(unique_hashes)
#         first_occurrence = torch.full((M,), N, dtype=torch.long, device=device)
#         first_occurrence.scatter_reduce_(0, inverse_indices, 
#                                         torch.arange(N, device=device), 
#                                         reduce='min')

#         down_points = points[first_occurrence]

#         current_size = down_points.shape[0]
#         if current_size >= self.num_points:
#             down_depths = down_points[:, 2]
#             weights = 1.0 / (down_depths + 0.1)
#             weights = weights / weights.sum()
#             indices = torch.multinomial(weights, self.num_points, replacement=False)
#             down_points = down_points[indices]
#         else:
#             extra_indices = torch.randint(0, current_size, (self.num_points - current_size,), device=device)
#             down_points = torch.cat([down_points, down_points[extra_indices]], dim=0)

#         return down_points

#     def save_point_cloud(self, points: torch.Tensor, filename: str):
#         """Save point cloud to file using Open3D."""
#         points_np = points.cpu().numpy()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_np)
#         o3d.io.write_point_cloud(filename, pcd)

#     def __call__(
#         self,
#         env: ManagerBasedEnv,
#         sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
#         depth_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
#         data_type: str = "rgb",
#         convert_perspective_to_orthogonal: bool = False,
#         model_zoo_cfg: dict | None = None,
#         model_name: str = "resnet18",
#         model_device: str | None = None,
#         inference_kwargs: dict | None = None,
#     ) -> torch.Tensor:

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"\n{'='*60}")
#             print(f"DEBUG: Step {self.step_count}")
#             print(f"{'='*60}")

#         # Get sensors
#         rgb_sensor = env.scene.sensors[sensor_cfg.name]
#         depth_sensor = env.scene.sensors[depth_cfg.name]

#         # RGB: ResNet-style processing with skip connection
#         rgb_data = rgb_sensor.data.output[data_type].float() / 255.0
#         batch_size = rgb_data.shape[0]
#         rgb_flat = rgb_data.reshape(batch_size, -1)

#         self.debug_tensor("RGB Input (normalized)", rgb_flat)

#         x = self.rgb_block1(rgb_flat)  # [B, 512]
#         self.debug_tensor("RGB Block1", x)

#         residual = x
#         x = self.rgb_block2(x)  # [B, 512]
#         self.debug_tensor("RGB Block2 (before skip)", x)

#         x = torch.nn.functional.relu(x + residual)  # Skip connection
#         self.debug_tensor("RGB Block2 (after skip)", x)

#         rgb_features = self.rgb_block3(x)  # [B, 128]
#         self.debug_tensor("RGB Features (final)", rgb_features)

#         # Depth: Point cloud processing
#         depth_data = depth_sensor.data.output["distance_to_image_plane"]
#         depth_data = depth_data.squeeze(-1)

#         H, W = depth_data.shape[1], depth_data.shape[2]

#         v, u = torch.meshgrid(
#             torch.arange(H, device=depth_data.device, dtype=torch.float32),
#             torch.arange(W, device=depth_data.device, dtype=torch.float32),
#             indexing='ij'
#         )

#         pcd_features_list = []
#         for i in range(batch_size):
#             depth = depth_data[i]

#             depth = depth.clone()
#             depth[~torch.isfinite(depth)] = 0.0
#             depth = depth.clamp(min=0.0)

#             valid_mask = (depth > 0) & (depth < 5.0)

#             if valid_mask.sum() > 0:
#                 Z = depth[valid_mask]
#                 X = (u[valid_mask] - self.cx) * Z / self.fx
#                 Y = (v[valid_mask] - self.cy) * Z / self.fy

#                 points_3d = torch.stack([X, -Y, Z], dim=-1)

#                 if i == 0:  # Debug only first batch
#                     self.debug_tensor(f"Points3D (batch {i})", points_3d)

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     filename_orig = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_original.ply"
#                     )
#                     self.save_point_cloud(points_3d, filename_orig)

#                 downsampled_points = self.voxel_downsample_gpu(points_3d)

#                 if i == 0:
#                     self.debug_tensor(f"Downsampled Points (batch {i})", downsampled_points)

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     filename_down = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_downsampled.ply"
#                     )
#                     self.save_point_cloud(downsampled_points, filename_down)

#                 # PointNet: per-point features -> max pool -> global features
#                 point_features = self.point_mlp(downsampled_points)  # [2048, 256]

#                 if i == 0:
#                     self.debug_tensor(f"Point Features (batch {i})", point_features)

#                 global_feature = torch.max(point_features, dim=0)[0]  # [256]

#                 if i == 0:
#                     self.debug_tensor(f"Global Feature (batch {i})", global_feature)

#                 pcd_feat = self.global_mlp(global_feature)  # [128]

#                 if i == 0:
#                     self.debug_tensor(f"PCD Features (batch {i})", pcd_feat)
#             else:
#                 pcd_feat = torch.zeros(128, device=depth.device)

#             pcd_features_list.append(pcd_feat)

#         pcd_features = torch.stack(pcd_features_list)  # [B, 128]
#         self.debug_tensor("PCD Features (all batches)", pcd_features)

#         # Concatenate
#         features = torch.cat([rgb_features, pcd_features], dim=-1)  # [B, 256]
#         self.debug_tensor("Final Features (concatenated)", features)

#         if self.debug_mode and self.step_count % self.debug_interval == 0:
#             print(f"{'='*60}\n")

#         # Increment step counter
#         self.step_count += 1

#         features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

#         return features 


# # 中间版本CNN + MaxPooling
# class image_features(ManagerTermBase):
#     """Simple but effective RGB + Point Cloud feature extraction."""

#     def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#         # PCD saving configuration
#         self.step_count = 0
#         self.save_interval = 100
#         self.save_pcd = True  # Set to False to disable saving
#         self.pcd_save_dir = "./pcd_logs"
#         os.makedirs(self.pcd_save_dir, exist_ok=True)

#         # Camera intrinsics
#         self.fx, self.fy = 140.0, 140.0
#         self.cx, self.cy = 200.5, 150.5

#         # Downsampling parameters
#         self.voxel_size = 0.01
#         self.num_points = 2048

#         # RGB Network: Simple CNN instead of flattening
#         # Input: [B, 3, H, W] -> Output: [B, 128]
#         self.rgb_net = torch.nn.Sequential(
#             # 400x300 -> 200x150
#             torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             # 200x150 -> 100x75
#             torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             # 100x75 -> 50x37
#             torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             # 50x37 -> 25x18
#             torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 64, 1, 1]
#             torch.nn.Flatten(),  # -> [B, 64]
#             torch.nn.Linear(64, 128)
#         ).to(env.device)

#         # Point Cloud Network: PointNet-style with max pooling
#         # Process each point independently, then aggregate
#         self.pcd_net = torch.nn.Sequential(
#             torch.nn.Linear(3, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128)
#         ).to(env.device)

#     def reset(self, env_ids: torch.Tensor | None = None):
#         pass

#     def voxel_downsample_gpu(self, points: torch.Tensor) -> torch.Tensor:
#         """Voxel size grows exponentially with DEPTH (Z-coordinate)."""
#         device = points.device
#         N = points.shape[0]

#         if N == 0:
#             return torch.zeros((self.num_points, 3), device=device)

#         # Use Z-coordinate (depth) as the metric, not Euclidean distance!
#         depths = points[:, 2:3]  # [N, 1] - Z coordinate only

#         # Exponential growth based on depth
#         base_voxel_size = 0.001  # Very fine at close depth
#         power = 2.5              # Exponential growth rate
#         depth_scale = 4.0        # How quickly voxel size increases with depth

#         # Voxel size = base_size * (1 + depth * scale)^power
#         adaptive_voxel_sizes = base_voxel_size * torch.pow(1 + depths * depth_scale, power)

#         # Clamp to avoid extreme values
#         adaptive_voxel_sizes = torch.clamp(adaptive_voxel_sizes, min=0.002, max=0.15)

#         # For X and Y, use adaptive size; for Z, also use adaptive size
#         normalized_points = points / adaptive_voxel_sizes
#         voxel_indices = torch.floor(normalized_points).to(torch.int64)

#         # Handle negative indices
#         min_idx = voxel_indices.min(dim=0)[0]
#         shifted_indices = voxel_indices - min_idx

#         # Create unique hash for voxels
#         voxel_hash = (shifted_indices[:, 0] * 1000000 + 
#                     shifted_indices[:, 1] * 1000 + 
#                     shifted_indices[:, 2])

#         unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)

#         # Get first point from each voxel
#         M = len(unique_hashes)
#         first_occurrence = torch.full((M,), N, dtype=torch.long, device=device)
#         first_occurrence.scatter_reduce_(0, inverse_indices, 
#                                         torch.arange(N, device=device), 
#                                         reduce='min')

#         down_points = points[first_occurrence]

#         # print(f"After adaptive voxelization: {down_points.shape[0]} points (from {N})")

#         # Adjust to target size with DEPTH-based weighting
#         current_size = down_points.shape[0]
#         if current_size >= self.num_points:
#             # Prefer closer points (smaller Z)
#             down_depths = down_points[:, 2]
#             weights = 1.0 / (down_depths + 0.1)
#             weights = weights / weights.sum()
#             indices = torch.multinomial(weights, self.num_points, replacement=False)
#             down_points = down_points[indices]
#         else:
#             extra_indices = torch.randint(0, current_size, (self.num_points - current_size,), device=device)
#             down_points = torch.cat([down_points, down_points[extra_indices]], dim=0)

#         return down_points
        

#     def save_point_cloud(self, points: torch.Tensor, filename: str):
#         """Save point cloud to file using Open3D."""
#         # Convert torch tensor to numpy
#         points_np = points.cpu().numpy()

#         # Create Open3D point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_np)

#         # Save to file
#         o3d.io.write_point_cloud(filename, pcd)

#     def __call__(
#         self,
#         env: ManagerBasedEnv,
#         sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
#         depth_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
#         data_type: str = "rgb",
#         convert_perspective_to_orthogonal: bool = False,
#         model_zoo_cfg: dict | None = None,
#         model_name: str = "resnet18",
#         model_device: str | None = None,
#         inference_kwargs: dict | None = None,
#     ) -> torch.Tensor:

#         # Get sensors
#         rgb_sensor = env.scene.sensors[sensor_cfg.name]
#         depth_sensor = env.scene.sensors[depth_cfg.name]

#         # RGB Processing
#         # Input: [B, H, W, 3] -> [B, 3, H, W] for Conv2d
#         rgb_data = rgb_sensor.data.output[data_type].float() / 255.0
#         batch_size = rgb_data.shape[0]
#         rgb_data = rgb_data.permute(0, 3, 1, 2)  # [B, 3, H, W]
#         rgb_features = self.rgb_net(rgb_data)  # [B, 128]

#         # Depth Processing
#         depth_data = depth_sensor.data.output["distance_to_image_plane"]
#         depth_data = depth_data.squeeze(-1)  # [B, H, W]

#         H, W = depth_data.shape[1], depth_data.shape[2]

#         # Create pixel coordinate grid
#         v, u = torch.meshgrid(
#             torch.arange(H, device=depth_data.device, dtype=torch.float32),
#             torch.arange(W, device=depth_data.device, dtype=torch.float32),
#             indexing='ij'
#         )

#         pcd_features_list = []
#         for i in range(batch_size):
#             depth = depth_data[i]  # [H, W]

#             # Handle invalid depth
#             depth = depth.clone()
#             depth[~torch.isfinite(depth)] = 0.0
#             depth = depth.clamp(min=0.0)

#             # Valid mask
#             valid_mask = (depth > 0) & (depth < 5.0)

#             if valid_mask.sum() > 0:
#                 # Back-project to 3D
#                 Z = depth[valid_mask]
#                 X = (u[valid_mask] - self.cx) * Z / self.fx
#                 Y = (v[valid_mask] - self.cy) * Z / self.fy

#                 points_3d = torch.stack([X, -Y, Z], dim=-1)  # [N, 3]

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     # Save original PCD (before downsampling)
#                     filename_orig = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_original.ply"
#                     )
#                     self.save_point_cloud(points_3d, filename_orig)

#                 # Voxel downsample to 1024 points
#                 downsampled_points = self.voxel_downsample_gpu(points_3d)  # [1024, 3]

#                 if self.save_pcd and self.step_count % self.save_interval == 0 and i == 0:
#                     # Save downsampled PCD
#                     filename_down = os.path.join(
#                         self.pcd_save_dir, 
#                         f"step_{self.step_count}_env_{i}_downsampled.ply"
#                     )
#                     self.save_point_cloud(downsampled_points, filename_down)

#                 # PointNet-style: process each point, then max-pool
#                 point_features = self.pcd_net(downsampled_points)  # [1024, 128]
#                 pcd_feat = torch.max(point_features, dim=0)[0]  # [128] - global feature
#             else:
#                 pcd_feat = torch.zeros(128, device=depth.device)

#             pcd_features_list.append(pcd_feat)
#             # pcd_features_list.append(torch.zeros(128, device=depth.device))

#         pcd_features = torch.stack(pcd_features_list)  # [B, 128]

#         # Concatenate RGB and point cloud features
#         features = torch.cat([rgb_features, pcd_features], dim=-1)  # [B, 256]

#         self.step_count += 1

#         return features


"""
Actions.
"""

def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        # with open('output_5142.txt', 'a') as f:
        #     f.write(f"obs5 m: {(env.action_manager.action).mean().item()}\n")
        #     f.write(f"obs5 s: {(env.action_manager.action).std().item()}\n")
        # print("obs5 m:",(env.action_manager.action).mean().item())
        # print("obs5 s:",(env.action_manager.action).std().item())
        return env.action_manager.action
    else:
        # with open('output_5142.txt', 'a') as f:
        #     f.write(f"obs5 m: {(env.action_manager.get_term(action_name).raw_actions).mean().item()}\n")
        #     f.write(f"obs5 s: {(env.action_manager.get_term(action_name).raw_actions).std().item()}\n")
        # print("obs5 m:",(env.action_manager.get_term(action_name).raw_actions).mean().item())
        # print("obs5 s:",(env.action_manager.get_term(action_name).raw_actions).std().item())
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""

    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs4 m: {(env.command_manager.get_command(command_name)).mean().item()}\n")
    #     f.write(f"obs4 s: {(env.command_manager.get_command(command_name)).std().item()}\n")
    # print("obs4 m:",(env.command_manager.get_command(command_name)).mean().item())
    # print("obs4 s:",(env.command_manager.get_command(command_name)).std().item())
    return env.command_manager.get_command(command_name)
