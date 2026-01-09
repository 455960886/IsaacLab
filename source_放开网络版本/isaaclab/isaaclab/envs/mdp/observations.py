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
import torchvision
import os
import time
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
# from .gripper_transform import add_gripper_labels_to_observation, debug_gripper_transformation, transform_world_to_camera
import open3d as o3d
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    depth_cfg : SceneEntityCfg = SceneEntityCfg("tiled_camera2"),
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
    # depth image conversion
    # images = images[:, :, images.shape[2] // 2:, :]

    # obs_image = torch.tensor(images).float().squeeze(0).cpu().numpy()  # Convert to tensor and float type
    # obs_bgr = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
    # os.makedirs("IMAGES2", exist_ok=True)
    # # os.makedirs("IMAGES17", exist_ok=True)
    # time1 = time.time()
    # # cv2.imwrite(f"./IMAGES16/observation_{time1}.png", obs_image)
    # cv2.imwrite(f"./IMAGES2/observation_{time1}.png", obs_bgr)
    # with open('output_formres9.txt', 'a') as f:
    #     f.write(f"observation_{time1}.png\n")
    # print(f"./IMAGES2/observation_{time1}.png")
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
    # 深度图与RGB图拼接
    images = torch.cat((images,depth),dim=-1)
    # print("image shape22:",images.shape)
    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        """这里只做两件事：
        1. 保存相机内参（如果你要用，也可以不用）
        2. 准备一个 frame_counter，用来给 debug 保存图片/点云编号
        """
        super().__init__(cfg, env)

        # 如果你之后想用固定内参，可以留着（当前 depth_to_pointcloud_batch_gpu 已经从相机 K 里读） 
        self.fx, self.fy = 117.78, 124.95
        self.cx, self.cy = 200.0, 150.0

        # 计数器：用在 _apply_domain_randomization / 保存 debug 图
        self._frame_counter = 0

    def _apply_domain_randomization(
        self,
        images: torch.Tensor,
        save_debug: bool = False,
        step_counter: int = 0
    ) -> torch.Tensor:
        """
        Domain randomization: blur + noise.
        输出必须保持 float32 且范围 [0,1]，禁止转回 uint8，否则 ActorCritic clamp 会把图变白图。
        """
        # 记录输入 layout（B,H,W,C 或 B,C,H,W）
        was_channels_last = (images.ndim == 4 and images.shape[-1] in [1, 3, 4])

        # 1) 统一转 float32，并保证范围在 0~1
        #    - uint8: /255
        #    - float 但范围可能 0~255: 也要 /255
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.max() > 1.5:   # 保险：如果还是 0~255
                images = images / 255.0

        # 2) 统一转为 (B,C,H,W) 做 torchvision blur
        if was_channels_last:
            images = images.permute(0, 3, 1, 2).contiguous()

        if save_debug:
            self._save_images(images, step_counter, prefix="before_aug")

        # 3) blur
        kernel_size = int(torch.randint(3, 8, (1,), device=images.device).item())
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = torch.rand(1, device=images.device).item() * 1.5 + 0.5
        images = torchvision.transforms.functional.gaussian_blur(
            images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        )

        # 4) noise
        noise_std = torch.rand(1, device=images.device).item() * 0.03
        images = torch.clamp(images + torch.randn_like(images) * noise_std, 0.0, 1.0)

        if save_debug:
            self._save_images(images, step_counter, prefix="after_aug")

        # 5) 转回输入 layout（如果输入是 BHWC，就还回 BHWC）
        if was_channels_last:
            images = images.permute(0, 2, 3, 1).contiguous()

        # ✅ 关键：绝对不要转回 uint8
        return images

    def _save_images(self, images: torch.Tensor, step: int, prefix: str = "img"):
        """Save images to disk for debugging."""
        import os
        import cv2
        import numpy as np

        save_dir = "debug_augmentation"
        os.makedirs(save_dir, exist_ok=True)

        # Save only the first image in the batch
        img_to_save = images[0]  # Take first image from batch

        # Convert from torch tensor to numpy
        img_np = img_to_save.detach().cpu().numpy()

        # Check the shape and handle accordingly
        print(f"Image shape: {img_np.shape}")  # Debug print

        # If shape is (C, H, W), convert to (H, W, C)
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:  # Channel first
            img_np = np.transpose(img_np, (1, 2, 0))

        # Ensure values are in [0, 1] range, then convert to [0, 255]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        # Handle different channel counts
        if img_np.shape[-1] == 3:  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif img_np.shape[-1] == 1:  # Grayscale
            img_bgr = img_np.squeeze(-1)
        elif img_np.shape[-1] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            print(f"Warning: Unexpected channel count {img_np.shape[-1]}, saving as-is")
            img_bgr = img_np

        save_path = os.path.join(save_dir, f"{prefix}_step_{step:06d}.png")
        cv2.imwrite(save_path, img_bgr)

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        return

    def depth_to_pointcloud(self, depth_image, fx, fy, cx, cy, rgb_image=None, output_path="pointcloud.ply"):
        """
        将深度图转换为点云（可选带颜色）

        参数:
            depth_image : np.ndarray
                深度图（H, W），单位为米。
            fx, fy, cx, cy : float
                相机内参。
            rgb_image : np.ndarray, optional
                彩色图（H, W, 3），与深度图对齐。
            output_path : str
                点云保存路径。
        """
        assert len(depth_image.shape) == 2, "深度图必须是单通道 (H, W)"
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 深度图中无效值置0（避免NaN）
        depth = np.nan_to_num(depth_image, nan=0.0)
        # depth = (depth.max() - depth)
        # print(depth_image.dtype)
        # print("min, max, median:", np.nanmin(depth_image), np.nanmax(depth_image), np.nanmedian(depth_image))
        # print("non-zero fraction:", np.count_nonzero(~np.isnan(depth_image) & (depth_image!=0)) / depth_image.size)
        mask = depth > 0  # 有效深度

        # 反投影到3D空间
        Z = depth[mask]
        X = (u[mask] - cx) * Z / fx
        Y = (v[mask] - cy) * Z / fy
        points = np.stack((X, -Y, Z), axis=-1)

        def save_ply(points, colors=None, output_path="pointcloud.ply"):
            """
            保存点云为 PLY 文件
            参数:
                points: (N, 3) numpy 数组
                colors: (N, 3) numpy 数组 (0~255 或 0~1)
                output_path: 输出文件路径
            """
            # 创建 open3d 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if colors is not None:
                if colors.max() > 1.0:
                    colors = colors / 255.0  # 归一化到 [0,1]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # 保存为 PLY 文件
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"✅ 点云已保存到: {output_path}")
       
        # save_ply(points, colors=None, output_path=output_path.replace(".ply","_0.ply"))
        # theta = np.deg2rad(0.5)
        # R_x = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(theta), -np.sin(theta)],
        #     [0, np.sin(theta),  np.cos(theta)]
        # ])

        # rotated_points = points @ R_x.T
        # save_ply(rotated_points, colors=None, output_path=output_path.replace(".ply","_1.ply"))
        # save_ply(points, colors=None, output_path=output_path)
        # ===== 距离筛选部分 =====
        # points = rotated_points[rotated_points[:,2]<0.16]
        # points = points[points[:,1]>-0.05]

        def voxel_down_sample_fixed(points, voxel_size=2.0, num_points=1024, seed=None):
            """
            对点云进行体素下采样，并确保输出固定数量的点。

            参数:
                points: np.ndarray, shape [N, 3]
                voxel_size: float, 体素大小
                num_points: int, 输出固定点数
                seed: int or None, 随机种子（可选）

            返回:
                down_points: np.ndarray, shape [num_points, 3]
            """
            if len(points) == 0:
                # 返回一个全零点云（或可选 raise）
                return np.zeros((num_points, 3), dtype=np.float32)

            if seed is not None:
                np.random.seed(seed)

            decay_rate = voxel_size / 2.0
            dist = np.linalg.norm(points, axis=1)
            p = np.exp(-dist / decay_rate)   # 近处概率大
            p /= p.sum()

            # ✅ 修复点：若点数不足，则允许放回采样
            replace_flag = len(points) < num_points
            indices = np.random.choice(len(points), num_points, replace=replace_flag, p=p)
            down_points = points[indices]

            # ✅ 第二步其实可以省略，但如果你想保持逻辑清晰：
            N = down_points.shape[0]
            if N < num_points:
                extra_indices = np.random.choice(N, num_points - N, replace=True)
                down_points = np.concatenate([down_points, down_points[extra_indices]], axis=0)

            return down_points

        downsampled_points = voxel_down_sample_fixed(trans_points, voxel_size=2.0)
        # save_ply(points, colors=None, output_path=output_path.replace(".ply","_downsampled8.ply"))
        if save_ply_debug:
            points_downsampled = downsampled_points[env_id][mask[env_id]].cpu().numpy()
            save_ply(points_filtered, "3_downsampled")
        return downsampled_points

    # GPU-accelerated version for batch processing
    def depth_to_pointcloud_batch_gpu(
        self,
        depth_batch,                      # (B,H,W)
        fx, fy, cx, cy,
        seg_batch=None,                   # <--- 新增：语义分割 (B,H,W) 或 (B,H,W,1)
        num_points=1024,
        save_ply_debug=True,
        env_id=0,
        frame_counter=None,
        save_dir="debug_pointclouds",
        env=None,
    ):
        """GPU-accelerated batch point cloud generation with optional semantic IDs."""
        B, H, W = depth_batch.shape
        device = depth_batch.device

        # 如果传进来的 seg 是 (B,H,W,1)，先 squeeze 掉最后一维
        if seg_batch is not None:
            if seg_batch.ndim == 4 and seg_batch.shape[-1] == 1:
                seg_batch = seg_batch[..., 0]
            # seg_flat: (B, H*W)
            seg_flat = seg_batch.reshape(B, H * W)

        if save_ply_debug:
            env_dir = os.path.join(save_dir, f"env_{env_id}")
            os.makedirs(env_dir, exist_ok=True)
            if frame_counter is not None:
                prefix = f"frame_{frame_counter:06d}"
            else:
                prefix = f"time_{int(time.time() * 1000)}"

        def save_ply(points_np, stage_name):
            if save_ply_debug and env_id < B:
                filepath = os.path.join(env_dir, f"{prefix}_{stage_name}.ply")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                o3d.io.write_point_cloud(filepath, pcd)
                print(f"✅ Saved: {filepath}")

        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v, u = torch.meshgrid(v_coords, u_coords, indexing='ij')

        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)

        Z = depth_batch
        X = -(u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = torch.stack([X, -Y, Z], dim=-1)        # (B,H,W,3)

        # points_flat = points.reshape(B, H * W, 3)       # (B,H*W,3)

        # theta = torch.deg2rad(torch.tensor(0.5, device=device))
        # cos_theta = torch.cos(theta)
        # sin_theta = torch.sin(theta)
        # R_x = torch.tensor(
        #     [[1, 0, 0],
        #     [0, cos_theta, -sin_theta],
        #     [0, sin_theta,  cos_theta]],
        #     device=device, dtype=torch.float32,
        # )
        def deg2rad(v, device):
            return torch.tensor(v, device=device) * torch.pi / 180.0
        roll = deg2rad(90.0, device)
        pitch = deg2rad(0.0, device)
        yaw = deg2rad(90.0, device)
        c1, s1 = torch.cos(roll), torch.sin(roll)
        c2, s2 = torch.cos(pitch), torch.sin(pitch)
        c3, s3 = torch.cos(yaw), torch.sin(yaw)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, c1, -s1],
            [0, s1, c1]
        ], device=device, dtype=torch.float32)
        Ry = torch.tensor([
            [c2, 0, s2],
            [0, 1, 0],
            [-s2, 0, c2]
        ], device=device, dtype=torch.float32)
        Rz = torch.tensor([
            [c3, -s3, 0],
            [s3, c3, 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        # -----------------------------
        # 构造 4x4 transformation 矩阵 T
        # -----------------------------
        x1 = deg2rad(0.011, device)
        c = torch.cos(x1)
        s = torch.sin(x1)
        Rx1 = torch.tensor([
            [1., 0., 0.],
            [0., c, -s],
            [0., s, c],
        ], device=device)
        R = Rx1 @ Rz @ Ry @ Rx
        # import pdb
        # pdb.set_trace()
        points_flat = points.reshape(B, H * W, 3)
        rotated_points = torch.matmul(points_flat, R.T)
        translation = torch.tensor([0.1654, 0.0, 0.049], device=device)
        trans_points = rotated_points + translation
        # Save Stage 1: After rotation
        if save_ply_debug:
            points_trans = trans_points[env_id].cpu().numpy()
            save_ply(points_trans, "1_rotated")
        # Apply distance filtering
        # mask1 = rotated_points[:, :, 2] < 0.21
        # mask2 = rotated_points[:, :, 1] > -0.0628
        # mask3 = rotated_points[:, :, 1] < 0.0428
        
        # mask1 = trans_points[:,:, 0] >=0.35
        mask2 = trans_points[:,:, 0] <= 0.42
        mask3 = trans_points[:,:, 2] >= 0.00

        mask = mask2 & mask3
        # Save Stage 2: After filtering
        if save_ply_debug:
            points_filtered = trans_points[env_id][mask[env_id]].cpu().numpy()
            save_ply(points_filtered, "2_filtered")

        # 几何过滤 mask: (B,H*W)
        # z = rotated_points[:, :, 2]
        # y = rotated_points[:, :, 1]
        # mask1 = z < 0.21
        # mask2 = y > -0.06
        # mask3 = y < 0.0428
        # mask = mask1 & mask2 & mask3           # (B,H*W)

        # if save_ply_debug:
        #     points_filtered = rotated_points[env_id][mask[env_id]].cpu().numpy()
        #     save_ply(points_filtered, "2_filtered")

        sampled_points_list = []
        sampled_semantic_list = [] if seg_batch is not None else None

        for b in range(B):
            # 这一行是关键：取出满足几何 mask 的索引
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)   # (M,)
            if valid_idx.numel() == 0:
                # 没有有效点：全 0 填充
                sampled_points = torch.zeros(num_points, 3, device=device)
                sampled_points_list.append(sampled_points)
                if sampled_semantic_list is not None:
                    sampled_semantic = torch.zeros(num_points, dtype=seg_flat.dtype, device=device)
                    sampled_semantic_list.append(sampled_semantic)
                continue

            valid_points = trans_points[b][valid_idx]         # (M,3)

            if seg_batch is not None:
                valid_semantic = seg_flat[b][valid_idx]         # (M,)

            M = valid_points.shape[0]
            if M >= num_points:
                dist = torch.norm(valid_points, dim=1)
                weights = torch.exp(-dist / 1.0)
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, num_points, replacement=False)
            else:
                indices = torch.randint(0, M, (num_points,), device=device)

            sampled_points = valid_points[indices]              # (num_points,3)
            sampled_points_list.append(sampled_points)

            if seg_batch is not None:
                sampled_semantic = valid_semantic[indices]      # (num_points,)
                sampled_semantic_list.append(sampled_semantic)

        result_points = torch.stack(sampled_points_list, dim=0)  # (B,N,3)

        if save_ply_debug:
            save_ply(result_points[env_id].cpu().numpy(), "3_downsampled")

        if seg_batch is not None:
            result_semantic = torch.stack(sampled_semantic_list, dim=0)   # (B,N)
            return result_points, result_semantic
        else:
            return result_points

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        depth_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera2"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
        save_augmentation_debug: bool = False,
    ) -> torch.Tensor:
        """
        从相机拿「原始图像 + 原始点云」，拉平成 1D 向量后返回。

        输出格式：
            obs_visual: [B, C*H*W + 3*N]
            - 前 C*H*W 维：按 (C,H,W) 的顺序 flatten 的图像
            - 后 3*N 维：按 (3,N) 的顺序 flatten 的点云 (x,y,z)
        """
        # ===== 1. 读取 RGB 图像，并做 domain randomization =====
        sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
        images = sensor.data.output[data_type]          # 通常是 [B, H, W, C]，dtype=uint8
        images = images[:, 120:, :, :]
        # 做 sim-to-real 的模糊 + 噪声增强（你之前写好的函数）
        # images = self._apply_domain_randomization(
        #     images,
        #     save_debug=save_augmentation_debug,
        #     step_counter=self._frame_counter,
        # )
        # ===== 强保险：确保 image 一定是 float32 且 0~1 =====
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.max() > 1.5:
                images = images / 255.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # ===================== DEBUG (first 5 calls, rank0 only) =====================
        def _rank0():
            try:
                import torch.distributed as dist
                return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
            except Exception:
                return True

        if _rank0() and self._frame_counter <= 5:
            # images 仍可能是 uint8 或 float；只看 batch 的第 0 张，避免开销
            img0 = images[0]
            # 为了统计统一转 float
            img0f = img0.float()
            print(
                f"[OBS_DBG] frame={self._frame_counter:02d} "
                f"after_DR dtype={images.dtype} shape={tuple(images.shape)} "
                f"min={img0f.min().item():.4f} max={img0f.max().item():.4f} "
                f"mean={img0f.mean().item():.4f} std={img0f.std().item():.4f}"
            )
        # =============================================================================
        self._frame_counter += 1

        device = images.device
        B = images.shape[0]

        # 确保变成 (B, C, H, W) 再 flatten
        if images.ndim == 4 and images.shape[-1] in [1, 3, 4]:
            # 原本是 (B, H, W, C) -> (B, C, H, W)
            images_chw = images.permute(0, 3, 1, 2).contiguous()
        else:
            # 已经是 (B, C, H, W)
            images_chw = images

        # ====== ADD: resize for speed (do this BEFORE flatten) ======
        target_h = 128
        # 方案A：保持宽高比（推荐）：按当前 H/W 自动算 target_w
        target_w = int(round(target_h * images_chw.shape[-1] / images_chw.shape[-2]))  # e.g. 360x640 -> 128x228
        images_chw = F.interpolate(images_chw, size=(target_h, target_w), mode="bilinear", align_corners=False)

        _, C, H, W = images_chw.shape

        # 按 (C,H,W) 顺序 flatten，每个环境一行
        img_flat = images_chw.view(B, -1)   # [B, C*H*W]

        # ===== 2. 读取深度图，生成点云（不依赖语义）=====
        depth_sensor = env.scene.sensors[depth_cfg.name]
        depth = depth_sensor.data.output["distance_to_image_plane"]   # [B,H,W,1] or [B,H,W]
        depth_tensor = depth.squeeze(-1)  # [B,H,W]

        # 从 depth_cfg 对应相机读取 K（别硬编码 "depth_camera"）
        cam = depth_sensor
        K = cam._data.intrinsic_matrices[0]
        fx = K[0][0]; fy = K[1][1]; cx = K[0][2]; cy = K[1][2]

        # 只返回点云： (B, N, 3)
        batch_points_tensor = self.depth_to_pointcloud_batch_gpu(
            depth_tensor,
            fx, fy, cx, cy,
            seg_batch=None,          # 关键：不传语义
            num_points=1024,
            save_ply_debug=False,
            env_id=0,
            frame_counter=self._frame_counter,
            save_dir="debug_pointclouds",
            env=env,
        )

        # cache：只保留 point cloud
        env.point_cloud_cache = batch_points_tensor.detach()
        env.point_cloud_semantic_cache = None   # 防止下游误用
        env._pcd_cache_step = env.common_step_counter

        # ===== 3. 把点云整理成 (B, 3, N) 再 flatten =====
        # depth_to_pointcloud_batch_gpu 返回的是 (B, N, 3)
        # 我们转成 (B, 3, N)，再按 (3,N) 顺序 flatten，和 ActorCritic 那边保持一致
        pcd_chw = batch_points_tensor.permute(0, 2, 1).contiguous()   # [B, 3, N]
        B_p, C_p, N_p = pcd_chw.shape
        assert B_p == B, "batch size 不一致，点云 B 和 图像 B 要相同！"

        pcd_flat = pcd_chw.view(B, -1)   # [B, 3*N]

        # ===== 4. 拼接图像向量 + 点云向量 =====
        # 最终的视觉 obs:  [ image_flat | pointcloud_flat ]
        obs_visual = torch.cat([img_flat, pcd_flat], dim=1)   # [B, C*H*W + 3*N]

        # 🙋‍♀️ 第一次可以打印一下形状帮你确认
        if not hasattr(self, "_printed_shape"):
            self._printed_shape = True
            print("[image_features] images_chw shape:", images_chw.shape)          # (B, C, H, W)
            print("[image_features] batch_points_tensor shape:", batch_points_tensor.shape)  # (B, N, 3)
            print("[image_features] obs_visual dim:", obs_visual.shape[1])
        
        env.visual_obs_info = {
            "img_shape": (C, H, W),   # 来自前面 images_chw.shape
            "pcd_points": N_p,        # depth_to_pointcloud_batch_gpu 采样的点数（例如 1024）
        }

        return obs_visual.to(device)


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
            # features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            # return features.last_hidden_state[:, 1:]
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
            model = torch.nn.Sequential(*list(model.children())[:-1])
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
            feats = model(image_proc)          # [N, 512, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [N, 512]
            return feats
            # forward the image through the model
            # return model(image_proc)
        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}
    
    def _prepare_pointnet_model(self) :
        import torch.nn as nn
        # experiment_dir = '/home/roborock/IsaacLab' 
        # classifier = MODEL.get_model(13).cuda()
        # checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        # classifier = classifier.eval()

        # self._point_encoder = classifier.feat
        # self._point_encoder.eval()
        # self._point_encoder.cuda()

        experiment_dir = '/home/roborock/data/private/shengmei/IsaacLab'
        ckpt_path = f"{experiment_dir}/best_model.pth"

        # ✅ 模型输入通道：原模型是 normal_channel=True（6 通道）
        classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()  

        # ✅ 加载 checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

        # 拿出权重字典
        state_dict = checkpoint['model_state_dict']

        # # ✅ 动态修正输入通道权重 mismatch（从 6 -> 3）
        # for key in list(state_dict.keys()):
        #     if 'sa1' in key and 'weight' in key and state_dict[key].dim() == 4:
        #         if state_dict[key].shape[1] == 6:
        #             print(f"[INFO] Trimming {key} from 6→3 input channels.")
        #             state_dict[key] = state_dict[key][:, :3, :, :]  # 截取前3个通道 (XYZ)

        # # ✅ 忽略分类头不匹配部分
        # ignore_keys = ['fc3.weight', 'fc3.bias']
        # for k in ignore_keys:
        #     if k in state_dict:
        #         print(f"[INFO] Removing {k} from checkpoint.")
        #         del state_dict[k]

        # ✅ 加载修正后的权重
        classifier.load_state_dict(state_dict, strict=False)
        # print("[INFO] Missing keys:", missing)
        # print("[INFO] Unexpected keys:", unexpected)

        classifier.eval()

        # ✅ 仅保留特征提取部分（encoder）
        class PointNet2Encoder(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.normal_channel = True  # 我们只输入 XYZ
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